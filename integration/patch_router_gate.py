#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pure pinned-source K1v3 transform; no writes, imports of vLLM or GPU work."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

MODEL = Path('models/glm5next/nvidia/model.py')
RUNNER = Path('model_executor/layers/fused_moe/runner/moe_runner.py')
HASHES = {
    str(MODEL): '2e0bb44a38a96cc788e6bb23bd819b1c642b16aedd0a5fe5e0fec9e7fcfec312',
    str(RUNNER): '80d4b53fdbbbaf2aa9408f22ea476bacbdbb9db5cef0e51fca62c36a4f384d25',
}
COMPANIONS = {'model_executor/layers/fused_moe/layer.py': 'ebdc4af2f553ce2a70e666f9da3ac647c79d538680c284ce98dd4ac64785661b'}
MODEL_OLD = '''        # The router is always external (self.gate); main's MoERunner expects
        # pre-computed router_logits, so compute them here unconditionally.
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )'''
MODEL_NEW = '''        # K1V3: keep the registered gate and let the runner evaluate it once,
        # after the shared-expert auxiliary stream has received input readiness.
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=None
        )'''
PUBLIC_ANCHOR = '        # Apply transform for routed experts (e.g., latent projection for\n'
PUBLIC_GUARD = '''        # K1V3: reject missing logits before transforms or device work.
        if router_logits is None and self.gate is None:
            raise ValueError("router_logits=None requires a runner-owned gate")

'''
IMPL_ANCHOR = '        # TODO(bnell): this can be removed after MK migration is complete.\n'
IMPL_GUARD = '''        # K1V3: custom-op entry can bypass the public forward validation.
        if router_logits is None and self.gate is None:
            raise ValueError("router_logits=None requires a runner-owned gate")

'''
GATE_END = '                router_logits, _ = self.gate(hidden_states)\n\n'
GATE_CHECK = '''        # K1V3: subsequent routing/dispatch still requires an actual tensor.
        if router_logits is None:
            raise RuntimeError("Runner-owned gate returned no router logits")

'''
TARGETS = {'_moe_forward','_moe_forward_fake','_moe_forward_shared','_moe_forward_shared_fake'}


def digest(raw): return hashlib.sha256(raw).hexdigest()


def annotate_router_arguments(source, optional):
    tree = ast.parse(source); nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in TARGETS: nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == 'MoERunner':
            nodes.extend(child for child in node.body if isinstance(child, ast.FunctionDef)
                         and child.name in ('forward','_forward_impl'))
    if len(nodes) != 6: raise ValueError('Expected six pinned op/fake/runner entry points')
    lines=source.splitlines(keepends=True); starts=[]; offset=0
    for line in lines: starts.append(offset); offset+=len(line)
    edits=[]
    expected='torch.Tensor' if optional else 'torch.Tensor | None'
    replacement='torch.Tensor | None' if optional else 'torch.Tensor'
    for node in nodes:
        argument=next(arg for arg in node.args.args if arg.arg=='router_logits')
        if ast.get_source_segment(source,argument.annotation)!=expected:
            raise ValueError(f'Unexpected router annotation: {node.name}')
        ann=argument.annotation
        edits.append((starts[ann.lineno-1]+ann.col_offset,
                      starts[ann.end_lineno-1]+ann.end_col_offset,replacement))
    for start,end,text in sorted(edits,reverse=True): source=source[:start]+text+source[end:]
    return source


def _change(name, source, reverse=False):
    if name==str(MODEL):
        old,new=(MODEL_NEW,MODEL_OLD) if reverse else (MODEL_OLD,MODEL_NEW)
        if source.count(old)!=1: raise ValueError('GLM caller source drift')
        return source.replace(old,new,1)
    if name!=str(RUNNER): raise ValueError(name)
    if reverse:
        for text in (PUBLIC_GUARD,IMPL_GUARD,GATE_CHECK):
            if source.count(text)!=1: raise ValueError('K1v3 guard source drift')
            source=source.replace(text,'',1)
        return annotate_router_arguments(source,optional=False)
    source=annotate_router_arguments(source,optional=True)
    for anchor,insert in ((PUBLIC_ANCHOR,PUBLIC_GUARD),(IMPL_ANCHOR,IMPL_GUARD)):
        if source.count(anchor)!=1: raise ValueError('Pinned validation anchor drift')
        source=source.replace(anchor,insert+anchor,1)
    if source.count(GATE_END)!=1: raise ValueError('Pinned internal gate anchor drift')
    return source.replace(GATE_END,GATE_END+GATE_CHECK,1)


def transform(name, source):
    name=str(name)
    if digest(source.encode())==HASHES[name]:
        candidate=_change(name,source)
        ast.parse(candidate)
        return candidate,'baseline'
    restored=_change(name,source,reverse=True)
    if digest(restored.encode())!=HASHES[name]: raise ValueError(f'Pinned source drift: {name}')
    ast.parse(source)
    return source,'candidate'


def verify_tree(root):
    for name,expected in COMPANIONS.items():
        if digest((root/name).read_bytes())!=expected: raise ValueError(f'Companion drift: {name}')

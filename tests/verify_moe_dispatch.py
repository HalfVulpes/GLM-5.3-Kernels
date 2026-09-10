#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU spy test of real MarlinExperts.apply -> patched fused_marlin_moe dispatch."""
import argparse
import ast
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace


def main(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
    from patch_moe_align import patch_source
    source = args.source_root / "model_executor/layers/fused_moe/experts/marlin_moe.py"
    original = source.read_text()
    patched = patch_source(original)
    assert patch_source(patched) == patched
    tree = ast.parse(patched)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fused_marlin_moe')
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MarlinExperts')
    apply = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'apply')
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'fused_marlin_moe'
               for n in ast.walk(apply))

    class Tensor:
        def __init__(self, *shape, dtype='bf16', contiguous=True, cuda=True):
            self.shape = shape; self.dtype = dtype; self.ndim = len(shape)
            self._contiguous = contiguous; self.is_cuda = cuda; self.device = 'cuda:0'
        def size(self, axis=None): return self.shape if axis is None else self.shape[axis]
        def numel(self): return math.prod(self.shape)
        def is_contiguous(self): return self._contiguous
    class ReachedGemm(Exception): pass
    routes = []
    def fast(*args, **kwargs):
        routes.append(('titan', kwargs.get('num_warps'))); return (None, None, None)
    def baseline(*args, **kwargs):
        routes.append(('baseline', None)); return (None, None, None)
    def stop_at_gemm(**kwargs): raise ReachedGemm()
    scalar = SimpleNamespace(uint4='u4', uint8b128='u8', uint4b8='u4b8',
                             float8_e4m3fn='f8', float4_e2m1f='f4')
    env = SimpleNamespace(VLLM_DETERMINISTIC_MOE_ALIGN=True)
    namespace = {'torch': SimpleNamespace(float16='fp16', bfloat16='bf16', float32='fp32', int32='i32', int64='i64'),
        'math': math, 'envs': env, 'scalar_types': scalar,
        'ScalarType': SimpleNamespace(from_id=lambda x: x),
        'MoEActivation': SimpleNamespace(SILU='silu'),
        'marlin_moe_intermediate_size': lambda w1, w2: w2.size(1) * 16,
        '_titan_moe_align_small': fast, 'moe_align_block_size': baseline,
        '_fused_marlin_moe': stop_at_gemm}
    for node in (function, apply):
        # Keep the installed function body intact; annotations are deferred.
        exec(compile('from __future__ import annotations\n' + ast.unparse(node),
                     str(source), 'exec'), namespace)
    checks = []
    def check(name, batch=8, experts=288, topk=8, hidden=4096, intermediate=512,
              enabled=True, dtype='bf16', input_dtype=None, expert_map=None,
              scale_groups=32, expected='baseline', expected_warps=None):
        env.VLLM_DETERMINISTIC_MOE_ALIGN = enabled
        self = SimpleNamespace(_lora_context=None, w1_bias=None, w2_bias=None,
            w1_scale=Tensor(experts, hidden // scale_groups, intermediate * 2),
            w2_scale=Tensor(experts, intermediate // scale_groups, hidden),
            g1_alphas=None, g2_alphas=None, a1_gscale=None, a2_gscale=None,
            w1_zp=None, w2_zp=None, quant_type_id=scalar.uint4b8,
            activation=lambda *a, **k: None, activation_config=None,
            moe_sum=lambda *a, **k: None, w13_g_idx=None, w2_g_idx=None,
            w13_g_idx_sort_indices=None, w2_g_idx_sort_indices=None,
            is_k_full=True, input_dtype=input_dtype, marlin_workspace=lambda device: None)
        weights1 = Tensor(experts, hidden // 16, intermediate * 4)
        weights2 = Tensor(experts, intermediate // 16, hidden * 2)
        routes.clear()
        try:
            namespace['apply'](self, output=None, hidden_states=Tensor(batch, hidden, dtype=dtype),
                w1=weights1, w2=weights2, topk_weights=Tensor(batch, topk, dtype='fp32'),
                topk_ids=Tensor(batch, topk, dtype='i32'), activation='silu',
                global_num_experts=experts, expert_map=expert_map,
                a1q_scale=None, a2_scale=None, workspace13=None, workspace2=None,
                expert_tokens_meta=None, apply_router_weight_on_input=False)
        except ReachedGemm: pass
        assert routes == [(expected, expected_warps)], (name, routes)
        checks.append({'case': name, 'route': expected, 'warps': expected_warps, 'passed': True})
    for batch in range(1, 9):
        check(f'Titan_B{batch}', batch=batch, expected='titan', expected_warps=1 if batch <= 2 else 4)
    check('B9_unchanged', batch=9)
    check('prefill_unchanged', batch=2048)
    check('deterministic_off_unchanged', enabled=False)
    check('other_expert_count', experts=256)
    check('other_topk', topk=4)
    check('other_hidden_size', hidden=2048)
    check('other_TP_intermediate', intermediate=1024)
    check('other_quant_groups', scale_groups=64)
    check('fp16_unchanged', dtype='fp16')
    check('activation_int8_unchanged', input_dtype=SimpleNamespace(itemsize=1))
    check('expert_parallel_map_unchanged', expert_map=object())
    print(json.dumps({'passed': True, 'scope': 'CPU call-chain spy; no Torch/GPU imports',
                      'caller': 'MarlinExperts.apply -> fused_marlin_moe -> selected aligner',
                      'cases': checks}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True, help='Pinned vLLM package source tree; read only')
    main(p.parse_args())

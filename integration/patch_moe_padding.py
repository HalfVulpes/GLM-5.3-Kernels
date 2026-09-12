# SPDX-License-Identifier: Apache-2.0
"""Canonicalize inactive MoE rows on the exact tested K1v3 source."""
import ast, hashlib
K1V3_SHA='3d1ee69e71ec0594d08cdf90cc292641825bfae204580a99bc74276e2811d87b'
def apply_candidate(source):
 if hashlib.sha256(source.encode()).hexdigest()!=K1V3_SHA:raise ValueError('Expected exact K1v3 source')
 anchor='        # K1V3: keep the registered gate and let the runner evaluate it once,\n'
 if source.count(anchor)!=1:raise ValueError('Missing pinned MoE anchor')
 source=source.replace('import torch\n','import torch\nimport vllm.envs as envs\nfrom vllm.forward_context import get_forward_context, is_forward_context_available\n',1)
 source=source.replace(anchor,'''        # Canonical inactive rows before router/Marlin consume the padded buffer.
        if envs.VLLM_MOE_SKIP_PADDING and not self.is_sequence_parallel and is_forward_context_available():
            padding = get_forward_context().is_padding
            if padding is not None:
                hidden_states.masked_fill_(padding[:hidden_states.shape[0], None], 0)

'''+anchor,1)
 ast.parse(source);return source

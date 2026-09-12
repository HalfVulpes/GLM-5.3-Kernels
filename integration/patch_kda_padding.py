# SPDX-License-Identifier: Apache-2.0
"""Pinned, pure source transform for deterministic inactive KDA output rows."""
import ast
import hashlib
BASELINE_SHA256='be510e17c1bd0f412a3fefd66b8372e3334233f462b4f447154fe9bf453dfd1f'
TARGET='models/glm5next/nvidia/kda.py'
def apply_candidate(source):
    if hashlib.sha256(source.encode()).hexdigest()!=BASELINE_SHA256:
        raise ValueError('Pinned model KDA source mismatch')
    anchor='        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens\n'
    if source.count(anchor)!=1:raise ValueError('Actual-token metadata anchor mismatch')
    after=source.replace(anchor,anchor+'        # Deterministic inactive rows: downstream norm/projection sees padded buffers.\n        if core_attn_out.shape[1] > num_actual_tokens:\n            core_attn_out[:, num_actual_tokens:].zero_()\n',1)
    ast.parse(after)
    return after

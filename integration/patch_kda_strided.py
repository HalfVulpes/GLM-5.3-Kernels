# SPDX-License-Identifier: Apache-2.0
"""Pure, composable pinned-source transformer for K2. No writes on import."""
from __future__ import annotations

import ast

from kda_strided_contract import (
    BASE_LAUNCHER_SHA256, BASE_WRAPPER_SHA256, WRAPPER_NAME,
    function_source, require_hash,
)

MARKER = "TITAN_KDA_STRIDED_K2_V1"
INSERT = '''    # TITAN_KDA_STRIDED_K2_V1: preserve the upstream fallback verbatim.
    # B1 and contiguous inputs do not pay packing costs. Keep their original
    # path without importing or evaluating the more expensive storage guard.
    if (q.ndim == 4 and 2 <= q.shape[1] <= 8 and q.stride(1) == 6416
            and k.ndim == 4 and k.stride(1) == 6416
            and v.ndim == 4 and v.stride(1) == 6416
            and beta.ndim == 3 and beta.stride(1) == 6416):
        from titan_kda_strided import maybe_kda_strided
        candidate = maybe_kda_strided(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=initial_state, inplace_final_state=inplace_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens, out=out,
            sigmoid_beta=sigmoid_beta, a_log=a_log, g_bias=g_bias,
            compute_gate=compute_gate, lower_bound=lower_bound,
        )
        if candidate is not None:
            return candidate

'''
ANCHOR = "    o, final_state = fused_recurrent_kda_fwd(\n"


def apply_candidate(source: str) -> str:
    """Patch only the recurrent wrapper; allow unrelated norm-function edits."""
    _, launcher = function_source(source, "fused_recurrent_kda_fwd")
    require_hash(launcher, BASE_LAUNCHER_SHA256, "KDA launcher")
    node, wrapper = function_source(source, WRAPPER_NAME)
    if MARKER in wrapper:
        if wrapper.count(INSERT) != 1:
            raise ValueError("Existing K2 insertion differs from this candidate")
        original = wrapper.replace(INSERT, "", 1)
        require_hash(original, BASE_WRAPPER_SHA256, "restored KDA wrapper")
        return source
    if MARKER in source:
        raise ValueError("K2 marker exists outside the expected function")
    require_hash(wrapper, BASE_WRAPPER_SHA256, "KDA wrapper")
    if wrapper.count(ANCHOR) != 1:
        raise ValueError("Expected one recurrent launch anchor")
    changed = wrapper.replace(ANCHOR, INSERT + ANCHOR, 1)
    lines = source.splitlines(keepends=True)
    result = "".join(lines[:node.lineno - 1]) + changed + "\n" + "".join(lines[node.end_lineno:])
    ast.parse(result)
    return result

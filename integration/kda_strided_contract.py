"""Pinned-source derivation for the K2 strided KDA experiment; CPU only."""
from __future__ import annotations

import ast
import hashlib

BASE_KERNEL_SHA256 = "085c1e75103c53488471b05ec888dacd9172844d78637dfe09de9f51d58e1366"
BASE_WRAPPER_SHA256 = "3992851be8834b23a72c53dd9d1df345cc3a92bd3a6312da0efade9ab497ddb0"
BASE_LAUNCHER_SHA256 = "f05630a9ea06f42c68045f3a7141c73a9977342e1f7c5b19e3e8779c8b7cad4b"
KERNEL_NAME = "fused_recurrent_gated_delta_rule_fwd_kernel"
WRAPPER_NAME = "fused_recurrent_kda"


def function_source(source: str, name: str) -> tuple[ast.FunctionDef, str]:
    matches = [n for n in ast.parse(source).body
               if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one top-level function {name}, got {len(matches)}")
    node = matches[0]
    return node, ast.get_source_segment(source, node)


def require_hash(source: str, expected: str, label: str) -> None:
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != expected:
        raise ValueError(f"Pinned {label} source drift: {actual} != {expected}")


# Only pointer addressing changes. All arithmetic, masks, state reads/writes,
# token iteration, beta/gate computation and launch tile parameters stay intact.
POINTER_REPLACEMENTS = (
    (f"def {KERNEL_NAME}(", "def _titan_kda_strided_kernel("),
    ("    stride_indices_tok: tl.constexpr,", "    stride_indices_tok: tl.constexpr,\n"
     "    stride_q_token: tl.constexpr,\n    stride_k_token: tl.constexpr,\n"
     "    stride_v_token: tl.constexpr,\n    stride_g_token: tl.constexpr,\n"
     "    stride_beta_token: tl.constexpr,"),
    ("p_q = q + (bos * H + i_h) * K + o_k",
     "p_q = q + bos * stride_q_token + i_h * K + o_k"),
    ("p_k = k + (bos * H + i_h) * K + o_k",
     "p_k = k + bos * stride_k_token + i_h * K + o_k"),
    ("p_v = v + (bos * HV + i_hv) * V + o_v",
     "p_v = v + bos * stride_v_token + i_hv * V + o_v"),
    ("p_beta = beta + bos * HV + i_hv",
     "p_beta = beta + bos * stride_beta_token + i_hv"),
    ("p_beta = beta + (bos * HV + i_hv) * V + o_v",
     "p_beta = beta + bos * stride_beta_token + i_hv * V + o_v"),
    ("p_g = g + bos * HV + i_hv", "p_g = g + bos * stride_g_token + i_hv"),
    ("p_gk = g + (bos * HV + i_hv) * K + o_k",
     "p_gk = g + bos * stride_g_token + i_hv * K + o_k"),
    ("        p_q += H * K", "        p_q += stride_q_token"),
    ("        p_k += H * K", "        p_k += stride_k_token"),
    ("        p_v += HV * V", "        p_v += stride_v_token"),
    ("            p_g += HV", "            p_g += stride_g_token"),
    ("            p_gk += HV * K", "            p_gk += stride_g_token"),
    ("        p_beta += HV * (V if IS_BETA_HEADWISE else 1)",
     "        p_beta += stride_beta_token"),
)


def derive_kernel(source: str) -> str:
    _, original = function_source(source, KERNEL_NAME)
    require_hash(original, BASE_KERNEL_SHA256, "recurrent kernel")
    result = original
    for old, new in POINTER_REPLACEMENTS:
        if result.count(old) != 1:
            raise ValueError(f"Expected one pointer anchor: {old}")
        result = result.replace(old, new, 1)
    ast.parse(result)
    return result


def recover_original_kernel(derived: str) -> str:
    result = derived
    for old, new in reversed(POINTER_REPLACEMENTS):
        if result.count(new) != 1:
            raise ValueError(f"Expected one transformed pointer anchor: {new}")
        result = result.replace(new, old, 1)
    require_hash(result, BASE_KERNEL_SHA256, "recovered recurrent kernel")
    return result

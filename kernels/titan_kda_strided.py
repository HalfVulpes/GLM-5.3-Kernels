# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Derived from the pinned vLLM/flash-linear-attention kernel (MIT provenance
# retained in the original source). Only token-pointer strides are changed.
"""K2: Titan SM80/TP4 plain-decode KDA without Q/K/V/beta packing copies.

Not activated by importing this module. The original recurrent token loop,
NULL-state return behavior and floating-point operations are preserved.
The serving caller guarantees one non-speculative token per request; device
cu_seqlens values are deliberately not copied to the host during dispatch.
"""
from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.third_party.flash_linear_attention.ops.op import exp, log

BASE_KERNEL_SHA256 = "085c1e75103c53488471b05ec888dacd9172844d78637dfe09de9f51d58e1366"

@triton.jit(do_not_specialize=["N", "T"])
def _titan_kda_strided_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    a_log,
    g_bias,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    stride_q_token: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    SIGMOID_BETA: tl.constexpr,  # beta holds raw logits; sigmoid at fp32 load
    COMPUTE_GATE: tl.constexpr,  # g holds raw logits; KDA gate computed in-kernel
    SAFE_GATE: tl.constexpr,  # bounded gate variant (only branch implemented)
    LOWER_BOUND: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q_token + i_h * K + o_k
    p_k = k + bos * stride_k_token + i_h * K + o_k
    p_v = v + bos * stride_v_token + i_hv * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * stride_beta_token + i_hv * V + o_v
    else:
        p_beta = beta + bos * stride_beta_token + i_hv

    if not IS_KDA:
        p_g = g + bos * stride_g_token + i_hv
    else:
        p_gk = g + bos * stride_g_token + i_hv * K + o_k

    # Per-head gate amplitude, hoisted out of the token loop (COMPUTE_GATE).
    if COMPUTE_GATE:
        b_a_log = tl.exp(tl.load(a_log + i_h).to(tl.float32))

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            # Load state index and check for invalid entries
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                tl.int64
            )
            # Skip if state index is invalid (NULL_BLOCK_ID=0)
            if state_idx <= 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        # [BV, BK]
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            if COMPUTE_GATE:
                # Replicates kda_gate_fwd_kernel's SAFE_GATE branch
                # bit-for-bit (same tl.exp, same fp32 math; the intermediate
                # gate value this replaces was stored/reloaded as fp32,
                # which is lossless): y = lb / (1 + exp(-exp(A)*(g+bias))).
                b_gk += tl.load(
                    g_bias + i_h * K + o_k, mask=mask_k, other=0.0
                ).to(tl.float32)
                b_gk = LOWER_BOUND / (1.0 + tl.exp(-(b_a_log * b_gk)))
            b_h *= exp(b_gk[None, :])
        # [BV]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        # Matches torch's `x.float().sigmoid()` pre-computation bit-for-bit
        # on the input side (bf16->fp32 is exact); only the sigmoid impl itself
        # can differ by <=1 ULP.
        if SIGMOID_BETA:
            b_beta = tl.sigmoid(b_beta)
        b_v *= b_beta
        # [BV, BK]
        b_h += b_v[:, None] * b_k[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if INPLACE_FINAL_STATE:
            # Load state index and check for invalid entries
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            # Only store if state index is valid (not NULL_BLOCK_ID=0)
            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += stride_q_token
        p_k += stride_k_token
        p_o += HV * V
        p_v += stride_v_token
        if not IS_KDA:
            p_g += stride_g_token
        else:
            p_gk += stride_g_token
        p_beta += stride_beta_token


def supports_kda_strided(
    *, q, k, v, g, beta, initial_state, inplace_final_state,
    use_qk_l2norm_in_kernel, cu_seqlens, ssm_state_indices,
    num_accepted_tokens, out, sigmoid_beta, a_log, g_bias,
    compute_gate, lower_bound,
):
    """Host-metadata-only guard. Unsupported inputs use the upstream wrapper.

    The existing GLM plain-decode caller guarantees unit cu_seqlens segments.
    Structural checks never synchronize or inspect CUDA tensor contents.
    """
    if (num_accepted_tokens is not None or not inplace_final_state
            or not use_qk_l2norm_in_kernel or not sigmoid_beta
            or not compute_gate or lower_bound != -5.0):
        return False
    tensors = (q, k, v, g, beta, initial_state, cu_seqlens,
               ssm_state_indices, out, a_log, g_bias)
    if any(not isinstance(x, torch.Tensor) for x in tensors):
        return False
    if q.device.type != "cuda" or any(x.device != q.device for x in tensors):
        return False
    if q.ndim != 4 or q.shape[0] != 1 or not 1 <= q.shape[1] <= 8:
        return False
    n = q.shape[1]
    if any(x.shape != (1, n, 16, 128) or x.dtype != torch.bfloat16
           or x.stride(-1) != 1 or x.stride(-2) != 128
           or x.stride(1) < 2048 for x in (q, k, v, g)):
        return False
    if (beta.shape != (1, n, 16) or beta.dtype != torch.bfloat16
            or beta.stride(-1) != 1 or beta.stride(1) < 16):
        return False
    if (initial_state.ndim != 4 or initial_state.shape[1:] != (16, 128, 128)
            or initial_state.dtype != torch.float32
            or initial_state.stride(-1) != 1
            or initial_state.stride(-2) != 128
            or initial_state.stride(-3) != 16384):
        return False
    if (cu_seqlens.shape != (n + 1,) or not cu_seqlens.is_contiguous()
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or ssm_state_indices.shape != (n,)
            or not ssm_state_indices.is_contiguous()
            or ssm_state_indices.dtype not in (torch.int32, torch.int64)):
        return False
    if (out.shape != q.shape or out.dtype != q.dtype or not out.is_contiguous()
            or a_log.numel() != 16 or a_log.dtype != torch.float32
            or not a_log.is_contiguous() or g_bias.numel() != 2048
            or g_bias.dtype != torch.float32 or not g_bias.is_contiguous()):
        return False
    # The original wrapper snapshots noncontiguous read inputs before launch.
    # Reading them in place is equivalent only when output/state writes cannot
    # overlap that storage. Conservatively reject shared storage even when the
    # visible slices are disjoint. Storage addresses are host metadata; this
    # performs no CUDA readback or synchronization, including during capture.
    read_storage = {x.untyped_storage().data_ptr() for x in
                    (q, k, v, g, beta, a_log, g_bias, cu_seqlens, ssm_state_indices)}
    out_storage = out.untyped_storage().data_ptr()
    state_storage = initial_state.untyped_storage().data_ptr()
    if (out_storage == state_storage or out_storage in read_storage
            or state_storage in read_storage):
        return False
    return True


def maybe_kda_strided(
    *, q, k, v, g, beta, scale, initial_state,
    inplace_final_state=True, use_qk_l2norm_in_kernel=True,
    cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
    out=None, sigmoid_beta=False, a_log=None, g_bias=None,
    compute_gate=False, lower_bound=-5.0,
):
    """Return (the caller's output, the same state), or None for fallback."""
    if not supports_kda_strided(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens, out=out,
        sigmoid_beta=sigmoid_beta, a_log=a_log, g_bias=g_bias,
        compute_gate=compute_gate, lower_bound=lower_bound,
    ):
        return None
    n = q.shape[1]
    _titan_kda_strided_kernel[(1, 16, n * 16)](
        q=q, k=k, v=v, g=g, beta=beta, o=out,
        h0=initial_state, ht=initial_state,
        cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None, a_log=a_log, g_bias=g_bias,
        scale=128 ** -0.5 if scale is None else scale,
        N=n, T=n, B=1, H=16, HV=16, K=128, V=128, BK=128, BV=8,
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0), stride_indices_tok=1,
        stride_q_token=q.stride(1), stride_k_token=k.stride(1),
        stride_v_token=v.stride(1), stride_g_token=g.stride(1),
        stride_beta_token=beta.stride(1),
        USE_INITIAL_STATE=True, INPLACE_FINAL_STATE=True,
        IS_BETA_HEADWISE=False, USE_QK_L2NORM_IN_KERNEL=True,
        IS_VARLEN=True, IS_CONTINUOUS_BATCHING=True, IS_SPEC_DECODING=False,
        IS_KDA=True, SIGMOID_BETA=True, COMPUTE_GATE=True, SAFE_GATE=True,
        LOWER_BOUND=-5.0, num_warps=1, num_stages=3,
    )
    return out, initial_state

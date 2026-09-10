# SPDX-License-Identifier: Apache-2.0
"""Single-CTA deterministic MoE alignment for Titan decode B=1..8.

Installed at the verified Marlin call site. Exact target: 288 local/global experts,
top-k 8, block M 8, no expert parallel map. Invalid routes are ignored.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _align_small(
    TOPK, SORTED, EXPERTS, POSTPAD,
    R: tl.constexpr, BR: tl.constexpr, BS: tl.constexpr,
):
    route = tl.arange(0, BR)
    raw = tl.load(TOPK + route, mask=route < R, other=288)
    valid = (route < R) & (raw >= 0) & (raw < 288)
    expert = tl.where(valid, raw, 288).to(tl.int32)

    # Each row j compares against every original route k. Original flat IDs
    # give the same tie order as the baseline's stable argsort(expert).
    same = (expert[:, None] == expert[None, :]) & valid[None, :]
    rank = tl.sum((same & (route[None, :] < route[:, None])).to(tl.int32), 1)
    count = tl.sum(same.to(tl.int32), 1)
    first = valid & (rank == 0)
    padded = ((count + 7) // 8) * 8
    offset = tl.sum(tl.where(
        first[None, :] & (expert[None, :] < expert[:, None]),
        padded[None, :], 0), 1)

    # Fully initialize padding/tails before any lanes scatter live entries.
    slot = tl.arange(0, BS)
    tl.store(SORTED + slot, R, mask=slot < R * 8)
    tl.store(EXPERTS + route, -1, mask=route < R)
    tl.debug_barrier()

    tl.store(SORTED + offset + rank, route, mask=valid)
    # A synthetic expert can receive >8 routes. Its first route writes every
    # required expert block, not only one block as in real top-8 B<=8 traffic.
    block = tl.arange(0, BR // 8)
    block_mask = first[:, None] & (block[None, :] < (padded[:, None] // 8))
    tl.store(EXPERTS + offset[:, None] // 8 + block[None, :],
             expert[:, None], mask=block_mask)
    tl.store(POSTPAD, tl.sum(tl.where(first, padded, 0), 0))


def titan_moe_align_small(
    topk_ids,
    block_size=8,
    num_experts=288,
    expert_map=None,
    pad_sorted_ids=False,
    ignore_invalid_experts=True,
    *,
    num_warps=None,
):
    """Return exactly the stable baseline's three tensors, including padding."""
    if (topk_ids.ndim != 2 or not 1 <= topk_ids.shape[0] <= 8
            or topk_ids.shape[1] != 8 or block_size != 8 or num_experts != 288
            or expert_map is not None or not ignore_invalid_experts):
        raise ValueError('Titan small alignment requires B1..8/top8/E288/block8/EPoff/ignore-invalid')
    if not topk_ids.is_cuda or not topk_ids.is_contiguous():
        raise ValueError('Titan small alignment requires contiguous CUDA route IDs')
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Route IDs must be int32 or int64')
    routes = topk_ids.numel()
    # For R<288, the baseline caps padded storage at R*block_size; already
    # divisible by eight, so pad_sorted_ids does not alter this target layout.
    sorted_ids = torch.empty(routes * 8, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty(routes, dtype=torch.int32, device=topk_ids.device)
    postpad = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    if num_warps is None:
        num_warps = 1 if routes <= 16 else 4
    if num_warps not in (1, 2, 4):
        raise ValueError('Candidate warp counts are 1, 2, or 4')
    _align_small[(1,)](topk_ids, sorted_ids, expert_ids, postpad,
        R=routes, BR=triton.next_power_of_2(routes),
        BS=triton.next_power_of_2(routes * 8), num_warps=num_warps, num_stages=1)
    return sorted_ids, expert_ids, postpad

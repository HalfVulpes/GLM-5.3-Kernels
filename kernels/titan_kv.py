# SPDX-License-Identifier: Apache-2.0
"""SM80-specific per-token symmetric INT8 latent KV; no calibration required."""
import torch
import triton
import triton.language as tl

@triton.jit
def _store_int8(X, CACHE, SLOTS, N: tl.constexpr, XS: tl.constexpr,
                CS0: tl.constexpr, CS1: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(SLOTS + row)
    if slot >= 0:
        g = tl.arange(0, 4)
        d = tl.arange(0, 128)
        x = tl.load(X + row * XS + g[:, None] * 128 + d[None, :]).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(x)) / 127.0, 1.0e-12)
        quant = tl.extra.cuda.libdevice.nearbyint(x / scale)
        quant = tl.minimum(tl.maximum(quant, -127), 127).to(tl.int8)
        dest = CACHE + (slot // BLOCK).to(tl.int64) * CS0 + (slot % BLOCK) * CS1
        tl.store(dest + g[:, None] * 128 + d[None, :], quant)
        tl.store((dest + 512).to(tl.pointer_type(tl.float32)) + g, tl.where(g == 0, scale, 0.0))

def store_int8(x, cache, slots):
    assert x.shape[-1] == 512 and cache.shape[-1] == 528
    slots = slots.flatten()
    assert slots.numel() <= x.shape[0]
    _store_int8[(slots.numel(),)](x, cache.view(torch.int8), slots, slots.numel(),
        x.stride(0), cache.stride(0), cache.stride(1), cache.shape[1], num_warps=4)

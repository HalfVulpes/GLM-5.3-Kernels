#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthetic per-token INT8 KV/MLA correctness; GPU work requires --run.

No checkpoint or vLLM is required. The CPU default checks core imports/syntax
and prints the GPU plan, without importing torch/triton. The GPU procedure
follows the per-token production verifier, not the older group-128 layout.
Only one FP32 scale at byte offset 512 applies to all 512 latent dimensions.
Quantization comparisons are chunked to avoid materializing a full FP32 KV.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def require_device(torch, device_name):
    device = torch.device(device_name)
    if device.type != "cuda": raise ValueError("GPU validation requires a CUDA device")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor, props.multi_processor_count) != (8, 0, 70):
        raise RuntimeError("This validation profile requires SM80 with 70 SMs; other devices are not validated by this profile")
    return device, props


def build_fixture(torch, tokens, device, seed):
    from kernels.titan_kv import store_int8
    generator = torch.Generator(device=device).manual_seed(seed)
    original = torch.randn(tokens, 512, device=device, dtype=torch.bfloat16, generator=generator)
    original[:, :128] *= 0.05; original[:, 256:384] *= 8; original[0] = 0
    cache = torch.zeros(tokens // 256, 256, 528, device=device, dtype=torch.int8)
    store_int8(original, cache, torch.arange(tokens, device=device, dtype=torch.int64))
    return original, cache, generator


def verify_gpu(args):
    import torch
    sys.path.insert(0, str(ROOT))
    from kernels.titan_kv import store_int8
    from kernels.titan_mla import triton_mla_sparse_attention
    device, props = require_device(torch, args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_mib * 1024**2 / props.total_memory, device)
    original, cache, generator = build_fixture(torch, args.kv_tokens, device, args.seed)
    rows = cache.view(args.kv_tokens, 528)
    error_sum = reference_sum = 0.0
    max_integer_error = 0.0
    for start in range(0, args.kv_tokens, 4096):
        values = original[start:start + 4096].float()
        packed = rows[start:start + 4096]
        scales = packed[:, 512:516].contiguous().view(torch.float32)
        expected_scale = values.abs().amax(dim=1, keepdim=True).clamp_min(127e-12) / 127
        torch.testing.assert_close(scales, expected_scale, rtol=1e-6, atol=1e-12)
        expected_int = torch.round(values / expected_scale).clamp(-127, 127)
        actual_int = packed[:, :512].float()
        max_integer_error = max(max_integer_error, (actual_int - expected_int).abs().max().item())
        dequant = (actual_int * scales).to(torch.bfloat16).float()
        error_sum += (dequant - values).square().sum().item()
        reference_sum += values.square().sum().item()
        assert bool((packed[:, 516:] == 0).all()), "KV row padding must be zero"
    quant_nmse = error_sum / reference_sum
    assert quant_nmse < 0.0003 and max_integer_error <= 1
    first = cache[0].clone()
    store_int8(original[:1], cache, torch.tensor([-1], device=device))
    assert torch.equal(first, cache[0]), "A negative slot modified cache contents"
    results = []
    for batch in (1, 6, 8):
        q = torch.randn(batch, 16, 512, generator=generator, device=device, dtype=torch.bfloat16) * 0.15
        indices = torch.randint(args.kv_tokens, (batch, 1, 2176), generator=generator,
                                device=device, dtype=torch.int32)
        indices[:, :, 2048:] = -1
        if batch > 1: indices[1, :, 2000:] = -1
        # Positive out-of-range indices and negative padding must both mask.
        indices[0, 0, 0] = args.kv_tokens + 5
        valid = (indices[:, 0] >= 0) & (indices[:, 0] < args.kv_tokens)
        chosen = original[indices[:, 0].clamp(0, args.kv_tokens - 1).long()].float()
        chosen.masked_fill_(~valid[:, :, None], 0)
        logits = torch.bmm(q.float(), chosen.transpose(1, 2)) * (512 ** -0.5)
        logits.masked_fill_(~valid[:, None, :], -float("inf"))
        reference = torch.bmm(logits.softmax(-1), chosen)
        for splits in (1, 4, 8, 16):
            output = triton_mla_sparse_attention(q, cache.view(args.kv_tokens, 1, 528), indices,
                                                 512 ** -0.5, num_kv_splits=splits, sm_count=70)
            nmse = ((output.float() - reference).square().sum() / reference.square().sum()).item()
            cosine = torch.nn.functional.cosine_similarity(output.float().flatten(), reference.flatten(), dim=0).item()
            assert bool(torch.isfinite(output).all()) and nmse < 0.001 and cosine > 0.999, (batch, splits, nmse, cosine)
            blank = triton_mla_sparse_attention(q, cache.view(args.kv_tokens, 1, 528), torch.full_like(indices, -1),
                                                512 ** -0.5, num_kv_splits=splits, sm_count=70)
            assert torch.equal(blank, torch.zeros_like(blank)), (batch, splits, "empty row")
            results.append({"batch": batch, "splits": splits, "nmse": nmse, "cosine": cosine, "empty_rows_zero": True})
    torch.cuda.synchronize(device)
    return {"passed": True, "gpu_tested": True, "device": props.name, "sm_count": props.multi_processor_count,
            "kv_tokens": args.kv_tokens, "quant_nmse": quant_nmse, "max_integer_error": max_integer_error,
            "results": results, "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            "scope": "Synthetic numerical correctness; not model quality, capacity or end-to-end throughput"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly enable GPU allocation and execution")
    parser.add_argument("--device", default="cuda:0", help="Logical device selected by the caller; no physical GPU is hardcoded")
    parser.add_argument("--kv-tokens", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--memory-limit-mib", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.kv_tokens < 256 or args.kv_tokens % 256 or args.memory_limit_mib <= 0:
        parser.error("KV token count must be a positive multiple of 256; memory limit must be positive")
    if args.run: report = verify_gpu(args)
    else:
        from verify_source_equivalence import check_core_imports
        report = {"cpu_passed": True, "gpu_tested": False, "core_import_checks": check_core_imports(ROOT / "kernels"),
                  "gpu_plan": {"kv_tokens": args.kv_tokens, "batches": [1, 6, 8], "splits": [1, 4, 8, 16],
                               "profile": "SM80 / 70 SMs", "enable_with": "--run"}}
    rendered = json.dumps(report, indent=2); print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(rendered + "\n")


if __name__ == "__main__": main()

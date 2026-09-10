#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Short synthetic MLA split comparison. GPU work requires explicit --run.

Uses only torch/triton and the public kernels. Run numerical validation first.
Measurements reuse one sparse working set and are warm-cache module timings,
not end-to-end decoding or proof of supported model concurrency.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def gpu_main(args):
    import torch
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
    from kernels import titan_mla
    from verify_mla import build_fixture, require_device
    device, props = require_device(torch, args.device)
    torch.cuda.set_per_process_memory_fraction(1024 * 1024**2 / props.total_memory, device)
    original, cache, generator = build_fixture(torch, 262144, device, 37)
    del original
    results = []
    for batch in args.batches:
        q = torch.randn(batch, 16, 512, device=device, dtype=torch.bfloat16, generator=generator) * 0.15
        indices = torch.randint(262144, (batch, 1, 2176), device=device, dtype=torch.int32, generator=generator)
        indices[:, :, 2048:] = -1
        for splits in args.splits:
            def run():
                return titan_mla.triton_mla_sparse_attention(q, cache.view(262144, 1, 528), indices,
                                                            512**-0.5, num_kv_splits=splits, sm_count=70)
            for _ in range(3): run()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph): output = run()
                for _ in range(5): graph.replay()
                samples = []
                for _ in range(args.repeats):
                    torch.cuda.synchronize(device)
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    wall = time.perf_counter(); begin.record()
                    for _ in range(args.iterations): graph.replay()
                    end.record(); end.synchronize()
                    samples.append({"cuda_us": begin.elapsed_time(end) * 1000 / args.iterations,
                                    "wall_us": (time.perf_counter() - wall) * 1e6 / args.iterations})
                assert bool(torch.isfinite(output).all())
                results.append({"batch": batch, "splits": splits,
                                "median_cuda_us": statistics.median(s["cuda_us"] for s in samples), "samples": samples})
            finally:
                torch.cuda.synchronize(device); graph.reset()
    source = Path(titan_mla.__file__)
    return {"gpu_tested": True, "device": props.name, "sm_count": props.multi_processor_count,
            "kernel_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "results": results,
            "scope": "Synthetic warm-cache graph replay; numerical reference validation is separate"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true"); p.add_argument("--device", default="cuda:0")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 6, 8])
    p.add_argument("--splits", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--iterations", type=int, default=100); p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if min(args.iterations, args.repeats, *args.batches) < 1 or any(s not in (1, 2, 4, 8, 16) for s in args.splits):
        p.error("Positive sizes/iterations and supported split counts are required")
    report = gpu_main(args) if args.run else {"gpu_tested": False, "run_required": True,
        "plan": {"kv_tokens": 262144, "batches": args.batches, "splits": args.splits, "profile": "SM80 / 70 SMs"}}
    rendered = json.dumps(report, indent=2); print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(rendered + "\n")


if __name__ == "__main__": main()

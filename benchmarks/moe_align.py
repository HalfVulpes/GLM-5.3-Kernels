#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stable small-batch MoE alignment correctness and eager/graph comparison.

Default: stdlib-only CPU algebra tests. --run explicitly enables one caller-
selected SM80/70-SM GPU. The default GPU reference uses only public PyTorch
operations. --baseline vllm optionally compares an installed compatible
deterministic vLLM aligner; it is not a core package dependency.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cpu_reference(flat):
    count = len(flat); output = []; experts = []
    for expert in range(288):
        members = [i for i, value in enumerate(flat) if value == expert]
        padded = (len(members) + 7) // 8 * 8
        output.extend(members + [count] * (padded - len(members)))
        experts.extend([expert] * (padded // 8))
    postpad = len(output)
    return output + [count] * (count * 8 - len(output)), experts + [-1] * (count - len(experts)), [postpad]


def cpu_pairwise(flat):
    size = len(flat); valid = [0 <= x < 288 for x in flat]
    counts = [sum(v and x == expert for x, v in zip(flat, valid)) for expert in flat]
    ranks = [sum(valid[k] and flat[k] == expert for k in range(j)) for j, expert in enumerate(flat)]
    first = [v and rank == 0 for v, rank in zip(valid, ranks)]
    padded = [(n + 7) // 8 * 8 for n in counts]
    offsets = [sum(padded[k] for k in range(size) if first[k] and flat[k] < expert) for expert in flat]
    output = [size] * (size * 8); experts = [-1] * size
    for j, expert in enumerate(flat):
        if valid[j]: output[offsets[j] + ranks[j]] = j
        if first[j]:
            for block in range(padded[j] // 8): experts[offsets[j] // 8 + block] = expert
    return output, experts, [sum(p for p, keep in zip(padded, first) if keep)]


def make_patterns(batch, rng):
    routes = batch * 8
    return {"distinct": list(range(routes)), "shared_top8": list(range(8)) * batch,
            "random_top8": [expert for _ in range(batch) for expert in rng.sample(range(288), 8)],
            "all_one_expert": [173] * routes, "seven_experts": [i % 7 for i in range(routes)],
            "invalid_mixed": [-1 if i % 5 == 0 else 288 if i % 7 == 0 else i % 11 for i in range(routes)],
            "all_invalid": [-1] * routes}


def cpu_checks():
    rng = random.Random(37); cases = 0
    for batch in range(1, 9):
        patterns = list(make_patterns(batch, rng).values())
        patterns += [[rng.randrange(-2, 292) for _ in range(batch * 8)] for _ in range(100)]
        for flat in patterns:
            assert cpu_pairwise(flat) == cpu_reference(flat)
            cases += 1
    return {"cpu_passed": True, "cases": cases, "gpu_tested": False,
            "scope": "CPU layout algebra only; no torch/CUDA imports"}


def torch_reference(torch, ids):
    """Fixed-shape reference intended for graph capture; no .item() in this path."""
    flat = ids.flatten().long(); count = flat.numel()
    position = torch.arange(count, device=ids.device, dtype=torch.int64)
    expert = torch.where((flat >= 0) & (flat < 288), flat, 288)
    counts_all = torch.zeros(289, device=ids.device, dtype=torch.int64)
    counts_all.scatter_add_(0, expert, torch.ones_like(expert))
    counts = counts_all[:288]
    padded = (counts + 7) // 8 * 8
    padded_start = padded.cumsum(0) - padded
    original_start = counts.cumsum(0) - counts
    order = torch.argsort(expert, stable=True)
    ordered_expert = expert[order]; safe = ordered_expert.clamp_max(287)
    rank = position - original_start[safe]
    valid = ordered_expert < 288
    # Dummy lanes get unique scratch destinations outside the returned views.
    destination = torch.where(valid, padded_start[safe] + rank, count * 8 + position)
    sorted_storage = torch.full((count * 9,), count, device=ids.device, dtype=torch.int32)
    sorted_storage.scatter_(0, destination, order.to(torch.int32))
    block_destination = torch.where(valid & (rank % 8 == 0), padded_start[safe] // 8 + rank // 8, count + position)
    expert_storage = torch.full((count * 2,), -1, device=ids.device, dtype=torch.int32)
    expert_storage.scatter_(0, block_destination, ordered_expert.to(torch.int32))
    return sorted_storage[:count * 8], expert_storage[:count], padded.sum().to(torch.int32).reshape(1)


def torch_cpu_checks():
    import torch
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    rng = random.Random(37); cases = 0
    for batch in range(1, 9):
        patterns = list(make_patterns(batch, rng).values())
        patterns += [[rng.randrange(-2, 292) for _ in range(batch * 8)] for _ in range(100)]
        for flat in patterns:
            reference = cpu_reference(flat)
            for dtype in (torch.int32, torch.int64):
                result = torch_reference(torch, torch.tensor(flat, dtype=dtype, device="cpu").view(batch, 8))
                assert all(t.tolist() == wanted for t, wanted in zip(result, reference))
                cases += 1
    assert not torch.cuda.is_initialized()
    return {"torch_cpu_passed": True, "cases": cases, "gpu_tested": False, "cuda_initialized": False}


def gpu_main(args):
    import torch
    sys.path.insert(0, str(ROOT))
    from kernels.titan_moe_align import titan_moe_align_small
    sys.path.insert(0, str(ROOT / "tests"))
    from verify_mla import require_device
    device, props = require_device(torch, args.device)
    if args.baseline == "vllm":
        os.environ["VLLM_DETERMINISTIC_MOE_ALIGN"] = "1"
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        baseline = lambda ids: moe_align_block_size(ids, 8, 288, expert_map=None, ignore_invalid_experts=True)
    else:
        baseline = lambda ids: torch_reference(torch, ids)
    report = {"passed": False, "gpu_tested": True, "device": props.name, "sm_count": props.multi_processor_count,
              "baseline": args.baseline, "correctness": [], "timings": [],
              "scope": "Alignment module only; not whole MoE/model performance"}

    def timed(fn, use_graph):
        for _ in range(5): fn()
        torch.cuda.synchronize(device)
        graph = None
        try:
            if use_graph:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = fn()
                for _ in range(5): graph.replay()
            samples = []
            for _ in range(args.repeats):
                torch.cuda.synchronize(device)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter(); start.record()
                for _ in range(args.iterations):
                    if graph is None: fn()
                    else: graph.replay()
                end.record(); end.synchronize()
                samples.append({"cuda_us": start.elapsed_time(end) * 1000 / args.iterations,
                                "wall_us": (time.perf_counter() - wall) * 1e6 / args.iterations})
            return {"cuda_us_median": statistics.median(s["cuda_us"] for s in samples),
                    "wall_us_median": statistics.median(s["wall_us"] for s in samples), "samples": samples}
        finally:
            torch.cuda.synchronize(device)
            if graph is not None: graph.reset()

    try:
        rng = random.Random(37)
        with torch.inference_mode():
            for batch in args.batches:
                patterns = make_patterns(batch, rng)
                for name, flat in patterns.items():
                    cpu = cpu_reference(flat)
                    for dtype in (torch.int32, torch.int64):
                        ids = torch.tensor(flat, dtype=dtype, device=device).view(batch, 8)
                        reference = baseline(ids)
                        assert all(t.cpu().tolist() == expected for t, expected in zip(reference, cpu)), (batch, name, "reference")
                        for warps in (1, 2, 4):
                            actual = titan_moe_align_small(ids, num_warps=warps)
                            assert all(torch.equal(x, y) for x, y in zip(actual, reference)), (batch, name, warps)
                            padded = titan_moe_align_small(ids, pad_sorted_ids=True, num_warps=warps)
                            assert all(torch.equal(x, y) for x, y in zip(padded, reference))
                            report["correctness"].append({"batch": batch, "pattern": name, "dtype": str(dtype), "warps": warps, "passed": True})
                if not args.correctness_only:
                    ids = torch.tensor(patterns["random_top8"], dtype=torch.int32, device=device).view(batch, 8)
                    variants = {"reference": lambda: baseline(ids),
                                **{f"triton_w{w}": lambda w=w: titan_moe_align_small(ids, num_warps=w) for w in (1, 2, 4)}}
                    # Rotate ordering by batch; each branch receives equal warmup.
                    names = list(variants); offset = batch % len(names)
                    timings = {}
                    for name in names[offset:] + names[:offset]:
                        timings[name] = {"eager": timed(variants[name], False), "graph": timed(variants[name], True)}
                    report["timings"].append({"batch": batch, "variants": timings})
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Explicitly enable GPU execution")
    mode.add_argument("--torch-cpu", action="store_true", help="Validate the torch reference on CPU; requires torch but never CUDA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline", choices=("torch", "vllm"), default="torch")
    parser.add_argument("--batches", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.iterations, args.repeats) < 1 or any(b < 1 or b > 8 for b in args.batches):
        parser.error("Positive iterations/repeats and B1..8 are required")
    print(json.dumps(gpu_main(args) if args.run else torch_cpu_checks() if args.torch_cpu else cpu_checks(), indent=2))


if __name__ == "__main__": main()

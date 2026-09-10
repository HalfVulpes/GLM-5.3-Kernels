#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the optional pinned-vLLM query-sharding integration.

No torch/vLLM import or device allocation is performed. Supply either an
explicit --source-root pointing at a vLLM package source tree or --snapshot-dir
containing kpool.py, metadata.py and utils.py. Unpatched inputs are transformed
in memory only. This checks partitions, collective order and output placement;
it does not establish CUDA/NCCL correctness or end-to-end performance.
"""

import argparse
import ast
import copy
import importlib.util
import json
import random
import textwrap
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[1]


class Scalar(int):
    def item(self):
        return int(self)


class Tensor:
    """Only the CPU tensor operations used by the extracted completion path."""

    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0])) if data and isinstance(data[0], list) else (len(data),)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
            return Tensor([r[cols] for r in self.data[rows]])
        value = self.data[key]
        return Tensor(value) if isinstance(key, slice) else Scalar(value)

    def __setitem__(self, key, value):
        rows, cols = key
        indices = range(*rows.indices(len(self.data)))
        assert len(indices) == value.shape[0]
        for row, data in zip(indices, value.data):
            assert len(self.data[row][cols]) == len(data)
            self.data[row][cols] = data

    def contiguous(self):
        return self

    def to(self, dtype):
        return self

    def __add__(self, value):
        return Tensor([x + value for x in self.data])

    def __ge__(self, value):
        return Tensor([[x >= value for x in row] for row in self.data])


def expand_row(pools, seq_len, pool_size, include_tail=True):
    history = [p * pool_size + offset if p >= 0 else -1
               for p in pools for offset in range(pool_size)]
    if include_tail:
        tail_start = seq_len // pool_size * pool_size
        history += [tail_start + offset if offset < seq_len % pool_size else -1
                    for offset in range(pool_size - 1)]
    return history


class PoolOps:
    @staticmethod
    def expand_pools_and_append_tail(pool_ids, seq_lens, pool_size):
        return Tensor([expand_row(row, n, pool_size) for row, n in zip(pool_ids.data, seq_lens.data)])

    @staticmethod
    def expand_pools_to_tokens(pool_ids, valid, topk_tokens, pool_size):
        return Tensor([expand_row(row, 0, pool_size, False) for row in pool_ids.data])


def extract(source, names, namespace):
    tree = ast.parse(source)
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError(f"Missing source definitions: {names}")
    tree = ast.Module(body=selected, type_ignores=[])
    exec(compile(tree, "<actual-vllm-source>", "exec"), namespace)


def load_patch():
    spec = importlib.util.spec_from_file_location("titan_indexer_patch", REPO / "integration/patch_indexer_sharding.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_workspace_in_memory(kpool, metadata, mqa):
    """Run the public workspace transform against memory-backed file objects."""
    spec = importlib.util.spec_from_file_location("indexer_workspace_transform", REPO / "integration/patch_indexer_workspace.py")
    workspace = importlib.util.module_from_spec(spec); spec.loader.exec_module(workspace)
    files = {"model_executor/layers/sparse_attn_indexer_kpool.py": kpool,
             "v1/attention/backends/mla/indexer.py": metadata,
             "v1/attention/ops/mqa_logits_triton.py": mqa}
    class MemoryPath:
        def __init__(self, value): self.value = PurePosixPath(value)
        def __truediv__(self, value): return MemoryPath(self.value / value)
        def read_text(self): return files[str(self.value)]
        def write_text(self, value): files[str(self.value)] = value
    workspace.Path = MemoryPath
    workspace.patch_tree("")
    return files["model_executor/layers/sparse_attn_indexer_kpool.py"], files["v1/attention/backends/mla/indexer.py"]


def run_tests(kpool, metadata, utils, mqa=None):
    prepared_workspace = "workspace=(logits_workspace, k_bf16_workspace, q_bf16_workspace)" not in kpool
    if prepared_workspace:
        if mqa is None:
            raise ValueError("Pristine inputs also require mqa_logits_triton.py; provide a complete source root or add it to the snapshot")
        kpool, metadata = prepare_workspace_in_memory(kpool, metadata, mqa)
    patch = load_patch()
    patched_kpool, patched_metadata = patch.patch_sources(kpool, metadata)
    assert patch.patch_sources(patched_kpool, patched_metadata) == (patched_kpool, patched_metadata)
    ns = {"NamedTuple": NamedTuple, "MIN_SHARD_TOKENS": 256,
          "torch": SimpleNamespace(Tensor=Tensor, int64="int64", int32="int32"),
          "kpool_ops": PoolOps}
    extract(utils, {"balanced_row_bounds", "balanced_row_counts"}, ns)
    extract(patched_metadata, {"ShardedChunkSpec", "shard_chunk_specs_by_query",
                              "indexer_shard_size_for_batch", "split_indexer_prefill_chunks",
                              "indexer_decode_shard_bounds"}, ns)
    extract(patched_kpool, {"_titan_gather_prefill_topk"}, ns)
    body = patched_kpool.split("            # TITAN_PREFILL_GATHER_BEGIN\n", 1)[1].split(
        "            # TITAN_PREFILL_GATHER_END", 1)[0]
    code = "def finish(topk_dst, chunk, index_kpool, positions, topk_indices_buffer, topk_tokens):\n"
    exec(code + textwrap.indent(textwrap.dedent(body), "    "), ns)
    assert ns["indexer_shard_size_for_batch"](255, 4) == 1
    for tokens in (256, 257, 2040, 2048):
        assert ns["indexer_shard_size_for_batch"](tokens, 4) == 4
    for rank in range(4):
        assert ns["indexer_decode_shard_bounds"](64, 64, rank, 4, 0) is None

    rng = random.Random(53)
    partition_cases = 0
    collective_calls = 0
    # Include tiny replicated chunks BETWEEN sharded chunks and reset the K
    # gather at request-group transitions. Test offsets in absolute batch rows.
    cases = [([(slice(1, 3), slice(0, 509)), (slice(1, 3), slice(509, 511)),
               (slice(1, 3), slice(511, 1022)), (slice(3, 4), slice(0, 7))], 4)]
    for _ in range(350):
        seq_lens = [rng.randint(0, 65536) for _ in range(rng.randint(1, 8))]
        query_lens = [rng.randint(1, 2048) for _ in seq_lens]
        specs = ns["split_indexer_prefill_chunks"](
            Tensor(seq_lens), Tensor(query_lens), 8 * 65536, 128 * 1024 * 1024, request_offset=2)
        cases.append((specs, 4))
        for req, query in specs:
            n = sum(seq_lens[req.start - 2:req.stop - 2])
            m = query.stop - query.start
            # The GLOBAL budget has not been enlarged. For the deployment's
            # <= 8 contexts even one row always fits, including ceil shards.
            assert m * n * 4 <= 128 * 1024 * 1024
            assert max(ns["balanced_row_counts"](m, 4)) * n * 4 <= 128 * 1024 * 1024
    for specs, size in cases:
        rank_specs = [ns["shard_chunk_specs_by_query"](specs, rank, size) for rank in range(size)]
        assert all(len(x) == len(specs) for x in rank_specs)
        signatures = []
        for rank in range(size):
            signatures.append([(i, s.gather_start, s.shard_row_counts) for i, s in enumerate(rank_specs[rank])
                               if s.shard_row_counts is not None])
            previous_req = None
            for spec in rank_specs[rank]:
                req = (spec.req_slice.start, spec.req_slice.stop)
                assert spec.skip_kv_gather == (req == previous_req)
                previous_req = req
        assert all(sig == signatures[0] for sig in signatures)
        for i, (_, query) in enumerate(specs):
            parts = [rs[i] for rs in rank_specs]
            if parts[0].shard_row_counts is None:
                assert query.stop - query.start < size
                assert all(p.query_slice == query for p in parts)
            else:
                counts = parts[0].shard_row_counts
                assert sum(counts) == query.stop - query.start
                assert min(counts) > 0 and max(counts) - min(counts) <= 1
                assert parts[0].query_slice.start == query.start
                assert parts[-1].query_slice.stop == query.stop
                assert all(a.query_slice.stop == b.query_slice.start for a, b in zip(parts, parts[1:]))
                collective_calls += 1
            partition_cases += 1

    # Execute the patch's REAL post-topk source, including its destination
    # slices and global position slice, on each simulated TP rank.
    output_cases = 0
    mismatch_guard = False
    for rows, select_k in [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (7, 5),
                           (255, 5), (256, 5), (257, 5), (509, 5), (511, 5),
                           (2040, 512), (2048, 5)]:
        for pool_size, with_positions in [(4, True), (4, False), (1, True)]:
            output_start = 13
            group_start = 7  # Decode-prefix + nonzero request-relative q slice.
            query = slice(output_start - group_start, output_start - group_start + rows)
            specs = [ns["shard_chunk_specs_by_query"]([(slice(2, 4), query)], rank, 4)[0]
                     for rank in range(4)]
            positions_data = [9000 + i for i in range(output_start)]
            # A reset in the middle is a request boundary; all four tail phases
            # and many different sequence lengths must survive reassembly.
            positions_data += [7000 + i if i < rows // 2 else 19000 + i - rows // 2 for i in range(rows)]
            positions_data += [42000] * 3
            pool_rows = [[((r * 17 + col * 3) % 997) if col % 11 else -1
                          for col in range(select_k)] for r in range(rows)]
            parts = []
            chunks = []
            for spec in specs:
                start = group_start + spec.query_slice.start
                end = group_start + spec.query_slice.stop
                chunks.append(SimpleNamespace(
                    token_start=start, token_end=end, shard_row_counts=spec.shard_row_counts,
                    gather_token_start=(start - (spec.query_slice.start - spec.gather_start)
                                        if spec.gather_start is not None else None)))
                parts.append(Tensor(copy.deepcopy(pool_rows[start - output_start:end - output_start])))
            for rank in range(4):
                calls = []

                def all_gatherv(value, dim, sizes):
                    assert dim == 0 and sizes == [p.shape[0] for p in parts]
                    assert value.data == parts[rank].data
                    # Wire payload is pool-width, never expanded token-width.
                    assert value.shape[1] == select_k
                    calls.append(tuple(sizes))
                    return Tensor([list(row) for part in parts for row in part.data])

                ns["get_tp_group"] = lambda: SimpleNamespace(world_size=4, rank_in_group=rank, all_gatherv=all_gatherv)
                width = select_k * pool_size + (pool_size - 1 if with_positions else 0)
                buffer_width = max(width, 2176 if select_k == 512 else width + 3)
                buffer = Tensor([[-777] * buffer_width for _ in positions_data])
                if pool_size == 1:
                    # In the actual function the topk kernel already wrote the
                    # local output-buffer view before the finish block.
                    buffer[chunks[rank].token_start:chunks[rank].token_end, :select_k] = parts[rank]
                ns["finish"](parts[rank], chunks[rank], pool_size,
                             Tensor(positions_data) if with_positions else None,
                             buffer, select_k * pool_size)
                assert len(calls) == int(rows >= 4)
                expected = [[-777] * buffer_width for _ in positions_data]
                for r, pools in enumerate(pool_rows):
                    result = (expand_row(pools, positions_data[output_start + r] + 1, pool_size, with_positions)
                              if pool_size > 1 else pools)
                    expected[output_start + r][:len(result)] = result
                assert buffer.data == expected, (rows, rank, pool_size, with_positions)
                output_cases += 1
                # Regression sensitivity: rank-local start must never survive
                # reassembly. The helper must reject stale metadata before NCCL.
                if rows == 509 and rank == 2 and pool_size == 4 and with_positions:
                    invalid_chunk = copy.copy(chunks[rank])
                    invalid_chunk.gather_token_start += 1
                    try:
                        ns["_titan_gather_prefill_topk"](parts[rank], invalid_chunk)
                    except AssertionError:
                        mismatch_guard = True
                    else:
                        raise AssertionError("Invalid global destination was accepted")
    assert mismatch_guard
    # A corrupt anchor must fail closed and never silently omit the gather.
    if patch.MARKER not in kpool:
        try:
            patch.patch_sources(kpool.replace(patch.OLD_FINISH, "# changed upstream\n"), metadata)
        except ValueError:
            pass
        else:
            raise AssertionError("Patch accepted source drift")
    return {"cpu_passed": True, "workspace_prepared_in_memory": prepared_workspace,
            "metadata_chunks_checked": partition_cases,
            "rank_uniform_collective_chunks": collective_calls,
            "full_output_comparisons": output_cases,
            "tp_size": 4, "target_pool_columns": 512, "global_logits_budget_mib": 128,
            "gpu_tested": False, "performance_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source-root", type=Path)
    inputs.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    patch = load_patch()
    if args.snapshot_dir:
        sources = [(args.snapshot_dir / n).read_text() for n in ("kpool.py", "metadata.py", "utils.py")]
        optional_mqa = args.snapshot_dir / "mqa_logits_triton.py"
    else:
        sources = [(args.source_root / n).read_text() for n in (patch.KPOOL_PATH, patch.METADATA_PATH, Path("distributed/utils.py"))]
        optional_mqa = args.source_root / "v1/attention/ops/mqa_logits_triton.py"
    sources.append(optional_mqa.read_text() if optional_mqa.is_file() else None)
    report = run_tests(*sources)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

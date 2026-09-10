#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for the GLM-5.3-Kernels SM80 integration; see docs/SOURCES.md.
"""Finish vLLM kpool prefill query sharding for Titan TP4.

Apply AFTER patch_indexer_workspace.py. This does not enable the feature or
restart a service. Set VLLM_INDEXER_QUERY_SHARD=1 and explicitly set
VLLM_INDEXER_DECODE_SHARD_MIN_REQS=0 in the worker environment when testing.
The 128 MiB GLOBAL chunk budget stays unchanged; sharding only reduces each
rank's rows, so ceil-division cannot exceed the existing logits workspace.

The metadata partition/all_gatherv contract comes from the Apache-2.0 vLLM
standard sparse_attn_indexer.py implementation in the deployed image. Gather
512 int32 pool IDs before expanding them to token IDs (2051 meaningful
columns, stored in the model's padded output buffer).

Default is a read-only check. --apply writes only the two named vLLM files,
with first-application backups, and is idempotent for this exact patch.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

KPOOL_PATH = Path("model_executor/layers/sparse_attn_indexer_kpool.py")
METADATA_PATH = Path("v1/attention/backends/mla/indexer.py")
MARKER = "TITAN_KPOOL_PREFILL_QUERY_SHARD_V1"

HELPER = '''
# TITAN_KPOOL_PREFILL_QUERY_SHARD_V1
def _titan_gather_prefill_topk(topk_dst, chunk):
    """Reassemble rows before pool expansion, preserving rank order exactly."""
    counts = chunk.shard_row_counts
    if counts is None:
        return topk_dst, chunk.token_start, chunk.token_end
    tp = get_tp_group()
    assert tp.world_size == 4, "Titan kpool query sharding requires TP4"
    assert len(counts) == tp.world_size and min(counts) > 0
    rank = tp.rank_in_group
    assert topk_dst.shape[0] == counts[rank]
    start = chunk.gather_token_start
    assert start is not None and start >= 0
    assert chunk.token_start == start + sum(counts[:rank])
    assert chunk.token_end == chunk.token_start + counts[rank]
    # Every rank visits the same metadata chunks; tiny chunks stay replicated
    # on ALL ranks. The upstream all_gatherv handles unequal counts without
    # padding or reducing integer indices. Pool IDs remain request-relative.
    gathered = tp.all_gatherv(topk_dst.contiguous(), dim=0, sizes=counts)
    end = start + sum(counts)
    assert gathered.shape == (end - start, topk_dst.shape[1])
    return gathered, start, end

'''

OLD_FINISH = '''            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                if positions is not None:
                    # Fused expand-pools + append-tail into one Triton kernel
                    # (replaces ~25 elementwise ops). seq_len is token-granular
                    # (pos+1); the kernel derives pool_len internally.
                    q_seq = (
                        positions[chunk.token_start : chunk.token_end].to(torch.int32)
                        + 1
                    )
                    expanded = kpool_ops.expand_pools_and_append_tail(
                        pool_ids, q_seq, index_kpool
                    )
                else:
                    valid = pool_ids >= 0
                    expanded = kpool_ops.expand_pools_to_tokens(
                        pool_ids, valid, topk_tokens, index_kpool
                    )
                topk_indices_buffer[
                    chunk.token_start : chunk.token_end, : expanded.shape[-1]
                ] = expanded
'''

NEW_FINISH = '''            # TITAN_PREFILL_GATHER_BEGIN
            topk_dst, output_start, output_end = _titan_gather_prefill_topk(
                topk_dst, chunk
            )
            if index_kpool > 1:
                pool_ids = topk_dst.to(torch.int64)
                if positions is not None:
                    # The collective restored GLOBAL batch-row order. Use
                    # the same global positions, including request boundaries
                    # and any decode prefix, to append each query's own tail.
                    q_seq = positions[output_start:output_end].to(torch.int32) + 1
                    expanded = kpool_ops.expand_pools_and_append_tail(
                        pool_ids, q_seq, index_kpool
                    )
                else:
                    valid = pool_ids >= 0
                    expanded = kpool_ops.expand_pools_to_tokens(
                        pool_ids, valid, topk_tokens, index_kpool
                    )
                topk_indices_buffer[
                    output_start:output_end, : expanded.shape[-1]
                ] = expanded
            elif chunk.shard_row_counts is not None:
                topk_indices_buffer[output_start:output_end, :topk_tokens] = topk_dst
            # TITAN_PREFILL_GATHER_END
'''


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Source drift: expected one anchor, got {source.count(old)}: {old[:100]!r}")
    return source.replace(old, new, 1)


def patch_sources(kpool, metadata):
    """Pure transformation for CPU verification before touching installed code."""
    if MARKER in kpool or MARKER in metadata:
        if MARKER not in kpool or MARKER not in metadata:
            raise ValueError("Partially applied sharding patch; inspect both files")
        if HELPER.strip() not in kpool or NEW_FINISH not in kpool:
            raise ValueError("Existing sharding patch differs from this version")
        if "MIN_SHARD_TOKENS = 256" not in metadata:
            raise ValueError("Existing patch has a different threshold")
        return kpool, metadata
    if "workspace=(logits_workspace, k_bf16_workspace, q_bf16_workspace)" not in kpool:
        raise ValueError("Apply patch_indexer_workspace.py before this patch")
    for contract in ("shard_row_counts", "gather_token_start", "spec.gather_start"):
        if contract not in metadata:
            raise ValueError(f"Missing upstream metadata contract: {contract}")
    kpool = replace_once(kpool, "from vllm.forward_context import get_forward_context",
                         "from vllm.distributed import get_tp_group\nfrom vllm.forward_context import get_forward_context")
    kpool = replace_once(kpool, "\n@eager_break_during_capture\ndef sparse_attn_indexer_kpool(",
                         "\n" + HELPER + "\n@eager_break_during_capture\ndef sparse_attn_indexer_kpool(")
    kpool = replace_once(kpool, OLD_FINISH, NEW_FINISH)
    # Fail at layer construction, before a request could use unsupported
    # decode-shard metadata. The feature flag defaults are image-dependent.
    kpool = replace_once(kpool, "        super().__init__()\n        self.k_cache = k_cache",
                         '''        super().__init__()
        if envs.VLLM_INDEXER_QUERY_SHARD and envs.VLLM_INDEXER_DECODE_SHARD_MIN_REQS != 0:
            raise ValueError(
                "Titan kpool implements PREFILL query sharding only; set "
                "VLLM_INDEXER_DECODE_SHARD_MIN_REQS=0 explicitly."
            )
        self.k_cache = k_cache''')
    comment_start = metadata.index("# Shard only when the top-k all-gather")
    threshold_end = metadata.index("MIN_SHARD_TOKENS = 2048", comment_start) + len("MIN_SHARD_TOKENS = 2048")
    metadata = metadata[:comment_start] + '''# TITAN_KPOOL_PREFILL_QUERY_SHARD_V1
# Titan TP4 target: include fair-scheduler batches of 2040 prefill tokens.
# Gather only 512 pool IDs per row, before token expansion. 256 is a target
# tuning point requiring an on-device A/B, not a measured universal crossover.
# sparse_attn_indexer_kpool has an explicit eager CUDA-graph break.
MIN_SHARD_TOKENS = 256''' + metadata[threshold_end:]
    ast.parse(kpool)
    ast.parse(metadata)
    return kpool, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    paths = [args.root / KPOOL_PATH, args.root / METADATA_PATH]
    before = [p.read_text() for p in paths]
    after = patch_sources(*before)  # Validate every anchor before any write.
    reports = []
    for p, old, new in zip(paths, before, after):
        changed = old != new
        if args.apply and changed:
            backup = p.with_name(p.name + ".before_titan_prefill_query_shard")
            if backup.exists() and backup.read_text() != old:
                raise ValueError(f"Backup differs from current unpatched source: {backup}")
    for p, old, new in zip(paths, before, after):
        changed = old != new
        if args.apply and changed:
            p.with_name(p.name + ".before_titan_prefill_query_shard").write_text(old)
            p.write_text(new)
        reports.append({"file": str(p), "changed": changed,
                        "sha256": hashlib.sha256(new.encode()).hexdigest()})
    print(json.dumps({"applied": args.apply, "files": reports,
                      "required_env": {"VLLM_INDEXER_QUERY_SHARD": "1", "VLLM_INDEXER_DECODE_SHARD_MIN_REQS": "0"}}, indent=2))


if __name__ == "__main__":
    main()

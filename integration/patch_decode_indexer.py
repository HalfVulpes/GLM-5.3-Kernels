#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for the GLM-5.3-Kernels SM80 integration; see docs/SOURCES.md.
"""Pinned GLM kpool decode optimization: slim logits and optional TP4 B8 rows.

Read-only by default; --apply modifies the selected vLLM source tree.
Apply after the workspace and prefill query-sharding patches. Production
measurements used --mode slim-query8 with QUERY_SHARD=1 and
DECODE_SHARD_MIN_REQS=8. See docs/INTEGRATION.md for the exact compatibility
boundary. No services are restarted and no GPU work is run by this script.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

RELATIVE_PATH = Path("model_executor/layers/sparse_attn_indexer_kpool.py")
SLIM_MARKER = "TITAN_DECODE_SLIM_CANDIDATE_V1"
QUERY_MARKER = "TITAN_DECODE_QUERY8_CANDIDATE_V1"

OLD_CALL = '''            logits = fp8_paged_mqa_logits_triton(
                padded_q_quant_cast,
                kv_cache,
                padded_weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                max_model_len=max_model_len,
                clean_logits=False,
            )'''

SLIM_CALL = '''            # TITAN_DECODE_SLIM_CANDIDATE_V1
            # Metadata already uses pool-granular context lengths. A fixed
            # upper bound keeps shape/stride stable during CUDA graph replay.
            logits = fp8_paged_mqa_logits_triton(
                padded_q_quant_cast,
                kv_cache,
                padded_weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                max_model_len=(max_model_len + index_kpool - 1) // index_kpool,
                clean_logits=False,
            )'''

HELPER = '''
# TITAN_DECODE_QUERY8_CANDIDATE_V1
def _titan_decode_query8_bounds(decode_metadata, batch_size, next_n):
    """Only the balanced B8/TP4 shape measured by the independent A/B."""
    bounds = decode_metadata.shard_bounds
    if (bounds is None or batch_size != 8 or next_n != 1
            or decode_metadata.requires_padding):
        return None
    tp = get_tp_group()
    assert tp.world_size == 4
    expected = (tp.rank_in_group * 2, tp.rank_in_group * 2 + 2)
    assert tuple(bounds) == expected, "Decode metadata does not match TP4 B8 rows"
    return expected

'''

QUERY_CALL = '''            # TITAN_DECODE_SLIM_CANDIDATE_V1
            # Select only query rows; K/tail writes above remain replicated.
            decode_query8_bounds = _titan_decode_query8_bounds(
                decode_metadata, batch_size, next_n
            )
            group_lo, group_hi = decode_query8_bounds or (0, batch_size)
            if decode_query8_bounds is not None:
                assert index_kpool == 4 and topk_tokens == 2048
            seq_lens = seq_lens[group_lo:group_hi]
            logits = fp8_paged_mqa_logits_triton(
                padded_q_quant_cast[group_lo:group_hi],
                kv_cache,
                padded_weights[group_lo * next_n:group_hi * next_n],
                seq_lens,
                decode_metadata.block_table[group_lo:group_hi],
                max_model_len=(max_model_len + index_kpool - 1) // index_kpool,
                clean_logits=False,
            )'''

OLD_GUARD = '''        if envs.VLLM_INDEXER_QUERY_SHARD and envs.VLLM_INDEXER_DECODE_SHARD_MIN_REQS != 0:
            raise ValueError(
                "Titan kpool implements PREFILL query sharding only; set "
                "VLLM_INDEXER_DECODE_SHARD_MIN_REQS=0 explicitly."
            )'''

NEW_GUARD = '''        if envs.VLLM_INDEXER_QUERY_SHARD and envs.VLLM_INDEXER_DECODE_SHARD_MIN_REQS not in (0, 8):
            raise ValueError(
                "Titan decode query candidate accepts "
                "VLLM_INDEXER_DECODE_SHARD_MIN_REQS=0 or 8 only."
            )'''

GATHER = '''        if decode_query8_bounds is not None:
            # Equal fixed counts use one NCCL AllGather on the current
            # stream. Allocation is captured once in FULL_DECODE_ONLY;
            # every TP rank issues the same collective before expansion.
            pool_topk = get_tp_group().all_gatherv(
                pool_topk, dim=0, sizes=[2, 2, 2, 2]
            )
            assert pool_topk.shape == (8, 512)

'''


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Source drift: expected one anchor, got {source.count(old)}: {old[:110]!r}")
    return source.replace(old, new, 1)


def patch_source(source, mode="slim"):
    if mode not in ("slim", "slim-query8"):
        raise ValueError(mode)
    if "TITAN_KPOOL_PREFILL_QUERY_SHARD_V1" not in source:
        raise ValueError("Apply the complete production patch_runtime first")
    if SLIM_MARKER not in source:
        source = replace_once(source, OLD_CALL, SLIM_CALL)
    elif SLIM_CALL not in source and QUERY_CALL not in source:
        raise ValueError("Existing decode slim patch differs from this version")
    if mode == "slim-query8":
        if QUERY_MARKER not in source:
            source = replace_once(source, "\n@eager_break_during_capture\ndef sparse_attn_indexer_kpool(",
                                  "\n" + HELPER + "\n@eager_break_during_capture\ndef sparse_attn_indexer_kpool(")
            source = replace_once(source, SLIM_CALL, QUERY_CALL)
            source = replace_once(source, "        num_padded_tokens = batch_size * next_n\n",
                                  "        num_padded_tokens = batch_size * next_n\n        decode_query8_bounds = None\n")
            source = replace_once(source, "        # Resolve to token-level indices in the output buffer.\n",
                                  GATHER + "        # Resolve to token-level indices in the output buffer.\n")
            source = replace_once(source, OLD_GUARD, NEW_GUARD)
        elif HELPER.strip() not in source or QUERY_CALL not in source or GATHER not in source or NEW_GUARD not in source:
            raise ValueError("Existing query8 candidate differs from this version")
    elif QUERY_MARKER in source:
        raise ValueError("Cannot downgrade query8 candidate to slim in place; use its backup/base image")
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("slim", "slim-query8"), default="slim")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    path = args.root / RELATIVE_PATH
    before = path.read_text()
    after = patch_source(before, args.mode)
    backup = path.with_name(path.name + ".before_titan_decode_candidate")
    if args.apply and before != after:
        if not backup.exists():
            backup.write_text(before)
        path.write_text(after)
    print(json.dumps({"mode": args.mode, "applied": args.apply, "changed": before != after,
                      "file": str(path), "sha256": hashlib.sha256(after.encode()).hexdigest(),
                      "required_decode_env": "0" if args.mode == "slim" else "8 to enable B8; 0 disables decode sharding",
                      "production_wiring_changed": False}, indent=2))


if __name__ == "__main__":
    main()

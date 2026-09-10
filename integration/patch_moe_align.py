# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for the GLM-5.3-Kernels SM80 integration; see docs/SOURCES.md.
"""Wire the verified Titan small aligner into the actual Marlin call site."""
import ast
from pathlib import Path

RELATIVE = Path('model_executor/layers/fused_moe/experts/marlin_moe.py')
MARKER = '# TITAN_SMALL_DETERMINISTIC_MOE_ALIGN_V1'
OLD = '''    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_size_m,
        global_num_experts,
        expert_map,
        ignore_invalid_experts=True,
    )'''
NEW = '''    # TITAN_SMALL_DETERMINISTIC_MOE_ALIGN_V1
    # Exact GLM-5.3 W4A16 group32 TP4 geometry. This changes only stable
    # route layout construction; all GEMMs/activation/reduction stay intact.
    if (
        envs.VLLM_DETERMINISTIC_MOE_ALIGN
        and 1 <= hidden_states.shape[0] <= 8
        and topk_ids.shape[0] == hidden_states.shape[0]
        and topk == 8 and block_size_m == 8
        and E == 288 and global_num_experts == 288
        and expert_map is None
        and K == 4096 and marlin_moe_intermediate_size(w1, w2) == 512
        and quant_type == scalar_types.uint4b8 and input_dtype is None
        and hidden_states.dtype == torch.bfloat16
        and w1_scale.shape[1] == 128 and w2_scale.shape[1] == 16
        and topk_ids.is_cuda and topk_ids.is_contiguous()
        and topk_ids.dtype in (torch.int32, torch.int64)
    ):
        sorted_token_ids, expert_ids, num_tokens_post_padded = _titan_moe_align_small(
            topk_ids, num_warps=1 if topk_ids.numel() <= 16 else 4
        )
    else:
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            block_size_m,
            global_num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )'''


def patch_source(source):
    if MARKER in source:
        if NEW not in source or 'from titan_moe_align import titan_moe_align_small as _titan_moe_align_small' not in source:
            raise ValueError('An incompatible Titan small-alignment patch is present')
        return source
    if source.count(OLD) != 1 or source.count('import torch\n') != 1:
        raise ValueError('Pinned Marlin call-site source drift')
    source = source.replace('import torch\n', 'import torch\nimport vllm.envs as envs\n'
        'from titan_moe_align import titan_moe_align_small as _titan_moe_align_small\n', 1)
    source = source.replace(OLD, NEW, 1)
    ast.parse(source)
    return source


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    target = args.root / RELATIVE
    before = target.read_text()
    after = patch_source(before)
    if args.apply and after != before:
        target.write_text(after)
    print('MoE source validation passed' + ('; applied' if args.apply else '; dry run'))

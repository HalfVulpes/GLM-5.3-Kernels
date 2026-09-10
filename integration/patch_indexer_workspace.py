# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for the GLM-5.3-Kernels SM80 integration; see docs/SOURCES.md.
"""Workspace source transformation; invoke through integration/apply.py."""
from pathlib import Path

def patch_tree(root):
    ROOT = Path(root)
    p = ROOT / 'v1/attention/backends/mla/indexer.py'
    s = p.read_text()
    a = '    return max_model_len * 40'
    b = (
        "    # At most max_num_seqs full contexts, compressed by the model's pool4.\n"
        '    return ((max_model_len + 3) // 4) * vllm_config.scheduler_config.max_num_seqs'
    )
    assert s.count(a) == 1
    p.write_text(s.replace(a, b))
    p = ROOT / 'model_executor/layers/sparse_attn_indexer_kpool.py'
    s = p.read_text()
    a = (
        '            values_spec,\n'
        '            scales_spec,\n'
        '            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),'
    )
    b = (
        '            values_spec,\n'
        '            scales_spec,\n'
        '            ((envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4,), torch.float32),\n'
        '            ((total_seq_lens, head_dim), torch.bfloat16),\n'
        '            ((q_quant.numel(),), torch.bfloat16),\n'
        '            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),'
    )
    assert s.count(a) == 1
    s = s.replace(a, b)
    a = (
        '        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(\n'
        '            values_spec,\n'
        '            scales_spec,\n'
        '        )'
    )
    b = (
        '        k_quant_full, k_scale_full, logits_workspace, k_bf16_workspace, q_bf16_workspace = workspace_manager.get_simultaneous(\n'
        '            values_spec,\n'
        '            scales_spec,\n'
        '            ((envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4,), torch.float32),\n'
        '            ((total_seq_lens, head_dim), torch.bfloat16),\n'
        '            ((q_quant.numel(),), torch.bfloat16),\n'
        '        )'
    )
    assert s.count(a) == 1
    s = s.replace(a, b)
    a = (
        '                logits = fp8_mqa_logits_triton(\n'
        '                    q_slice_cast,\n'
        '                    (k_quant_cast, k_scale_cast),\n'
        '                    weights[chunk.token_start : chunk.token_end],\n'
        '                    chunk.cu_seqlen_ks,\n'
        '                    chunk.cu_seqlen_ke,\n'
        '                    clean_logits=False,\n'
        '                )'
    )
    b = a.replace('                    clean_logits=False,', (
        '                    clean_logits=False,\n'
        '                    workspace=(logits_workspace, k_bf16_workspace, q_bf16_workspace),'
    ))
    assert s.count(a) == 1
    p.write_text(s.replace(a, b))
    p = ROOT / 'v1/attention/ops/mqa_logits_triton.py'
    s = p.read_text()
    start = s.index('def fp8_mqa_logits_triton(')
    end = s.index('def warmup_fp8_mqa_logits_triton(', start)
    part = s[start:end]
    part = part.replace('    clean_logits: bool = True,', (
        '    clean_logits: bool = True,\n'
        '    workspace=None,'
    ), 1)
    part = part.replace('        _select_prefill_kv_group(q.shape[0], kv[0].shape[0]),', (
        '        _select_prefill_kv_group(q.shape[0], kv[0].shape[0]),\n'
        '        workspace,'
    ), 1)
    part = part.replace('    kv_group: int,', (
        '    kv_group: int,\n'
        '    workspace=None,'
    ), 1)
    a = '    logits = torch.empty((M, N), dtype=torch.float32, device=q.device)'
    b = (
        '    if workspace is None:\n'
        '        logits = torch.empty((M, N), dtype=torch.float32, device=q.device)\n'
        '    else:\n'
        '        assert M * N <= workspace[0].numel()\n'
        '        logits = workspace[0][:M*N].view(M, N)'
    )
    assert part.count(a) == 1
    part = part.replace(a, b)
    a = (
        '    q_bf16 = q.to(torch.bfloat16)\n'
        '    k_bf16 = k_fp8.to(torch.bfloat16)'
    )
    b = (
        '    if workspace is None:\n'
        '        q_bf16 = q.to(torch.bfloat16)\n'
        '        k_bf16 = k_fp8.to(torch.bfloat16)\n'
        '    else:\n'
        '        k_bf16 = workspace[1][:N]\n'
        '        q_bf16 = workspace[2][:q.numel()].view(q.shape)\n'
        '        k_bf16.copy_(k_fp8)\n'
        '        q_bf16.copy_(q)'
    )
    assert part.count(a) == 1
    part = part.replace(a, b)
    s = s[:start] + part + s[end:]
    p.write_text(s)

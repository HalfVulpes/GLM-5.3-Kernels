"""Qualified source transforms; staging only."""
def transform(site):
    root = site
    p = root / 'vllm/model_executor/layers/linear.py'
    s = p.read_text()
    old = '        return self._gemm_impl(layer, x, layer.weight, bias)'
    assert s.count(old) == 1
    new = '        # Qualified Morrowmake 74-SM BF16 kernel: no draft tokens or cache changes.\n        if (os.environ.get("TITAN_RECIPE_THIN_GEMM", "0") == "1"\n                and bias is None and x.ndim == 2 and layer.weight.ndim == 2\n                and 1 <= x.shape[0] <= 8 and x.dtype == torch.bfloat16\n                and layer.weight.dtype == torch.bfloat16\n                and x.stride(1) == 1 and layer.weight.stride(1) == 1\n                and tuple(layer.weight.shape) in {\n                    (6416,4096),(6144,4096),(4096,4096),(4096,3072),\n                    (2048,4096),(4096,2048),(4096,1536),(1024,4096)}):\n            from vllm.titan_recipe_thin import thin_gemm\n            return thin_gemm(x, layer.weight)\n        return self._gemm_impl(layer, x, layer.weight, bias)'
    if '\nimport os\n' not in s:
        s = s.replace('import torch', 'import os\nimport torch', 1)
    p.write_text(s.replace(old, new))
    p = site / 'vllm/models/glm5next/nvidia/kda.py'
    s = p.read_text()
    assert s.count('    chunk_kda_with_fused_gate,') == 1
    s = s.replace('    chunk_kda_with_fused_gate,\n', '').replace('from vllm.transformers_utils.configs.glm5_next import', 'from titan_prefill.dispatch import chunk_kda_with_fused_gate\nfrom vllm.transformers_utils.configs.glm5_next import')
    p.write_text(s)
    p = site / 'vllm/model_executor/layers/fused_moe/experts/marlin_moe.py'
    s = p.read_text()
    old = '        ctx = self._lora_context\n        if ctx is None:\n            fused_marlin_moe('
    assert s.count(old) == 1
    new = '        ctx = self._lora_context\n        if ctx is None:\n            from titan_prefill.dispatch import maybe_marlin\n            if maybe_marlin(self, output, hidden_states, w1, w2, topk_weights,\n                            topk_ids, activation, global_num_experts, expert_map,\n                            apply_router_weight_on_input, workspace13, workspace2):\n                return\n            fused_marlin_moe('
    p.write_text(s.replace(old, new))

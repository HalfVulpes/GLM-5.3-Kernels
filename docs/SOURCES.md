# Source map

Upstream: [vLLM](https://github.com/vllm-project/vllm), commit `3bec275739c6f4cc7c2ff403d0556d477e4d0f33` (Apache-2.0).

| Published component | Upstream origin / relationship |
| --- | --- |
| `kernels/titan_mla.py` | Adapted `vllm/v1/attention/ops/triton_mla_sparse_kernel.py`; INT8 cache arithmetic, NoPE path, tiling and split/merge tuning. |
| `kernels/titan_kv.py` | Project-specific Triton writer for the packed per-token scale layout. |
| `kernels/titan_moe_align.py` | Project-specific pairwise stable routing kernel matching the deterministic contract in `moe_align_block_size.py`. No Marlin GEMM code is vendored. |
| `integration/titan_backend.py` | Adapted Triton/XPU sparse MLA backend interface; supplies the specialized cache writer and reader. |
| Indexer integration patches | Modify the kpool indexer, shared indexer metadata and Triton MQA workspace allocation; gather contract follows upstream TP communicator/standard indexer behavior. |
| INT8 layout patch | Modifies `mla_attention.py` and `kv_cache_interface.py` to account for 528 bytes consistently. |
| MoE call-site patch | Modifies `experts/marlin_moe.py` to select the specialized route aligner only for its verified shape. |

The public package keeps the previously GPU-validated computation bodies. Its MLA imports/constants are expressed directly in PyTorch/Triton; packaging, CLI path handling and source preflight are new. A private deployment runtime is not included and is not asserted to be a pristine upstream wheel.

## Upstream SHA256

These hashes identify source provenance. The installer enforces the seven transformed input files listed in `integration/upstream-manifest.json`.

| Source | SHA256 |
| --- | --- |
| [vllm/model_executor/layers/attention/mla_attention.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/model_executor/layers/attention/mla_attention.py) | `6efbbb7f1913eb40516586a1a9dbfaa7be80e96c7135bf5a39a6cd6d3ce6f514` |
| [vllm/model_executor/layers/fused_moe/experts/marlin_moe.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py) | `6abb28d2eb1929d441ba58e85f16951bda1869e7f4decdc5d02d333f7961a215` |
| [vllm/model_executor/layers/fused_moe/moe_align_block_size.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/model_executor/layers/fused_moe/moe_align_block_size.py) | `faa9e8e8f72d6b8d513957b22259739ef641c3fafa8d8f63ea6075f99f4cf245` |
| [vllm/model_executor/layers/sparse_attn_indexer_kpool.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/model_executor/layers/sparse_attn_indexer_kpool.py) | `ba31be4feba9a10a9818c8fee777bf3f79c5c94108f312e2a2f869f2af6acc4a` |
| [vllm/triton_utils/__init__.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/triton_utils/__init__.py) | `e248f5a8555d414a35f1f8fb2392284a57c61d5e837ef462468e3d4a356156c3` |
| [vllm/v1/attention/backends/mla/indexer.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/v1/attention/backends/mla/indexer.py) | `eb6b082a527fe5d0f442e5fa2fae299738f5e1b2f5f940dd0f6bf6b6a89fa70e` |
| [vllm/v1/attention/backends/mla/triton_mla_sparse.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/v1/attention/backends/mla/triton_mla_sparse.py) | `2aa00f65339f518036c9a0afca8a6a7e19fd9556b5d72f9e0e30af6ea313a371` |
| [vllm/v1/attention/backends/mla/xpu_mla_sparse.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/v1/attention/backends/mla/xpu_mla_sparse.py) | `1527f91b313b4cc343abdc20dd393db0f302c7bc6793cd2fed2d1f76174ab6dc` |
| [vllm/v1/attention/ops/mqa_logits_triton.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/v1/attention/ops/mqa_logits_triton.py) | `9aaa6d0e955d272eb49910f915f62c3d4e74da6b9484d77601d2c1e8e5fce419` |
| [vllm/v1/attention/ops/triton_mla_sparse_kernel.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/v1/attention/ops/triton_mla_sparse_kernel.py) | `9f024c3b3b20d59c6fa4bba8bf0daa275a0a21da4a8200971b4130166de7aeb7` |
| [vllm/v1/kv_cache_interface.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/v1/kv_cache_interface.py) | `7cc794b1863a993dfd04897450640e392bc32525dc9a4c168a76707c99e836fa` |

License SHA256: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`. The pinned upstream root has no NOTICE file.

## Optional router/KDA/padding sources

The following files were independently fetched from the same public revision
and their bytes matched the isolated tested runtime inputs exactly. The
optional installer enforces these source hashes plus the tested output hashes
in `integration/decode-manifest.json`. A private image is not needed.

| Public source, relative to `vllm/` | SHA256 |
| --- | --- |
| [models/glm5next/nvidia/model.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/models/glm5next/nvidia/model.py) | `2e0bb44a38a96cc788e6bb23bd819b1c642b16aedd0a5fe5e0fec9e7fcfec312` |
| [models/glm5next/nvidia/kda.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/models/glm5next/nvidia/kda.py) | `be510e17c1bd0f412a3fefd66b8372e3334233f462b4f447154fe9bf453dfd1f` |
| [model_executor/layers/fused_moe/runner/moe_runner.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/model_executor/layers/fused_moe/runner/moe_runner.py) | `80d4b53fdbbbaf2aa9408f22ea476bacbdbb9db5cef0e51fca62c36a4f384d25` |
| [model_executor/layers/fused_moe/layer.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/model_executor/layers/fused_moe/layer.py) | `ebdc4af2f553ce2a70e666f9da3ac647c79d538680c284ce98dd4ac64785661b` |
| [third_party/flash_linear_attention/ops/kda.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/third_party/flash_linear_attention/ops/kda.py) | `8508d5e091e645baa86ba698ef1366298fed2070bfa63b40522ce15a434a2c68` |
| [third_party/flash_linear_attention/ops/fused_recurrent.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py) | `0b5d2bac9463c2711a718084134b5d00f4c0166fcd7758f3db10c7f2b8a05241` |
| [third_party/flash_linear_attention/ops/op.py](https://github.com/vllm-project/vllm/blob/3bec275739c6f4cc7c2ff403d0556d477e4d0f33/vllm/third_party/flash_linear_attention/ops/op.py) | `456da84fd12411cf53c96a90dd4f78f49afb454c3d51d2c524ea8e7106fe8ae5` |

The published runtime `titan_kda_strided.py` is byte-identical to the isolated
tested module: `ef39789fe50ff69ce95769c318a0a771213a5c22c018d804ec4f458002abaaf9`.
Its pointer substitutions reverse to original recurrence function SHA256
`085c1e75103c53488471b05ec888dacd9172844d78637dfe09de9f51d58e1366`.
The copied upstream FLA MIT license SHA256 is
`1350bfbef13ce4d3d3bdaa2f1fc4b1d1117d846732a2a62f891a67f3d5356d0d`.
The existing standalone KV/MLA/MoE files are unchanged by this addition.

The inactive-row corrections modify the same pinned GLM model plus the model
KDA layer listed above. The installer verifies four targets and three unchanged
companions. Its composed output hashes are `b6ce3802…16f2f` for the model,
`9404da52…8b193` for the MoE runner, `76bdb286…73f22` for the model KDA layer,
and `979e979f…740a3` for the recurrence wrapper; full hashes are in
`integration/decode-manifest.json`. Transformations reproduce the isolated
test payload byte-for-byte. Qualification is tracked separately from source
verification.

The final model adds an environment-feature guard to the earlier tested
padding source `7b57f99a64f1399f3b1d0c4273e8e41ca5ebc375edd676c7da5a74263f0134ca`.
Removing only that import and predicate operand exactly recovers the earlier
source. Its true branch is unchanged under the pinned
`VLLM_MOE_SKIP_PADDING=1` configuration; false-flag behavior is covered by CPU
regression checks. GPU receipts identify the earlier tested source separately
from this source-equivalence qualification; see
[the guard evidence](evidence/padding-env-guard.json).

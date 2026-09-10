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

# Pinned vLLM source integration

## Obtain the source

```bash
git clone https://github.com/vllm-project/vllm.git
git -C vllm checkout 3bec275739c6f4cc7c2ff403d0556d477e4d0f33
python integration/apply.py --vllm-root ./vllm/vllm
python integration/apply.py --vllm-root ./vllm/vllm --apply
```

Build/install vLLM using that revision's own instructions and dependencies. Install this repository's standalone modules into the same Python environment. The source preflight is validated against the public commit; full-model performance was measured on a tuned serving runtime based on it, not on a newly rebuilt unmodified upstream environment.

`--vllm-root` names the package directory containing `v1/` and `model_executor/`. The top-level installer checks all seven input SHA256 values in `integration/upstream-manifest.json`. It applies:

1. 528-byte MLA layer/cache-spec accounting.
2. Fixed-size indexer logits and FP8-to-BF16 workspaces.
3. TP4 prefill query partitioning and pool-index gathering.
4. Slim pool-granular decode logits and the B8 query partition.
5. Exact small MoE routing at the Marlin call site.
6. The specialized sparse MLA backend adapter.

All transformations and Python syntax checks finish in a temporary tree before the first target write. `--apply` saves originals under `.glm53-kernels-original/` and an applied manifest under `.glm53-kernels-applied.json`. Repeating the same command checks the resulting hashes and reports that the patch is already applied. Source drift fails before patching. Individual patch scripts are useful for review, but the aggregate installer is the supported entry point.

Do this in an offline build tree or stopped environment: multiple file writes are not atomic to running workers. To undo, start from a fresh checkout of the pinned revision, or restore the saved original files and remove the applied manifest/backup after retaining any unrelated changes. The installer never restarts a process, changes GPU visibility or connects to an inference API.

## Relevant runtime choices

The measured configuration used:

```bash
export VLLM_USE_DEEP_GEMM=0
export VLLM_DETERMINISTIC_MOE_ALIGN=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128
export VLLM_INDEXER_QUERY_SHARD=1
export VLLM_INDEXER_DECODE_SHARD_MIN_REQS=8
```

Model/engine choices included TP4, BF16 activations, Marlin W4A16 group32 experts, `TRITON_MLA_SPARSE`, `int8_per_token_head`, context length 262144 and maximum eight active sequences. The model's hybrid planner resolved 2304-token blocks. Merely changing a dtype name does not install the packed layout.

The MoE specialization is guarded by the real call-site geometry: B1–8, E288, top8, block M8, hidden4096, per-rank expert intermediate512, group32 scales, no expert-parallel map and deterministic alignment enabled. Other call-site shapes retain the upstream aligner.

Prefill query sharding activates at 256 or more query rows. B8 decode sharding requires exactly eight requests, one next token and no padding; each rank processes two complete query rows. Other decode shapes use slim replicated logits. K/V cache writes remain replicated. The gather carries pool indices, not the full logits or latent cache.

## What is not installed

No scheduler/admission patches, raw checkpoint cache, image/video processor changes, API/harness fixes, system services, Docker images or model configuration are installed. In particular, this repository alone does not reproduce the original eight-full-context memory/admission result. Check memory headroom and model behavior in your complete engine configuration.

No NCCL transport setting is forced by the installer. The reference machine had four independent PCIe Gen2 x16 paths, no NVLink, and disabled P2P after failed correctness tests. Select and validate transport settings for the actual machine rather than assuming that every SM80 device has the same connectivity.

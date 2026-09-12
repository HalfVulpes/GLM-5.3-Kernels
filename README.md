# GLM-5.3 Kernels

Shape-specialized Triton kernels and optional vLLM source patches for **GLM-5.3 Flash W4A16 on SM80**. Developed for a four-GPU, **70-SM-per-device** system running 262,144-token contexts.

This is an inference-only kernel release (no backward/autograd implementation), not a model checkpoint or a complete serving distribution. The three standalone KV/MLA/MoE modules require PyTorch and Triton. The optional `titan_kda_strided` module additionally requires the pinned vLLM utility and recurrence helpers; installing the package does not activate it.

## Included

| Component | What changes |
| --- | --- |
| `titan_kv.store_int8` | Dynamically quantizes each 512-value MLA latent into a **528-byte row**: 512 INT8 values, one FP32 scale, 12 zero padding bytes. |
| `titan_mla.triton_mla_sparse_attention` | Fuses INT8 dequantization into sparse NoPE MLA; reuses the shared K/V latent and uses split-KV decode. BF16 tensor-core products, FP32 accumulation/softmax. |
| `titan_moe_align.titan_moe_align_small` | Single-CTA, stable MoE route layout for **B1–8, top-8, 288 experts, block M=8**. It does not replace expert GEMMs. |
| `integration/` | Bounded indexer workspaces, TP4 prefill query partitioning, pool-sized decode logits, B8 decode query partitioning, and the MLA cache-planner/backend adapter. |
| Optional K1v3 router integration | Preserves registered gate/weight aliases; the GLM caller delegates its one gate evaluation to the existing MoE runner after auxiliary-stream input readiness. |
| Optional K2 `titan_kda_strided` | Reads the merged6416 Q/K/V/beta views directly for ordinary B2–8 KDA decode, preserving the upstream recurrent arithmetic and fallback. Requires vLLM. |
| Inactive-row corrections | Initializes the unused KDA output suffix and clears MoE input rows selected by the existing padding mask. Active-token formulas and state are preserved. |

The tested MLA contract is **NoPE latent 512, 16 query heads per rank, one KV head, 2176 index slots**. Negative index/slot padding is supported. The inherited RoPE branches are outside the tested contract. Do not interpret the reference machine as a stock 108-SM A100 or the modules as universally tuned kernels.

## Install the standalone kernels

Use an existing CUDA-enabled PyTorch/Triton environment. The reference measurements used Python 3.12, PyTorch **2.13.0+cu130**, Triton **3.7.1**, CUDA 13.0 and SM80. Dependency declarations alone are not a compatibility guarantee for other versions or hardware.

```bash
git clone https://github.com/HalfVulpes/GLM-5.3-Kernels.git
cd GLM-5.3-Kernels
python -m pip install --no-deps .
```

```python
import torch
from titan_kv import store_int8
from titan_mla import triton_mla_sparse_attention
from titan_moe_align import titan_moe_align_small

# Run only on a GPU allocated for this work.
device = 'cuda:0'
n, block = 262144, 256
latent = torch.randn(n, 512, dtype=torch.bfloat16, device=device)
cache = torch.empty(n // block, block, 528, dtype=torch.int8, device=device)
slots = torch.arange(n, dtype=torch.int64, device=device)
store_int8(latent, cache, slots)

q = torch.randn(1, 16, 512, dtype=torch.bfloat16, device=device)
indices = torch.randint(n, (1, 1, 2176), dtype=torch.int32, device=device)
indices[:, :, 2048:] = -1
output = triton_mla_sparse_attention(
    q, cache.view(n, 1, 528), indices, 512 ** -0.5, sm_count=70
)  # [1, 16, 512]

routes = torch.randint(288, (8, 8), dtype=torch.int32, device=device)
sorted_ids, expert_ids, padded_count = titan_moe_align_small(routes)
```

The cache is a custom binary layout, **not** an arbitrary INT8 tensor or another engine's INT8 KV format. Writer, attention reader and cache planner must agree on all 528 bytes. The sparse indices above are synthetic; in a model they come from the actual indexer. Use BF16 latents contiguous in the last dimension, the contiguous INT8 cache shape shown above, and valid cache-row indices or negative padding. Inputs must be on the same allocated CUDA device; arbitrary strided cache layouts and out-of-range positive indices are outside this low-level API contract.

## Optional vLLM integration

The adapter targets the publicly available vLLM commit
[`3bec275739c6f4cc7c2ff403d0556d477e4d0f33`](https://github.com/vllm-project/vllm/commit/3bec275739c6f4cc7c2ff403d0556d477e4d0f33).
Its seven input files are fingerprinted. The installer validates and transforms a temporary copy first; **the default command writes nothing to the target tree**.

```bash
python integration/apply.py --vllm-root /path/to/vllm/vllm
# Only for an offline build tree or stopped environment:
python integration/apply.py --vllm-root /path/to/vllm/vllm --apply
```

Install the kernel package into the worker environment before using that build. See [integration instructions](docs/INTEGRATION.md) for patch order, runtime flags, backup/rollback and the boundaries of the published subset. No private base image is required to apply these source patches; this release does not claim a newly built pristine vLLM wheel has repeated the full deployment benchmark.

The four-part profile—K1v3, K2, KDA padding and MoE padding—has a separate, explicit installer so the existing integration defaults remain unchanged:

```bash
python -B integration/apply_decode.py --vllm-root /path/to/vllm/vllm
# Only for an offline build tree or stopped environment:
python -B integration/apply_decode.py --vllm-root /path/to/vllm/vllm --apply
```

Its public source inputs match the isolated tested runtime byte-for-byte; the four resulting vLLM source files and K2 module are pinned to the final output hashes. Read the [decode contracts and qualification status](docs/DECODE_OPTIMIZATIONS.md) before enabling it. Six matched cold/warm comparisons reproduce the first 64 token/log-probability values exactly, and all 20 functional boundary cases pass. Eight concurrent full-context requests, each with four images and one video, also pass all capacity checks. A separate eight-lane→single-lane comparison still differs; its raw failure is preserved.

The reference configuration meets the predefined release criterion based on
quality, function and stability. In High mode, the candidate matches the
baseline's **17/18 completed common checks**, with natural completion of all
five candidate coding responses and two valid candidate tool calls. In Off
mode, it scores **19/21 versus the baseline's 21/21**, including a real
topological-sort regression. This is a bounded acceptance with a disclosed
tradeoff, not a claim that all quality checks pass. The baseline's sixth High
task was interrupted and is excluded from parity claims. See
[the quality comparison](docs/evidence/quality-comparison.json).

Capacity and short-throughput receipts identify the padding-v2 source; the
final v3 adds an equivalent enabled-branch environment guard, documented
separately. The final High-mode quality run uses v3.

## Measurements and validation

Historical measurements on the reference system:

- Small MoE route alignment: approximately **74–76 µs → 1.6–5.1 µs**; 336 GPU tensor-equality cases.
- TP4 prefill indexer logits/top-k/gather: approximately **20.63 ms → 6.64 ms** at 2048 query rows and 65536 pool entries (**3.10× for that operator path**).
- INT8 MLA: maximum output NMSE approximately **4.97e-5**, minimum cosine similarity approximately **0.999975**, compared with the BF16 source latents in the tested cases.

These are different tests, not multipliers to combine. The full application also used scheduling, admission, checkpoint loading and multimodal/API work that is deliberately outside this repository. Its reported eight-stream capacity and throughput are **not promises made by installing this kernel package**.

For the final padding profile, the isolated 8192-input/512-output serving tests
observed median decode changes of **+0.08% at C1, +2.73% at C4 and +2.70% at C8**
against the recorded baseline. Each workload has three repetitions in separate,
noninterleaved server runs; these are observations, not statistical
non-inferiority or single-kernel causal estimates. See the
[final-profile measurements](docs/BENCHMARKS.md#final-routerkdapadding-profile).

See [benchmark methodology and limits](docs/BENCHMARKS.md) and the public numeric summary in [`docs/measurements.json`](docs/measurements.json). GPU verification is opt-in with `--run`; the default checks/plans do not launch GPU work. CPU checks do not replace GPU correctness tests.

## Source and license

Apache-2.0. Derived vLLM copyright notices are retained. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [SOURCES.md](docs/SOURCES.md) identify the upstream code, changes and pinned source hashes. No model weights, service credentials, private request logs, host addresses, or frontend are included.

## 中文概要

本仓库公开 GLM-5.3 Flash 在 SM80／70 SM、TP4、262K 上下文目标下的部分内核与接入补丁：动态 INT8 MLA KV、稀疏 MLA、单 CTA MoE 路由对齐、indexer workspace/query 分片，以及可选的单次router计算、KDA跨步读取和两处无效填充行规范化。六项同条件冷暖比较前64个token/logprob完全一致，20项功能检查通过；跨并发数值差异单独保留，不改写原失败记录。最终发布按回答质量、功能与稳定性验收，完整状态见资格记录。安装脚本默认只检查；GPU 测试需显式指定 `--run`。算子加速和整机吞吐是不同口径。

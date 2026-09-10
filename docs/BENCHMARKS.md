# Measurement scope

The numeric observations in [measurements.json](measurements.json) are selected from the reference system's **2026-09-10 historical GPU tests**. Host identifiers, addresses, deployment paths, application prompts and credentials are not part of that export. This repository preparation ran CPU/source/package checks; it did not rerun GPU benchmarks on the active inference service.

Reference hardware: SM80, 70 SMs and 64 GiB reported memory per device, four devices with independent PCIe Gen2 x16 links, no NVLink, P2P disabled. These are CMP170HX-derived devices, not stock 108-SM A100s. PyTorch 2.13.0+cu130 and Triton 3.7.1 were used. Kernel/module timing is not a measurement of SM occupancy, maximum board power or whole-server throughput.

## INT8 MLA correctness

The test uses 262144 synthetic BF16 source latents, per-token quantization, B1/6/8 and split counts 1/4/8/16. Outputs are compared against attention computed using the unquantized BF16 source latents with FP32 reference arithmetic. Negative slots, padded indices and fully masked requests are checked separately.

- Quantized latent NMSE: approximately **1.535e-4**.
- Maximum attention-output NMSE: approximately **4.967e-5**.
- Minimum attention-output cosine similarity: approximately **0.999975**.

These are numerical tests, not evidence of unchanged accuracy on every coding or visual task. The standalone verification script uses the production per-token scale layout; its error budget must not be confused with groupwise quantization.

## MoE route alignment

The historical baseline is vLLM's **deterministic stable Torch aligner**. The comparison covers all three output tensors, including unused/padded entries. Seven routing patterns, two integer dtypes, three warp counts and B1–8 give **336 GPU equality cases**.

The often quoted **74–76 µs → 1.6–5.1 µs** range describes graph-replayed route alignment, not the MoE GEMMs or complete layer. Eager dispatch has different timings and is retained separately in the JSON. The selected policy uses one warp for up to 16 routes and four otherwise. The historical script alternated/compared candidate settings; the JSON records graph unroll and timing samples.

The new public benchmark defaults to a graph-safe Torch reference and a CPU oracle so the kernel can be tested without vLLM. That reference is **not the same timing baseline** as the historical vLLM path. Use its explicit `--baseline vllm` option for a compatible installed reference; do not relabel a different baseline as a reproduction of the historical speedup.

## TP4 prefill indexer

At 65536 pool entries (pool4 compression of 262144 context tokens), the measured path includes logits, CUDA top-k and TP gathering of 512 int32 pool indices per query.

| Query rows | Replicated wall median | TP4 query-sharded wall median | Ratio |
| ---: | ---: | ---: | ---: |
| 2040 | 20.556 ms | 6.630 ms | 3.100× |
| 2048 | 20.632 ms | 6.639 ms | 3.108× |

The historical run used two warmups per branch, six samples per branch and three iterations per sample. A/B order alternated. Each sample uses the maximum synchronized wall time across the four ranks. Six cases checked exact top-k order on all ranks, including unequal row partitions, mixed ranges and a replicated tiny tail. Selected raw samples are retained in `measurements.json`.

This excludes query projection, K-cache gathering, pool expansion and the rest of the model. A 3.1× improvement here is **not** a 3.1× improvement in prefill TTFT. The public CPU indexer checker validates source-level row/collective semantics, not NCCL performance; a distributed GPU rerun needs the complete compatible vLLM runtime and an allocated maintenance window.

## Full serving deployment is a separate result

The larger deployment validated eight independent 262144-total-token requests and recorded approximately **78.62 generated tokens/s at C1** and **337.52 aggregate generated tokens/s at C8**, about 25.05%/23.02% above its earlier configuration. Those runs used different fixed output lengths and the C8 case included visual input. The comparison changes a configuration/kernel bundle, not one variable.

That stack also contained scheduler/admission, periodic hybrid-state prefix caching, checkpoint loading, multimodal and API work not shipped here. The numbers are context about where the kernels came from, not a reproduction claim for the standalone package or the seven-file public source patch. Cold prefill, warm-prefix prefill, graph-forward timing and decode throughput must be reported separately.

## Running your own measurements

See [tests/README.md](../tests/README.md) and [benchmarks/README.md](../benchmarks/README.md). The GPU commands require `--run`, user-selected logical device visibility and the measured SM80/70-SM profile. Do not co-schedule them with a production model and then interpret the result as an isolated benchmark. Record the full shape, dtype, split count, reference implementation, graph/eager mode, memory accounting and software versions with each result.

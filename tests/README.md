# Validation

Run these commands from the repository root. The default suite needs only
Python's standard library and performs **no GPU execution**:

```bash
python -m unittest discover -s tests -v
python tests/verify_mla.py
python tests/verify_source_equivalence.py
python benchmarks/moe_align.py
```

The default checks cover the stable alignment algebra, core import boundaries,
lazy CUDA access using dependency doubles, and explicit GPU opt-in. They do
not replace real torch/triton import or GPU numerical validation.

To compare the computational functions with a source directory you already
have, supply it explicitly. The test does not bundle or discover a private
checkout and does not print its absolute path:

```bash
python tests/verify_source_equivalence.py --reference-dir /path/to/reference-kernels
```

That directory must contain `titan_kv.py`, `titan_mla.py` and
`titan_moe_align.py`. Computational function ASTs, signatures, decorators and
MLA launch constants must match; imports and docstrings may differ. AST
equivalence is not a GPU correctness result.

## Explicit GPU validation

Choose the GPU using your own `CUDA_VISIBLE_DEVICES` mapping and pass its
logical `--device`. Every GPU entry point requires `--run` and rejects devices
outside the measured **SM80 / 70-SM** profile. No particular physical GPU is
selected by the repository.

```bash
python tests/verify_mla.py --run --device cuda:0 --output results/mla-correctness.json
python benchmarks/moe_align.py --run --device cuda:0 --correctness-only \
  --output results/moe-correctness.json
```

MLA validation uses synthetic BF16 data, per-token INT8 scales, masked indices,
empty rows and split counts 1/4/8/16. Its default 262144-row cache and numerical
reference do not load a model checkpoint. Quantization comparisons are
chunked, and the PyTorch allocator defaults to a 1024 MiB cap; driver memory
outside that allocator is not included.

The MoE test checks all three output tensors, including padding, against a
stable CPU oracle and a torch-only GPU reference. B1–8, seven routing patterns,
two index dtypes and three warp counts yield 336 GPU comparisons. An optional
`--baseline vllm` uses an installed compatible deterministic vLLM aligner;
vLLM is **not** needed for the default core tests. With torch installed,
`python benchmarks/moe_align.py --torch-cpu` independently compares the tensor
reference with the CPU oracle in 1712 cases, without initializing CUDA.

## Optional integration source checks

These CPU tests require the source layout expected by `integration/`. A generic
vLLM release may differ and is intentionally rejected rather than patched
approximately. The provided tree is read only; transforms execute in memory.
Pristine, workspace-only, prefill-patched and fully integrated trees are
accepted when their expected source contracts match. For pristine indexer
sources, the workspace transformer runs first against memory-backed files.
The full decode-query patch's constructor guard does not affect the prefill
checker: an already-present prefill patch is verified and reused in place.

```bash
python tests/verify_moe_dispatch.py --source-root /path/to/vllm
python tests/verify_indexer_sharding.py --source-root /path/to/vllm
```

For the indexer checker, `--snapshot-dir` alternatively accepts a directory
containing `kpool.py`, `metadata.py` and `utils.py`; pristine snapshots also
need `mqa_logits_triton.py` for the prerequisite workspace transform. No source snapshot is bundled.
The tests verify dispatch guards, row partitions, collective order and global
output placement using CPU doubles. They do not run NCCL or start a service.

The public preparation pass ran CPU checks only. The GPU scripts need a new
explicit run on suitable hardware; do not label their new torch reference or
graph timings as newly measured until such a run is performed.

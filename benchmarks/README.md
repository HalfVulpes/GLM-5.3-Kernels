# Standalone benchmarks

Both scripts default to CPU checks or a printed plan. Install the core
torch/triton dependencies, select your own logical CUDA device, and explicitly
add `--run` to execute GPU work on the SM80 / 70-SM target:

```bash
python benchmarks/moe_align.py --run --device cuda:0 \
  --output results/moe-align.json
python benchmarks/mla.py --run --device cuda:0 --batches 1 6 8 \
  --output results/mla-splits.json
```

Run the numerical checks in `tests/README.md` before interpreting timing results.
The MLA benchmark measures synthetic warm-cache split/merge graph replay.
The MoE benchmark measures eager and graph execution, with identical warmup
and an explicit graph reset after each case. Its default reference is a
fixed-shape torch-only implementation intended for graph capture, checked against the stable CPU oracle;
it is not claimed to reproduce historical vLLM reference timings.

If a compatible vLLM is explicitly installed, `moe_align.py --baseline vllm`
compares its deterministic aligner instead. This optional comparison does not
make vLLM a dependency of the core kernels.

Results contain synthetic numeric metrics, not checkpoints, real prompts or
service credentials. Module latency is not server throughput, model quality,
full-context concurrency, power efficiency, or a fresh validation of other
hardware. No GPU benchmark was newly executed while preparing these scripts.

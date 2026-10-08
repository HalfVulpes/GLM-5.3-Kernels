# V20: SM80 / 74-SM inference profile

This optional source profile contains the operators qualified for the V20
deployment. It preserves TP4, eight 262144-token request slots, INT8 KV,
BF16 activations and FP32 recurrent state. **No MTP or DFLASH.** Installing
the root package alone does not activate this profile.

Source: [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx/tree/ab60b723ada254a442a4ba5ff27bf837aa27ef83),
commit `ab60b723ada254a442a4ba5ff27bf837aa27ef83`. Apache-2.0 and FLA MIT
notices are retained. See `../../licenses/sm74-*`.

## Operators

* `titan_recipe_thin.py`: upstream 74-SM BF16 thin GEMM, unchanged. Split-K
  increases parallel work for narrow decode matrices; the last arriving CTA
  reduces FP32 partials in fixed order without another kernel launch.
  The integration selects only M=1..8 and eight measured TP4 weight shapes.
* `titan_prefill/kda.py`: fused chunk KDA prefill; selected only for H16/D128,
  at most 2304 tokens and eight sequences, with FP32 state. Decode remains on
  the existing recurrence. Fusion reduces intermediate memory traffic and
  launches. Empty-sequence IDs and reused metadata buffers are covered by
  the private index/cache helpers.
* `titan_prefill/marlin.py` and `moe_split_align.py`: stable route lists split
  across 64/48/32/16-row blocks, avoiding excessive padding of small experts;
  explicit SM80 Marlin tiles improve utilization. Our weights use **group32**,
  unlike the upstream recipe's group128. The native template adaptation
  changes the tile group-block parameter to 2; no weight requantization.
  Eligibility is BF16/uint4b8/E288/top8/K4096/N512 at M384..2304, without
  bias, act-order or LoRA. Unsupported calls retain the incumbent path.

Metadata is immutable within one model forward; cache entries are invalidated
across forwards. Inference tensors outside a forward bypass metadata caching.
Production prefill CUDA graphs are disabled; decode graphs are unchanged.

## Native build

Use a fresh checkout of the pinned source and a CUDA development environment
matching the qualified PyTorch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0, CXX ABI1.
The runtime loader verifies native ABI, schedule version2 and group size32.

```sh
python profiles/sm74/prepare_source.py /path/to/pinned-vllm
VLLM_BUILD_AMPERE_MARLIN=1 MAX_JOBS=2 python \
  /path/to/pinned-vllm/csrc/libtorch_stable/moe/ampere_marlin/build_standalone.py \
  --out native --build-dir marlin-build
```

The preparer verifies pristine source fingerprints. It excludes unused decode
extension sources and fast-dequant variants. Legacy wide group128 kernels
remain unchanged and are never selected by this profile. It does not enable
speculation. Supply a writable isolated build cache and the upstream build
dependencies. No native compilation runs during serving.

## Integration

`integration-manifest.json` pins three input files and the exact tested output
hashes. This is an addon to an already integrated engine, **not a claim that
arbitrary pristine vLLM versions are supported**. Unknown source is rejected.
The tool stages transformations and never writes into the input installation:

```sh
python profiles/sm74/prepare_integration.py --site-packages /path/to/site-packages
python profiles/sm74/prepare_integration.py --site-packages /path/to/site-packages \
  --output /new/offline-overlay
```

In an offline image build, install the overlay and place the built
`_ampere_marlin_C.abi3.so` under `vllm/`. Enable `TITAN_RECIPE_THIN_GEMM=1`,
`TITAN_KDA_PREFILL=1`, `TITAN_MARLIN_PREFILL=1`. Revalidate a newly built binary
on the target system. The private model-serving/container distribution and
weight-dependent captures are intentionally excluded.

Stop/drain serving and invalidate previous numerical session caches and
compiled caches before changing kernels. KV shapes alone do not establish
numerical cache compatibility. Do not delete model checkpoints or account data.

See [measurements and limitations](../../docs/SM74_V20.md).

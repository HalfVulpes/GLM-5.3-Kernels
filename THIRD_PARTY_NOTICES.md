# Third-party notices

This repository contains modifications and adaptations of vLLM source code, licensed under Apache License 2.0. Copyright contributors to the vLLM project. The unmodified upstream license is included as `LICENSE`.

The source revision is `3bec275739c6f4cc7c2ff403d0556d477e4d0f33`. A file-by-file map and checksums are in [docs/SOURCES.md](docs/SOURCES.md) and `integration/upstream-manifest.json`. The upstream root did not contain a NOTICE file at this revision; this document describes the provenance of this release.

Changes include per-token INT8 cache storage/attention, modified MLA layout planning and backend wiring, bounded indexer workspace reuse, query-row partition/gather integration, and a specialized deterministic MoE route aligner. Public packaging replaces the sparse MLA module's vLLM utility imports with equivalent PyTorch/Triton imports and constants, without changing the Triton computation bodies.

The FlashMLA combine-kernel pattern comment is inherited from vLLM's sparse MLA source. This repository does not separately vendor FlashMLA code. It also does not publish or replace Marlin CUDA GEMMs; the MoE patch changes route alignment before the existing vLLM/Marlin operations. PyTorch, Triton, vLLM and any model checkpoints are separate dependencies with their own licenses.

The optional `kernels/titan_kda_strided.py` derives its recurrent Triton body
from the pinned vLLM vendored flash-linear-attention implementation. Songlin
Yang and Yu Zhang's copyright notice is retained, and the upstream MIT text
is included in `licenses/flash-linear-attention-MIT.txt`. Its modifications
add token strides and conservative dispatch/alias guards. The optional router
patch changes ownership of an existing gate evaluation and its nullable
custom-operator interface; it does not vendor a new gate GEMM implementation.
Two additional Apache-2.0 source transforms define inactive KDA output rows
and masked MoE input rows in the pinned GLM model. They retain the original
source notices and do not change model weights.

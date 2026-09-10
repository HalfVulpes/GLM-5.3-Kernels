# Third-party notices

This repository contains modifications and adaptations of vLLM source code, licensed under Apache License 2.0. Copyright contributors to the vLLM project. The unmodified upstream license is included as `LICENSE`.

The source revision is `3bec275739c6f4cc7c2ff403d0556d477e4d0f33`. A file-by-file map and checksums are in [docs/SOURCES.md](docs/SOURCES.md) and `integration/upstream-manifest.json`. The upstream root did not contain a NOTICE file at this revision; this document describes the provenance of this release.

Changes include per-token INT8 cache storage/attention, modified MLA layout planning and backend wiring, bounded indexer workspace reuse, query-row partition/gather integration, and a specialized deterministic MoE route aligner. Public packaging replaces the sparse MLA module's vLLM utility imports with equivalent PyTorch/Triton imports and constants, without changing the Triton computation bodies.

The FlashMLA combine-kernel pattern comment is inherited from vLLM's sparse MLA source. This repository does not separately vendor FlashMLA code. It also does not publish or replace Marlin CUDA GEMMs; the MoE patch changes route alignment before the existing vLLM/Marlin operations. PyTorch, Triton, vLLM and any model checkpoints are separate dependencies with their own licenses.

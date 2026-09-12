# Optional router/KDA optimizations and inactive-row corrections

This integration targets the existing four-device SM80 setup with
70 SMs/device and the GLM-5.3 Flash W4A16 TP4 layout. It is opt-in and does not
change the original standalone kernel APIs. Qualification status is recorded
in [decode-qualification.json](decode-qualification.json). The four components
are K1v3 router ownership, K2 strided recurrence, KDA output padding and MoE
input padding. Raw numerical and quality failures remain visible.

## K1v3: one registered router gate

The pinned GLM MoE caller computes `self.gate(hidden_states)` and supplies its
logits to a runner that also owns `self.gate` and evaluates it again. K1v3 keeps
the registered gate, correction bias, dtype handling and checkpoint aliases.
The GLM caller supplies `router_logits=None`; the runner computes the gate at
its original internal location, after its existing shared-expert auxiliary
stream synchronization.

```python
# The factory still receives gate=self.gate, preserving registered aliases.
output = self.experts(hidden_states=hidden_states, router_logits=None)
```

The two custom operators, their fake handlers, `MoERunner.forward`, and
`MoERunner._forward_impl` consistently accept `Tensor | None`. Both real entry
paths reject `None` when no gate is owned, before transforms or device work;
routing still requires a real tensor after the internal gate call. Existing
tensor callers retain the prior behavior. No synthetic/uninitialized logits
tensor is used. Gate source ownership is not a measured runtime call count;
actual dispatch and output/state checks belong in qualification evidence.

## K2: read the actual merged KDA layout

The model's BF16 merged projection leaves Q/K/V/beta views with row stride6416.
At B>1 the original wrapper packs these views before the recurrence. The
optional wrapper selects the new path only for B2–8 and that merged layout.
`titan_kda_strided` then verifies CUDA device consistency, H16/K128/V128,
BF16 activations, FP32 state, output layout, gate flags, and conservative
storage non-aliasing. B1, contiguous inputs and unsupported cases keep the
original wrapper.

Only recurrence token-pointer addressing changes. The original arithmetic,
token loop, masks, NULL-state behavior, output/state objects, BK128/BV8,
one-warp/three-stage launch and gate operations remain intact. Reversing the
published pointer substitutions reproduces the pinned original function
hash. This CPU source proof complements numerical tests; it cannot establish
GPU equality by itself.

The existing ordinary-decode caller guarantees unit `cu_seqlens` segments.
Host metadata guards do not inspect tensor contents or synchronize CUDA to
prove that value-level condition. Do not call the low-level helper with
arbitrary multi-token/speculative sequences. The helper's direct B1 entry is
available for controls; the production wrapper's specialization begins at B2.

Unlike `titan_kv`, `titan_mla`, and `titan_moe_align`, this module requires
`vllm.triton_utils` and
`vllm.third_party.flash_linear_attention.ops.op`. The imports and entire tested
runtime file are unchanged. vLLM is not automatically added as an unpinned
package dependency; obtain/build the documented public revision.

## Source installation and tests

See [INTEGRATION.md](INTEGRATION.md). `integration/decode-manifest.json` records
the public revision, all before/after hashes and companion hashes. The new
installer changes four vLLM files; the KDA runtime module is provided by
the Python package. It never contacts an inference endpoint or restarts a
service.

```bash
python -B -m unittest discover -s tests -v
python -B tests/verify_decode_source.py --vllm-root /path/to/vllm/vllm \
  --exercise-installer
```

The second command copies only the pinned files into a temporary directory,
then verifies apply and idempotence there. It leaves the supplied source
untouched. The standard suite additionally tests rollback after an injected
write failure, symlink/path rejection, missing-source rejection, the exact
runtime payload hash, and pointer-only source derivation. Neither command
launches GPU work or imports the runtime module.

## Deterministic inactive rows

The model can process 2303 real tokens in a 2304-row buffer. The KDA output
previously left its inactive suffix unwritten, and the MoE router/expert path
could consume unspecified values in padded rows. Two source changes define
these inactive values before downstream processing:

```python
# Model KDA layer: initialize only the output suffix outside real tokens.
if core_attn_out.shape[1] > num_actual_tokens:
    core_attn_out[:, num_actual_tokens:].zero_()

# GLM MoE caller: use the existing mask on the tested non-SP path.
if (envs.VLLM_MOE_SKIP_PADDING and not self.is_sequence_parallel
        and is_forward_context_available()):
    padding = get_forward_context().is_padding
    if padding is not None:
        hidden_states.masked_fill_(padding[:hidden_states.shape[0], None], 0)
```

These changes preserve active-token formulas, weights, and recurrent state.
They do not sanitize active NaNs or infer padding from input values. Missing
forward context, missing masks, sequence parallelism, and disabled
`VLLM_MOE_SKIP_PADDING` retain their previous path. The final guard checks this
environment feature before accessing a possibly stale padding mask. Set
`VLLM_MOE_SKIP_PADDING=1` for the tested TP4 profile. The model KDA file has its
own pinned input hash; the MoE correction is
composed after the exact K1v3 transform. All four resulting vLLM files are
checked against the final-profile hashes. The final environment guard was
added after the v2 GPU qualification source: removing only its import and
predicate term recovers that tested source exactly, and its enabled branch
is unchanged. [The CPU/source bridge](evidence/padding-env-guard.json) records
this distinction; it is not a new GPU measurement of v3.

Diagnostics found matching retained/restored KDA state and complete MLA/index
cache pages. The first differing valid activation appeared at the first MoE
layer despite matching MoE input. KDA padding alone fixed self-origin warm
replay but left a longer-prime difference; adding the masked MoE input fix
produced exact matched comparisons. This supports the inactive-row
explanation without identifying the particular Marlin instruction responsible.

## Evidence limits

The public staging pass reproduces the exact tested runtime bytes from a
public source checkout. It has not rebuilt a fresh upstream wheel or repeated
full-model tests on that fresh build. Earlier full-model comparisons were
separate server processes with small repetition counts and some output
variability; they do not establish statistical non-inferiority.

On the final padding profile, all 20 functional boundary/cache/cancellation
cases pass with zero preemptions. All six matched serial cold/warm comparisons
have identical first 64 token signatures and log probabilities, maximum
absolute difference 0. The mixed B8→B1 comparison still has token divergence
and a maximum log-probability difference of about 0.393528 within its common
39-token prefix; that comparison changes both batch and cache conditions.
This difference is not an answer-error rate. The source receipt retains
`passed=false`, `matched_cache_passed=true` and `cross_batch_passed=false`.
The [metadata-only boundary record](evidence/matched-cache-boundaries.json)
preserves the original receipt hash and each comparison outcome.

The release acceptance criterion is answer quality, function and stability;
0.05 cross-batch numerical equality is an advisory diagnostic, not a blocking
gate. This policy decision does not change raw test outcomes or claim a
statistical quality guarantee.

The final padding-profile capacity rerun passed all eight gates: eight requests
with 260096 input + 2048 output tokens each, four images and one video per
request, all 24 text retrieval values, all 40 unique encoder features actually
scheduled, eight near-full contexts resident simultaneously, and zero
preemptions or logged OOM/engine errors. Wall time was 1007.440s; aggregate
first-to-last-content decode was 196.885 tokens/s. This all-eight-MM workload
differs from the historical one-MM-plus-seven-text run. It is a load/encoding
and text retrieval test, not visual reasoning accuracy. See
[the independently checked metadata](evidence/full-context-mm8.json).

The final-profile 8192-input/512-output tests passed all nine benchmark rows.
Median changes from the recorded baseline were +0.08%, +2.73% and +2.70% at
C1/C4/C8. The parent receipt retains `passed=false` because a canary produced
the correct arithmetic equation rather than the requested bare number; this
is preserved as a formatting failure. [Performance metadata](evidence/short-decode-performance.json)
retains source hashes, samples and output variability. These noninterleaved,
three-repeat measurements do not prove statistical non-inferiority.

## Bounded quality acceptance

The final v3 High-mode candidate meets the release criterion recorded before
its result: at least the baseline's 17 of 18 completed common checks, natural
completion of all five candidate coding responses and two valid candidate
tool calls. The candidate returns exactly the same pass/fail vector as that
baseline: **17/18**, not 18/18. Settings are temperature 1.0, top-p 0.95, High
effort, 16384 thinking tokens, 65536 maximum output tokens and seed 37.

The baseline's sixth High-mode SSE task was stopped before natural completion.
It is censored, not passed, and this comparison establishes no High-mode SSE
parity. The baseline High run also completed no tool cases, so the two valid
candidate tool calls are not a paired High-tool comparison. No broad
statistical quality equivalence is claimed.

The Off-mode, temperature 0 comparison remains a real regression: baseline
**21/21**, candidate **19/21**, with one topological-sort task failing two
functional checks. This is not a scorer or formatting failure. The release
accepts that disclosed limitation under its quality/function/stability
criterion; it does not rename the underlying failed results as passes.
See [quality-comparison.json](evidence/quality-comparison.json) for source
receipt hashes, the predefined criterion and exact comparison scope.

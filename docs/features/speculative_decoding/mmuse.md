# MMuse reference adapters

## DeepSeek-V4: reference implementation

For `method="mmuse"` and a `deepseek_v4` target, the V2 worker selects
`MMuseReferenceSpeculator`; Ascend selects `AscendMMuseReferenceSpeculator`.
Qwen3 targets retain the existing implementation described below.

This adapter follows the training checkout's `speculators_dsv4/training.py`,
`contract.py`, `models/mmuse/compat.py` and
`DSparkOfflineRunner._single_anchor_backbone`. DeepSeek is the teacher; the
trained dense decoder is Qwen3. Auxiliary IDs are one-based post-layer mean-HC
slots and exclude the final teacher HC-head slot.

`DeepseekMMuseReference` uses unpacked linear projections, the exported
`head_dim`, RMSNorm, default RoPE, grouped-query SDPA, dynamic convolution and
trained residual order. Fusion, Selector, Markov and Correction use the existing
inference-only MMuse modules; serving does not import the training package.
Missing, unexpected and incorrectly shaped trained tensors fail loading.
The three scalar gates (`layer_fusion_gate`, `context_hidden_gate`, and
`correction_markov_scale`) accept both legacy scalar `[]` and FSDP2 `[1]`
checkpoint layouts without changing their values. Other shapes stay strict.
Legacy enhanced DSpark exports follow the training migration flags; baseline
DSpark retains its own loader.

The proposer retains only confirmed projected target features, indexed by
persistent request slot and request ID. Rejected suffixes are excluded. Query
blocks are `[anchor, mask, ...]` at absolute positions; slot k predicts
anchor+k+1. Sliding layers expose the same `[anchor-window, anchor)` context to
every query slot. Synthetic query K/V is ephemeral. No draft attention objects
or paged KV groups are registered, so the native Target cache layout is retained.

Each Target TP rank executes a complete replicated draft. Without QuaRot, IO
matrices are supplied by the serving Target and gathered once under TP. They
must be ordinary floating matrices and are converted to the draft's compute dtype.
For an Ascend Target declaring a QuaRot global rotation, each auxiliary layer
is inverse-rotated in FP32 before any trained normalization or gated fusion.
The draft must contain the original teacher's `embed_tokens.weight` and
`lm_head.weight`; these are retained for draft IO. A deployed head may already
contain the teacher's final norm and cannot be restored by inverse rotation
alone. Missing teacher IO or a declared but unavailable rotation matrix causes
a loading error instead of silently borrowing incompatible Target IO.
The declared rotation path is tried first, then `optional/<filename>` inside
the Target directory for exports with a stale `../optional/` relative path.
These interface checks do not establish numerical parity with the training
teacher; verify acceptance and feature parity on the real checkpoint.
Replication and optional context recomputation increase memory and compute costs: this
is a correctness reference, not a throughput optimization. Target backbone
quantization remains the backend's responsibility. Interface checks do not prove
that the serving Target weights match the training teacher numerically.

Require BF16 activations, an unquantized draft,
`enable_prefix_caching=False`, and PP/DP/context parallel sizes of one. Target TP
may exceed one; Draft computation is replicated. Adaptive verification,
watermarking and local argmax reduction are rejected. Eager remains the
reference path; the optional FULL query path is described below.

### DeepSeek MMuse platform attention bridge

DeepSeek Draft attention defaults to the independent SDPA reference. Ascend
provides an explicit dense FIA bridge using TorchNPU's
`npu_fused_infer_attention_score`, the inference operator family used by its
native attention backend. Select it when constructing a new engine:

```python
additional_config = {"mmuse_attention_backend": "ascend_fia"}
```

Use `"reference"` (or omit the key) for the reference path. Other platforms reject
`ascend_fia`; this selector applies to DeepSeek MMuse, not Qwen3 or Target
attention. Recreate the engine/graphs when changing backends.

The bridge preserves the explicit per-layer mask: fixed-anchor context window,
optional causal query block, and padded context exclusion. It converts SDPA's
allowed boolean mask to FIA's excluded boolean mask and passes compact GQA KV
heads directly. BF16/FP16 tensors use dense BNSD layout. Eager and FULL Draft
paths bind the same callable before warmup/capture; runtime errors are not
silently redirected to SDPA.

This is the first kernel-integration stage, not a paged-cache conversion. Existing
verified-feature history, context KV storage, query KV concatenation and graph
ownership remain unchanged. It does not register Draft `AttentionLayerBase` or
updatable attention tasks. Synthetic tests cover hidden/logits/token parity at
the platform boundary; optional NPU tests exercise real FIA and changing inputs
and masks across native graph replay. Trained-model acceptance and TPOT must be
compared on the deployment hardware before selecting this backend for service.

### DeepSeek MMuse FULL query graphs

Target retains the existing DeepSeek/Ascend FULL graph path, including W4A8
quantization and attention/cache handling. With Draft `enforce_eager=True`,
`MMuseReferenceSpeculator` remains eager and can consume graph-produced Target
features. With Draft `enforce_eager=False`, Model Runner V2 selects
`DeepseekMMuseFullSpeculator` / `AscendDeepseekMMuseFullSpeculator`.

The new path reuses the reference model, trained teacher IO and QuaRot binding.
Declared rotations are still undone in FP32 before feature normalization/fusion.
The Draft configuration disables Dynamo compilation on a copy and clears the
Target quantizer. It does not change Target compilation or quantization.

Confirmed history update, feature projection, context normalization and K/V
projection run outside capture. FULL takes one device-to-host metadata snapshot
per proposal, retaining request-ID and contiguous-position validation. Confirmed
feature history updates its reserved storage in place. Context capacities use powers of two from
128 up to the reserved history capacity, including a final non-power-of-two
capacity when necessary. Negative context positions identify padding; both full
and sliding attention mask them. Query anchor conditioning gathers the last
valid context row instead of the end of the padded buffer.

Each native CUDA/NPU graph captures one K-token query decoder and complete MMuse
rollout, including native probability sampling and logits writes when selected.
Requests replay serially through stable input/context buffers; output is copied
before the next request reuses them. There is no Draft paged attention layer and
no Draft updatable-attention task: native Torch operators use fixed shapes.
Target graph updates are unchanged. Native graph ownership participates in the
runner's memory-profiling cleanup so profiling captures are not retained.

For a K=7 checkpoint, preserve the working W4A8 Target settings and use:

```python
enforce_eager = False  # Target
enable_prefix_caching = False
speculative_config = {
    "method": "mmuse",
    "model": "/path/to/deepseek-v4-flash-mmuse",
    "num_speculative_tokens": 7,
    "enforce_eager": False,
    "draft_sample_method": "greedy",  # Or "probabilistic".
}
compilation_config = {
    "cudagraph_mode": "FULL",
    "cudagraph_capture_sizes": [8],  # Target BS1 verify shape; BS4 also needs 32.
}
```

Preserve the baseline Target's compilation mode and other compilation settings;
the example only shows the FULL mode and verification size. Retain smaller
Target capture sizes needed by the workload, including prefill.
Draft capture shapes are context buckets with one complete K-token query, not
the Target token-size list. Probabilistic mode uses request seeds, temperatures
and absolute query positions, with pre-temperature logits cached by persistent
request slot. Capture uses slot -1 and does not write live probability caches.
NPU FP64 Gumbel is rejected. The older optional `mmuse_draft_kv_reuse` must be
disabled for this new path; it remains available for eager Draft execution.

Extra buffers per TP rank include approximately
`(capacity * hidden_size + 2 * layers * capacity * kv_heads * head_dim) * dtype_bytes`,
in addition to the existing per-request feature history and native graph
workspaces. The shared buffers cache the last active request's normalized context
and context K/V: only newly confirmed or overwritten rows are projected, and a
sliding prefix moves within existing storage. A request/slot change recomputes
valid context rows. Padding is zeroed without projection. This avoids allocating
a separate K/V cache for every request, but alternating requests do not benefit
from incremental reuse. Padding in captured query attention, the remaining
metadata synchronization and Python request processing may still limit speed.
Native device validation and throughput measurements remain required.

CPU tests cover padded/unpadded decoder parity, mixed full/sliding layers,
convolution, context anchor conditioning, overwritten contexts and independent
request outputs. Optional native tests capture decoder plus real MMuse heads and
sampling, changing context lengths, seeds and request slots across replays:

```bash
pytest --noconftest --import-mode=importlib tests/v1/spec_decode/test_mmuse_deepseek_graph.py -q
```

These synthetic tests do not validate the W4A8 Target or TP serving. On the
server compare Eager/Eager, Target FULL/Draft Eager, then FULL/FULL using the same
checkpoint, requests and TP layout. Check mean-HC layer IDs/order, live auxiliary
features, QuaRot alignment, rejection rollback, EOS and reused request slots.
Only after correctness checks measure unprofiled steady-state throughput and
memory. Existing DSpark FULL success is the Target execution baseline, not proof
of MMuse feature or probability correctness.

### Optional confirmed-context KV reuse

Set `additional_config["mmuse_draft_kv_reuse"] = True` to cache the replicated
DeepSeek draft's context K/V projections. This is disabled by default. Each
layer keeps only confirmed context K/V after normalization and RoPE; synthetic
query K/V is always recomputed. Cache identity includes the persistent request
slot and request ID. Overwritten suffixes are reprojected and sliding windows
evict old positions. Dummy/profile calls do not commit request state.

The buffers are reserved before memory profiling. Additional memory is
`2 * layers * max_num_seqs * capacity * kv_heads * head_dim * dtype_bytes`,
where capacity is the sliding window for all-sliding decoders and otherwise
`max_model_len`. For 5 layers, 4 requests, a 2048 window, 8 KV heads and head
dimension 256 in BF16 this is 320 MiB per Target TP rank. Prefix movement still
copies cached rows; measured throughput gains depend on the real workload.
Compare enabled/disabled logits, acceptance and steady-state throughput on NPU
before adopting it. This does not enable Target prefix caching or FULL graphs.

### Nonzero temperature validation

Target `temperature=1.0` can use either `draft_sample_method="greedy"` or
`"probabilistic"`. The latter caches final Correction logits in persistent
request/column slots before temperature scaling. Native verification applies
the request temperature to those logits and performs probability-ratio
acceptance plus residual resampling. Correction feedback uses the actual sampled
token, not an unrelated argmax. Keep eager mode and begin with `top_p=1`,
`top_k=-1` and repetition penalty 1 when validating the interface.

CPU tests verify cache routing/feedback and context-KV parity, but do not prove
the native NPU sampling distribution. A focused one-card kernel test is in
`vllm-ascend/tests/e2e/pull_request/one_card/test_gumbel_sampling.py`, selected by
`-k mmuse_probabilistic_proposal`. It calls the actual MMuse NPU sampler binding
and native rejection/resampling, checking proposal and output frequencies at
temperatures 1.0 and 0.7 with FP32/BF16 caches. Run in the matching installed
Ascend test environment; no large model weights are needed. This is not a
substitute for end-to-end acceptance and performance tests with the actual models.

Retain the working Target quantization/tokenizer settings and set:

```python
enforce_eager = True
enable_prefix_caching = False
speculative_config = {
    "method": "mmuse",
    "model": "/home/data/DeepSeek-V4-Flash-mmuse",
    "num_speculative_tokens": 7,  # Must equal this export's block_size.
    "enforce_eager": True,
    "draft_sample_method": "greedy",
}
```

CPU tests compare synthetic checkpoint weights, the complete decoder and
Correction rollout to the literal training offline evaluator, including long
sliding contexts and mixed full/sliding layers. Native GPU/NPU loading, TP
execution, acceptance and throughput still require server verification.

## Qwen3 reference implementation

This experimental adapter includes standalone MMuse inference components for
Model Runner V2. It has not been validated with a trained
`qwen3_4b_mmuse` checkpoint or a GPU/NPU serving run. CPU component comparisons
do not establish end-to-end correctness or speedup.

## Supported first-stage contract

- Qwen3 target and Qwen3 draft with matching hidden width and full vocabulary.
- Speculators-format `speculators_model_type="mmuse"` configuration.
- `sample_from_anchor=True`, full-attention draft layers, enabled Correction.
- Fixed proposal length equal to the trained block size (typically 7).
- Greedy proposals with eager or FULL query execution, one device.
- Probabilistic proposals with eager or FULL query execution, one device.
- No quantization, watermarking or adaptive verification.

These are implementation limits; commenting out a configuration check does not
add support for that mode. Model Runner V1 rejects MMuse explicitly.
The target's generation temperature is separate from the draft sampling mode;
start evaluation with greedy target requests as well.

For probabilistic proposals, set `draft_sample_method` to `"probabilistic"`
in the speculative config and use a positive request temperature. MMuse calls
the backend Gumbel sampler after each final Correction/Markov step, using the
request-state mapping, request temperature, seed and predicted token position.
Pre-temperature logits are cached in the framework's `draft_logits` buffer for
rejection sampling. The sampled token is fed into subsequent online Correction
steps. Static Selector conditioning remains static as specified by the checkpoint.
As with the existing V2 drafter, draft sampling uses temperature, not request
top-k/top-p; target sampling retains its own processing. Zero-temperature requests
remain greedy, including within a mixed batch. Watermarking and adaptive
verification are rejected for this new path. GPU/NPU end-to-end validation of
probabilistic MMuse remains required; previous greedy serving results do not
validate this extension.

## Implementation

Greedy Qwen3 uses `MMuseGreedySpeculator`, directly derived from the generic
`DraftModelSpeculator`. Ascend uses `AscendMMuseGreedySpeculator` and its own
`MMuseAclGraphManager`. Input preparation, model loading and graph capture do
not call the DFlash proposer or DFlash graph managers. Probabilistic execution
selects `MMuseProbabilisticSpeculator` / `AscendMMuseProbabilisticSpeculator`.
These extend our independent query engine and graph managers with request-scoped
sampling buffers; they do not inherit the DFlash proposer.

`MMuseQwen3ForCausalLM` uses native vLLM Qwen3/DFlash Attention and projections.
Its decoder wraps Attention and MLP with the source dynamic convolution, preserving
the trained prepare/sublayer/finish/residual order. Source layer fusion and query
conditioning are retained. Last accepted normalized context features are gathered
per request before query preparation; rejected target features are not used for
the context residual.

`build_mmuse_runtime` constructs local MMuse heads and feature modules, not
a second decoder or verifier. Its inference-only `rollout_correction()` preserves
Selector conditioning, causal Correction cache, hidden/logit feedback and optional
Markov collaboration. Correction cache and feedback are initialized within each
rollout, rather than reused across requests or proposal rounds.

Checkpoint `layers/fc/norm/hidden_norm` keys map to the native backbone; extension
keys map to `runtime.*`. Decoder QKV and MLP packing reuse the DFlash weight mapper.
Missing trainable parameters fail loading. Full-vocabulary embedding and LM-head
weights are intentionally taken from the target; this is distinct from missing
trained draft parameters. Confidence weights are loaded but are not used for
adaptive proposal length in this first version.

### Qwen3 FULL graph boundary

Target uses the framework's original model graph manager. This adapter does not
change Target attention, capture sizes, synchronization or graph-update ordering.
Draft has an independent capture lifecycle:

1. Project the current Target auxiliary features and select the last confirmed
   context anchor, excluding rejected suffixes.
2. Prepare K query rows per request, context/cache slots and absolute positions
   in persistent buffers. Null blocks and rejected/padded rows receive PAD slots.
3. Project and store confirmed context K/V outside the graph. Long prefill
   contexts follow the same path as decode context.
4. Build current-step query attention metadata once.
5. Replay query decoder, complete Correction/Selector/Markov rollout and
   output-buffer write together. Shapes absent from the capture set run eagerly.

Thus FULL covers the draft query forward and rollout, not the whole proposal
pipeline, context projection, Target verification or the scheduling loop.
The model backbone still uses native Qwen3/DFlash attention and weight packing;
shared MMuse prediction modules are not duplicated.

The Draft's copied compilation configuration uses `CompilationMode.NONE` to
avoid compiling the eager MMuse Python rollout through Dynamo. This does not
disable device graph capture and does not change the Target configuration.
Ascend uses native `UpdatableGraph` task updates with the already-built current
metadata. It does not remove required events or promise to fix MEM_WAIT overhead.
Only the updatable full-attention backend is supported by this independent NPU
path; unsupported graph backends fail explicitly.

For a checkpoint with K=7 and `max_num_seqs=1`, retain the existing model/dtype/
tokenizer settings and use:

```python
enforce_eager = False  # Target
speculative_config = {
    "method": "mmuse",
    "model": "/path/to/qwen3_4b_mmuse",
    "num_speculative_tokens": 7,
    "draft_sample_method": "greedy",
    "enforce_eager": False,
}
compilation_config = {
    "mode": 0,
    "cudagraph_mode": "FULL",
    "cudagraph_capture_sizes": [7, 8],
}
```

Use Model Runner V2. Sizes count tokens, not requests: Draft needs a complete
7-row block; Target verification can need 8 rows. For BS4 include 28 and 32,
and preserve smaller capture sizes to exercise shrinking batches. K must equal
the checkpoint's trained block size. Keep TP/PP/DP/CP=1. Context anchor storage
is a nonpersistent model buffer, moved with the model and stable during replay.

Before deployment compare the same requests with NONE, Target FULL + Draft
eager, and both FULL. Warm up separately, disable profiling for timing, compare
output tokens/logits, acceptance, EOS and repeated request-slot reuse. Measure
prefill and decode separately. Graph capture succeeding is not performance or
correctness acceptance.

`tests/v1/spec_decode/test_mmuse_greedy_graph.py` covers independent query layout,
rejection, null blocks, shrinking/reordered batches and compilation isolation.
`test_mmuse_rollout_graph.py` exercises the real prediction runtime with synthetic
weights, including native CUDA/NPU capture when hardware is available. Ascend's
`tests/ut/spec_decode/test_mmuse_greedy_graph.py` covers current-step task resolution
and capture-context restoration. These tests do not validate paged attention or
trained-checkpoint quality; full-model GPU/NPU acceptance is still required.

### Probabilistic FULL execution

Use the FULL configuration above and explicitly change
`speculative_config["draft_sample_method"] = "probabilistic"`. Both Target and
Draft may have `enforce_eager=False`; Target uses its native FULL graph path.
Begin with request `temperature=0.7`, `top_p=1.0`, `top_k=-1`, repetition penalty
1 and a fixed request seed. Draft retains temperature-only proposal sampling;
Target retains its normal processing.

The probability extension captures native Gumbel sampling, final-logit cache
writes and sampled-token feedback together with the MMuse rollout. Request
mapping, seeds, temperatures and absolute positions are updated in persistent
buffers before replay. A query at P predicts token P+1, so its random key uses
P plus the native draft-stream salt. Random state is not frozen at capture
time; repeating the same seed and position intentionally reproduces the draw.

Cache storage is `[max_num_seqs, K, vocab_size]`, in the verifier's head dtype.
Sampling first casts logits to this same precision, then caches them before
temperature scaling. Only live request slots may be written. Capture, dummy
execution and padded rows use state index -1, leaving other requests' caches
unchanged. Zero-temperature requests use argmax within mixed batches.
NPU rejects FP64 Gumbel explicitly. Adaptive verification, dynamic proposal
lengths and watermarking remain unsupported.

`test_mmuse_probabilistic_graph.py` covers slot remapping, rejection-adjusted
positions, cache precision and inactive rows. Optional CUDA/NPU cases capture
real MMuse heads and the native sampler with synthetic weights, then compare
replay against eager execution while changing seeds, anchors, features and
request occupancy. Run from the vLLM repository in the matching backend
environment:

```bash
pytest --noconftest --import-mode=importlib tests/v1/spec_decode/test_mmuse_probabilistic_graph.py -q
```

These checks do not execute full-model paged attention or Target rejection.
Before claiming correctness or speedup, compare the trained checkpoint with
NONE, Target FULL + eager Draft, and both FULL. Check multi-round verification,
first/middle rejection, EOS, reused request slots and output distributions at
nonzero temperature. GPU and NPU need not produce identical random streams.
Measure steady-state throughput without profiling after a separate warmup.

## Environment and launch template

Deployment needs vLLM, vLLM-Ascend when using NPU, their runtime dependencies,
and the target/draft checkpoints. It does not need the training checkout or the
`speculators`/`hs_connectors` packages. Checkpoint identity still uses the existing
`speculators_model_type` field; that field is a format marker, not a Python import.

Inference modules live in `vllm/model_executor/models/mmuse/`: config parsing,
feature fusion, dynamic convolution, Selector, Correction, Markov/confidence heads,
and rollout feedback. Training losses, data loading and optimizers are excluded.
`NOTICE`, `LICENSE`, and `source_manifest.json` record source attribution and hashes.
Preserve training provenance, tokenizer and target/checkpoint hashes for evaluation.

Example on a server with the correct vLLM/Ascend installation and dependencies:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve /path/to/Qwen3-4B \
  --enforce-eager \
  --speculative-config '{"method":"mmuse","model":"/path/to/qwen3_4b_mmuse","num_speculative_tokens":7,"enforce_eager":true,"draft_sample_method":"greedy"}'
```

This is a launch template, not a verified deployment command. Use the block size
in the actual checkpoint, and keep the target identical to the training verifier.
Do not edit a checkpoint's feature switches to bypass missing-weight errors.

## Validation still required

The development-only test `checks/test_mmuse_adapter.py` compares source/runtime heads
on fixed synthetic weights, query conditioning, decoder convolution/residual
ordering with a substituted Attention sublayer, request anchor selection, and
proposal output layout. Its subprocess smoke test blocks all `speculators` imports
while constructing config, loading synthetic weights and executing rollout and
dynamic convolution. Only the numerical reference tests need the training checkout.
These tests do not execute native vLLM Attention kernels.

Before claiming service support, compare the complete backbone and logits on
the actual backend, then validate at least two proposal/verification rounds,
full acceptance, first/middle rejection, EOS and request cancellation. Native
weight packing, cache writes/rollback and GPU/NPU serving remain integration
checks. Only then measure acceptance length, latency, memory and throughput.

## Adapter code layout

- `models/mmuse/validation.py` dispatches model-specific execution checks;
  `models/mmuse_runtime.py` keeps runtime construction, the public validation
  import and legacy rollout helpers used by compatibility/parity checks.
- `worker/gpu/spec_decode/mmuse/greedy.py` and `probabilistic.py` implement the
  current Qwen3 worker paths. Ascend uses corresponding subclasses. The older
  DFlash-based `mmuse/speculator.py` classes are retained import entry points;
  neither worker factory selects them. They are distinct from the checkpoint
  architecture string `MMuseSpeculator`, which remains a model-registry alias
  for `MMuseQwen3ForCausalLM` and must not be removed as an unused worker class.
- `models/mmuse/deepseek_contract.py` validates the DeepSeek training export.
  `models/mmuse/context.py` owns confirmed feature history and optional context
  K/V reuse; the former module retains compatibility imports for these classes.
- `models/deepseek_mmuse_reference.py` loads trained tensors and binds teacher
  or deployed Target IO without changing checkpoint parameter names.
- `worker/gpu/spec_decode/mmuse/reference.py` prepares verified context, query
  blocks and proposals for eager DeepSeek drafting. `deepseek_full.py` adds its
  fixed-context graph path. Both support greedy and probabilistic proposals.
  The Ascend subclasses supply NPU sampling, graph backend and QuaRot binding.

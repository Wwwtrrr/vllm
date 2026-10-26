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

Require BF16 activations, an unquantized draft, eager Target and Draft,
`enable_prefix_caching=False`, and PP/DP/context parallel sizes of one. Target TP
may exceed one; Draft computation is replicated. Adaptive verification,
watermarking and local argmax reduction are rejected. FULL graphs are not
implemented for this path.

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
- Greedy or probabilistic draft proposals on one device. Probabilistic remains
  eager; greedy has an outer FULL capture path described below.
- No quantization, watermarking or adaptive verification.

These are implementation limits; commenting out a configuration check does not
add support for that mode. Model Runner V1 rejects MMuse explicitly.
The target's generation temperature is separate from the draft sampling mode;
start evaluation with greedy target requests as well.

## Qwen3 greedy FULL graph path

Target and Draft have separate graphs. The Target retains the framework's
existing graph path; MMuse captures the Draft decoder and the complete fixed
Correction/Selector rollout through DFlash's outer `_generate_draft` capture.
The inner decoder runs with graph mode NONE to avoid nested graph replay. This
does not disable capture of the outer function. Python configuration branches
are resolved during capture, while tensor operations consume the current input
buffers on each replay. `torch.compiler.disable` on the reference heads disables
Dynamo tracing, not device graph capture.

Draft FULL requires `draft_sample_method="greedy"`, `sample_from_anchor=True`,
and `num_speculative_tokens` equal to the checkpoint's trained `block_size`.
The existing single-device/full-attention restrictions still apply. Unsupported
Attention graph capabilities retain DFlash's existing eager fallback. Draft
`enforce_eager=True` disables its graph manager without forcing Target eager.
The context anchor is a non-persistent module buffer: it moves with the model,
is updated in place before drafting, and does not add checkpoint state.
Ascend's capture wrapper explicitly forwards the MMuse rollout, including on
versions without generic model attribute forwarding.

For a checkpoint trained with block size seven, test in this order:

```python
# Target FULL, Draft eager first; then change only Draft enforce_eager to False.
llm = LLM(
    model="/path/to/Qwen3-4B",
    enforce_eager=False,
    max_num_seqs=1,
    compilation_config={
        "mode": 0,
        "cudagraph_mode": "FULL",
        "cudagraph_capture_sizes": [7, 8],
    },
    speculative_config={
        "method": "mmuse",
        "model": "/path/to/qwen3_4b_mmuse",
        "num_speculative_tokens": 7,
        "draft_sample_method": "greedy",
        "enforce_eager": True,
    },
)
```

MMuse uses seven query rows per request; Target verification usually uses eight.
For four simultaneous requests include the required multiples, for example
`[7, 8, 14, 16, 21, 24, 28, 32]`. These are token counts, not request counts.
Keep capture sizes bounded by scheduler limits. Actual graph dispatch remains
subject to backend and batch compatibility. No experimental changes to the
shared Target graph manager or UpdatableGraph event ordering are required.

`tests/v1/spec_decode/test_mmuse_full_graph.py` tests lifecycle contracts and
replays the real greedy heads with changed inputs/anchors, both with and without
LM-head fusion. CPU checks run without accelerators; CUDA/NPU replay cases skip
when unavailable. These synthetic head tests deliberately substitute decoder
outputs: they do not establish native paged-Attention, scheduling, trained-model
accuracy or end-to-end throughput. Validate actual checkpoint outputs, rejection,
request padding/reuse and acceptance against eager before deployment. FULL
capture success is not a speed guarantee, particularly with Ascend dynamic
Attention updates. DeepSeek reference and probabilistic MMuse keep their eager
contracts.

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

`MMuseSpeculator` inherits `DFlashSpeculator`; `AscendMMuseSpeculator` inherits
`AscendDFlashSpeculator`. Both mix in `MMuseGreedyRolloutMixin`. They retain the
DFlash context insertion and Attention metadata, while selecting an anchor plus
`block_size - 1` mask queries. All `block_size` positions predict tokens.

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
  `models/mmuse_runtime.py` keeps the public validation import and Qwen3 rollout
  helpers.
- `models/mmuse/deepseek_contract.py` validates the DeepSeek training export.
  `models/mmuse/context.py` owns confirmed feature history and optional context
  K/V reuse; the former module retains compatibility imports for these classes.
- `models/deepseek_mmuse_reference.py` loads trained tensors and binds teacher
  or deployed Target IO without changing checkpoint parameter names.
- `worker/gpu/spec_decode/mmuse/reference.py` prepares verified context, query
  blocks and proposals. The Ascend subclass supplies NPU sampling and QuaRot
  matrix loading.

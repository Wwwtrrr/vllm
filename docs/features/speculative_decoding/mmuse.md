# MMuse reference adapter for Qwen3

This experimental adapter includes standalone MMuse inference components for
Model Runner V2. Greedy serving with a real checkpoint has been reported by the
user. FULL graphs and probabilistic graph serving have not been validated in
the development environment. CPU component comparisons do not establish
end-to-end correctness or speedup.

## Supported first-stage contract

- Qwen3 target and Qwen3 draft with matching hidden width and full vocabulary.
- Speculators-format `speculators_model_type="mmuse"` configuration.
- `sample_from_anchor=True`, full-attention draft layers, enabled Correction.
- Fixed proposal length equal to the trained block size (typically 7).
- Greedy or probabilistic draft proposals, one device.
- Eager or experimental FULL graph execution; no piecewise draft graphs.
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

## Target and Draft FULL graphs (experimental)

Target forward uses the existing V2 target graph manager. MMuse query forward,
Correction and draft sampling use the DFlash query graph manager; Ascend keeps
its ACL graph metadata and update-stream implementation. MMuse passes the actual
capture mode to its forward context. The DFlash manager selects full decode-only
query graphs and may fall back to eager if its attention backend cannot capture
them. Target sampling/rejection is not implied to be inside the target graph.

Use these switches to isolate behavior:

| Test | Target `--enforce-eager` | Draft `enforce_eager` | Compilation graph mode |
| --- | --- | --- | --- |
| Baseline | present | `true` | `NONE` |
| Target FULL, Draft eager | absent | `true` | `FULL` |
| Target FULL, Draft FULL | absent | `false` | `FULL` |

Target eager with Draft FULL is not provided as an independent mode: the parent
capture flow initializes and captures the draft from the target's graph setup.
This change does not introduce a separate draft-only graph driver.

Example (change paths and block length to match the actual checkpoints):

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve /path/to/Qwen3-4B \
  --tensor-parallel-size 1 \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --speculative-config '{"method":"mmuse","model":"/path/to/qwen3_4b_mmuse","num_speculative_tokens":15,"enforce_eager":false,"draft_sample_method":"probabilistic"}'
```

The graph path requires full trained blocks and `sample_from_anchor=True`.
Static global Selector still requires draft eager because its validation reads
a device tensor from Python. Corrected online feedback does not execute that
whole-block global-path validation. Existing single-device/full-attention
restrictions remain; graph support does not add quantization or parallelism.

Logs distinguish planning, completed capture and real-request replay submission:

```text
MMuse target graph setup: ...
MMuse draft graph setup: ...
MMuse target graph capture completed: ...
MMuse draft FULL graph capture completed: ...
MMuse target FULL graph replay submitted.
MMuse draft FULL graph replay submitted.
```

Replay logs exclude dummy/profile batches and print once. Submission is not a
device synchronization or numerical-correctness check. Eager execution observed
for an individual batch is also logged once; a large prefill may use eager even
when decode batches replay FULL graphs. Capture planning or startup alone is
not proof of replay. No extra device-to-host synchronization is added for logs.

Run real-checkpoint tests separately so graph and eager engines do not occupy
device memory at the same time:

```bash
python examples/offline_inference/mmuse_full_graph.py \
  --target /path/to/Qwen3-4B --draft /path/to/qwen3_4b_mmuse \
  --block-size 15 --mode eager --output eager.json
python examples/offline_inference/mmuse_full_graph.py \
  --target /path/to/Qwen3-4B --draft /path/to/qwen3_4b_mmuse \
  --block-size 15 --mode target-full --output target-full.json
python examples/offline_inference/mmuse_full_graph.py \
  --target /path/to/Qwen3-4B --draft /path/to/qwen3_4b_mmuse \
  --block-size 15 --mode both-full --output both-full.json
```

The script exercises batches 1/2/4/2/1 with fixed per-request seeds and stores
token IDs and finish reasons. Start with its greedy defaults, compare outputs
and inspect both replay messages. Then repeat with
`--draft-sample-method probabilistic --temperature 0.7` and new output filenames.
Stochastic output equality alone does not prove distribution correctness, and
floating-point differences can change greedy decisions near ties. Validate
proposal logits/cache, full/partial rejection, EOS and cancellation separately.
The script is a serving smoke test, not a latency or throughput benchmark.

CPU dispatch tests are in `tests/v1/spec_decode/test_mmuse_graph.py`; they do not
capture CUDA/ACL graphs. Accelerator and real-checkpoint validation remain
required before claiming FULL support or speedup.

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

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

### Native FULL versus compiled FULL

Graph replay and torch compilation must be controlled independently. Start with
`compilation_config={"mode":0,"cudagraph_mode":"FULL", ...}` to capture native
operators without adding compilation passes. This is not `enforce_eager=True`,
which disables graphs too. Mode 3 is a separate opt-in compiled-FULL experiment.
Neither route has a measured speedup guarantee in this development environment.

Use the explicit configuration builder to cover both query widths:

```python
from vllm.model_executor.models.mmuse.graph_config import (
    build_mmuse_full_graph_config,
)

enforce_eager = False
compilation_config = build_mmuse_full_graph_config(
    block_size=7, max_num_seqs=4, compilation_mode=0,
)
# Pass both to LLM; set speculative_config["enforce_eager"] = False for Draft FULL.
```

For block size 7 this produces `[1,2,3,4,7,8,14,16,21,24,28,32]`; for 15 it produces
`[1,2,3,4,15,16,30,32,45,48,60,64]`. Ordinary one-token decode batch sizes are
included too. Sizes are shared by target/draft managers; draft
queries use K positions and typical target verification uses K+1. The builder
does not alter checkpoint fields or global compilation defaults.

Ascend's current platform setup forces compilation mode NONE when graph mode
is NONE, so setting mode=3 with graph NONE does not provide a reliable
"compiled-only" experiment on this version. `compilation_config=None` means
defaults, which can enable graphs; it is not an off switch.

Target forward uses the existing V2 target graph manager. MMuse query forward,
Correction and draft sampling use the DFlash query graph manager, the same FULL
replay path as DSpark. Ascend rebuilds draft-decoder attention metadata on each
replay with `build_draft_attn_metadatas`. Correction is a block-local SDPA loop
and does not use that metadata or the paged KV cache. The DFlash manager selects
full decode-only query graphs and may fall back to eager if its attention
backend cannot capture them. Target sampling/rejection is not implied to be
inside the target graph.

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
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,3,4,15,16,30,32,45,48,60,64]}' \
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
token IDs, finish reasons and warmed generation times. It disables prefix
caching and warms batch sizes 1/2/4 before measurement. Model initialization,
graph capture and warmup are not timed. Start with its greedy defaults, compare outputs
and inspect both replay messages. Then repeat with
`--draft-sample-method probabilistic --temperature 0.7` and new output filenames.
Stochastic output equality alone does not prove distribution correctness, and
floating-point differences can change greedy decisions near ties. Validate
proposal logits/cache, full/partial rejection, EOS and cancellation separately.
The script is a serving smoke test, not a latency or throughput benchmark.

The script now defaults to native FULL (`--compilation-mode 0`); repeat with
`--compilation-mode 3` and different output names to isolate compiled FULL.
Use `--without-draft --mode target-full` to test the plain target without any
MMuse execution (omit `--draft`). This is important when native target-only
graph execution already exhibits the regression. Keep the same physical
device, requests and profiler settings across comparisons. Profile separately
from timing; synchronous profiler parsing must not be included in generation
performance results.

CPU dispatch tests are in `tests/v1/spec_decode/test_mmuse_graph.py`; they do not
capture CUDA/ACL graphs. Accelerator and real-checkpoint validation remain
required before claiming FULL support or speedup.

### Throughput acceptance before deployment

The earlier server measurements regressed even with no draft and compilation
mode 0. That target-only regression has not been diagnosed from a device trace;
the MMuse metadata changes do not establish a fix. Do not deploy FULL solely
because capture/replay logs appear.

Run the staged acceptance harness on the same idle device with the real weights:

```bash
python examples/offline_inference/mmuse_full_graph_acceptance.py \
  --target /home/data/Qwen3-4B --draft /home/data/qwen3_4b_mmuse \
  --block-size 7 --batch-sizes 1 4 2 --tokens 128 \
  --warmups 2 --repeats 3 --min-ratio 0.95 \
  --output-dir mmuse-full-acceptance
```

Each configuration runs in a fresh process. The order is plain target eager,
plain target native FULL, MMuse eager, target FULL/draft eager, both FULL. A
target-only regression stops the run before testing the MMuse graph path.
Every FULL case must reach 95% of its eager reference median tokens/second for
each batch size (a configurable tolerance, not a promised speedup). MMuse eager
tokens are also compared to target-only eager. All comparisons use exact greedy
token IDs; mismatches stop acceptance even if caused by rounding near ties.

The harness excludes initialization, capture, RPC counter reads and warmup from
generation timing. Prefix caching and profiling are disabled, requests use fixed
token budgets with EOS ignored, and batch sizes grow then shrink. This measures
fixed-budget throughput, not natural EOS behavior or a representative production
benchmark. The existing smoke test remains useful for EOS and mixed temperatures.

Benchmark-only probes count actual successful `run_fullgraph` submissions for
target and draft during each timed repeat. They are installed after capture and
reset after warmup; requested FULL roles must have a nonzero count in every
repeat. No probes or per-step device synchronization are installed in serving.
These counts prove submission, not device kernel correctness; token comparison
is a separate check. They do not assert that every prefill/decode batch is FULL.

Inspect `summary.json`, per-case JSON and `.log` files. A nonzero exit status,
timeout, token mismatch, absent replay or throughput regression is a failed
acceptance, not permission to silently fall back to eager and report FULL.
Timeouts include startup; adjust `--timeout` for slow checkpoint loading.
Completed batches are saved even when a later batch times out. Keep the known
working eager deployment until the real-device test passes. A pass only covers
this workload; validate EOS, rejection/cache rollback, longer contexts and
probabilistic sampling separately before broadening support.

For Ascend 0.30 compatibility, the capture wrapper explicitly forwards
`rollout_mmuse` after decoder forward; relying on its version-conditional generic
attribute delegation would omit this method on 0.30. Decoder, dynamic convolution,
Selector and Correction math is otherwise unchanged.

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

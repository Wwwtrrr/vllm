# MMuse reference adapter for Qwen3

This experimental adapter includes standalone MMuse inference components for
Model Runner V2. It has not been validated with a trained
`qwen3_4b_mmuse` checkpoint or a GPU/NPU serving run. CPU component comparisons
do not establish end-to-end correctness or speedup.

## Supported first-stage contract

- Qwen3 target and Qwen3 draft with matching hidden width and full vocabulary.
- Speculators-format `speculators_model_type="mmuse"` configuration.
- `sample_from_anchor=True`, full-attention draft layers, enabled Correction.
- Fixed proposal length equal to the trained block size (typically 7).
- Greedy or probabilistic draft proposals, eager execution, one device.
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

### DeepSeek-V4 dense MMuse candidate adapter

`DeepseekV4MMuseSpeculator` is a separate model entry for the dense drafter
exported by this checkout's `speculators_dsv4` training path. It consumes
`deepseek_v4_mean_hc_head` auxiliary features with one-based post-layer IDs.
It does not consume the wide MTP HC residual. The teacher's final HC-head slot
is used by training and must not appear in the auxiliary ID list.

The trained draft decoder is Qwen3 dense attention, not the target's MLA/DSA
architecture. Its full/sliding attention causality follows the training config.
The export must contain `speculators_model_type="mmuse"`,
`transformer_layer_config`, and `target_hidden_state_format`; an unrelated
checkpoint is not converted by changing its model name. Raw MMuse exports are
recognized even when their root `model_type` names the dense decoder.
The DSpark-labelled export (`DSparkDraftModel`, `speculators_model_type="dspark"`)
is also recognized when the DeepSeek mean-HC format, Qwen3 decoder, Correction,
dynamic convolution and candidate Selector are explicitly present and enabled.
Its original metadata is preserved; the in-memory runtime identity becomes
MMuse. Baseline DSpark exports keep their existing loader. The actual tensors
must still pass the strict MMuse trained-weight loading checks.

This initial path requires BF16 activations and eager execution for both models.
Target W4A8 is independent of the unquantized draft and its BF16 KV cache.
Borrowed target embedding and output-head weights must be ordinary floating
matrices; packed or quantized IO weights are rejected. Under TP, full embedding
and head matrices are gathered once per rank for Correction, increasing memory.
Target/draft TP must agree; pipeline/data/context parallelism remains restricted
to one. The runtime proposal length must equal the checkpoint's `block_size`.
Ascend's cache planner retains both native DeepSeek cache groups and dense draft
cache groups in the native shared backing allocation.

For a DeepSeek MMuse draft with sliding-attention layers, real eager proposals
use explicit anchor-relative SDPA attention. Training exposes the same context
interval `[anchor-window, anchor)` to every slot, while ordinary paged SWA moves
the interval with each query. The eager path retains verified target features
per persistent request slot and ID, excludes rejected rows, and recomputes the
dense draft K/V from that history. It does not read the paged draft SWA cache.
Native draft attention remains in dummy/profile runs. This path requires
`enable_prefix_caching=False`, since reused target prefix KV does not supply
the intermediate features needed to reconstruct draft context.

Feature-history storage is reserved at model initialization for memory profiling.
All-sliding drafts reserve `max_num_seqs * sliding_window * hidden_size` elements;
mixed full/sliding drafts retain up to the draft maximum model length instead.
This is an eager correctness path, with additional memory, host synchronization
and context projection cost. Native NPU execution, throughput and acceptance
still need server validation; this change does not promise an acceptance-rate fix.

Run `examples/offline_inference/mmuse_deepseek_smoke.py` on the server with the
same environment and target quantization settings that already run the target:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
python examples/offline_inference/mmuse_deepseek_smoke.py \
  --target /home/data/deepseek-V4-flash-Dspark-w4a8-test \
  --draft /home/data/deepseek-V4-flash-mmuse --tp 1
```

Use the TP degree required by the target hardware. The script reads the trained
block size instead of assuming seven tokens. This is an integration smoke test,
not a benchmark or proof of speculative sampling equivalence. Local CPU tests
exercise configuration, synthetic IO weights, hidden-state routing and cache
planning; they do not execute native kernels, real checkpoint loading, cache
rollback or quantized serving. The optional training contract checks interfaces,
not target checkpoint fingerprints. Verify target identity separately.

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

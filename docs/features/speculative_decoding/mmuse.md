# MMuse reference adapters

## DeepSeek-V4: stateless reference implementation

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
Legacy enhanced DSpark exports follow the training migration flags; baseline
DSpark retains its own loader.

The proposer retains only confirmed projected target features, indexed by
persistent request slot and request ID. Rejected suffixes are excluded. Query
blocks are `[anchor, mask, ...]` at absolute positions; slot k predicts
anchor+k+1. Sliding layers expose the same `[anchor-window, anchor)` context to
every query slot. Synthetic query K/V is ephemeral. No draft attention objects
or paged KV groups are registered, so the native Target cache layout is retained.

Each Target TP rank executes a complete replicated draft. Draft checkpoint
embedding and output-head tensors are ignored even when serialized. Both IO
matrices are supplied by the serving Target and gathered once under TP. They
must be ordinary floating matrices and are converted to the draft's compute dtype.
This ensures shared serving IO, but does not establish numerical equivalence
with the original training teacher's IO or hidden states.
Replication and context recomputation increase memory and compute costs: this
is a correctness reference, not a throughput optimization. Target backbone
quantization remains the backend's responsibility. Interface checks do not prove
that the serving Target weights match the training teacher numerically.

Require BF16 activations, an unquantized draft, eager Target and Draft,
`enable_prefix_caching=False`, and PP/DP/context parallel sizes of one. Target TP
may exceed one; Draft computation is replicated. Adaptive verification,
watermarking and local argmax reduction are rejected. FULL graphs are not
implemented for this path.

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

### Diagnosing zero acceptance

`examples/offline_inference/mmuse_deepseek_audit.py` runs two short requests at
temperature zero with four audited proposal steps. Use the Target TP,
quantization and tokenizer settings that already work on your server:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
python examples/offline_inference/mmuse_deepseek_audit.py \
  --target /home/data/DeepSeek-V4-Flash-DSpark-w4a8-test \
  --draft /home/data/DeepSeek-V4-Flash-mmuse \
  --tp 4 --quantization ascend --tokenizer-mode deepseek_v4 \
  --output ./mmuse-audit
```

The TP value above is an example; retain your working deployment's TP. The
script needs local checkpoint directories. Add `--teacher /path/to/original`
to compare original training IO as well, or use `--io-only` for CPU checkpoint
comparison without creating the inference engine. It compares complete IO
matrices in row chunks after BF16 conversion, reporting exact inequality counts,
max/mean absolute differences and nonfinite values. Missing Draft IO is an
ordinary omitted-export case, not a mismatch.

`io-comparison.json` reports checkpoint IO comparisons. Per-worker JSONL files
report configured/captured auxiliary layers, sampled feature statistics, actual
anchor/mask inputs and positions, proposals, verification candidates and actual
sampling/rejection counts. `transport_matches=false` means an observed candidate
differs from the previously recorded proposal at that absolute position.
`transport_matches=null` means no recorded proposal covers that position.
Negative placeholder candidates are excluded. `target_raw_argmax` and
`target_raw_probability` refer to logits before sampling filters; `raw_match`
is not a substitute for the framework's accepted-token metric. A match after an
earlier rejection is not itself an accepted token.

Callbacks are installed only by this opt-in RPC, execute no additional TP
collectives and restore themselves after the configured steps. They synchronize
devices and are unsuitable for throughput measurement. These tools have CPU
coverage; actual GPU/NPU RPC execution still needs server validation. Checkpoint
IO agreement does not establish intermediate feature agreement.

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

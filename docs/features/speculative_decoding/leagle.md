# LEAGLE

LEAGLE is an autoregressive Qwen3 drafter with target-feature KV injection and
shared decoder loops. Use a `looped_eagle` training checkpoint whose
`speculators_model_type` is `leagle`. EAGLE3 and MMUSE checkpoints are different
architectures and cannot be substituted.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /path/to/Qwen3-4B \
  --enforce-eager --no-enable-prefix-caching \
  --tensor-parallel-size 1 --max-num-seqs 1 --max-model-len 4096 \
  --speculative-config '{"method":"leagle","model":"/path/to/leagle/checkpoints/checkpoint_best","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

Use the verifier checkpoint used during training. Physical depth, loop count,
normalization, attention geometry/windows and auxiliary feature indices come
from the saved training config. They are not inferred from the directory name.
HF auxiliary indices are preserved, including embedding slot zero; capture is
sorted and features are reordered back to the checkpoint's concatenation order.
Draft vocabulary mappings use `absolute_target_id = draft_id + d2t[draft_id]`.
Weights load by their original training names, with missing weights and invalid
mappings rejected. No `speculators` package is needed for serving.

Each proposal step fuses the bonus/known token with the previous draft state,
then visits the shared physical decoder stack `num_loops` times. Each logical
`(loop, layer)` has independent draft KV. The verified target context stays fixed
during a proposal round. Context K/V projections are reused by all loops and
draft token steps in that round. Rejected target positions are excluded from
the next round; partial prefills accumulate context without emitting proposals.

Both eager and FULL rollouts call the same public single-step `model(...)`
interface. `forward(input_ids, positions, *, hidden_states, context_kvs,
attention_masks, past_draft_kv=None)` returns `LEagleForwardOutput` containing
normalized `hidden_states` for `compute_logits`, `recurrent_state` for the next
token, and immutable `past_draft_kv` per logical `(loop, layer)`. Context K/V is
prepared incrementally from verified target features by the serving executor.
The caller provides visibility masks and
performs token selection after `compute_logits`.

This interface requires LEAGLE-specific context and recurrent state. Serving
still uses `LEagleSpeculator`; this does not turn the drafter into a standalone
language model or enable generic EAGLE3 paged attention. Checkpoint tensor names
and FULL graph boundaries are unchanged by the single-step interface.

The implementation uses Model Runner V2, PyTorch SDPA, a Qwen3 target,
TP=PP=DCP=PCP=1, and greedy draft proposals. Target sampling still uses
vLLM's normal verifier/rejection sampler; greedy draft probabilities are one-hot.
Prefix caching, remote KV transfer, target LoRA, watermarking, quantized drafts,
parallel drafting and adaptive verification are rejected. Paged draft attention
is not implemented.

For FULL target and draft graphs, remove `--enforce-eager` and add:

```bash
--compilation-config '{"cudagraph_mode":"FULL"}'
```

`FULL_DECODE_ONLY` also works: target prefill stays eager and target verification
uses FULL graphs. Leave draft `enforce_eager` unset or set it to false. Setting
draft `enforce_eager=true` keeps target FULL but runs the drafter eagerly.
The target uses the framework's existing FULL capture and auxiliary hidden-state
outputs. The draft captures target-feature FC/RMSNorm projection in token-count
buckets, and complete K-token autoregressive rollouts in context-length buckets.
Each rollout includes all shared loops, draft KV, output
normalization, LM head, argmax and vocabulary mapping. Dynamic context lengths
are device tensors; padding and rejected tail positions remain masked. RoPE
variants that resize their cache at runtime are rejected for draft FULL.

Buffers are allocated before memory profiling. Draft graph managers participate
in the framework's throwaway profiling/capture lifecycle, use a private graph
pool, and are recaptured before serving. Runtime replay never captures lazily
and does not silently fall back to eager for an unsupported shape.

Projected target features reserve
`max_num_seqs * max_model_len * hidden_size * dtype_size` bytes. Start with the
small limits above. FULL adds one padded projected-context buffer and feature
projection input/output buffers, plus graph-pool memory. Request scheduling,
verdict/position synchronization and staging copies remain outside graphs.
For the optimized executor, projection outputs are copied into persistent
feature/KV buffers inside the projection graph. A separate token-count-bucketed
cache-update graph scatters each request's verified feature/KV suffix into its
persistent slot. Padded writes use a private sink row and never touch live
requests. Cache-update graphs run even for partial prefills without proposals.
The rollout graph gathers anchors by device slot indices. Slot/length metadata
is uploaded together; host validation, request grouping and external input
copies still remain outside capture. Each nonempty request suffix adds one
cache-update replay; measure end-to-end latency rather than assuming a speedup.
Requests are batched into batch-size and context-length buckets; each batch's
K-token rollout is one FULL replay. Measure acceptance and output parity
before using its latency as a serving benchmark.

## Inference optimizations

Ascend also supports target input preparation graphs via
`leagle_target_input_graphs` (default `true` on eligible FULL decode steps).
Request-map expansion, positions/lengths, verification token assembly, block
table gathering and slot mappings run in a separate native preparation graph
before target FULL replay. The shared block-table gather API accepts stable
block-count buffers to avoid capturing rotating UVA pointers. CUDA preparation
is unchanged. See the Ascend LEAGLE guide for eligibility, fallback and A/B
evaluation; this does not capture CPU scheduling or eliminate all transfers.

### Inference-only SVD head

Dense LEAGLE checkpoints can derive an SVD retrieval head at startup, without
candidate training, checkpoint conversion or random initialization:

```python
speculative_config = {
    "method": "leagle",
    "model": "/path/to/existing/leagle/checkpoint",
    "num_speculative_tokens": 7,
    "draft_sample_method": "greedy",
    "leagle_candidate_svd_rank": 256,
    "leagle_candidate_top_k": 256,
}
```

After all dense weights load, CPU FP64 vocabulary chunks accumulate `W.T @ W`;
`eigh` extracts leading right singular vectors `Q`. The inference buffers are
`up_weight = W @ Q` and `down_weight = Q.T`, matching `looped_eagle`'s SVD
implementation. They are nonpersistent, retain the LM head dtype/device and
are built before memory profiling and FULL capture. No SVD enters a replay.
The full original LM head stays available for exact candidate scoring.

Default SVD scoring is low-rank Top-K retrieval followed by exact rescoring.
Set `leagle_candidate_rerank=false` and omit `leagle_candidate_top_k` to predict
directly from low-rank full-vocabulary logits instead. This second mode skips
both Top-K and exact rescoring and can change predictions more substantially.
Checkpoint `candidate_rerank` is also preserved for trained low-rank heads;
an explicit SVD rank defaults to reranking regardless of checkpoint mode.

Rank must not exceed the smaller weight dimension. SVD rank cannot be combined
with `leagle_use_candidate_head=false`. For dense A/B runs, remove the SVD rank
and set `leagle_use_candidate_head=false`; restart the engine for each variant.
Changing rank/scoring or hot-reloading dense weights after SVD setup requires
restart and recapture, preventing stale factors or captured pointers.

Large weights can make FP64 CPU initialization take substantial time. Logs
report initialization start/end; factors are rebuilt on each startup and are
not saved to the checkpoint or a disk cache. Account for CPU workspaces and
additional device factor memory: this does not reduce resident dense weights.
Repeated eigenvalues may yield different bases across environments. Evaluate
real hidden-state recall, acceptance, output quality and TPOT on the NPU; the
factorization alone guarantees neither identical tokens nor a speedup.

### Trained candidate head

Checkpoints with `candidate_head=true` now use the trained
`candidate_head.down_proj.weight` and `candidate_head.up_proj.weight` for low-rank
retrieval. Each step computes coarse scores, selects the checkpoint's
`candidate_head_top_k`, gathers only those LM-head rows and scores the candidates
at full hidden width. Greedy selection reduces the candidate scores directly;
it does not materialize full-vocabulary logits. `compute_logits` remains
available and scatters candidate scores into a vocabulary tensor with `-inf`
elsewhere, matching the training implementation. Ties select the smallest draft
vocabulary ID before applying the existing draft-to-target mapping.

The checkpoint architecture is the default. Optional `speculative_config`
overrides are `leagle_use_candidate_head=false` for dense A/B evaluation, and
`leagle_candidate_top_k` for an enabled candidate head. Overrides are applied
before graph capture; changing retrieval at replay time is unsupported.
Missing, mismatched or non-finite candidate weights are rejected. Dense
checkpoints cannot enable this path without trained candidate weights, and
remain unchanged by default. No random candidate head is added to old models.

Candidate retrieval is shared by eager and FULL, including batched cached
rollouts. It changes the proposal support and may reduce acceptance; evaluate
recall, acceptance and end-to-end TPOT rather than assuming a speedup. Greedy
draft proposals still use the existing one-hot rejection-sampling path.
CPU parity tests cover the actual training candidate implementation; native
NPU Top-K, gather, small matmul and graph replay require hardware validation.

The following `speculative_config` switches default to `true` for LEAGLE:

| Option | Behavior |
| --- | --- |
| `leagle_reuse_context_kv` | Project only newly verified target features into K/V, retaining each request's prefix across rounds. |
| `leagle_static_draft_kv` | Preallocate logical-layer draft KV and physical-layer attention workspaces; avoid concatenating the full context at every draft step. |
| `leagle_batch_drafts` | Draft active requests together with per-request positions and visibility masks. |
| `leagle_fused_projections` | Pack Q/K/V, context K/V and MLP gate/up projections into fewer matrix multiplications. |
| `leagle_fused_rmsnorm` | Use the device RMSNorm kernel instead of separate reduction/elementwise operations. |

For an A/B comparison, set all five to `false` to select the reference executor.
Keep the checkpoint, draft length, request limits and input workload identical.
Both executors support eager and FULL. The optimized FULL projection graph also
computes new target K/V; the rollout graph reads cached K/V and stages a padded
batch. Autoregressive drafting still requires K token steps and all configured
shared loops.

Only verified target features enter the cross-round cache. Slot reuse, suffix
rewinds, rejected tails and partial prefills follow the same context ledger as
the reference path. Synthetic draft KV is rebuilt each round. Cached K/V also
requires RoPE whose scaling does not change with runtime sequence length.

These switches require no retraining or checkpoint conversion. Packed weights
are nonpersistent buffers; original checkpoint names remain unchanged. Packing
and fused kernels can change floating-point rounding, so evaluate acceptance
and output quality with the actual checkpoint as well as token parity tests.
Additional memory includes packed weights, persistent K/V per physical layer,
batched attention workspaces, draft KV per logical layer and graph buckets.
Captured cache writes also reserve one private sink slot for features and K/V.
Increase `max_num_seqs` and `max_model_len` gradually while checking memory.
This is dense cache execution, without paged attention or a fused attention
kernel. Training-only chunked loss and DDP settings belong to the training
repository and are not serving options.

CPU contracts (no CUDA imports):

```bash
LEAGLE_TRAINING_ROOT=/path/to/looped_eagle \
  .venv/bin/python -m pytest --confcutdir=tests/standalone \
  -o addopts='' tests/standalone/test_leagle.py -q
```

Install the training package's dependencies to run the optional numerical
parity cases. They load the real training model and compare all proposal logits
for different physical depths, loops, normalization modes, sliding windows and
vocabulary sizes, including padded static rollouts. CPU replay tests simulate
buffer orchestration and do not validate a hardware graph runtime.
GPU/NPU end-to-end serving and model-quality evaluation must
also be run on the deployment machine with the actual trained checkpoint.

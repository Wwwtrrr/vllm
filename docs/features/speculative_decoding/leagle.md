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
Each rollout includes target K/V projection, all shared loops, draft KV, output
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
Requests are handled sequentially by the drafter; each K-token rollout is one
FULL replay. Measure acceptance and output parity
before using its latency as a serving benchmark.

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

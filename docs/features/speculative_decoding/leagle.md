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

The initial implementation uses Model Runner V2, eager PyTorch SDPA, a Qwen3
target, TP=PP=DCP=PCP=1, and greedy draft proposals. Target sampling still uses
vLLM's normal verifier/rejection sampler; greedy draft probabilities are one-hot.
Prefix caching, remote KV transfer, target LoRA, watermarking, quantized drafts,
parallel drafting and adaptive verification are rejected. CUDA/ACL graph drafting
and paged draft attention are not implemented.

Projected target features reserve
`max_num_seqs * max_model_len * hidden_size * dtype_size` bytes. Start with the
small limits above. The eager proposer synchronizes verdicts/positions once per
batch and handles requests sequentially. Measure acceptance and output parity
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
vocabulary sizes. GPU/NPU end-to-end serving and model-quality evaluation must
also be run on the deployment machine with the actual trained checkpoint.

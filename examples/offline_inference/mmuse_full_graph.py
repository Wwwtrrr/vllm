# SPDX-License-Identifier: Apache-2.0
"""Run Qwen3/MMuse with eager, target FULL, or target+draft FULL execution.

Run each mode in a separate process on the server. Inspect MMuse capture and
replay logs; a requested graph mode does not prove that a request replayed it.
This uses real checkpoints and saves token IDs for comparison, not a benchmark.
"""

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("eager", "target-full", "both-full"), required=True
    )
    parser.add_argument(
        "--draft-sample-method", choices=("greedy", "probabilistic"), default="greedy"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.block_size <= 0:
        parser.error("--block-size must match the checkpoint and be positive")
    if args.output.exists():
        parser.error("--output already exists; choose a new file")

    target_eager = args.mode == "eager"
    draft_eager = args.mode != "both-full"
    llm = LLM(
        model=args.target,
        enforce_eager=target_eager,
        compilation_config={"cudagraph_mode": "NONE" if target_eager else "FULL"},
        speculative_config={
            "method": "mmuse",
            "model": args.draft,
            "num_speculative_tokens": args.block_size,
            "draft_sample_method": args.draft_sample_method,
            "enforce_eager": draft_eager,
        },
        tensor_parallel_size=1,
        max_num_seqs=4,
        max_model_len=args.max_model_len,
        seed=42,
    )
    prompts = [
        "Explain speculative decoding in three sentences.",
        "Write a Python function that adds two integers.",
        "What is the capital of France? Answer briefly.",
        "Explain why the sky appears blue.",
    ]
    results = []
    # Repeated shapes, growing/shrinking batches and mixed temperatures.
    for iteration, count in enumerate((1, 2, 4, 2, 1)):
        sampling = [
            SamplingParams(
                temperature=args.temperature if row % 2 == 0 else 0.0,
                max_tokens=48,
                seed=100 + iteration * 4 + row,
            )
            for row in range(count)
        ]
        outputs = llm.generate(prompts[:count], sampling, use_tqdm=False)
        results.append(
            {
                "batch_size": count,
                "token_ids": [list(item.outputs[0].token_ids) for item in outputs],
                "finish_reasons": [item.outputs[0].finish_reason for item in outputs],
            }
        )
    args.output.write_text(
        json.dumps({"mode": args.mode, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {args.mode} outputs to {args.output}. Check FULL replay logs.")


if __name__ == "__main__":
    main()

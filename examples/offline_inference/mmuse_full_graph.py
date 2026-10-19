# SPDX-License-Identifier: Apache-2.0
"""Run Qwen3/MMuse with eager, target FULL, or target+draft FULL execution.

Run each mode in a separate process on the server. Inspect MMuse capture and
replay logs; a requested graph mode does not prove that a request replayed it.
This uses real checkpoints and saves token IDs for comparison, not a benchmark.
"""

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.model_executor.models.mmuse.graph_config import (
    build_mmuse_full_graph_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft")
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--without-draft", action="store_true")
    parser.add_argument(
        "--compilation-mode",
        type=int,
        choices=(0, 3),
        default=0,
        help="0: native operators plus FULL capture; 3: compiled FULL capture",
    )
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
    if not args.without_draft and not args.draft:
        parser.error("--draft is required unless --without-draft is set")
    if args.without_draft and args.mode == "both-full":
        parser.error("Use --mode target-full when --without-draft is set")

    target_eager = args.mode == "eager"
    draft_eager = args.mode != "both-full"
    compilation = build_mmuse_full_graph_config(
        1 if args.without_draft else args.block_size,
        4,
        compilation_mode=args.compilation_mode,
    )
    if target_eager:
        compilation = {"mode": 0, "cudagraph_mode": "NONE"}
    speculative = (
        None
        if args.without_draft
        else {
            "method": "mmuse",
            "model": args.draft,
            "num_speculative_tokens": args.block_size,
            "draft_sample_method": args.draft_sample_method,
            "enforce_eager": draft_eager,
        }
    )
    llm = LLM(
        model=args.target,
        enforce_eager=target_eager,
        compilation_config=compilation,
        speculative_config=speculative,
        tensor_parallel_size=1,
        max_num_seqs=4,
        max_model_len=args.max_model_len,
        seed=42,
        enable_prefix_caching=False,
        cudagraph_metrics=True,
        disable_log_stats=False,
    )
    prompts = [
        "Explain speculative decoding in three sentences.",
        "Write a Python function that adds two integers.",
        "What is the capital of France? Answer briefly.",
        "Explain why the sky appears blue.",
    ]
    results = []
    # Exercise the same shapes before timing; initialization/capture is excluded.
    for count in (1, 2, 4):
        llm.generate(
            prompts[:count],
            SamplingParams(temperature=0.0, max_tokens=16, seed=42),
            use_tqdm=False,
        )
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
        started = time.perf_counter()
        outputs = llm.generate(prompts[:count], sampling, use_tqdm=False)
        elapsed = time.perf_counter() - started
        results.append(
            {
                "batch_size": count,
                "elapsed_seconds": elapsed,
                "output_tokens": sum(
                    len(item.outputs[0].token_ids) for item in outputs
                ),
                "token_ids": [list(item.outputs[0].token_ids) for item in outputs],
                "finish_reasons": [item.outputs[0].finish_reason for item in outputs],
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "compilation_config": compilation,
                "speculative_config": speculative,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {args.mode} outputs to {args.output}. Check FULL replay logs.")


if __name__ == "__main__":
    main()

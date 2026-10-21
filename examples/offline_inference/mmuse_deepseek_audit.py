# SPDX-License-Identifier: Apache-2.0
"""Bounded eager zero-acceptance audit; requires the real GPU/NPU environment."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument(
        "--teacher", help="Optional original training teacher checkpoint"
    )
    parser.add_argument("--output", default="mmuse-audit")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--quantization")
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--io-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    from vllm.model_executor.models.mmuse.io_comparison import compare_io

    comparisons = {"draft_vs_target": compare_io(args.draft, args.target)}
    if args.teacher:
        comparisons["teacher_vs_target"] = compare_io(args.teacher, args.target)
    (root / "io-comparison.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparisons, indent=2))
    if args.io_only:
        return

    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.mmuse.reference_audit import install_reference_audit

    source = json.loads((Path(args.draft) / "config.json").read_text(encoding="utf-8"))
    llm = LLM(
        model=args.target,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        quantization=args.quantization,
        tokenizer_mode=args.tokenizer_mode,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_num_seqs=2,
        max_model_len=args.max_model_len,
        speculative_config=dict(
            method="mmuse",
            model=args.draft,
            num_speculative_tokens=source["block_size"],
            enforce_eager=True,
            draft_sample_method="greedy",
        ),
    )
    print(
        llm.collective_rpc(
            install_reference_audit,
            kwargs=dict(directory=str(root), steps=args.steps, requests=2),
        )
    )
    outputs = llm.generate(
        [
            "Explain why the sky is blue.",
            "Calculate 17 plus 25 and explain your answer.",
        ],
        SamplingParams(
            temperature=0.0, seed=0, max_tokens=args.max_tokens, ignore_eos=True
        ),
    )
    (root / "outputs.json").write_text(
        json.dumps(
            [
                dict(
                    prompt=x.prompt,
                    token_ids=list(x.outputs[0].token_ids),
                    text=x.outputs[0].text,
                )
                for x in outputs
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

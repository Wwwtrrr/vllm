# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint eager smoke test; requires the server's GPU/NPU environment."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    source = json.loads((Path(args.draft) / "config.json").read_text())
    from vllm.transformers_utils.configs.speculators.algos import (
        is_deepseek_mmuse_export,
    )

    if source.get("speculators_model_type") != "mmuse" and not is_deepseek_mmuse_export(
        source
    ):
        parser.error("draft config must be a real MMuse training export")
    if source.get("target_hidden_state_format") != "deepseek_v4_mean_hc_head":
        parser.error("draft must use the DeepSeek-V4 mean-HC training contract")
    block_size = source["block_size"]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.target,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        quantization=args.quantization,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_num_seqs=1,
        speculative_config={
            "method": "mmuse",
            "model": args.draft,
            "num_speculative_tokens": block_size,
            "enforce_eager": True,
            "draft_sample_method": "greedy",
            "kv_cache_dtype": "auto",
        },
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    # Two calls exercise reuse after request completion, not just model loading.
    for prompt in ("Explain why the sky is blue.", "计算 17 加 25，并简要说明。"):
        output = llm.generate([prompt], params)[0]
        print(
            json.dumps(
                {"prompt": prompt, "text": output.outputs[0].text}, ensure_ascii=False
            )
        )


if __name__ == "__main__":
    main()

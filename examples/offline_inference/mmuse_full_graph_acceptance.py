# SPDX-License-Identifier: Apache-2.0
"""Validate warmed FULL throughput in isolated processes with real checkpoints.

Stops at the first regression. No profiler, model initialization or graph
capture is included in measured generate calls. A passing result applies only
to these prompts, batch sizes, token budget, checkpoints and device/software.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

CASES = (
    "target-eager",
    "target-full",
    "mmuse-eager",
    "mmuse-target-full",
    "mmuse-full",
)


def graph_replay_probe(worker, action):
    """Install benchmark-only counters after capture; never change graph choice."""
    runner = worker.model_runner
    managers = {"target": runner.cudagraph_manager}
    speculator = getattr(runner, "speculator", None)
    if speculator is not None:
        managers["draft"] = speculator.query_cudagraph_manager
    result = {}
    for name, manager in managers.items():
        if manager is None:
            continue
        if action == "install":
            if hasattr(manager, "_acceptance_replays"):
                raise RuntimeError("Replay probe already installed")
            manager._acceptance_replays = 0
            original = manager.run_fullgraph

            def counted(desc, _manager=manager, _original=original):
                output = _original(desc)
                _manager._acceptance_replays += 1
                return output

            manager.run_fullgraph = counted
        elif action == "reset":
            manager._acceptance_replays = 0
        elif action != "read":
            raise ValueError(action)
        result[name] = manager._acceptance_replays
    return result


def compare_results(baseline, candidate, min_ratio):
    """Require identical greedy outputs and non-regressing median throughput."""
    failures = []
    ratios = {}
    if baseline["batch_sizes"] != candidate["batch_sizes"]:
        return {"passed": False, "failures": ["batch sizes differ"], "ratios": {}}
    for batch in baseline["batch_sizes"]:
        key = str(batch)
        before, after = baseline["batches"][key], candidate["batches"][key]
        if before["token_ids"] != after["token_ids"]:
            failures.append(f"batch={batch}: greedy token IDs differ")
        ratio = after["median_tokens_per_second"] / before["median_tokens_per_second"]
        ratios[key] = ratio
        if ratio < min_ratio:
            failures.append(
                f"batch={batch}: throughput ratio {ratio:.3f} < {min_ratio}"
            )
        for role in candidate["required_replays"]:
            if any(row.get(role, 0) <= 0 for row in after["replays"]):
                failures.append(f"batch={batch}: no measured {role} FULL replay")
    return {"passed": not failures, "failures": failures, "ratios": ratios}


def run_worker(args):
    # Keep controller and report inspection usable without importing vLLM/NPU.
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.mmuse.graph_config import (
        build_mmuse_full_graph_config,
    )

    case = args.worker_case
    use_draft = case.startswith("mmuse")
    target_full = case not in ("target-eager", "mmuse-eager")
    draft_full = case == "mmuse-full"
    compilation = (
        build_mmuse_full_graph_config(
            args.block_size if use_draft else 1, max(args.batch_sizes)
        )
        if target_full
        else {"mode": 0, "cudagraph_mode": "NONE"}
    )
    speculative = None
    if use_draft:
        speculative = {
            "method": "mmuse",
            "model": args.draft,
            "num_speculative_tokens": args.block_size,
            "enforce_eager": not draft_full,
            "draft_sample_method": "greedy",
        }
    llm = LLM(
        model=args.target,
        speculative_config=speculative,
        enforce_eager=not target_full,
        compilation_config=compilation,
        tensor_parallel_size=1,
        max_num_seqs=max(args.batch_sizes),
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
        seed=42,
        disable_log_stats=False,
        profiler_config={"profiler": None},
    )
    llm.collective_rpc(graph_replay_probe, args=("install",))
    base_prompts = [
        "Explain how speculative decoding accelerates language model generation.",
        "Write and explain a Python function for binary search.",
        "Describe the water cycle and why it matters for agriculture.",
        "Compare solar and wind energy with concrete examples.",
    ]
    # Fixed token budgets remove EOS/output-length differences from throughput.
    sampling = SamplingParams(
        temperature=0.0, max_tokens=args.tokens, ignore_eos=True, seed=42
    )
    report = {
        "case": case,
        "batch_sizes": args.batch_sizes,
        "required_replays": (["target"] if target_full else [])
        + (["draft"] if draft_full else []),
        "compilation_config": compilation,
        "speculative_config": speculative,
        "target": args.target,
        "tokens_per_request": args.tokens,
        "resolved_compilation_mode": str(
            llm.llm_engine.vllm_config.compilation_config.mode
        ),
        "resolved_graph_mode": str(
            llm.llm_engine.vllm_config.compilation_config.cudagraph_mode
        ),
        "resolved_target_enforce_eager": llm.llm_engine.vllm_config.model_config.enforce_eager,
        "batches": {},
    }
    # Growing and shrinking batches exercise graph descriptor changes.
    for batch in args.batch_sizes:
        prompts = [base_prompts[i % len(base_prompts)] for i in range(batch)]
        for _ in range(args.warmups):
            llm.generate(prompts, sampling, use_tqdm=False)
        samples, replays, reference = [], [], None
        for _ in range(args.repeats):
            llm.collective_rpc(graph_replay_probe, args=("reset",))
            started = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - started
            token_ids = [list(output.outputs[0].token_ids) for output in outputs]
            if any(len(ids) != args.tokens for ids in token_ids):
                raise RuntimeError(
                    "Output was truncated; fixed-token comparison is invalid"
                )
            if reference is not None and token_ids != reference:
                raise RuntimeError(f"{case}: repeated greedy outputs differ")
            reference = token_ids
            samples.append(sum(map(len, token_ids)) / elapsed)
            snapshot = llm.collective_rpc(graph_replay_probe, args=("read",))
            if len(snapshot) != 1:
                raise RuntimeError("Acceptance currently requires one worker")
            replays.append(snapshot[0])
        report["batches"][str(batch)] = {
            "token_ids": reference,
            "median_tokens_per_second": statistics.median(samples),
            "samples_tokens_per_second": samples,
            "replays": replays,
        }
        # Preserve completed batches if a later batch times out or fails.
        args.result.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 2])
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-ratio", type=float, default=0.95)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(
            args.batch_sizes
            + [args.block_size, args.tokens, args.warmups, args.repeats, args.timeout]
        )
        <= 0
    ):
        parser.error(
            "batch sizes, block size, tokens, warmups, repeats and timeout must be positive"
        )
    if not 0 < args.min_ratio <= 1:
        parser.error("--min-ratio must be in (0, 1]")
    if args.worker_case:
        run_worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"passed": False, "checks": {}, "failure": None}
    reports = {}
    environment = dict(os.environ, VLLM_USE_V2_MODEL_RUNNER="1")
    for case in CASES:
        result = args.output_dir / f"{case}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker-case",
            case,
            "--result",
            str(result),
        ]
        print(f"Running {case}; log: {args.output_dir / (case + '.log')}", flush=True)
        try:
            with (args.output_dir / f"{case}.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=True,
                )
            reports[case] = json.loads(result.read_text(encoding="utf-8"))
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            summary["failure"] = f"{case}: {error}"
            break
        baseline = (
            "target-eager"
            if case in ("target-full", "mmuse-eager")
            else "mmuse-eager"
            if case in ("mmuse-target-full", "mmuse-full")
            else None
        )
        if baseline:
            # MMuse eager is a correctness reference, not a graph speed claim.
            threshold = 0.0 if case == "mmuse-eager" else args.min_ratio
            check = compare_results(reports[baseline], reports[case], threshold)
            summary["checks"][case] = check
            if not check["passed"]:
                summary["failure"] = (
                    f"{case}: acceptance failed; see checks and case log"
                )
                break
    else:
        summary["passed"] = True
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

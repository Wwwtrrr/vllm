# SPDX-License-Identifier: Apache-2.0
"""CPU tests for acceptance failures; these do not measure graph performance."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def acceptance():
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/offline_inference/mmuse_full_graph_acceptance.py"
    )
    spec = importlib.util.spec_from_file_location("mmuse_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(*, speed=100, tokens=None, replays=3, required=True):
    return {
        "batch_sizes": [1],
        "required_replays": ["target", "draft"] if required else [],
        "batches": {
            "1": {
                "median_tokens_per_second": speed,
                "token_ids": [[1, 2]] if tokens is None else tokens,
                "replays": [{"target": replays, "draft": replays}],
            }
        },
    }


@pytest.mark.parametrize(
    "candidate,passed",
    [
        (report(speed=99), True),
        (report(speed=10), False),
        (report(tokens=[[1, 3]]), False),
        (report(replays=0), False),
    ],
)
def test_acceptance_requires_throughput_correctness_and_actual_replay(
    acceptance, candidate, passed
):
    assert acceptance.compare_results(report(), candidate, 0.95)["passed"] is passed


def test_probe_excludes_capture_and_warmup_and_counts_completed_submissions(acceptance):
    calls = []
    manager = SimpleNamespace(run_fullgraph=lambda desc: calls.append(desc) or "ok")
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(cudagraph_manager=manager, speculator=None)
    )
    acceptance.graph_replay_probe(worker, "install")
    assert manager.run_fullgraph("warmup") == "ok"
    acceptance.graph_replay_probe(worker, "reset")
    manager.run_fullgraph("measured")
    assert acceptance.graph_replay_probe(worker, "read") == {"target": 1}
    assert calls == ["warmup", "measured"]


@pytest.mark.parametrize("regress", [True, False])
def test_controller_stops_before_draft_when_plain_target_regresses(
    acceptance, monkeypatch, tmp_path, regress
):
    directory = tmp_path / "acceptance"
    monkeypatch.setattr(
        acceptance.sys,
        "argv",
        [
            "acceptance.py",
            "--target",
            "target",
            "--draft",
            "draft",
            "--batch-sizes",
            "1",
            "--output-dir",
            str(directory),
        ],
    )
    cases = []

    def run(command, **kwargs):
        case = command[command.index("--worker-case") + 1]
        cases.append(case)
        result = Path(command[command.index("--result") + 1])
        result.write_text(
            json.dumps(
                report(
                    speed=1 if regress and case == "target-full" else 100,
                    required="full" in case,
                )
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    if regress:
        with pytest.raises(SystemExit) as error:
            acceptance.main()
        assert error.value.code == 1
        assert cases == ["target-eager", "target-full"]
    else:
        acceptance.main()
        assert tuple(cases) == acceptance.CASES
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["passed"] is (not regress)

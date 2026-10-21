# SPDX-License-Identifier: Apache-2.0
"""CPU tests for audit alignment and streaming checkpoint comparisons."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save_file

DIRECTORY = Path(__file__).resolve().parents[3] / "vllm/model_executor/models/mmuse"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_verification_compares_next_candidate_and_preserves_first_rejection():
    audit = load("reference_audit")
    batch = NS(
        req_ids=["r"],
        cu_num_logits_np=[0, 4],
        logits_indices=torch.arange(4),
        positions=torch.arange(10, 14),
    )
    logits = torch.zeros(4, 16)
    for row, token in enumerate([7, 2, 9, 0]):
        logits[row, token] = 10
    previous = {"r": dict(anchor_position=10, tokens=[7, 8, 9])}
    rows = audit.verification_rows(
        logits, batch, torch.tensor([5, 7, 8, 9]), previous, 2
    )
    comparisons = rows[0]["comparisons"]
    assert [item["raw_match"] for item in comparisons] == [True, False, True]
    assert all(item["transport_matches"] for item in comparisons)
    assert comparisons[0]["source_position"] == 10
    assert comparisons[0]["candidate_position"] == 11
    assert comparisons[0]["target_raw_probability"] > 0.99
    wrong = audit.verification_rows(
        logits, batch, torch.tensor([5, 6, 8, 9]), previous, 1
    )
    assert wrong[0]["comparisons"][0]["transport_matches"] is False
    masked = audit.verification_rows(
        logits, batch, torch.tensor([5, -1, -1, -1]), previous, 1
    )
    assert masked[0]["comparisons"] == []


def test_io_comparison_streams_shards_and_resolves_target_aliases(tmp_path):
    comparison = load("io_comparison")
    left, right = tmp_path / "draft", tmp_path / "target"
    left.mkdir()
    right.mkdir()
    weight = torch.arange(24).reshape(8, 3).float()
    save_file(
        {"embed_tokens.weight": weight, "lm_head.weight": weight.clone()},
        str(left / "model.safetensors"),
    )
    changed = weight.clone()
    changed[-1, -1] += 1
    save_file(
        {"embed.weight": weight, "head.weight": changed},
        str(right / "part.safetensors"),
    )
    (right / "model.safetensors.index.json").write_text(
        json.dumps(
            dict(
                weight_map={
                    "embed.weight": "part.safetensors",
                    "head.weight": "part.safetensors",
                }
            )
        )
    )
    result = comparison.compare_io(left, right, chunk_rows=2)
    assert result["embedding"]["equal"]
    assert result["head"]["unequal"] == 1
    assert result["head"]["max_abs"] == 1
    assert result["head"]["total"] == 24
    with pytest.raises(ValueError, match="positive"):
        comparison.compare_io(left, right, chunk_rows=0)


def test_missing_and_incompatible_io_are_reported(tmp_path):
    comparison = load("io_comparison")
    left, right = tmp_path / "a", tmp_path / "b"
    left.mkdir()
    right.mkdir()
    save_file({"head.weight": torch.zeros(3, 2)}, str(left / "model.safetensors"))
    save_file({"head.weight": torch.zeros(4, 2)}, str(right / "model.safetensors"))
    result = comparison.compare_io(left, right)
    assert result["embedding"]["status"] == "missing"
    assert result["head"]["status"] == "shape_mismatch"
    with pytest.raises(FileNotFoundError):
        comparison.compare_io(tmp_path / "not_here", right)


def test_io_summary_does_not_materialize_the_full_matrix():
    summary = load("reference_audit").tensor_summary(torch.ones(1000, 1000))
    assert summary["shape"] == [1000, 1000]
    assert summary["sample_numel"] == 65536
    assert summary["rms"] == 1


def test_worker_audit_records_proposals_and_restores_callbacks(tmp_path):
    audit = load("reference_audit")

    class Runtime:
        def rollout_correction(self, hidden, anchors):
            return torch.tensor([[4, 5, 6]]), None

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = NS(mmuse_source_config={"aux_hidden_state_layer_ids": [1]})
            self.history = object()
            self.runtime = Runtime()
            self.embedding_weight = torch.ones(8, 3)
            self.output_weight = torch.ones(8, 3)

        def forward(self, ids, positions, context_positions, features):
            return torch.zeros(3, 3)

    class Proposer:
        model = Model()
        num_query_per_req = 3

        def propose(
            self,
            input_batch,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            dummy_run=False,
        ):
            hidden = self.model(
                torch.tensor([3, 7, 7]),
                torch.arange(10, 13),
                torch.arange(10),
                torch.zeros(10, 3),
            )
            return self.model.runtime.rollout_correction(
                hidden[None], torch.tensor([3])
            )[0]

    class Verifier:
        def _verify_in_chunks(self, logits, batch, draft_logits, draft_sampled):
            return "verified"

    proposer, verifier = Proposer(), Verifier()
    runner = NS(
        speculator=proposer,
        rejection_sampler=verifier,
        model=NS(model=NS(aux_hidden_state_layers=(1,))),
    )
    originals = (
        proposer.propose,
        proposer.model.runtime.rollout_correction,
        verifier._verify_in_chunks,
    )
    path = audit.install_reference_audit(
        NS(model_runner=runner, rank=0), str(tmp_path), steps=2
    )
    batch = NS(
        req_ids=["r"],
        num_draft_tokens_per_req=torch.tensor([3]),
        cu_num_logits_np=[0, 4],
        logits_indices=torch.arange(4),
        positions=torch.arange(10, 14),
    )
    for _ in range(2):
        verifier._verify_in_chunks(
            torch.randn(4, 8), batch, None, torch.tensor([3, 4, 5, 6])
        )
        proposer.propose(
            batch, [torch.ones(4, 3)], torch.tensor([1]), torch.tensor([3])
        )
    assert proposer.propose == originals[0]
    assert proposer.model.runtime.rollout_correction == originals[1]
    assert verifier._verify_in_chunks == originals[2]
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert records[-1]["kind"] == "audit_completed"
    verification = [x for x in records if x["kind"] == "verification"][-1]
    assert all(
        row["transport_matches"] for row in verification["rows"][0]["comparisons"]
    )
    assert any(x["kind"] == "proposal" and x["anchor_position"] == 10 for x in records)

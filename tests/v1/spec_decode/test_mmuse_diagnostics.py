# SPDX-License-Identifier: Apache-2.0
"""CPU checks for opt-in real-weight and token-alignment diagnostics."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch


def diagnostics():
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm/model_executor/models/mmuse/diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("mmuse_diagnostics_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tp_size", [1, 2, 8])
@pytest.mark.parametrize("projection", ["q", "k", "v"])
def test_checkpoint_qkv_mapping_handles_tp_and_replicated_kv(tp_size, projection):
    module = diagnostics()
    rank = tp_size - 1
    q = torch.arange(32 * 6).reshape(32, 6).float()
    k = torch.arange(8 * 6).reshape(8, 6).float() + 1000
    v = k + 1000
    q_width, kv_width = 32 // tp_size, max(4, 8 // tp_size)
    kv_rank = rank if tp_size <= 2 else rank // (tp_size // 2)
    fused = torch.cat(
        (
            q[rank * q_width : (rank + 1) * q_width],
            k[kv_rank * kv_width : (kv_rank + 1) * kv_width],
            v[kv_rank * kv_width : (kv_rank + 1) * kv_width],
        )
    )
    destination = "model.layers.0.self_attn.qkv_proj.weight"
    model = NS(
        model=NS(layers=[NS(self_attn=NS(q_size=q_width, kv_size=kv_width))]),
        named_parameters=lambda: [(destination, fused)],
    )
    _, actual, expected = module.checkpoint_parameter(
        model,
        f"layers.0.self_attn.{projection}_proj.weight",
        {"q": q, "k": k, "v": v}[projection],
        rank,
        tp_size,
    )
    assert torch.equal(actual, expected)


def test_checkpoint_audit_detects_numerically_wrong_loaded_weights(tmp_path):
    from safetensors.torch import save_file

    module = diagnostics()
    source = torch.arange(12).reshape(3, 4).float()
    save_file({"fc.weight": source}, tmp_path / "model.safetensors")
    actual = source.to(torch.bfloat16).clone()
    model = NS(named_parameters=lambda: [("model.fc.weight", actual)])
    assert module.audit_checkpoint(model, tmp_path, 0, 1)["mismatches"] == []
    actual[0, 0] = 17
    result = module.audit_checkpoint(model, tmp_path, 0, 1)
    assert result["checked"] == 1
    assert result["mismatches"][0]["source"] == "fc.weight"
    assert result["mismatches"][0]["max_abs_error"] == 17


@pytest.mark.parametrize("rank", [0, 1])
def test_checkpoint_gate_up_and_down_match_separate_tp_axes(rank):
    module = diagnostics()
    gate = torch.arange(8 * 6).reshape(8, 6).float()
    up = gate + 100
    down = torch.arange(6 * 8).reshape(6, 8).float()
    parameters = {
        "model.layers.0.mlp.gate_up_proj.weight": torch.cat(
            (gate.chunk(2)[rank], up.chunk(2)[rank])
        ),
        "model.layers.0.mlp.down_proj.weight": down.chunk(2, dim=1)[rank],
    }
    model = NS(named_parameters=lambda: parameters.items())
    for name, value in (("gate", gate), ("up", up), ("down", down)):
        _, actual, expected = module.checkpoint_parameter(
            model, f"layers.0.mlp.{name}_proj.weight", value, rank, 2
        )
        assert torch.equal(actual, expected)


def test_serialized_embedding_and_head_are_compared_to_borrowed_io(tmp_path):
    from safetensors.torch import save_file

    module = diagnostics()
    source = torch.ones(16, 4)
    save_file(
        {"embed_tokens.weight": source, "lm_head.weight": source.clone()},
        tmp_path / "model.safetensors",
    )
    model = NS(
        named_parameters=lambda: [
            ("model.embed_tokens.weight", source.clone()),
            ("lm_head.weight", source.clone() + 1),
        ]
    )
    result = module.audit_checkpoint(model, tmp_path, 0, 1)
    assert result["checked"] == 2
    assert len(result["mismatches"]) == 1
    assert result["mismatches"][0]["source"] == "lm_head.weight"


def test_verification_trace_checks_next_input_and_counts_only_matching_prefix():
    module = diagnostics()
    logits = torch.zeros(4, 16)
    logits[torch.arange(4), torch.tensor([5, 0, 7, 8])] = 5
    rows = module.verification_rows(
        logits,
        torch.tensor([4, 5, 6, 7]),
        torch.arange(10, 14),
        torch.tensor([0, 4]),
        torch.tensor([3]),
        torch.tensor([2]),
    )
    row = rows[0]
    assert row["slot"] == 3
    assert row["draft_ids"] == [5, 6, 7]
    assert row["target_argmax_ids"] == [5, 0, 7]
    assert (
        row["predicted_token_positions"] == row["draft_input_positions"] == [11, 12, 13]
    )
    assert row["matches"] == [True, False, True]
    assert row["greedy_matched_prefix"] == 1


def test_placeholder_draft_is_not_counted_as_a_match():
    module = diagnostics()
    row = module.verification_rows(
        torch.zeros(2, 16),
        torch.tensor([4, -1]),
        torch.tensor([20, 21]),
        torch.tensor([0, 2]),
        torch.tensor([0]),
        torch.tensor([1]),
    )[0]
    assert row["valid_draft"] == [False]
    assert row["greedy_matched_prefix"] == 0


def test_worker_diagnostics_is_bounded_and_preserves_outputs(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    module = diagnostics()
    distributed = ModuleType("vllm.distributed")
    distributed.get_tensor_model_parallel_rank = lambda: 0
    distributed.get_tensor_model_parallel_world_size = lambda: 1
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    checkpoint = tmp_path / "draft"
    checkpoint.mkdir()
    source = torch.ones(4, 4)
    save_file({"fc.weight": source}, checkpoint / "model.safetensors")
    embedding = torch.nn.Embedding(16, 4)
    head = torch.nn.Linear(4, 16, bias=False)
    tokens = torch.tensor([[5, 6, 7]])
    model = NS(
        named_parameters=lambda: [("model.fc.weight", source)],
        config=NS(
            mask_token_id=15,
            vocab_size=16,
            block_size=3,
            mmuse_source_config={"aux_hidden_state_layer_ids": [1]},
        ),
        model=NS(embed_tokens=embedding, layers=[]),
        lm_head=head,
        embed_input_ids=embedding,
        compute_logits=head,
        combine_hidden_states=lambda x: x,
        rollout_mmuse=lambda hidden, anchors, **kwargs: (tokens, torch.zeros(1, 3, 16)),
    )
    target = NS(model=NS(embed_tokens=embedding), compute_logits=head)
    spec = NS(
        use_mmuse=lambda: True,
        draft_sample_method="greedy",
        rejection_sample_method="standard",
        synthetic_acceptance_rates=None,
    )

    def original_propose(batch):
        hidden = model.combine_hidden_states(torch.ones(1, 3, 4))
        return model.rollout_mmuse(hidden, torch.tensor([9]))

    proposer = NS(model=model, propose=original_propose)
    verifier = NS(
        _verify=lambda *args: (args[0], torch.tensor([[5, 0]]), torch.tensor([2])),
        sampler=NS(sampling_states=NS(temperature=NS(gpu=torch.zeros(1)))),
    )
    runner = NS(
        vllm_config=NS(model_config=NS(enforce_eager=True, quantization=None)),
        speculative_config=spec,
        speculator=proposer,
        model=target,
        rejection_sampler=verifier,
    )
    output = tmp_path / "trace"
    module.install_diagnostics(NS(model_runner=runner), output, checkpoint, max_steps=1)
    actual, _ = proposer.propose(NS(idx_mapping_np=torch.tensor([0])))
    assert torch.equal(actual, tokens)
    logits = torch.zeros(4, 16)
    logits[torch.arange(4), torch.tensor([5, 0, 7, 8])] = 5
    arguments = (
        logits,
        None,
        torch.tensor([9, 5, 6, 7]),
        torch.arange(10, 14),
        torch.tensor([0, 4]),
        torch.tensor([0]),
    )
    verifier._verify(*arguments)
    verifier._verify(*arguments)
    records = [
        json.loads(line) for line in (output / "rank-0.jsonl").read_text().splitlines()
    ]
    verified = [record for record in records if record["event"] == "verification"]
    assert len(verified) == 1
    assert verified[0]["rows"][0]["anchor_matches_previous_proposal"]
    assert verified[0]["rows"][0]["draft_ids_match_previous_proposal"]

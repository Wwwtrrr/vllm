# SPDX-License-Identifier: Apache-2.0
"""CPU tests of opt-in log alignment, request reordering and logging bounds."""

import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


def load_trace():
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm/model_executor/models/mmuse/trace.py"
    )
    spec = importlib.util.spec_from_file_location("mmuse_trace_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records(caplog):
    return [
        json.loads(item.message.split("MMUSE_TRACE ", 1)[1])
        for item in caplog.records
        if "MMUSE_TRACE " in item.message
    ]


def test_trace_pairs_requests_by_id_and_positions_not_batch_order(caplog):
    module = load_trace()
    caplog.set_level(logging.INFO, logger=module.__name__)
    trace = module.MMuseTrace(2)
    trace.proposal(
        NS(req_ids=["a", "b"], idx_mapping_np=[1, 0]),
        torch.zeros(2, 3, 4),
        torch.tensor([10, 11]),
        torch.tensor([[20, 21, 22], [30, 31, 32]]),
        torch.tensor([[4, 5, 6], [7, 8, 9]]),
    )
    batch = NS(
        req_ids=["b", "a"],
        idx_mapping_np=[0, 1],
        num_reqs=2,
        num_draft_tokens=6,
        logits_indices=torch.arange(8),
        cu_num_logits_np=torch.tensor([0, 4, 8]),
        prefill_len_np=torch.tensor([12, 10]),
        positions=torch.tensor([30, 31, 32, 33, 20, 21, 22, 23]),
        input_ids=torch.tensor([11, 7, 8, 9, 10, 4, 5, 6]),
        num_draft_tokens_per_req=torch.tensor([3, 3]),
    )
    logits = torch.zeros(8, 16)
    for row, token in enumerate([7, 2, 9, 0, 4, 5, 6, 0]):
        logits[row, token] = 10
    before = logits.clone()
    trace.target(logits, batch)
    trace.result(batch, torch.tensor([2, 4]), torch.tensor([2, 0]))
    output = records(caplog)
    assert output[0]["rows"][0]["prediction_positions"] == [21, 22, 23]
    rows = output[1]["rows"]
    assert [x["request_id"] for x in rows] == ["b", "a"]
    assert all(x["transport_matches"] for row in rows for x in row["comparisons"])
    assert [x["argmax_matches"] for x in rows[0]["comparisons"]] == [True, False, True]
    assert output[2]["num_sampled"] == [2, 4]
    torch.testing.assert_close(before, logits)
    batch.input_ids[5] = 12
    trace.target(logits, batch)
    assert (
        records(caplog)[-1]["rows"][1]["comparisons"][0]["transport_matches"] is False
    )
    trace.target(None, None)  # Verification budget exhausted: no tensor access.
    assert len(records(caplog)) == 4


def test_prefill_placeholders_are_not_reported_as_valid_draft_matches(caplog):
    module = load_trace()
    caplog.set_level(logging.INFO, logger=module.__name__)
    trace = module.MMuseTrace(1)
    batch = NS(
        req_ids=["a"],
        idx_mapping_np=[0],
        num_draft_tokens=2,
        logits_indices=torch.arange(3),
        positions=torch.arange(9, 12),
        input_ids=torch.tensor([1, 2, 3]),
        cu_num_logits_np=torch.tensor([0, 3]),
        prefill_len_np=torch.tensor([10]),
    )
    trace.target(torch.zeros(3, 8), batch)
    comparisons = records(caplog)[0]["rows"][0]["comparisons"]
    assert all(x["placeholder"] for x in comparisons)
    assert all(
        x["transport_matches"] is None and x["argmax_matches"] is None
        for x in comparisons
    )


def test_disabled_trace_never_reads_tensors_or_emits_logs(caplog):
    trace = load_trace().MMuseTrace(0)
    trace.proposal(None, None, None, None, None)
    trace.target(None, None)
    trace.result(None, None, None)
    assert not records(caplog)


@pytest.mark.parametrize("steps", [-1, 33, True, "4"])
def test_trace_step_limit_requires_a_bounded_integer(steps):
    with pytest.raises(ValueError, match="mmuse_trace_steps"):
        load_trace().MMuseTrace(steps)

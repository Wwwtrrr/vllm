# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for MMuse graph dispatch; no accelerator capture is simulated."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def adapter(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm/model_executor/models/mmuse_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("mmuse_graph_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compilation = ModuleType("vllm.config.compilation")
    compilation.CUDAGraphMode = SimpleNamespace(NONE="none", FULL="full")
    logger_module = ModuleType("vllm.logger")
    logger = Mock()
    logger_module.init_logger = lambda name: logger
    monkeypatch.setitem(sys.modules, "vllm.config.compilation", compilation)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    return module, logger


def config(*, target_eager=False, draft_eager=False, method="probabilistic"):
    source = dict(
        transformer_layer_config={"model_type": "qwen3"},
        aux_hidden_state_layer_ids=[1, 2],
        block_size=7,
        sample_from_anchor=True,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=target_eager,
            hf_config=SimpleNamespace(model_type="qwen3"),
            get_total_num_hidden_layers=lambda: 36,
        ),
        parallel_config=SimpleNamespace(),
        watermark_config=None,
        speculative_config=SimpleNamespace(
            enforce_eager=draft_eager,
            draft_sample_method=method,
            num_speculative_tokens=7,
            enable_adaptive_verification=False,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    mmuse_source_config=source, layer_types=["full_attention"]
                )
            ),
        ),
    )


@pytest.mark.parametrize(
    "target_eager,draft_eager", [(True, True), (False, True), (False, False)]
)
@pytest.mark.parametrize("method", ["greedy", "probabilistic"])
def test_target_and_draft_graph_configs(adapter, target_eager, draft_eager, method):
    module, _ = adapter
    module.validate_mmuse_config(
        config(target_eager=target_eager, draft_eager=draft_eager, method=method)
    )


@pytest.mark.parametrize("invalid", ["anchor", "length", "global_static"])
def test_draft_graph_rejects_incompatible_layouts(adapter, invalid):
    module, _ = adapter
    cfg = config()
    source = cfg.speculative_config.draft_model_config.hf_config.mmuse_source_config
    if invalid == "anchor":
        source["sample_from_anchor"] = False
        message = "sample_from_anchor"
    elif invalid == "length":
        cfg.speculative_config.num_speculative_tokens = 6
        message = "block_size"
    else:
        source.update(
            dflash2_candidate_selector=True,
            selector_correction_feedback="static",
            dflash2_selector_search_mode="global",
        )
        message = "static global Selector"
    with pytest.raises(ValueError, match=message):
        module.validate_mmuse_config(cfg)


def test_corrected_feedback_does_not_run_global_path_check(adapter):
    module, _ = adapter
    cfg = config()
    source = cfg.speculative_config.draft_model_config.hf_config.mmuse_source_config
    source.update(
        dflash2_candidate_selector=True,
        selector_correction_feedback="corrected",
        dflash2_selector_search_mode="global",
    )
    module.validate_mmuse_config(cfg)


@pytest.mark.parametrize(
    "eager,supported", [(True, True), (False, True), (False, False)]
)
def test_manager_fallback_and_capture_status(adapter, eager, supported):
    module, logger = adapter

    class Parent:
        def init_cudagraph_manager(self, mode):
            self.mode = mode
            self.query_cudagraph_manager = SimpleNamespace(
                needs_capture=lambda: mode == "full" and supported,
                captured_token_counts=lambda: [7, 14],
            )

        def capture(self):
            self.captured = True

    class Speculator(module.MMuseGreedyRolloutMixin, Parent):
        pass

    obj = Speculator()
    obj.speculative_config = SimpleNamespace(enforce_eager=eager)
    obj.num_query_per_req = 7
    obj.captured = False
    obj.init_cudagraph_manager("full")
    assert obj.mode == ("none" if eager else "full")
    obj.capture()
    assert obj.captured == (not eager and supported)
    messages = [call.args[0] for call in logger.info.call_args_list]
    assert any("setup" in message for message in messages)
    assert any(
        ("capture completed" if obj.captured else "skipped") in message
        for message in messages
    )


@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("block_size", [7, 15])
@pytest.mark.parametrize("mode", [None, "none", "full"])
@pytest.mark.parametrize("probabilistic", [False, True])
def test_forward_mode_block_shape_and_sampling(
    adapter, num_reqs, block_size, mode, probabilistic
):
    module, _ = adapter
    hidden = torch.randn(num_reqs * block_size, 16)
    seen = {}

    def run_model(*args):
        seen["mode"] = args[-1]
        return hidden

    def rollout(block, anchors, *, sample_token=None):
        torch.testing.assert_close(block, hidden.reshape(num_reqs, block_size, 16))
        assert anchors.shape == (num_reqs,)
        if sample_token is None:
            return torch.ones(num_reqs, block_size, dtype=torch.long), None
        return torch.stack(
            [
                sample_token(torch.zeros(num_reqs, 32), step)
                for step in range(block_size)
            ],
            dim=1,
        ), None

    obj = SimpleNamespace(
        _run_model=run_model,
        num_query_per_req=block_size,
        input_buffers=SimpleNamespace(input_ids=torch.zeros(num_reqs * block_size)),
        model=SimpleNamespace(rollout_mmuse=rollout),
        draft_tokens=torch.full((5, block_size), -1, dtype=torch.long),
        draft_logits=object() if probabilistic else None,
        _sample_mmuse_token=lambda logits, position, count: torch.full(
            (count,), position + 1, dtype=torch.long
        ),
    )
    module.MMuseGreedyRolloutMixin._generate_draft(
        obj, num_reqs, num_reqs * block_size, None, None, None, mode
    )
    assert seen["mode"] == (mode or "none")
    expected = (
        torch.arange(1, block_size + 1).expand(num_reqs, -1)
        if probabilistic
        else torch.ones(num_reqs, block_size, dtype=torch.long)
    )
    torch.testing.assert_close(obj.draft_tokens[:num_reqs], expected)
    assert (obj.draft_tokens[num_reqs:] == -1).all()

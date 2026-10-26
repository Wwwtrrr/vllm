# SPDX-License-Identifier: Apache-2.0
"""Native head capture with synthetic weights; excludes paged attention."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

MODELS = Path(__file__).resolve().parents[3] / "vllm/model_executor/models"


def runtime(device, fusion, monkeypatch):
    for package, path in (
        ("vllm", MODELS.parents[1]),
        ("vllm.model_executor", MODELS.parent),
        ("vllm.model_executor.models", MODELS),
        ("vllm.model_executor.models.mmuse", MODELS / "mmuse"),
    ):
        root = ModuleType(package)
        root.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, package, root)
    name = "vllm.model_executor.models.mmuse.runtime"
    spec = importlib.util.spec_from_file_location(name, MODELS / "mmuse/runtime.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    config = {
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
            "vocab_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
        },
        "block_size": 3,
        "draft_vocab_size": 32,
        "mask_token_id": 31,
        "aux_hidden_state_layer_ids": [1, 2],
        "sample_from_anchor": True,
        "enable_correction_head": True,
        "correction_hidden_size": 16,
        "correction_rank": 8,
        "correction_num_heads": 4,
        "correction_hidden_feedback": True,
        "correction_lm_head_fusion": fusion,
        "dflash2_candidate_selector": True,
        "dflash2_selector_rank": 8,
        "dflash2_selector_top_k": 4,
        "selector_correction_feedback": "corrected",
    }
    model = module.MMuseRuntime(config, torch.nn.Linear(32, 16))
    model.embed_tokens = torch.nn.Embedding(32, 16)
    model.lm_head = torch.nn.Linear(16, 32, bias=False)
    return model.eval().to(device)


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu"])
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_rollout_graph_reads_changed_anchors_and_hidden_rows(
    device, fusion, dtype, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("NPU hardware required")
    backend = getattr(torch, device, None)
    torch.manual_seed(12)
    model = runtime(device, fusion, monkeypatch).to(dtype=dtype)
    hidden = torch.randn(4, 3, 16, device=device, dtype=dtype)
    anchors = torch.tensor([1, 2, 3, 4], device=device)
    output = torch.empty(4, 3, dtype=torch.long, device=device)
    scores = torch.empty(4, 3, 32, device=device, dtype=dtype)

    def forward():
        tokens, logits = model.rollout_correction(hidden, anchors)
        output.copy_(tokens)
        scores.copy_(logits)

    graph = None
    if device != "cpu":
        stream = backend.Stream()
        stream.wait_stream(backend.current_stream())
        with backend.stream(stream):
            for _ in range(3):
                forward()
        backend.current_stream().wait_stream(stream)
        backend.synchronize()
        graph = backend.CUDAGraph() if device == "cuda" else backend.NPUGraph()
        with backend.graph(graph):
            forward()
    for step in range(3):
        hidden.copy_(torch.randn_like(hidden))
        anchors.copy_((anchors + 5) % 32)
        if step == 1:
            hidden[1:].zero_()
            anchors[1:].fill_(31)
        expected, logits = model.rollout_correction(hidden, anchors)
        if graph is None:
            forward()
        else:
            graph.replay()
            backend.synchronize()
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(scores, logits, atol=2e-4, rtol=2e-4)
        single, _ = model.rollout_correction(hidden[:1], anchors[:1])
        torch.testing.assert_close(output[:1], single)

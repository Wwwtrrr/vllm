# SPDX-License-Identifier: Apache-2.0
"""MMuse FULL lifecycle contracts and native head-rollout replay checks.

Native tests cover the shared Decoder-output-to-token capture boundary using
synthetic weights. They do not validate paged Attention or model accuracy.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch

MODELS = Path(__file__).resolve().parents[3] / "vllm/model_executor/models"


@pytest.fixture
def mixin(monkeypatch):
    class Mode(str):
        def decode_mode(self):
            return self

    modes = NS(NONE=Mode("NONE"), FULL=Mode("FULL"))
    fake = ModuleType("vllm.config.compilation")
    fake.CUDAGraphMode = modes
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    tree = ast.parse((MODELS / "mmuse_runtime.py").read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MMuseGreedyRolloutMixin"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), "mmuse_rollout", "exec"),
        namespace,
    )
    return namespace[cls.name]


@pytest.mark.parametrize("eager", [False, True])
def test_draft_eager_is_respected_without_disabling_target(mixin, eager):
    class Parent:
        def init_cudagraph_manager(self, mode):
            self.selected = mode

    class Proposer(mixin, Parent):
        pass

    full = NS(decode_mode=lambda: "FULL")
    proposer = Proposer()
    proposer.speculative_config = NS(enforce_eager=eager, draft_sample_method="greedy")
    proposer.draft_model_config = NS(
        hf_config=NS(mmuse_source_config={"block_size": 3})
    )
    proposer.num_query_per_req = 3
    proposer.init_cudagraph_manager(full)
    assert proposer.selected == ("NONE" if eager else full)


@pytest.mark.parametrize(
    "field,value", [("block_size", 4), ("sample_from_anchor", False)]
)
def test_incompatible_full_block_is_rejected_before_capture(mixin, field, value):
    proposer = mixin()
    proposer.speculative_config = NS(enforce_eager=False, draft_sample_method="greedy")
    source = {"block_size": 3, "sample_from_anchor": True, field: value}
    proposer.draft_model_config = NS(hf_config=NS(mmuse_source_config=source))
    proposer.num_query_per_req = 3
    with pytest.raises(ValueError, match="MMuse FULL"):
        proposer.init_cudagraph_manager(NS(decode_mode=lambda: "FULL"))


def test_incomplete_padded_block_is_rejected(mixin):
    proposer = mixin()
    proposer._run_model = lambda *args: torch.zeros(5, 16)
    proposer.num_query_per_req = 3
    with pytest.raises(ValueError, match="complete padded query blocks"):
        proposer._generate_draft(2, 5, None, None, None)


def test_context_anchor_moves_with_model_but_is_not_checkpoint_state():
    class Parent(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.config = NS(hidden_size=16)

    tree = ast.parse((MODELS / "qwen3_mmuse.py").read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MMuseQwen3Model"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    minimal = ast.ClassDef(
        name="Model",
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=[init],
        decorator_list=[],
    )
    namespace = {"Parent": Parent, "torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[minimal], type_ignores=[])),
            "anchor_buffer",
            "exec",
        ),
        namespace,
    )
    model = namespace["Model"](
        vllm_config=NS(
            scheduler_config=NS(max_num_seqs=4), model_config=NS(dtype=torch.float32)
        )
    )
    assert model.context_anchor.shape == (4, 16)
    assert "context_anchor" in dict(model.named_buffers())
    assert "context_anchor" not in model.state_dict()
    assert model.to(dtype=torch.float64).context_anchor.dtype == torch.float64


def make_runtime(device, fusion, monkeypatch):
    # Load only pure-Torch model modules, without importing vLLM extensions.
    for package, path in (
        ("vllm", MODELS.parents[1]),
        ("vllm.model_executor", MODELS.parent),
        ("vllm.model_executor.models", MODELS),
        ("vllm.model_executor.models.mmuse", MODELS / "mmuse"),
    ):
        root = ModuleType(package)
        root.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, package, root)
    package = "vllm.model_executor.models.mmuse"
    name = f"{package}.runtime"
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
    runtime = sys.modules[name].MMuseRuntime(config, torch.nn.Linear(32, 16))
    runtime.embed_tokens = torch.nn.Embedding(32, 16)
    runtime.lm_head = torch.nn.Linear(16, 32, bias=False)
    return runtime.eval().to(device)


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu"])
@pytest.mark.parametrize("fusion", [False, True])
def test_full_rollout_reads_new_inputs_and_keeps_rows_independent(
    mixin, device, fusion, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("NPU required")
    backend = getattr(torch, device, None)
    torch.manual_seed(12)
    runtime = make_runtime(device, fusion, monkeypatch)
    hidden = torch.randn(4, 3, 16, device=device)
    anchors = torch.tensor([1, 2, 3, 4], device=device)
    proposer = mixin()
    proposer.num_query_per_req = 3
    proposer.draft_logits = None
    proposer.draft_tokens = torch.empty(4, 3, dtype=torch.long, device=device)
    proposer.input_buffers = NS(
        input_ids=torch.zeros(12, dtype=torch.long, device=device)
    )
    proposer.input_buffers.input_ids[::3].copy_(anchors)
    proposer._run_model = lambda *args: hidden.flatten(0, 1)
    logits_output = torch.empty(4, 3, 32, device=device)

    def rollout(block, anchor_ids):
        tokens, logits = runtime.rollout_correction(block, anchor_ids)
        logits_output.copy_(logits)
        return tokens, logits

    proposer.model = NS(rollout_mmuse=rollout)

    def forward():
        proposer._generate_draft(4, 12, None, None, None)

    graph = None
    with torch.inference_mode():
        if device == "cpu":
            forward()
        else:
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
                # A single live row in a four-request capture must not depend
                # on the contents of inactive rows; next step reuses them.
                hidden[1:].zero_()
                anchors[1:].fill_(31)
            proposer.input_buffers.input_ids[::3].copy_(anchors)
            expected, expected_logits = runtime.rollout_correction(hidden, anchors)
            if graph is None:
                forward()
            else:
                graph.replay()
                backend.synchronize()
            torch.testing.assert_close(proposer.draft_tokens, expected)
            torch.testing.assert_close(
                logits_output, expected_logits, atol=2e-4, rtol=2e-4
            )
            single, _ = runtime.rollout_correction(hidden[:1], anchors[:1])
            torch.testing.assert_close(proposer.draft_tokens[:1], single)

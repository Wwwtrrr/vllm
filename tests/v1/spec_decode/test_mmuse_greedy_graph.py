# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for independent MMuse query buffers and graph boundaries."""

import ast
import copy
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm/v1/worker/gpu/spec_decode/mmuse"


def prepare_inputs():
    spec = importlib.util.spec_from_file_location(
        "mmuse_query_inputs", SOURCE / "inputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_mmuse_inputs


def method(name, **namespace):
    tree = ast.parse((SOURCE / "greedy.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MMuseGreedySpeculator"
    )
    node = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    scope = {"torch": torch, **namespace}
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), str(SOURCE / "greedy.py"), "exec"
        ),
        scope,
    )
    return scope[name]


@pytest.fixture(params=["cpu", "cuda", "npu"])
def query_device(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("NPU hardware required")
    with torch.device(device):
        yield device


@pytest.mark.parametrize("column_sampled", [False, True])
def test_query_layout_rejects_suffix_and_preserves_batch_to_state_mapping(
    query_device, column_sampled
):
    buffers = NS(
        input_ids=torch.full((12,), 999),
        positions=torch.full((12,), 999),
        seq_lens=torch.full((3,), 999),
        query_start_loc=torch.full((4,), 999),
    )
    batch = NS(
        num_reqs=2,
        num_tokens=6,
        query_start_loc=torch.tensor([0, 4, 6]),
        idx_mapping=torch.tensor([2, 0]),
        positions=torch.tensor([10, 11, 12, 13, 20, 21]),
    )
    contexts, context_slots, queries = [torch.full((12,), 999) for _ in range(3)]
    table = torch.tensor([[0, 0, 3, 4, 5, 6, 7], [0, 0, 8, 9, 10, 11, 0]])
    prepare = prepare_inputs()
    args = (
        torch.tensor([1, 0]),
        torch.tensor([2, 0]),
        torch.tensor([70, 80, 90]),
        torch.tensor([[7, 8, 9]]),
    )
    if column_sampled:
        args = (*args[:2], args[2].reshape(-1, 1), args[3])
    prepare(
        buffers, batch, *args, contexts, context_slots, queries, table, 4, 3, 99, 128
    )
    assert buffers.input_ids.tolist() == [90, 99, 99, 7, 99, 99, 0, 0, 0, 0, 0, 0]
    assert buffers.positions[:6].tolist() == [12, 13, 14, 22, 23, 24]
    assert buffers.seq_lens.tolist() == [15, 25, 0]
    assert buffers.query_start_loc.tolist() == [0, 3, 6, 6]
    assert contexts[:6].tolist() == [10, 11, 0, 0, 20, 21]
    assert context_slots.tolist() == [14, 15, -1, -1, 44, 45] + [-1] * 6
    assert queries.tolist() == [16, 17, 18, 46, 47, -1] + [-1] * 6

    # Reuse the allocations for a smaller, reordered batch: no stale padded K/V.
    pointers = (buffers.input_ids.data_ptr(), queries.data_ptr())
    batch.num_reqs, batch.num_tokens = 1, 2
    batch.query_start_loc = torch.tensor([0, 2])
    batch.idx_mapping = torch.tensor([1])
    batch.positions = torch.tensor([10, 11])
    prepare(
        buffers,
        batch,
        torch.tensor([0]),
        torch.tensor([0]),
        args[2],
        args[3],
        contexts,
        context_slots,
        queries,
        table,
        4,
        3,
        99,
        128,
    )
    assert buffers.input_ids.tolist() == [8, 99, 99] + [0] * 9
    assert queries.tolist() == [16, 17, 18] + [-1] * 9
    assert context_slots.tolist() == [14, 15] + [-1] * 10
    assert pointers == (buffers.input_ids.data_ptr(), queries.data_ptr())


def test_draft_compilation_choice_does_not_mutate_target_config():
    class Base:
        draft_sample_method = "greedy"

        def __init__(self, config, device):
            self.vllm_config = config
            self.num_speculative_steps = 3
            self.max_num_reqs, self.max_num_tokens = 2, 12
            self.hidden_size, self.dtype = 4, torch.float32
            self.draft_model_config = config.speculative_config.draft_model_config

    tree = ast.parse((SOURCE / "greedy.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MMuseGreedySpeculator"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    node = ast.ClassDef(
        name="Proposer",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[init],
        decorator_list=[],
    )
    scope = dict(
        Base=Base,
        copy=copy,
        torch=torch,
        validate_mmuse_config=lambda c: None,
        CompilationMode=NS(NONE=0),
        get_parallel_drafting_token_id=lambda c: 99,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            "greedy",
            "exec",
        ),
        scope,
    )
    spec = NS(
        draft_sample_method="greedy",
        num_speculative_tokens=3,
        enable_adaptive_verification=False,
        uses_dynamic_speculative_decoding=lambda: False,
        draft_model_config=NS(hf_config=NS(mmuse_source_config={"block_size": 3})),
    )
    config = NS(
        speculative_config=spec, compilation_config=NS(mode=3, cudagraph_mode="FULL")
    )
    proposer = scope["Proposer"](config, "cpu")
    assert config.compilation_config.mode == 3
    assert proposer.vllm_config.compilation_config.mode == 0
    assert proposer.vllm_config.compilation_config.cudagraph_mode == "FULL"
    spec.num_speculative_tokens = 2
    with pytest.raises(ValueError, match="block_size"):
        scope["Proposer"](config, "cpu")


def test_query_forward_includes_rollout_and_writes_persistent_tokens():
    class Model:
        def __call__(self, input_ids, positions, inputs_embeds):
            return torch.stack([input_ids.float(), positions.float()], -1)

        def rollout_mmuse(self, hidden, anchors):
            return hidden[..., 1].long() + anchors[:, None], None

    proposer = NS(
        model=Model(),
        input_buffers=NS(
            input_ids=torch.tensor([10, 99, 99, 20, 99, 99]), positions=torch.arange(6)
        ),
        draft_tokens=torch.zeros(2, 3, dtype=torch.long),
        num_query_per_req=3,
        vllm_config=None,
        query_context=nullcontext,
    )
    proposer.rollout_queries = lambda block, anchors: proposer.model.rollout_mmuse(
        block, anchors
    )
    forward = method(
        "forward_queries",
        set_forward_context=lambda *a, **k: nullcontext(),
        BatchDescriptor=lambda **k: k,
        CUDAGraphMode=NS(NONE=0),
    )
    pointer = proposer.draft_tokens.data_ptr()
    forward(proposer, 2, 6, {}, {})
    assert proposer.draft_tokens.tolist() == [[10, 11, 12], [23, 24, 25]]
    proposer.input_buffers.input_ids[0] = 30
    forward(proposer, 2, 6, {}, {})
    assert proposer.draft_tokens[0].tolist() == [30, 31, 32]
    assert proposer.draft_tokens.data_ptr() == pointer


@pytest.mark.parametrize("use_graph", [False, True])
def test_proposal_precomputes_confirmed_context_before_replay(use_graph):
    events = []
    metadata = {"current": object()}
    desc = NS(cg_mode="FULL", num_tokens=6, num_reqs=2)
    batch = NS(
        num_reqs=1,
        num_tokens=4,
        query_start_loc=torch.tensor([0, 4]),
        idx_mapping=torch.tensor([1]),
        positions=torch.arange(4),
        seq_lens_cpu_upper_bound=torch.tensor([4]),
    )
    buffers = NS(
        input_ids=torch.zeros(6, dtype=torch.long),
        positions=torch.zeros(6, dtype=torch.long),
        seq_lens=torch.zeros(2, dtype=torch.long),
        query_start_loc=torch.zeros(3, dtype=torch.long),
    )
    anchor = torch.full((2, 2), 999.0)
    hidden = torch.arange(8).view(4, 2).float()

    def precompute(states, positions, slots=None):
        events.append("context")
        assert positions.tolist() == [0, 1, 0, 0]
        assert slots[0].tolist() == [4, 5, -1, -1]

    model = NS(
        model=NS(context_anchor=anchor, hidden_norm=lambda x: x),
        precompute_and_store_context_kv=precompute,
    )
    proposer = NS(
        num_query_per_req=3,
        model=model,
        hidden_states=torch.empty(6, 2),
        max_model_len=128,
        context_positions=torch.zeros(6, dtype=torch.long),
        context_slots={0: torch.full((6,), -1)},
        group_ids=[0],
        layer_groups=[0],
        mask_token_id=99,
        input_buffers=buffers,
        kv_cache_config=None,
        draft_tokens=torch.zeros(2, 3, dtype=torch.long),
        block_tables=NS(
            slot_mappings=torch.full((1, 6), -1),
            input_block_tables=[torch.tensor([[1, 2], [0, 0]])],
            kernel_block_sizes=[4],
        ),
    )
    proposer.prepare_sampling = lambda *args: None

    def eager(count, total, meta, slots):
        events.append("eager")
        assert count == 1 and total == 3 and meta is metadata

    def replay(current_desc):
        events.append("replay")
        assert current_desc is desc and proposer.replay_metadata is metadata
        assert proposer.input_buffers.input_ids.tolist() == [20, 99, 99, 0, 0, 0]

    proposer.query_cudagraph_manager = (
        NS(dispatch=lambda *a, **k: desc, run_fullgraph=replay) if use_graph else None
    )
    proposer.build_query_metadata = lambda *a: metadata
    proposer.forward_queries = eager
    propose = method(
        "propose",
        prepare_mmuse_inputs=prepare_inputs(),
        CUDAGraphMode=NS(FULL="FULL", NONE="NONE"),
        BatchExecutionDescriptor=lambda **k: NS(**k),
        build_slot_mappings_by_layer=lambda slots, config: slots,
    )
    result = propose(
        proposer,
        batch,
        {},
        {},
        hidden,
        None,
        torch.tensor([1]),
        torch.tensor([2]),
        torch.tensor([10, 20]),
        torch.tensor([[1, 2]]),
        None,
        None,
    )
    assert result.shape == (1, 3)
    assert events == ["context", "replay" if use_graph else "eager"]
    torch.testing.assert_close(anchor, torch.tensor([[2.0, 3.0], [0.0, 0.0]]))

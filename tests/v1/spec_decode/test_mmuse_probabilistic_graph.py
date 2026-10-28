# SPDX-License-Identifier: Apache-2.0
"""Probability cache/state contracts and optional native MMuse sampling graphs.

The CPU sampler is an oracle for callback routing, not the production RNG.
Native CUDA/NPU cases execute the actual sampler and real MMuse heads.
Paged attention and Target verification require separate full-model validation.
"""

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PROBABILITY = ROOT / "vllm/v1/worker/gpu/spec_decode/mmuse/probabilistic.py"


def sampling_class():
    tree = ast.parse(PROBABILITY.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    scope = {"torch": torch}
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(PROBABILITY), "exec"),
        scope,
    )
    return scope[cls.name]


def cpu_sample(
    logits,
    mapping,
    temperature,
    seeds,
    positions,
    *,
    logits_cache,
    logits_cache_col,
    apply_temperature,
    is_drafting,
    use_fp64,
):
    assert apply_temperature and is_drafting and not use_fp64
    result = torch.zeros(len(mapping), dtype=torch.long)
    for row, state in enumerate(mapping.tolist()):
        if state < 0:
            continue
        logits_cache[state, logits_cache_col[row]] = logits[row]
        if temperature[state] == 0:
            result[row] = logits[row].argmax()
        else:
            generator = torch.Generator().manual_seed(
                int(seeds[state]) + int(positions[row]) * 7919
            )
            probabilities = torch.softmax(logits[row].float() / temperature[state], -1)
            result[row] = torch.multinomial(probabilities, 1, generator=generator)[0]
    return result


def proposer(device, dtype, vocab=32):
    cls = sampling_class()
    obj = cls.__new__(cls)
    obj.num_query_per_req, obj.max_num_reqs = 3, 4
    obj.idx_mapping = torch.full((4,), -1, dtype=torch.int32, device=device)
    obj.sample_positions = torch.zeros(4, 3, dtype=torch.int64, device=device)
    obj.sample_columns = tuple(
        torch.full((4,), col, dtype=torch.int32, device=device) for col in range(3)
    )
    obj.sample_offsets = torch.arange(3, device=device)
    obj.temperature = torch.zeros(4, device=device)
    obj.seeds = torch.zeros(4, dtype=torch.int64, device=device)
    obj.draft_logits = torch.full((4, 3, vocab), -77, dtype=dtype, device=device)
    obj.use_fp64_gumbel = False
    return obj


def test_probability_constructor_uses_our_query_engine_and_inert_buffers():
    class Base:
        def __init__(self, config, device):
            self.vllm_config = config
            self.num_speculative_steps = 3
            self.max_num_reqs, self.max_num_tokens = 4, 32
            self.hidden_size, self.dtype = 16, torch.float32
            self.draft_model_config = config.speculative_config.draft_model_config
            self.idx_mapping = torch.zeros(4, dtype=torch.int32)
            self.temperature = torch.zeros(4)
            self.seeds = torch.zeros(4, dtype=torch.int64)

    tree = ast.parse(PROBABILITY.with_name("greedy.py").read_text())
    base = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MMuseGreedySpeculator"
    )
    base.body = [
        n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    ]
    tree = ast.parse(PROBABILITY.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    scope = {
        "torch": torch,
        "copy": copy,
        "DraftModelSpeculator": Base,
        "validate_mmuse_config": lambda c: None,
        "CompilationMode": NS(NONE=0),
        "get_parallel_drafting_token_id": lambda c: 31,
    }
    exec(
        compile(
            ast.Module(body=[base, *classes], type_ignores=[]),
            "probability_init",
            "exec",
        ),
        scope,
    )
    spec = NS(
        draft_sample_method="probabilistic",
        num_speculative_tokens=3,
        draft_model_config=NS(hf_config=NS(mmuse_source_config={"block_size": 3})),
        enable_adaptive_verification=False,
        uses_dynamic_speculative_decoding=lambda: False,
    )
    config = NS(
        speculative_config=spec, compilation_config=NS(mode=3, cudagraph_mode="FULL")
    )
    obj = scope["MMuseProbabilisticSpeculator"](config, "cpu")
    assert config.compilation_config.mode == 3
    assert obj.vllm_config.compilation_config.mode == 0
    assert obj.draft_sample_method == "probabilistic"
    assert obj.idx_mapping.tolist() == [-1] * 4
    assert obj.sample_positions.shape == (4, 3)
    assert obj.sample_offsets.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    "target_eager,draft_eager", [(False, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("sampling", ["greedy", "probabilistic"])
def test_query_contract_allows_target_full_and_keeps_sampling_checks(
    target_eager, draft_eager, sampling
):
    path = ROOT / "vllm/model_executor/models/mmuse/validation.py"
    tree = ast.parse(path.read_text())
    scope = {}
    exec(compile(tree, str(path), "exec"), scope)
    spec = NS(
        draft_sample_method=sampling,
        enforce_eager=draft_eager,
        enable_adaptive_verification=False,
        num_speculative_tokens=3,
        uses_dynamic_speculative_decoding=lambda: False,
    )
    config = NS(speculative_config=spec, model_config=NS(enforce_eager=target_eager))
    source = {"sample_from_anchor": True, "block_size": 3}
    scope["_validate_qwen3_sampling"](config, source)
    spec.enable_adaptive_verification = True
    with pytest.raises(ValueError, match="adaptive"):
        scope["_validate_qwen3_sampling"](config, source)
    spec.enable_adaptive_verification = False
    source["block_size"] = 4
    with pytest.raises(ValueError, match="block_size"):
        scope["_validate_qwen3_sampling"](config, source)
    source["block_size"] = 3
    source["sample_from_anchor"] = False
    with pytest.raises(ValueError, match="sample_from_anchor"):
        scope["_validate_qwen3_sampling"](config, source)
    source["sample_from_anchor"] = True
    spec.uses_dynamic_speculative_decoding = lambda: True
    with pytest.raises(ValueError, match="fixed trained query block"):
        scope["_validate_qwen3_sampling"](config, source)
    spec.uses_dynamic_speculative_decoding = lambda: False
    config.watermark_config = object()
    with pytest.raises(ValueError, match="watermarking"):
        scope["_validate_qwen3_sampling"](config, source)


def test_sampling_positions_cache_precision_and_padding_are_request_scoped():
    obj = proposer("cpu", torch.bfloat16, vocab=4)
    obj._gumbel_sample = cpu_sample
    batch = NS(
        num_reqs=2,
        idx_mapping=torch.tensor([2, 0]),
        query_start_loc=torch.tensor([0, 4, 8]),
        positions=torch.tensor([10, 11, 12, 13, 20, 21, 22, 23]),
    )
    temperature = torch.tensor([0.0, 0.5, 1.5, 0.7])
    seeds = torch.tensor([11, 22, 33, 44])
    obj.prepare_sampling(batch, torch.tensor([2, 1]), temperature, seeds, False)
    assert obj.idx_mapping.tolist() == [2, 0, -1, -1]
    assert obj.sample_positions.tolist() == [
        [12, 13, 14],
        [23, 24, 25],
        [0, 0, 0],
        [0, 0, 0],
    ]
    logits = torch.tensor(
        [
            [0.1223, 0.2567, 0.99, -0.1],
            [1.1, 2.2, 0.0, -1.0],
            [9.0, 8.0, 7.0, 6.0],
            [6.0, 7.0, 8.0, 9.0],
        ]
    )
    output = obj.sample_token(logits, 1, 4)
    assert output.shape == (4,)
    assert output[1] == 1  # A zero-temperature request remains greedy.
    torch.testing.assert_close(obj.draft_logits[2, 1], logits[0].bfloat16())
    torch.testing.assert_close(obj.draft_logits[0, 1], logits[1].bfloat16())
    assert (obj.draft_logits[[1, 3]] == -77).all()
    assert (obj.draft_logits[:, [0, 2]] == -77).all()
    batch.num_reqs = 1
    batch.idx_mapping = torch.tensor([1])
    obj.prepare_sampling(batch, torch.tensor([0]), temperature, seeds + 1, False)
    assert obj.idx_mapping.tolist() == [1, -1, -1, -1]
    assert obj.sample_positions.tolist() == [
        [14, 15, 16],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert obj.seeds.tolist() == [12, 23, 34, 45]
    before = obj.draft_logits.clone()
    obj.reset_sampling()
    obj.sample_token(logits, 2, 4)
    torch.testing.assert_close(obj.draft_logits, before)


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fusion", [False, True])
@torch.inference_mode()
def test_real_rollout_probability_replay_updates_state_and_logits(
    device, dtype, fusion, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("NPU hardware required")
    helper_spec = importlib.util.spec_from_file_location(
        "mmuse_rollout_helper", Path(__file__).with_name("test_mmuse_rollout_graph.py")
    )
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    torch.manual_seed(33)
    runtime = helper.runtime(device, fusion, monkeypatch).to(dtype=dtype)
    obj = proposer(device, dtype)
    if device == "cpu":
        obj._gumbel_sample = cpu_sample
    elif device == "cuda":
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        obj._gumbel_sample = gumbel_sample
    else:
        from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

        obj._gumbel_sample = gumbel_sample
    obj.model = NS(rollout_mmuse=runtime.rollout_correction)
    hidden = torch.randn(4, 3, 16, dtype=dtype, device=device)
    anchors = torch.tensor([1, 2, 3, 4], device=device)
    output = torch.empty(4, 3, dtype=torch.int64, device=device)

    def forward():
        tokens, _ = obj.rollout_queries(hidden, anchors)
        output.copy_(tokens)

    graph = None
    backend = getattr(torch, device, None)
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
        backend.synchronize()
        assert (obj.draft_logits == -77).all()  # Capture must be inert.
    pointers = (
        obj.idx_mapping.data_ptr(),
        obj.seeds.data_ptr(),
        obj.sample_positions.data_ptr(),
    )
    for step, states in enumerate(([2, 0], [1], [0, 3])):
        count = len(states)
        batch = NS(
            num_reqs=count,
            idx_mapping=torch.tensor(states, device=device),
            query_start_loc=torch.arange(count + 1, device=device) * 4,
            positions=torch.arange(count * 4, device=device) + 10 + step * 11,
        )
        temperature = torch.tensor([0.0, 0.7, 1.2, 1.5], device=device)
        seeds = torch.tensor([11, 22, 33, 44], device=device) + step * 13
        obj.prepare_sampling(
            batch,
            torch.ones(count, dtype=torch.int64, device=device),
            temperature,
            seeds,
            False,
        )
        hidden.copy_(torch.randn_like(hidden))
        anchors.copy_((anchors + 7) % 32)
        before = obj.draft_logits.clone()
        expected, scores = obj.rollout_queries(hidden, anchors)
        cache = obj.draft_logits.clone()
        torch.testing.assert_close(
            cache[batch.idx_mapping], scores[:count].to(dtype), rtol=0, atol=0
        )
        obj.draft_logits.copy_(before)
        if graph is None:
            forward()
        else:
            graph.replay()
            backend.synchronize()
        torch.testing.assert_close(output[:count], expected[:count])
        torch.testing.assert_close(obj.draft_logits, cache, rtol=0, atol=0)
        assert pointers == (
            obj.idx_mapping.data_ptr(),
            obj.seeds.data_ptr(),
            obj.sample_positions.data_ptr(),
        )

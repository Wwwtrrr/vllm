# SPDX-License-Identifier: Apache-2.0
"""Padded-context parity and optional native DeepSeek MMuse query graphs.

Synthetic weights cover decoder, convolution, query conditioning and real MMuse
heads. They do not validate quantized Target attention or its feature outputs.
"""

import ast
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch

MODELS = Path(__file__).resolve().parents[3] / "vllm/model_executor/models"


def helper(filename):
    spec = importlib.util.spec_from_file_location(
        filename, Path(__file__).with_name(filename + ".py")
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def pure_module(name):
    package = "deepseek_mmuse_graph_test"
    if package not in sys.modules:
        root = ModuleType(package)
        root.__path__ = [str(MODELS / "mmuse")]
        sys.modules[package] = root
    spec = importlib.util.spec_from_file_location(
        f"{package}.{name}", MODELS / "mmuse" / f"{name}.py"
    )
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def model(device, dtype, windows, monkeypatch):
    decoder = pure_module("deepseek_decoder")
    runtime = helper("test_mmuse_rollout_graph").runtime(device, False, monkeypatch)
    runtime.context_hidden_proj = torch.nn.Linear(16, 16, bias=False).to(device)
    runtime.context_hidden_gate = torch.nn.Parameter(torch.tensor(0.4, device=device))
    runtime.block_position_embedding = torch.nn.Embedding(3, 16).to(device)
    tree = ast.parse((MODELS / "deepseek_mmuse_reference.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekMMuseReference"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("forward", "forward_fixed_context")
    ]
    scope = {"torch": torch, "F": torch.nn.functional}
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), "fixed_decoder", "exec"),
        scope,
    )

    class Decoder(torch.nn.Module):
        forward = scope["forward"]
        forward_fixed_context = scope["forward_fixed_context"]

    obj = Decoder()
    obj.config = NS(mask_token_id=31)
    obj.runtime = runtime
    obj.register_buffer("feature_storage", torch.zeros(2, 16, 16))
    obj.register_buffer(
        "embedding_weight", runtime.embed_tokens.weight.detach().clone()
    )
    obj.hidden_norm, obj.norm = decoder.RMS(16, 1e-6), decoder.RMS(16, 1e-6)
    config = NS(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000.0},
    )
    obj.layers = torch.nn.ModuleList(
        [
            decoder.BlockLayer(
                config,
                {
                    "block_size": 3,
                    "dflash2_dynamic_conv": True,
                    "dflash2_conv_group_size": 4,
                },
                window,
            )
            for window in windows
        ]
    )
    obj.context_kv_cache = None
    return obj.eval().to(device=device, dtype=dtype)


@pytest.mark.parametrize("windows", [(None, None), (4, 4), (4, None)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_fixed_context_masks_padding_and_matches_reference_after_overwrite(
    windows, dtype, monkeypatch
):
    torch.manual_seed(43)
    owner = model("cpu", dtype, windows, monkeypatch)
    driver = pure_module("deepseek_graph").DeepseekMMuseGraphs(
        owner,
        3,
        None,
        None,
        torch.zeros(2),
        torch.zeros(2, dtype=torch.int64),
        False,
        None,
    )
    features = torch.randn(16, 16, dtype=dtype)
    output_pointer = driver.output.data_ptr()
    for start, end in [(0, 3), (0, 9), (2, 12), (0, 2), (0, 7)]:
        features[end - 2 : end] += 0.2
        context_positions = torch.arange(start, end)
        anchor = torch.tensor([5 + start])
        size = driver.prepare(context_positions, features[start:end], anchor, slot=0)
        expected = owner(
            driver.input_ids, driver.positions, context_positions, features[start:end]
        )
        fixed = owner.forward_fixed_context(
            driver.input_ids,
            driver.positions,
            driver.context_positions[:size],
            driver.context[:size],
            tuple(kv[:, :size] for kv in driver.context_kv),
            driver.anchor_index,
        )
        tol = 0.03 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(fixed, expected, atol=tol, rtol=tol)
        tokens, _ = owner.runtime.rollout_correction(fixed.unsqueeze(0), anchor)
        driver.forward(size)
        torch.testing.assert_close(driver.output, tokens[0])
        assert (driver.context_positions[end - start :] == -1).all()
        assert driver.output.data_ptr() == output_pointer


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("probabilistic", [False, True])
@torch.inference_mode()
def test_native_query_graph_reads_new_context_and_request_sampling_state(
    device, dtype, probabilistic, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("NPU required")
    torch.manual_seed(9)
    owner = model(device, dtype, (4, None), monkeypatch)
    if device == "cpu":
        sample = helper("test_mmuse_probabilistic_graph").cpu_sample
    elif device == "cuda":
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample as sample
    else:
        from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample as sample
    cache = (
        torch.full((2, 3, 32), -77, dtype=dtype, device=device)
        if probabilistic
        else None
    )
    temps = torch.tensor([0.0, 0.7], device=device)
    seeds = torch.tensor([4, 8], dtype=torch.int64, device=device)
    driver = pure_module("deepseek_graph").DeepseekMMuseGraphs(
        owner, 3, sample, cache, temps, seeds, False, getattr(torch, device, None)
    )
    if device != "cpu":
        driver.capture()
        if cache is not None:
            assert (cache == -77).all()
    for turn, (slot, length) in enumerate([(0, 3), (1, 12), (0, 2), (1, 8)]):
        context_positions = torch.arange(length, device=device) + turn
        features = torch.randn(length, 16, dtype=dtype, device=device)
        anchor = torch.tensor([turn + 5], device=device)
        seeds.add_(13)
        size = driver.prepare(context_positions, features, anchor, slot)
        # Compare native replay to eager execution with the same padded shape.
        # Unpadded reference parity is checked separately, within dtype tolerance.
        hidden = owner.forward_fixed_context(
            driver.input_ids,
            driver.positions,
            driver.context_positions[:size],
            driver.context[:size],
            tuple(kv[:, :size] for kv in driver.context_kv),
            driver.anchor_index,
        )
        before = cache.clone() if cache is not None else None
        expected, _ = owner.runtime.rollout_correction(
            hidden.unsqueeze(0),
            anchor,
            sample_token=driver.sample_token if probabilistic else None,
        )
        expected_cache = cache.clone() if cache is not None else None
        if cache is not None:
            cache.copy_(before)
        if device == "cpu":
            driver.forward(size)
        else:
            driver.replay(size)
            getattr(torch, device).synchronize()
        torch.testing.assert_close(driver.output, expected[0])
        if cache is not None:
            tol = 0.03 if dtype == torch.bfloat16 else 2e-5
            torch.testing.assert_close(
                cache[slot], expected_cache[slot], atol=tol, rtol=tol
            )
            torch.testing.assert_close(
                cache[1 - slot], before[1 - slot], atol=0, rtol=0
            )


def test_context_bucket_boundaries():
    cls = pure_module("deepseek_graph").DeepseekMMuseGraphs
    obj = cls.__new__(cls)
    obj.context = torch.empty(300, 1)
    obj.sizes = (128, 256, 300)
    assert [obj.bucket(n) for n in (1, 128, 129, 257, 300)] == [128, 128, 256, 300, 300]
    for length in (0, 301):
        with pytest.raises(ValueError, match="capacity"):
            obj.bucket(length)


def test_full_draft_config_is_isolated_and_legacy_kv_reuse_is_rejected():
    path = MODELS.parents[1] / "v1/worker/gpu/spec_decode/mmuse/deepseek_full.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekMMuseFullSpeculator"
    )

    class Base:
        def __init__(self, config, device):
            self.vllm_config = config

    scope = {
        "MMuseReferenceSpeculator": Base,
        "copy": copy,
        "torch": torch,
        "CompilationMode": NS(NONE=0),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    config = NS(compilation_config=NS(mode=3), additional_config={})
    obj = scope[cls.name](config, "cpu")
    assert obj.vllm_config.compilation_config.mode == 0
    assert config.compilation_config.mode == 3
    config.additional_config["mmuse_draft_kv_reuse"] = True
    with pytest.raises(ValueError, match="disable mmuse_draft_kv_reuse"):
        scope[cls.name](config, "cpu")


def test_graph_ownership_is_detached_from_persistent_inputs_after_capture():
    path = MODELS.parents[1] / "v1/worker/gpu/spec_decode/mmuse/deepseek_full.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekMMuseGraphManager"
    )

    class Base:
        def __init__(self, *args):
            pass

    scope = {"CudaGraphManager": Base, "CUDAGraphMode": NS(NONE=0)}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    calls = []
    graph = NS(replay=lambda: calls.append("replay"))
    inputs = NS(block_size=3, graphs={}, pool="private-pool")

    def capture():
        assert not inputs.graphs
        inputs.graphs = {16: graph}

    inputs.capture = capture
    manager = scope[cls.name](None, "cpu", inputs)
    manager.capture()
    assert inputs.graphs == {}
    manager.replay(16)
    assert calls == ["replay"]
    # Profiling cleanup can drop manager without input buffers retaining graphs.
    del manager
    replacement = scope[cls.name](None, "cpu", inputs)
    replacement.capture()
    replacement.replay(16)
    assert calls == ["replay", "replay"]


@torch.inference_mode()
def test_proposal_replays_each_request_and_owns_outputs_before_next_replay(monkeypatch):
    torch.manual_seed(41)
    owner = model("cpu", torch.float32, (4, None), monkeypatch)
    owner.history = pure_module("context").VerifiedPrefix(16, owner.feature_storage)
    driver = pure_module("deepseek_graph").DeepseekMMuseGraphs(
        owner,
        3,
        None,
        None,
        torch.zeros(2),
        torch.zeros(2, dtype=torch.int64),
        False,
        None,
    )
    path = MODELS.parents[1] / "v1/worker/gpu/spec_decode/mmuse/deepseek_full.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekMMuseFullSpeculator"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "propose"
    )
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    obj = NS(
        full_enabled=True,
        model=owner,
        fixed_graphs=driver,
        query_cudagraph_manager=NS(replay=driver.forward),
        temperature=driver.temperature,
        seeds=driver.seeds,
        draft_tokens=torch.empty(2, 3, dtype=torch.int64),
        _project_target_features=lambda batch, aux, dummy: aux[0],
    )
    batch = NS(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 3, 5]),
        idx_mapping_np=torch.tensor([1, 0]).numpy(),
        req_ids=["b", "a"],
        positions=torch.tensor([0, 1, 2, 0, 1]),
    )
    features = torch.randn(5, 16)
    anchors = torch.tensor([[5], [6]])
    result = scope["propose"](
        obj,
        batch,
        {},
        {},
        None,
        [features],
        torch.ones(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        anchors,
        torch.tensor([[4, 8]]),
        torch.zeros(2),
        torch.tensor([1, 2]),
    )
    for row, slot in enumerate([1, 0]):
        _, positions, values = owner.history.requests[slot]
        ids = torch.tensor([anchors[slot].item(), 31, 31])
        query_pos = positions[-1] + 1 + torch.arange(3)
        hidden = owner(ids, query_pos, positions, values)
        expected, _ = owner.runtime.rollout_correction(hidden.unsqueeze(0), ids[:1])
        torch.testing.assert_close(result[row], expected[0])
    snapshot = result.clone()
    driver.output.fill_(31)
    torch.testing.assert_close(result, snapshot)

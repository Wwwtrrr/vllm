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
    for layer in obj.layers:
        for conv in (layer.attention_conv, layer.mlp_conv):
            torch.nn.init.normal_(conv.base_kernel, std=0.1)
    return obj.eval().to(device=device, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_incremental_context_projects_only_changed_rows(dtype, monkeypatch):
    torch.manual_seed(17)
    owner = model("cpu", dtype, (4, None), monkeypatch)
    graph = pure_module("deepseek_graph").DeepseekMMuseGraphs
    args = (
        owner,
        3,
        None,
        None,
        torch.zeros(2),
        torch.zeros(2, dtype=torch.long),
        False,
        None,
    )
    incremental, reference = graph(*args), graph(*args)
    features = torch.randn(30, 16, dtype=dtype)
    rows = []
    hook = owner.layers[0].self_attn.k_proj.register_forward_pre_hook(
        lambda _module, args: rows.append(args[0].shape[0])
    )
    # Append, eviction, corrected suffix, rollback, slot switch and slot reuse.
    rounds = [
        (0, "a", 0, 5, 0, 5),
        (0, "a", 0, 9, 5, 4),
        (0, "a", 2, 18, 9, 9),
        (0, "a", 2, 18, 14, 4),
        (0, "a", 2, 8, 8, 0),
        (1, "b", 0, 4, 0, 4),
        (0, "new", 0, 5, 0, 5),
    ]
    try:
        for slot, request, start, end, update, expected_rows in rounds:
            features[update:end] += 0.1
            positions = torch.arange(start, end)
            values = features[start:end]
            rows.clear()
            size = incremental.prepare(
                positions,
                values,
                torch.tensor([7]),
                slot,
                request_id=request,
                start=start,
                end=end,
                update_start=update,
            )
            assert rows == ([expected_rows] if expected_rows else [])
            reference.prepare(positions, values, torch.tensor([7]), slot)
            tolerance = 0.03 if dtype == torch.bfloat16 else 2e-6
            for actual, expected in zip(incremental.context_kv, reference.context_kv):
                torch.testing.assert_close(
                    actual[:, :size], expected[:, :size], atol=tolerance, rtol=tolerance
                )
            incremental.forward(size)
            reference.forward(size)
            torch.testing.assert_close(incremental.output, reference.output)
    finally:
        hook.remove()


@torch.inference_mode()
def test_platform_attention_matches_decoder_logits_after_history_changes(monkeypatch):
    torch.manual_seed(23)
    owner = model("cpu", torch.bfloat16, (4, None), monkeypatch)
    graph = pure_module("deepseek_graph").DeepseekMMuseGraphs
    driver = graph(
        owner,
        3,
        None,
        None,
        torch.zeros(2),
        torch.zeros(2, dtype=torch.long),
        False,
        None,
    )
    observed = []

    def platform_attention(q, k, v, visible):
        # Boundary oracle: the platform receives unexpanded KV heads and the
        # exact mask used by the reference, including per-layer window rules.
        assert k.shape[1] == v.shape[1] == 2
        observed.append(visible.clone())
        groups = q.shape[1] // k.shape[1]
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1).repeat_interleave(groups, 0)[None],
            v.transpose(0, 1).repeat_interleave(groups, 0)[None],
            attn_mask=visible[None, None],
        )[0].transpose(0, 1)

    features = torch.randn(16, 16, dtype=torch.bfloat16)
    for start, end in [(0, 3), (0, 12), (2, 9), (0, 2)]:
        features[start:end] += 0.1
        size = driver.prepare(
            torch.arange(start, end), features[start:end], torch.tensor([7]), 0
        )
        args = (
            driver.input_ids,
            driver.positions,
            driver.context_positions[:size],
            driver.context[:size],
            tuple(kv[:, :size] for kv in driver.context_kv),
            driver.anchor_index,
        )
        for layer in owner.layers:
            layer.self_attn.attention_impl = None
        expected = owner.forward_fixed_context(*args)
        expected_tokens, expected_logits = owner.runtime.rollout_correction(
            expected[None], driver.input_ids[:1]
        )
        for layer in owner.layers:
            layer.self_attn.attention_impl = platform_attention
        actual = owner.forward_fixed_context(*args)
        actual_tokens, actual_logits = owner.runtime.rollout_correction(
            actual[None], driver.input_ids[:1]
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_logits, expected_logits)
        torch.testing.assert_close(actual_tokens, expected_tokens)
    assert len(observed) == 8


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

        def propose(self, *args):
            return "eager-profile"

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
    for full, dummy, profile in [
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ]:
        obj.full_enabled = full
        obj.fixed_graphs = NS(context_entry=(0, "a", 0, 16))
        assert (
            obj.propose(*([None] * 11), dummy_run=dummy, is_profile=profile)
            == "eager-profile"
        )
        assert obj.fixed_graphs.context_entry is None
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
    feature_calls = []
    features = NS(graphs={}, pool="feature-pool")

    def capture_features():
        assert not features.graphs
        features.graphs = {8: graph}

    features.capture = capture_features
    features.project = lambda states, graphs: (
        feature_calls.append((states, graphs)) or "projected"
    )
    inputs.feature_inputs = features
    manager = scope[cls.name](None, "cpu", inputs)
    manager.capture()
    assert inputs.graphs == {} and features.graphs == {}
    assert manager.feature_pool == "feature-pool"
    assert manager.project_features("aux") == "projected"
    assert feature_calls == [("aux", {8: graph})]
    manager.replay(16)
    assert calls == ["replay"]
    # Profiling cleanup can drop manager without input buffers retaining graphs.
    del manager
    replacement = scope[cls.name](None, "cpu", inputs)
    replacement.capture()
    assert features.graphs == {} and replacement.feature_graphs == {8: graph}
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
        _project_target_features=lambda batch, aux, dummy, **kwargs: aux[0],
        _project_captured_features=None,
        prepare_request_attention=lambda *args: None,
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_framework_decoder_preserves_loaded_weights_padding_and_block_outputs(
    dtype, bias, monkeypatch
):
    from contextlib import contextmanager

    torch.manual_seed(71)
    owner = model("cpu", dtype, (4, None), monkeypatch)
    owner.fc = owner.runtime.fc
    assert torch.is_inference(owner.fc.weight)
    if bias:
        for layer in owner.layers:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                linear = getattr(layer.self_attn, name)
                linear.bias = torch.nn.Parameter(
                    torch.randn(linear.out_features, dtype=dtype)
                )
    baseline = copy.deepcopy(owner)
    original = {name: value.clone() for name, value in owner.state_dict().items()}
    reference = pure_module("deepseek_decoder")
    calls = []
    config_module = ModuleType("vllm.config")

    @contextmanager
    def context(config):
        assert config.quant_config is None
        yield

    config_module.set_current_vllm_config = context
    norms = ModuleType("vllm.model_executor.layers.layernorm")

    class Norm(reference.RMS):
        def __init__(self, width, eps, dtype):
            super().__init__(width, eps)
            self.to(dtype=dtype)

    norms.RMSNorm = Norm
    linears = ModuleType("vllm.model_executor.layers.linear")

    class Linear(torch.nn.Linear):
        def __init__(
            self,
            inputs,
            outputs,
            bias,
            params_dtype,
            quant_config,
            prefix,
            return_bias,
            disable_tp,
        ):
            assert quant_config is None and disable_tp and not return_bias
            super().__init__(inputs, outputs, bias=bias, dtype=params_dtype)
            calls.append(prefix)

            def process(layer):
                assert not torch.is_inference_mode_enabled()
                assert not torch.is_inference(layer.weight)
                assert layer.weight._version >= 0
                # Simulate a platform layout conversion during postprocessing.
                layer.weight.data = layer.weight.detach().clone()

            self.quant_method = NS(process_weights_after_loading=process)

    linears.ReplicatedLinear = Linear
    acts = ModuleType("vllm.model_executor.layers.activation")

    class Activation(torch.nn.Module):
        def forward(self, x):
            gate, up = x.chunk(2, -1)
            return torch.nn.functional.silu(gate) * up

    acts.SiluAndMul = Activation
    rotary = ModuleType("vllm.model_executor.layers.rotary_embedding.common")

    class Apply(torch.nn.Module):
        def __init__(self, is_neox_style):
            super().__init__()
            assert is_neox_style

        def forward(self, x, cos, sin):
            left, right = x.chunk(2, -1)
            return (
                x * torch.cat((cos, cos), -1)[:, None]
                + torch.cat((-right, left), -1) * torch.cat((sin, sin), -1)[:, None]
            )

    rotary.ApplyRotaryEmb = Apply
    for module in (config_module, norms, linears, acts, rotary):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    framework = pure_module("framework_decoder")
    config = NS(
        quant_config=object(),
        model_config=NS(max_model_len=64),
        speculative_config=NS(num_speculative_tokens=3),
    )
    framework.bind_framework_decoder(owner, config)
    assert torch.is_inference_mode_enabled()
    framework.bind_framework_decoder(owner, config)  # No duplicate conversion.
    assert len(calls) == 1 + 7 * len(owner.layers)
    assert config.quant_config is not None
    assert owner.fc is owner.runtime.fc
    for layer in (
        owner.fc,
        owner.hidden_norm,
        owner.norm,
        *(
            item
            for block in owner.layers
            for item in (
                block.input_layernorm,
                block.post_attention_layernorm,
                block.self_attn.q_norm,
                block.self_attn.k_norm,
                block.self_attn.q_proj,
                block.self_attn.k_proj,
                block.self_attn.v_proj,
                block.self_attn.o_proj,
                block.mlp.gate_proj,
                block.mlp.up_proj,
                block.mlp.down_proj,
            )
        ),
    ):
        assert not torch.is_inference(layer.weight)
        assert layer.weight._version >= 0
        if getattr(layer, "bias", None) is not None:
            assert not torch.is_inference(layer.bias)
            assert layer.bias._version >= 0
    assert owner.state_dict().keys() == original.keys()
    for name, value in owner.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    positions = torch.tensor([-1, 0, 7, 66])
    k = torch.randn(4, 2, 4, dtype=dtype)
    actual = owner.layers[0].self_attn.apply_rotary(k, positions)
    expected = reference.rotate(k, positions, 10000.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    context = torch.randn(9, 16, dtype=dtype)
    context_positions = torch.arange(9)
    context_positions[-2:] = -1
    query_positions = torch.arange(7, 10)
    query = torch.randn(3, 16, dtype=dtype)
    for ref_layer, layer in zip(baseline.layers, owner.layers):
        expected = ref_layer(query, context, query_positions, context_positions)
        actual = layer(query, context, query_positions, context_positions)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # CPU dependency oracles check binding and tensor contracts. Native fused
    # kernel numerical behavior still requires the device tests/server run.


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rotation,fusion", [(False, False), (True, True)])
@torch.inference_mode()
def test_feature_graph_projection_matches_eager_across_row_buckets(
    device, dtype, rotation, fusion, monkeypatch
):
    from contextlib import nullcontext
    from types import MethodType

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("NPU required")
    torch.manual_seed(931)
    runtime = helper("test_mmuse_rollout_graph").runtime(device, False, monkeypatch)
    if fusion:
        runtime.config.dflash_gated_layer_fusion = True
        runtime._init_layer_fusion(runtime.config, 16, 2)
        runtime.layer_fusion_gate.data.fill_(0.7)
    runtime.to(device=device, dtype=dtype)
    owner = torch.nn.Module()
    owner.config = NS(hidden_size=16)
    owner.runtime = runtime
    owner.register_buffer(
        "feature_storage", torch.zeros(2, 16, 16, dtype=dtype, device=device)
    )
    matrix = torch.linalg.qr(torch.randn(16, 16))[0].to(device) if rotation else None
    owner.register_buffer("target_rotation", matrix)
    tree = ast.parse((MODELS / "deepseek_mmuse_reference.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekMMuseReference"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "project_features"
    )
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "features", "exec"), scope)
    owner.project_features = MethodType(scope["project_features"], owner)

    class Stream:
        def wait_stream(self, other):
            pass

    class Backend:
        current_stream = staticmethod(Stream)
        stream = staticmethod(lambda stream: nullcontext())
        synchronize = staticmethod(lambda: None)
        graph_pool_handle = staticmethod(lambda: object())
        CUDAGraph = staticmethod(lambda: NS())
        graph = staticmethod(lambda *args, **kwargs: nullcontext())

    Backend.Stream = Stream
    backend = Backend if device == "cpu" else getattr(torch, device)
    driver = pure_module("deepseek_features").DeepseekMMuseFeatureGraphs(
        owner, 2, 16, 8, backend
    )
    driver.capture()
    graphs, driver.graphs = driver.graphs, {}
    if device == "cpu":
        graphs = {
            size: NS(replay=lambda size=size: driver.forward(size)) for size in graphs
        }
    input_address, output_address = driver.inputs.data_ptr(), driver.output.data_ptr()
    for rows in (7, 1, 8, 3, 6):
        states = [
            torch.randn(rows * 2, 16, dtype=dtype, device=device)[::2] for _ in range(2)
        ]
        expected = owner.project_features(torch.cat(states, -1))
        driver.inputs.fill_(float("nan"))
        actual = driver.project(states, graphs)
        if device != "cpu":
            backend.synchronize()
        assert actual.shape == (rows, 16)
        torch.testing.assert_close(
            actual,
            expected,
            atol=0.02 if dtype == torch.bfloat16 else 1e-5,
            rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
        )
        assert driver.inputs.data_ptr() == input_address
        assert driver.output.data_ptr() == output_address
    # Prefill/unsupported input shapes and absent captures retain eager dispatch.
    assert (
        driver.project([torch.empty(9, 16, dtype=dtype, device=device)] * 2, graphs)
        is None
    )
    assert driver.project(states, {}) is None
    assert driver.project(states[:1], graphs) is None
    assert (
        driver.project(
            [torch.zeros(1, 16, dtype=torch.int32, device=device)] * 2, graphs
        )
        is None
    )
    assert driver.graphs == {}  # Persistent buffers do not retain graph handles.


@torch.inference_mode()
def test_feature_projection_dispatch_uses_eager_when_uncaptured():
    path = MODELS.parents[1] / "v1/worker/gpu/spec_decode/mmuse/deepseek_full.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekMMuseFullSpeculator"
    )

    class Base:
        def _project_feature_tensors(self, states):
            return "eager"

    scope = dict(MMuseReferenceSpeculator=Base, copy=copy, torch=torch)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    owner = scope[cls.name].__new__(scope[cls.name])
    owner.full_enabled = True
    owner.query_cudagraph_manager = None
    assert owner._project_captured_features([]) == "eager"
    owner.query_cudagraph_manager = NS(project_features=lambda states: None)
    assert owner._project_captured_features([]) == "eager"
    owner.query_cudagraph_manager = NS(project_features=lambda states: "graph")
    assert owner._project_captured_features([]) == "graph"
    owner.full_enabled = False
    assert owner._project_captured_features([]) == "eager"

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts; run with --confcutdir=tests/standalone without CUDA imports.

Set LEAGLE_TRAINING_ROOT to the looped_eagle checkout for real training parity.
"""

import ast
import copy
import importlib.util
import inspect
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import Qwen3Config

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_module = load_file("leagle_model", "vllm/model_executor/models/qwen3_leagle.py")
context_module = load_file(
    "leagle_context", "vllm/v1/worker/gpu/spec_decode/leagle/context.py"
)
algos = load_file(
    "leagle_algos", "vllm/transformers_utils/configs/speculators/algos.py"
)
graph_base = ModuleType("vllm.v1.worker.gpu.cudagraph_utils")
graph_base.CudaGraphManager = type("CudaGraphManager", (), {})
with patch.dict(sys.modules, {graph_base.__name__: graph_base}):
    graph_module = load_file(
        "leagle_graph", "vllm/v1/worker/gpu/spec_decode/leagle/graph.py"
    )
with patch.dict(
    sys.modules,
    {
        graph_base.__name__: graph_base,
        "vllm.v1.worker.gpu.spec_decode.leagle.graph": graph_module,
    },
):
    execution_module = load_file(
        "leagle_execution", "vllm/v1/worker/gpu/spec_decode/leagle/execution.py"
    )


@pytest.fixture(autouse=True)
def seed_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def architecture(layers=2, loops=2, reduced=False, norm=True, window=None):
    decoder = Qwen3Config(
        vocab_size=24,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        attention_dropout=0.0,
        sliding_window=window,
        layer_types=["full_attention" if window is None else "sliding_attention"]
        * layers,
    )
    return dict(
        transformer_layer_config=decoder.to_dict(),
        aux_hidden_state_layer_ids=[1, 0],
        num_loops=loops,
        norm_output=norm,
        draft_vocab_size=12 if reduced else 24,
    )


def make_model(config):
    draft = SimpleNamespace(hf_config=SimpleNamespace(leagle_config=config))
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=draft)
    )
    return model_module.LEagleDraftModel(vllm_config=config).eval()


def test_config_preserves_training_geometry_and_feature_order():
    config = architecture(3, 4, reduced=True, norm=False, window=8)
    converted = copy.deepcopy(config["transformer_layer_config"])
    algos.update_leagle(config, converted)
    assert converted["leagle_config"] == config
    assert converted["eagle_aux_hidden_state_layer_ids"] == [0, 1]
    assert converted["architectures"] == ["LEagleDraftModel"]


@pytest.mark.parametrize("norm", [True, False])
def test_public_forward_recurs_with_immutable_logical_depth_caches(norm):
    model = make_model(architecture(2, 3, norm=norm))
    context = torch.randn(1, 5, 16)
    anchor = torch.tensor([[3]])
    positions = torch.arange(5)[None]
    rope = model.rotary_emb(context, positions)
    context_kvs = [
        layer.self_attn.project_context(context, rope) for layer in model.layers
    ]
    first = model(
        anchor,
        torch.tensor([[5]]),
        hidden_states=context[:, -1:],
        context_kvs=context_kvs,
        attention_masks={None: None},
    )
    token = model.compute_logits(first.hidden_states).argmax(-1)
    snapshot = [(key.clone(), value.clone()) for key, value in first.past_draft_kv]
    second = model(
        token,
        torch.tensor([[6]]),
        hidden_states=first.recurrent_state,
        context_kvs=context_kvs,
        attention_masks={None: None},
        past_draft_kv=first.past_draft_kv,
    )
    assert len(second.past_draft_kv) == 6
    for before, original, updated in zip(
        snapshot, first.past_draft_kv, second.past_draft_kv
    ):
        for saved, previous, current in zip(before, original, updated):
            torch.testing.assert_close(saved, previous)
            assert previous.shape[-2] == 1 and current.shape[-2] == 2
    expected = torch.cat(
        (token, model.compute_logits(second.hidden_states).argmax(-1)), 1
    )
    calls = []
    handle = model.register_forward_hook(lambda *_args: calls.append(True))
    actual = model.propose_context(context, anchor, 2)
    torch.testing.assert_close(actual, expected)
    assert len(calls) == 2
    calls.clear()
    actual = model.propose_static(context, torch.tensor([5]), anchor, 2)
    handle.remove()
    torch.testing.assert_close(actual, expected)
    assert len(calls) == 2


def test_saved_speculators_config_selects_leagle_without_mutating_checkpoint(
    monkeypatch,
):
    name = "vllm.transformers_utils.configs.speculators.algos"
    monkeypatch.setitem(sys.modules, name, algos)
    utils = ModuleType("vllm.transformers_utils.utils")
    utils.without_trust_remote_code = lambda kwargs: kwargs
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    module = load_file(
        "leagle_speculators_config",
        "vllm/transformers_utils/configs/speculators/base.py",
    )
    source = architecture()
    source.update(
        speculators_model_type="leagle",
        speculators_config={
            "verifier": {"name_or_path": "target"},
            "proposal_methods": [{"speculative_tokens": 7}],
        },
    )
    before = copy.deepcopy(source)
    flattened = module.SpeculatorsConfig.extract_transformers_pre_trained_config(source)
    assert source == before
    speculative = module.SpeculatorsConfig.extract_vllm_speculative_config(source)
    assert speculative == {"method": "leagle", "num_speculative_tokens": 7}
    hf = module.SpeculatorsConfig(**flattened)
    assert hf.leagle_config["num_loops"] == 2
    assert hf.eagle_aux_hidden_state_layer_ids == [0, 1]


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_loops", 0),
        ("num_loops", True),
        ("draft_vocab_size", 0),
        ("aux_hidden_state_layer_ids", []),
        ("aux_hidden_state_layer_ids", [1, 1]),
        ("aux_hidden_state_layer_ids", [-1]),
    ],
)
def test_invalid_training_geometry_is_rejected(key, value):
    config = architecture()
    config[key] = value
    with pytest.raises(ValueError):
        algos.update_leagle(config, copy.deepcopy(config["transformer_layer_config"]))


@pytest.mark.parametrize("reduced", [False, True])
def test_checkpoint_loader_requires_all_inference_weights_and_valid_mapping(reduced):
    config = architecture(reduced=reduced)
    original, restored = make_model(config), make_model(config)
    if reduced:
        original.d2t.copy_(torch.arange(12))  # absolute ids = 0, 2, ..., 22
    weights = original.state_dict()
    restored.load_weights(weights.items())
    for name, value in weights.items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    missing = dict(weights)
    del missing["input_fc.weight"]
    with pytest.raises(ValueError, match="Missing"):
        restored.load_weights(missing.items())
    if reduced:
        weights["d2t"] = torch.full((12,), 24)
        with pytest.raises(ValueError, match="mapping"):
            restored.load_weights(weights.items())


def test_context_rollback_chunked_prefill_and_slot_reuse():
    cache = context_module.LEagleContext(2, 16, 4, dtype=torch.float32, device="cpu")
    cache.update(0, "first", 0, torch.ones(3, 4))
    cache.update(0, "first", 3, torch.ones(2, 4) * 2)
    value = cache.update(0, "first", 3, torch.ones(1, 4) * 7)
    assert value.shape == (1, 4, 4)
    assert torch.all(value[:, -1] == 7)
    with pytest.raises(RuntimeError, match="missing target context"):
        cache.update(1, "gap", 2, torch.ones(1, 4))
    value = cache.update(0, "replacement", 0, torch.ones(1, 4) * 9)
    assert value.shape == (1, 1, 4)


@pytest.fixture
def runtime(monkeypatch):
    # Isolate scheduler behavior from CUDA/Triton model-runner initialization.
    dependencies = {
        "vllm.compilation.backends": {"set_model_tag": lambda _: nullcontext()},
        "vllm.model_executor.model_loader": {"get_model": lambda **_: None},
        "vllm.v1.worker.gpu.spec_decode.leagle.context": {
            "LEagleContext": context_module.LEagleContext,
        },
        "vllm.v1.worker.gpu.spec_decode.leagle.graph": vars(graph_module),
        "vllm.v1.worker.gpu.spec_decode.leagle.execution": vars(execution_module),
        "vllm.v1.worker.gpu.spec_decode.speculator": {"DraftModelSpeculator": object},
    }
    for name, attributes in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return load_file(
        "leagle_runtime", "vllm/v1/worker/gpu/spec_decode/leagle/speculator.py"
    )


def test_proposer_accepts_actual_runner_profile_and_decode_calls(runtime):
    source = ROOT / "vllm/v1/worker/gpu/model_runner.py"
    calls = []
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "self.speculator.propose"
        ):
            calls.append(node)
    assert len(calls) >= 2  # Profiling and real decoding must both bind.
    signature = inspect.signature(runtime.LEagleSpeculator.propose)
    for call in calls:
        signature.bind(
            object(),
            *[object() for _ in call.args],
            **{kw.arg: object() for kw in call.keywords},
        )


@pytest.mark.parametrize("full_draft", [False, True])
def test_proposer_uses_verified_prefix_and_bonus_anchor(runtime, full_draft):
    proposer = runtime.LEagleSpeculator.__new__(runtime.LEagleSpeculator)
    proposer.model = make_model(architecture())
    proposer.device = torch.device("cpu")
    proposer.num_speculative_steps = 3
    proposer.max_model_len = 32
    proposer.draft_tokens = torch.empty((2, 3), dtype=torch.long)
    proposer.context = context_module.LEagleContext(
        2, 32, 16, dtype=torch.float32, device="cpu"
    )
    proposer.full_draft = full_draft
    if full_draft:
        inputs = graph_module.LEagleGraphInputs.allocate(
            proposer.model, 32, 16, dtype=torch.float32, device="cpu"
        )
        proposer.graph_runner = graph_module.LEagleGraphRunner(
            proposer.model, inputs, 3, CPUReplayBackend()
        )
        proposer.graph_runner.capture()
    anchor = torch.tensor([[3], [7]])
    history = []

    def call(start, length, rejected, sampled=1):
        aux = [torch.randn(length, 16) for _ in range(2)]
        batch = SimpleNamespace(
            num_reqs=1,
            req_ids=["request"],
            idx_mapping_np=np.array([1]),
            positions=torch.arange(start, start + length),
            query_start_loc_np=np.array([0, length]),
        )
        output = proposer.propose(
            batch,
            {},
            {},
            None,
            aux,
            torch.tensor([sampled]),
            torch.tensor([rejected]),
            anchor,
            None,
            None,
            None,
        ).clone()
        history.append(proposer.model.project_context(torch.cat(aux[::-1], -1)))
        return output

    assert (call(0, 3, 0, sampled=0) == -1).all()  # Partial prefill.
    output = call(3, 2, 0)
    context = torch.cat(history, dim=0)[None]
    expected = proposer.model.propose_context(context, anchor[1:], 3)
    torch.testing.assert_close(output, expected)
    output = call(5, 4, 2)  # Two rejected target feature positions are discarded.
    context = torch.cat((context, history[-1][:2][None]), dim=1)
    assert proposer.context.lengths[1] == 7
    torch.testing.assert_close(
        output, proposer.model.propose_context(context, anchor[1:], 3)
    )
    if full_draft:
        assert proposer.graph_runner.proposal_replays == 2
        assert proposer.graph_runner.projection_replays == 3


def runtime_config():
    trained = architecture()
    hf = SimpleNamespace(leagle_config=trained)
    target = Qwen3Config(**trained["transformer_layer_config"])
    target.num_hidden_layers = 4
    return SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(enforce_eager=True, hf_config=target),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        speculative_config=SimpleNamespace(
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=1),
            draft_sample_method="greedy",
            parallel_drafting=False,
            enable_adaptive_verification=False,
            use_local_argmax_reduction=False,
            disable_padded_drafter_batch=False,
            draft_model_config=SimpleNamespace(hf_config=hf, quantization=None),
        ),
    )


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("use_v2_model_runner", False, "Runner V2"),
        ("model_config.enforce_eager", False, "FULL"),
        ("cache_config.enable_prefix_caching", True, "prefix-caching"),
        ("parallel_config.tensor_parallel_size", 2, "TP=PP"),
        (
            "speculative_config.draft_parallel_config.tensor_parallel_size",
            2,
            "draft TP",
        ),
        ("speculative_config.draft_sample_method", "probabilistic", "greedy"),
        ("speculative_config.parallel_drafting", True, "parallel drafting"),
        ("speculative_config.draft_model_config.quantization", "fp8", "quantization"),
        ("model_config.hf_config.hidden_size", 32, "hidden sizes"),
        ("model_config.hf_config.vocab_size", 32, "vocabularies"),
        ("kv_transfer_config", object(), "remote KV"),
        ("lora_config", object(), "LoRA"),
    ],
)
def test_unsupported_runtime_fails_explicitly(runtime, path, value, message):
    config = runtime_config()
    runtime.validate_leagle_runtime(config)
    parent = config
    *parts, name = path.split(".")
    for part in parts:
        parent = getattr(parent, part)
    setattr(parent, name, value)
    with pytest.raises(ValueError, match=message):
        runtime.validate_leagle_runtime(config)


def test_full_target_and_draft_configuration_is_accepted(runtime):
    config = runtime_config()
    config.model_config.enforce_eager = False
    config.compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: True)
    )
    runtime.validate_leagle_runtime(config)
    config.compilation_config.cudagraph_mode.has_full_cudagraphs = lambda: False
    with pytest.raises(ValueError, match="FULL"):
        runtime.validate_leagle_runtime(config)


class CPUReplayBackend:
    """Exercise buffer/replay orchestration; this is not a hardware graph."""

    def __init__(self):
        self.captures = 0

    def capture(self, forward):
        self.captures += 1
        output = forward().clone()
        return SimpleNamespace(replay=lambda: output.copy_(forward())), output


class TreeCPUReplayBackend(CPUReplayBackend):
    def capture(self, forward):
        from torch.utils._pytree import tree_map

        self.captures += 1
        output = tree_map(lambda tensor: tensor.clone(), forward())
        return SimpleNamespace(
            replay=lambda: tree_map(
                lambda target, value: target.copy_(value), output, forward()
            )
        ), output


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_projections_preserve_logits_and_checkpoint_names(dtype):
    reference = make_model(architecture(2, 2, window=4)).to(dtype=dtype)
    optimized = copy.deepcopy(reference)
    keys = set(optimized.state_dict())
    optimized.configure_inference(fused_projections=True)
    assert set(optimized.state_dict()) == keys
    context = torch.randn(1, 7, 16, dtype=dtype)
    anchor = torch.tensor([[3]])
    expected_logits, actual_logits = [], []
    reference.lm_head.register_forward_hook(
        lambda _m, _a, value: expected_logits.append(value)
    )
    optimized.lm_head.register_forward_hook(
        lambda _m, _a, value: actual_logits.append(value)
    )
    expected = reference.propose_context(context, anchor, 3)
    actual = optimized.propose_context(context, anchor, 3)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.cat(actual_logits, 1), torch.cat(expected_logits, 1)
    )
    with torch.no_grad():
        reference.layers[0].self_attn.q_proj.weight.add_(0.1)
    optimized.load_weights(reference.state_dict().items())
    torch.testing.assert_close(
        optimized.layers[0].self_attn._qkv_weight[:16],
        reference.layers[0].self_attn.q_proj.weight,
    )


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize(
    "reuse,static,batch",
    [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("candidate", [False, True, "svd", "svd_lowrank"])
def test_optimized_serving_matches_reference_across_requests_and_rounds(
    runtime, full, reuse, static, batch, dtype, candidate
):
    config = architecture(2, 2, window=4)
    if candidate is True:
        config.update(
            candidate_head=True, candidate_head_rank=4, candidate_head_top_k=3
        )
    reference = make_model(config).to(dtype=dtype)
    if candidate in ("svd", "svd_lowrank"):
        reference.configure_output_head(
            svd_rank=4,
            top_k=3 if candidate == "svd" else None,
            rerank=candidate == "svd",
        )
    model = copy.deepcopy(reference)
    model.configure_inference(fused_projections=True)
    context = context_module.LEagleContext(3, 16, 16, dtype=dtype, device="cpu")
    options = execution_module.LEagleExecutionOptions(
        reuse_context_kv=reuse, static_draft_kv=static, batch_drafts=batch
    )
    execution = execution_module.LEagleExecution(
        model, context, 16, 3, options, dtype=dtype, device="cpu"
    )
    proposer = runtime.LEagleSpeculator.__new__(runtime.LEagleSpeculator)
    proposer.model, proposer.context, proposer.execution = model, context, execution
    proposer.full_draft = full
    proposer.num_speculative_steps = 3
    proposer.draft_tokens = torch.empty(3, 3, dtype=torch.long)
    proposer.graph_runner = None
    if full:
        proposer.graph_runner = execution_module.LEagleOptimizedGraphRunner(
            execution, TreeCPUReplayBackend()
        )
        proposer.graph_runner.capture()
    histories = {}
    projected_sizes = []
    prepare_context_kv = model.prepare_context_kv

    def record_projection(context_values, position_values):
        projected_sizes.append(context_values.shape[1])
        return prepare_context_kv(context_values, position_values)

    model.prepare_context_kv = record_projection
    anchors = torch.tensor([[5], [7], [3]])

    def run(rows):
        aux, positions, boundaries = [], [], [0]
        for slot, request_id, start, count, rejected, sampled in rows:
            features = torch.randn(count, 32, dtype=dtype)
            aux.append(features)
            positions.extend(range(start, start + count))
            boundaries.append(boundaries[-1] + count)
            kept = count - rejected
            if start == 0 or slot not in histories or histories[slot][0] != request_id:
                histories[slot] = (request_id, features[:kept].clone())
            else:
                histories[slot] = (
                    request_id,
                    torch.cat((histories[slot][1][:start], features[:kept])),
                )
        features = torch.cat(aux)
        # Capture order is sorted [0,1]; checkpoint feature order is [1,0].
        batch_input = SimpleNamespace(
            num_reqs=len(rows),
            req_ids=[row[1] for row in rows],
            idx_mapping_np=np.array([row[0] for row in rows]),
            positions=torch.tensor(positions),
            query_start_loc_np=np.array(boundaries),
        )
        output = proposer.propose(
            batch_input,
            {},
            {},
            None,
            [features[:, 16:], features[:, :16]],
            torch.tensor([row[5] for row in rows]),
            torch.tensor([row[4] for row in rows]),
            anchors,
            None,
            None,
            None,
        ).clone()
        for index, (slot, request_id, start, count, rejected, sampled) in enumerate(
            rows
        ):
            prefix = reference.project_context(histories[slot][1][None])
            if sampled:
                expected = reference.propose_context(
                    prefix, anchors[slot : slot + 1], 3
                )
                torch.testing.assert_close(output[index : index + 1], expected)
            else:
                assert (output[index] == -1).all()
            if reuse:
                position = torch.arange(prefix.shape[1])[None]
                kv = reference.prepare_context_kv(prefix, position)
                for cached, expected_pair in zip(execution.prefix_kv, kv):
                    for tensor, expected_value in zip(cached, expected_pair):
                        torch.testing.assert_close(
                            tensor[slot : slot + 1, :, : prefix.shape[1]],
                            expected_value,
                        )

    run([(2, "a", 0, 4, 0, 1), (0, "b", 0, 2, 0, 0), (1, "c", 0, 3, 0, 1)])
    run([(1, "c", 3, 4, 2, 1), (2, "a", 4, 4, 1, 1), (0, "b", 2, 2, 0, 1)])
    run([(1, "replacement", 0, 1, 0, 1)])
    # Rewind a suffix; neither accepted nor rejected synthetic draft KV is reused.
    run([(2, "a", 5, 2, 1, 1), (0, "b", 4, 3, 0, 1)])
    if full:
        assert proposer.graph_runner.proposal_replays == (4 if batch else 8)
        assert proposer.graph_runner.update_replays == 4
    if reuse:
        # Exactly one projection of the new batch per round, never a projection
        # of complete request prefixes from inside the autoregressive rollout.
        assert len(projected_sizes) == 4


@torch.inference_mode()
def test_cache_graph_padding_and_recapture_preserve_live_requests():
    model = make_model(architecture(2, 2, window=4))
    context = context_module.LEagleContext(3, 16, 16, dtype=torch.float32, device="cpu")
    execution = execution_module.LEagleExecution(
        model,
        context,
        16,
        3,
        execution_module.LEagleExecutionOptions(),
        dtype=torch.float32,
        device="cpu",
    )
    context.values.fill_(13)
    for pair in execution.prefix_kv:
        for tensor in pair:
            tensor.fill_(17)
    manager = execution_module.LEagleOptimizedGraphRunner(
        execution, TreeCPUReplayBackend()
    )
    manager.capture()
    assert (context.values == 13).all()
    assert all((tensor == 17).all() for pair in execution.prefix_kv for tensor in pair)
    features = torch.randn(16, 32)
    # The 9 valid source tokens occupy the end of the buffer; padding must not
    # read beyond that buffer or write to another live request.
    execution.project(features, torch.arange(16), manager)
    expected_features = execution.projected[7:16].clone()
    expected_kv = tuple(
        tuple(t[:, :, 7:16].clone() for t in pair) for pair in execution.new_kv
    )
    execution.update(2, "request", 0, 7, 16, manager)
    torch.testing.assert_close(context.values[2, :9], expected_features)
    assert (context.values[:2] == 13).all()
    assert (context.values[2, 9:] == 13).all()
    for cached, expected in zip(execution.prefix_kv, expected_kv):
        for storage, value in zip(cached, expected):
            torch.testing.assert_close(storage[2:3, :, :9], value)
            assert (storage[:2] == 17).all()
            assert (storage[2, :, 9:] == 17).all()
    snapshot = context.values.clone()
    kv_snapshot = tuple(tuple(t.clone() for t in pair) for pair in execution.prefix_kv)
    manager = execution_module.LEagleOptimizedGraphRunner(
        execution, TreeCPUReplayBackend()
    )
    manager.capture()
    torch.testing.assert_close(context.values, snapshot)
    for cached, expected in zip(execution.prefix_kv, kv_snapshot):
        for storage, value in zip(cached, expected):
            torch.testing.assert_close(storage, value)


@pytest.mark.parametrize("loops,window,reduced", [(1, None, False), (3, 4, True)])
def test_static_rollout_matches_eager_for_different_lengths_and_stale_padding(
    loops, window, reduced
):
    model = make_model(architecture(2, loops, reduced=reduced, window=window))
    if reduced:
        model.d2t.copy_(torch.arange(12))
    context = torch.randn(3, 16, 16)
    lengths = torch.tensor([9, 3, 1])
    anchor = torch.tensor([[2], [5], [7]])
    expected = torch.cat(
        [
            model.propose_context(
                context[row : row + 1, :length], anchor[row : row + 1], 7
            )
            for row, length in enumerate(lengths.tolist())
        ]
    )
    for row, length in enumerate(lengths.tolist()):
        context[row, length:] = torch.randn_like(context[row, length:]) * 1000
    # Graph code must not synchronize device values to the host.
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")):
        actual = model.propose_static(context, lengths, anchor, 7)
    torch.testing.assert_close(actual, expected)


def test_graph_replay_updates_lengths_features_and_anchors_without_recapture():
    model = make_model(architecture(2, 3, window=4))
    inputs = graph_module.LEagleGraphInputs.allocate(
        model, 257, 17, dtype=torch.float32, device="cpu"
    )
    backend = CPUReplayBackend()
    manager = graph_module.LEagleGraphRunner(model, inputs, 7, backend)
    context = torch.randn(1, 129, 16)
    anchor = torch.tensor([[3]])
    with pytest.raises(RuntimeError, match="not captured"):
        manager.propose(context, anchor, 7)
    manager.capture()
    captures = backend.captures
    pointers = (
        inputs.context.data_ptr(),
        inputs.features.data_ptr(),
        inputs.anchor.data_ptr(),
    )
    for length, token in [(129, 3), (1, 6), (257, 7), (5, 2)]:
        context = torch.randn(1, length, 16)
        anchor.fill_(token)
        expected = model.propose_context(context, anchor, 7)
        torch.testing.assert_close(manager.propose(context, anchor, 7), expected)
        features = torch.randn(3, 32)
        torch.testing.assert_close(
            manager.project_context(features), model.project_context(features)
        )
    assert manager.proposal_replays == manager.projection_replays == 4
    manager.capture()
    assert backend.captures == captures
    assert pointers == (
        inputs.context.data_ptr(),
        inputs.features.data_ptr(),
        inputs.anchor.data_ptr(),
    )
    with pytest.raises(ValueError, match="capacity"):
        manager.propose(torch.zeros(1, 258, 16), anchor, 7)
    with pytest.raises(ValueError, match="length"):
        manager.propose(context, anchor, 8)
    with pytest.raises(ValueError, match="capacity"):
        manager.project_context(torch.zeros(18, 32))


def test_graph_manager_can_be_recreated_after_memory_profiling(runtime):
    proposer = runtime.LEagleSpeculator.__new__(runtime.LEagleSpeculator)
    proposer.full_draft = True
    proposer.model = make_model(architecture())
    proposer.num_speculative_steps = 3
    proposer.graph_inputs = graph_module.LEagleGraphInputs.allocate(
        proposer.model, 8, 4, dtype=torch.float32, device="cpu"
    )
    proposer.create_graph_backend = CPUReplayBackend
    mode = SimpleNamespace(has_full_cudagraphs=lambda: True)
    proposer.init_cudagraph_manager(mode)
    proposer.capture()
    assert isinstance(proposer.graph_runner, graph_base.CudaGraphManager)
    assert not proposer.graph_runner.needs_capture()
    inputs = proposer.graph_inputs
    proposer.graph_runner = None  # Framework profiling teardown.
    proposer.init_cudagraph_manager(mode)
    proposer.capture()
    assert proposer.graph_inputs is inputs
    assert proposer.graph_runner.proposal_replays == 0


def test_full_draft_workspace_is_profiled_without_target_kv(runtime):
    proposer = runtime.LEagleSpeculator.__new__(runtime.LEagleSpeculator)
    proposer.full_draft = True
    proposer.num_speculative_steps = 3
    proposer.draft_tokens = torch.zeros((1, 3), dtype=torch.long)
    proposer.model = make_model(architecture())
    proposer.graph_inputs = graph_module.LEagleGraphInputs.allocate(
        proposer.model, 8, 4, dtype=torch.float32, device="cpu"
    )
    calls = []
    handle = proposer.model.lm_head.register_forward_hook(
        lambda *_args: calls.append(True)
    )
    proposer.propose(
        SimpleNamespace(num_reqs=1),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        dummy_run=True,
        skip_attn_for_dummy_run=True,
        is_profile=True,
    )
    handle.remove()
    assert len(calls) == 3


@pytest.mark.parametrize(
    "layers,loops,norm,window,reduced",
    [
        (1, 1, True, None, False),
        (1, 4, False, None, False),
        (2, 2, True, 4, True),
        (4, 1, False, None, True),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("candidate", [False, True, "low_rank", "svd", "svd_lowrank"])
def test_logits_match_real_training_code(
    layers, loops, norm, window, reduced, dtype, candidate, monkeypatch
):
    training_root = os.environ.get("LEAGLE_TRAINING_ROOT")
    if not training_root:
        pytest.skip("Set LEAGLE_TRAINING_ROOT for training/inference numerical parity")
    sys.path.insert(0, str(Path(training_root) / "src"))
    if sys.platform == "win32":
        # Load the real model modules without unrelated Linux file-transfer
        # imports in the training package's eager __init__ modules.
        for suffix in ("", ".models", ".models.eagle3", ".models.dflash"):
            name = "speculators" + suffix
            package = ModuleType(name)
            package.__path__ = [
                str(
                    Path(training_root)
                    / "src"
                    / "speculators"
                    / suffix.lstrip(".").replace(".", "/")
                )
            ]
            monkeypatch.setitem(sys.modules, name, package)
    from speculators.config import SpeculatorsConfig, VerifierConfig, reload_schemas
    from speculators.models.leagle import LEagleDraftModel, LEagleSpeculatorConfig
    from speculators.proposals.greedy import GreedyTokenProposalConfig

    reload_schemas()
    config = architecture(layers, loops, reduced, norm, window)
    if candidate is True or candidate == "low_rank":
        config.update(
            candidate_head=True,
            candidate_head_rank=4,
            candidate_head_top_k=3,
            candidate_rerank=candidate != "low_rank",
        )
    decoder = Qwen3Config(**config["transformer_layer_config"])
    decoder._attn_implementation = "sdpa"
    training = LEagleDraftModel(
        LEagleSpeculatorConfig(
            **{**config, "transformer_layer_config": decoder},
            speculators_config=SpeculatorsConfig(
                algorithm="leagle",
                default_proposal_method="greedy",
                proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=7)],
                verifier=VerifierConfig(
                    name_or_path=None, architectures=["Qwen3ForCausalLM"]
                ),
            ),
        )
    ).eval()
    with torch.no_grad():
        training.embed_tokens.weight.normal_(std=0.3)
        training.lm_head.weight.normal_(std=0.3)
    if reduced:
        selected = torch.arange(0, 24, 2)
        t2d = torch.zeros(24, dtype=torch.bool)
        t2d[selected] = True
        training.load_vocab_mappings(t2d, selected - torch.arange(12))
    saved_config = training.config.to_dict()
    converted = copy.deepcopy(saved_config["transformer_layer_config"])
    algos.update_leagle(saved_config, converted)
    inference = make_model(converted["leagle_config"])
    inference.load_weights(training.state_dict().items())
    training.to(dtype=dtype)
    inference.to(dtype=dtype)
    if candidate in ("svd", "svd_lowrank"):
        training.set_svd_candidate_head(4, top_k=3)
        inference.set_svd_candidate_head(4, top_k=3)
        if candidate == "svd_lowrank":
            training.set_candidate_head(True, rerank=False)
            inference.configure_candidate_head(rerank=False)
    features, anchor = torch.randn(1, 9, 32, dtype=dtype), torch.tensor([[5]])
    expected, logits = training.propose(features, anchor, num_tokens=7)
    actual_logits = []
    original_select = inference.select_greedy_token

    def record_selection(hidden):
        actual_logits.append(inference.compute_logits(hidden))
        return original_select(hidden)

    inference.select_greedy_token = record_selection
    actual = inference.propose_context(inference.project_context(features), anchor, 7)
    torch.testing.assert_close(actual, expected)
    tolerance = {"atol": 2e-6, "rtol": 2e-5} if dtype == torch.float32 else {}
    torch.testing.assert_close(torch.cat(actual_logits, 1), logits, **tolerance)
    # The same real training logits must hold with padded static graph inputs.
    actual_logits.clear()
    padded = torch.randn(1, 16, 16, dtype=dtype) * 100
    padded[:, :9] = inference.project_context(features)
    actual = inference.propose_static(padded, torch.tensor([9]), anchor, 7)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.cat(actual_logits, 1), logits, **tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reduced", [False, True])
@torch.inference_mode()
def test_candidate_selection_loader_ties_and_dense_override(dtype, reduced):
    config = architecture(1, 2, reduced=reduced)
    config.update(candidate_head=True, candidate_head_rank=4, candidate_head_top_k=3)
    converted = copy.deepcopy(config["transformer_layer_config"])
    algos.update_leagle(config, converted)
    assert converted["leagle_config"]["candidate_head"] is True
    model = make_model(converted["leagle_config"]).to(dtype)
    checkpoint = {name: value.clone() for name, value in model.state_dict().items()}
    if reduced:
        checkpoint["d2t"] = torch.arange(12)
    model.load_weights(checkpoint.items())
    hidden = torch.randn(2, 1, 16, dtype=dtype)
    expected = model.compute_logits(hidden).argmax(-1)
    # Candidate inference must never execute the full dense LM-head module.
    with patch.object(
        model.lm_head, "forward", side_effect=AssertionError("dense head called")
    ):
        torch.testing.assert_close(model.select_greedy_token(hidden), expected)
        # Exact scores tied at zero: unsorted candidate order must not affect ID.
        model.lm_head.weight.zero_()
        torch.testing.assert_close(
            model.select_greedy_token(hidden), model.compute_logits(hidden).argmax(-1)
        )
    broken = dict(checkpoint)
    del broken["candidate_head.down_proj.weight"]
    with pytest.raises(ValueError, match="Missing LEAGLE"):
        model.load_weights(broken.items())
    broken = dict(checkpoint)
    broken["candidate_head.up_proj.weight"] = torch.full_like(
        broken["candidate_head.up_proj.weight"], float("nan")
    )
    with pytest.raises(ValueError, match="finite"):
        model.load_weights(broken.items())
    model.load_weights(checkpoint.items())
    model.configure_candidate_head(enabled=False)
    torch.testing.assert_close(model.compute_logits(hidden), model.lm_head(hidden))
    dense = make_model(architecture())
    with pytest.raises(ValueError, match="no trained candidate"):
        dense.configure_candidate_head(enabled=True)
    with pytest.raises(ValueError, match="requires an enabled"):
        dense.configure_candidate_head(top_k=3)


@pytest.mark.parametrize(
    "key,value",
    [
        ("candidate_head_rank", 0),
        ("candidate_head_rank", 17),
        ("candidate_head_top_k", 0),
        ("candidate_head_top_k", 25),
    ],
)
def test_invalid_candidate_geometry_rejected(key, value):
    config = architecture()
    config.update(candidate_head=True, candidate_head_rank=4, candidate_head_top_k=3)
    config[key] = value
    with pytest.raises(ValueError, match="candidate"):
        make_model(config)


@torch.inference_mode()
def test_svd_factor_geometry_optimal_projection_and_ephemeral_state():
    weight = torch.zeros(20, 6)
    weight[:6] = torch.diag(torch.tensor([8.0, 5.0, 3.0, 2.0, 1.0, 0.5]))
    head = model_module.LEagleSVDHead(weight, 2)
    expected = weight.clone()
    expected[2:] = 0
    torch.testing.assert_close(head.up_weight @ head.down_weight, expected)
    assert not head.state_dict()
    assert not list(head.parameters())
    full = model_module.LEagleSVDHead(weight, 6)
    torch.testing.assert_close(full.up_weight @ full.down_weight, weight)
    for invalid in (0, 7, True):
        with pytest.raises(ValueError, match="rank"):
            model_module.LEagleSVDHead(weight, invalid)
    weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model_module.LEagleSVDHead(weight, 2)


@torch.inference_mode()
def test_svd_runtime_configuration_dense_fallback_and_checkpoint_unchanged():
    model = make_model(architecture(1, 1))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    model.configure_output_head(svd_rank=4, top_k=3)
    assert model.candidate_head is None
    assert model.candidate_head_enabled and model.candidate_rerank
    assert model.svd_candidate_head.down_weight.shape == (4, 16)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert model.state_dict().keys() == before.keys()
    hidden = torch.randn(2, 1, 16)
    expected = model.compute_logits(hidden).argmax(-1)
    with patch.object(
        model.lm_head, "forward", side_effect=AssertionError("dense head called")
    ):
        torch.testing.assert_close(model.select_greedy_token(hidden), expected)
        model.configure_candidate_head(rerank=False)
        torch.testing.assert_close(
            model.compute_logits(hidden), model.svd_candidate_head(hidden)
        )
        torch.testing.assert_close(
            model.select_greedy_token(hidden), model.compute_logits(hidden).argmax(-1)
        )
    model.configure_candidate_head(enabled=False)
    torch.testing.assert_close(model.compute_logits(hidden), model.lm_head(hidden))
    with pytest.raises(ValueError, match="restart"):
        model.load_weights(before.items())
    fresh = make_model(architecture())
    with pytest.raises(ValueError, match="cannot disable"):
        fresh.configure_output_head(svd_rank=4, enabled=False)
    with pytest.raises(ValueError, match="requires candidate reranking"):
        fresh.configure_output_head(svd_rank=4, top_k=3, rerank=False)
    fresh.train()
    with pytest.raises(ValueError, match="eval"):
        fresh.set_svd_candidate_head(4, top_k=3)


@pytest.mark.parametrize("full", [False, True])
@torch.inference_mode()
def test_sixteen_request_cache_update_is_one_replay_and_validation_is_atomic(full):
    model = make_model(architecture(1, 1))
    context = context_module.LEagleContext(16, 8, 16, dtype=torch.float32, device="cpu")
    execution = execution_module.LEagleExecution(
        model,
        context,
        64,
        3,
        execution_module.LEagleExecutionOptions(),
        dtype=torch.float32,
        device="cpu",
    )
    context.values.fill_(13)
    for pair in execution.prefix_kv:
        for tensor in pair:
            tensor.fill_(17)
    manager = None
    if full:
        manager = execution_module.LEagleOptimizedGraphRunner(
            execution, TreeCPUReplayBackend()
        )
        manager.capture()
    # 48 valid tokens round up to a 64-token graph, exceeding one request's
    # 8-token capacity. Padding must use multiple distinct private sink rows.
    features = torch.randn(64, 32)
    execution.project(features, torch.arange(64) % 3, manager)
    updates = [(slot, str(slot), 0, slot * 3, slot * 3 + 3) for slot in range(16)]
    execution.update_batch(updates, manager)
    if full:
        assert manager.update_replays == 1
        staged = execution.update_metadata[:64]
        assert (staged[48:, 0] >= 16).all()
        destinations = staged[:, 0] * 8 + staged[:, 1]
        assert destinations.unique().numel() == 64
    for slot in range(16):
        source = slice(slot * 3, slot * 3 + 3)
        torch.testing.assert_close(
            context.values[slot, :3], execution.projected[source]
        )
        assert (context.values[slot, 3:] == 13).all()
        for cached, new in zip(execution.prefix_kv, execution.new_kv):
            for storage, values in zip(cached, new):
                torch.testing.assert_close(
                    storage[slot : slot + 1, :, :3], values[:, :, source]
                )
                assert (storage[slot, :, 3:] == 17).all()
    snapshot = context.values.clone()
    ids, lengths = context.request_ids.copy(), context.lengths.copy()
    # A valid first request followed by a context gap must not partially
    # advance host lengths or write device data for either request.
    with pytest.raises(RuntimeError, match="missing target context"):
        execution.update_batch([(0, "0", 3, 0, 1), (1, "1", 6, 1, 2)], manager)
    with pytest.raises(ValueError, match="distinct"):
        execution.update_batch([(0, "0", 3, 0, 1), (0, "0", 4, 1, 2)], manager)
    assert context.lengths == lengths and context.request_ids == ids
    torch.testing.assert_close(context.values, snapshot)
    execution.update_batch([], manager)
    if full:
        assert manager.update_replays == 1
    # Reorder slots, replace one request, and advance others in one batch.
    updates = [
        (
            slot,
            "replacement" if slot == 2 else str(slot),
            0 if slot == 2 else 3,
            15 - slot,
            16 - slot,
        )
        for slot in reversed(range(16))
    ]
    execution.update_batch(updates, manager)
    for slot, request_id, start, begin, end in updates:
        torch.testing.assert_close(
            context.values[slot, start : start + 1], execution.projected[begin:end]
        )
    assert context.lengths[2] == 1 and context.request_ids[2] == "replacement"
    if full:
        assert manager.update_replays == 2


@torch.inference_mode()
def test_private_quantized_head_dispatch_and_restart_guards():
    model = make_model(architecture(1, 1))
    checkpoint = {key: value.clone() for key, value in model.state_dict().items()}

    class PrivateHead(torch.nn.Module):
        hidden_size, vocab_size = 16, 24

        def forward(self, hidden):
            # Controlled logits distinguish the private path from the dense head.
            output = hidden.new_zeros(*hidden.shape[:-1], self.vocab_size)
            output[..., 7] = 1
            return output

    model.set_quantized_lm_head(PrivateHead())
    with pytest.raises(ValueError, match="restart"):
        model.set_quantized_lm_head(PrivateHead())
    hidden = torch.randn(2, 1, 16)
    with patch.object(
        model.lm_head, "forward", side_effect=AssertionError("dense head called")
    ):
        assert (model.select_greedy_token(hidden) == 7).all()
    assert checkpoint.keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint[name])
    with pytest.raises(ValueError, match="restart"):
        model.load_weights(checkpoint.items())
    with pytest.raises(ValueError, match="cannot combine"):
        model.set_svd_candidate_head(4, top_k=3)
    with pytest.raises(ValueError, match="cannot combine"):
        model.configure_candidate_head(enabled=True)
    config = architecture(1, 1)
    config.update(candidate_head=True, candidate_head_rank=4, candidate_head_top_k=3)
    candidate = make_model(config)
    with pytest.raises(ValueError, match="cannot combine"):
        candidate.set_quantized_lm_head(PrivateHead())


@torch.inference_mode()
@pytest.mark.parametrize("rerank", [False, True])
def test_quantized_svd_projection_preserves_bf16_reranking_and_restart_guards(rerank):
    model = make_model(architecture(1, 1))
    model.configure_output_head(svd_rank=8, top_k=3 if rerank else None, rerank=rerank)
    hidden = torch.randn(2, 1, 16)
    expected = model.select_greedy_token(hidden)
    dense = model.lm_head.weight.clone()

    class Projection(torch.nn.Module):
        hidden_size, vocab_size = 8, 24
        calls = 0

        def __init__(self):
            super().__init__()
            self.register_buffer("weight", model.svd_candidate_head.up_weight.clone())

        def forward(self, reduced):
            self.calls += 1
            return torch.nn.functional.linear(reduced, self.weight)

    head = Projection()
    model.set_quantized_svd_head(head)
    torch.testing.assert_close(model.select_greedy_token(hidden), expected)
    torch.testing.assert_close(model.lm_head.weight, dense)
    assert head.calls == 1
    assert model.quantized_lm_head is None
    for change in (
        lambda: model.set_quantized_svd_head(head),
        lambda: model.configure_output_head(svd_rank=4),
        lambda: model.set_svd_candidate_head(4, top_k=3),
        lambda: model.configure_candidate_head(enabled=False),
        lambda: model.load_weights([]),
    ):
        with pytest.raises(ValueError, match="restart"):
            change()

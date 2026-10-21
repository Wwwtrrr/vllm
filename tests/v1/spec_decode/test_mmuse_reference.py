# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for stateless block attention and confirmed-feature history."""

import ast
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

DIRECTORY = Path(__file__).resolve().parents[3] / "vllm/model_executor/models/mmuse"


def module(name):
    package = "mmuse_reference_cpu"
    if package not in sys.modules:
        root = ModuleType(package)
        root.__path__ = [str(DIRECTORY)]
        sys.modules[package] = root
    spec = importlib.util.spec_from_file_location(
        f"{package}.{name}", DIRECTORY / f"{name}.py"
    )
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("window,noncausal", [(None, False), (4, False), (4, True)])
def test_block_attention_ignores_context_outside_fixed_anchor_window(window, noncausal):
    decoder = module("deepseek_decoder")
    config = SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_size=16,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 1000000.0},
    )
    torch.manual_seed(12)
    attention = decoder.BlockAttention(config, window, noncausal)
    context, query = torch.randn(9, 16), torch.randn(7, 16)
    positions = torch.arange(9, 16)
    result = attention(query, context, positions, torch.arange(9))
    assert result.shape == (7, 16)  # head_dim intentionally differs from hidden/heads.
    if window is not None:
        trimmed = attention(
            query, context[-window:], positions, torch.arange(9)[-window:]
        )
        torch.testing.assert_close(result, trimmed)
    if window is not None and not noncausal:
        changed = query.clone()
        changed[1:] += 10
        torch.testing.assert_close(
            result[:1], attention(changed, context, positions, torch.arange(9))[:1]
        )


def test_verified_history_replaces_rejected_suffix_and_isolates_requests():
    history = module("deepseek_contract").VerifiedPrefix(4, torch.empty(2, 4, 3))
    first = torch.randn(6, 3)
    history.append(0, "a", torch.arange(6), first)
    replacement = torch.randn(2, 3)
    positions, values = history.append(0, "a", torch.arange(5, 7), replacement)
    assert positions.tolist() == [3, 4, 5, 6]
    torch.testing.assert_close(values[-2:], replacement)
    with pytest.raises(ValueError, match="missing"):
        history.append(0, "b", torch.arange(7, 9), replacement)
    positions, values = history.append(0, "b", torch.arange(2), replacement)
    assert positions.tolist() == [0, 1]
    torch.testing.assert_close(values, replacement)


@pytest.mark.parametrize(
    "bad", [None, "eager", "prefix", "teacher", "order", "geometry", "contract"]
)
def test_training_contract_rejects_incompatible_serving_inputs(bad):
    decoder = dict(
        model_type="qwen3",
        hidden_size=16,
        vocab_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    source = dict(
        transformer_layer_config=decoder,
        target_hidden_state_format="deepseek_v4_mean_hc_head",
        aux_hidden_state_layer_ids=[1, 3],
        block_size=7,
        enable_correction_head=True,
    )
    target = SimpleNamespace(
        enforce_eager=True,
        dtype=torch.bfloat16,
        get_total_num_hidden_layers=lambda: 5,
        get_hidden_size=lambda: 16,
        get_vocab_size=lambda: 32,
    )
    spec = SimpleNamespace(
        enforce_eager=True,
        num_speculative_tokens=7,
        draft_model_config=SimpleNamespace(quantization=None),
        enable_adaptive_verification=False,
        use_local_argmax_reduction=False,
    )
    config = SimpleNamespace(
        model_config=target,
        speculative_config=spec,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(),
    )
    if bad == "eager":
        target.enforce_eager = False
    elif bad == "prefix":
        config.cache_config.enable_prefix_caching = True
    elif bad == "teacher":
        source["aux_hidden_state_layer_ids"] = [1, 5]
    elif bad == "order":
        source["aux_hidden_state_layer_ids"] = [3, 1]
    elif bad == "geometry":
        decoder["head_dim"] = 7
    elif bad == "contract":
        source["target_training_contract"] = {}
    validate = module("deepseek_contract").validate_deepseek_export
    if bad is None:
        validate(config, source)
    else:
        with pytest.raises(ValueError):
            validate(config, source)


def test_reference_decoder_registers_no_paged_attention():
    decoder = module("deepseek_decoder")
    config = SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_size=16,
        intermediate_size=32,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_parameters={},
    )
    layer = decoder.BlockLayer(config, {"block_size": 7}, 4)
    assert not any(
        cls.__name__ == "AttentionLayerBase"
        for child in layer.modules()
        for cls in type(child).__mro__
    )
    assert layer.self_attn.q_proj.weight.shape == (32, 16)


def reference_proposer():
    path = DIRECTORY.parents[2] / "v1/worker/gpu/spec_decode/mmuse/reference.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = dict(torch=torch, copy=copy, DraftModelSpeculator=object)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    proposer = namespace[cls.name].__new__(namespace[cls.name])
    proposer.device = torch.device("cpu")
    proposer.num_query_per_req = 3
    proposer.draft_tokens = torch.zeros(2, 3, dtype=torch.long)
    proposer.draft_logits = None
    proposer.hidden_size = 3

    class Model:
        config = SimpleNamespace(
            mask_token_id=31, mmuse_source_config={"aux_hidden_state_layer_ids": [1]}
        )
        feature_storage = torch.empty(2, 4, 3)
        history = module("deepseek_contract").VerifiedPrefix(4, feature_storage)
        calls = []

        def project_features(self, features):
            return features

        def __call__(self, ids, positions, context_positions, features):
            self.calls.append(
                (ids.clone(), positions.clone(), context_positions.clone())
            )
            return torch.zeros(3, 3)

    class Runtime:
        def rollout_correction(self, hidden, anchors, **kwargs):
            return anchors[:, None] + torch.arange(1, 4), None

    proposer.model = Model()
    proposer.model.runtime = Runtime()
    return proposer


def test_proposer_preserves_anchor_alignment_after_rejection_and_reordering():
    proposer = reference_proposer()
    batch = SimpleNamespace(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 2, 4]),
        positions=torch.tensor([0, 1, 0, 1]),
        idx_mapping_np=[1, 0],
        idx_mapping=torch.tensor([1, 0]),
        req_ids=["a", "b"],
    )
    common = dict(
        input_batch=batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=None,
        aux_hidden_states=[torch.randn(4, 3)],
        num_sampled=torch.tensor([1, 0]),
        num_rejected=torch.tensor([0, 0]),
        last_sampled=torch.tensor([8, 9]),
        next_prefill_tokens=torch.tensor([[10, 11]]),
        temperature=torch.zeros(2),
        seeds=torch.zeros(2, dtype=torch.long),
    )
    assert proposer.propose(**common).tolist() == [[10, 11, 12], [11, 12, 13]]
    assert proposer.model.calls[0][0].tolist() == [9, 31, 31]
    assert proposer.model.calls[0][1].tolist() == [2, 3, 4]
    batch.idx_mapping_np, batch.idx_mapping, batch.req_ids = (
        [0, 1],
        torch.tensor([0, 1]),
        ["b", "a"],
    )
    batch.query_start_loc = torch.tensor([0, 3, 6])
    batch.positions = torch.tensor([2, 3, 4, 2, 3, 4])
    common.update(
        aux_hidden_states=[torch.randn(6, 3)],
        num_rejected=torch.tensor([2, 1]),
        num_sampled=torch.tensor([1, 1]),
    )
    proposer.propose(**common)
    assert proposer.model.calls[-2][1].tolist() == [3, 4, 5]
    assert proposer.model.calls[-1][1].tolist() == [4, 5, 6]
    assert proposer.model.history.requests[0][1].tolist() == [0, 1, 2]
    assert proposer.model.history.requests[1][1].tolist() == [0, 1, 2, 3]
    saved = proposer.model.feature_storage.clone()
    proposer.propose(**common, dummy_run=True)
    torch.testing.assert_close(proposer.model.feature_storage, saved)


def test_draft_loader_isolates_target_quantizer_without_rebuilding_target_config(
    monkeypatch,
):
    proposer = reference_proposer()
    target_quantizer, target = object(), object()
    proposer.vllm_config = SimpleNamespace(
        quant_config=target_quantizer, load_config="target"
    )
    proposer.draft_model_config = object()
    seen = {}
    loader = ModuleType("vllm.model_executor.model_loader")
    utilities = ModuleType("vllm.model_executor.model_loader.utils")
    utilities.get_draft_load_config = lambda _: "draft"

    def get_model(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(bind_target=lambda value: seen.update(target=value))

    loader.get_model = get_model
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setitem(sys.modules, utilities.__name__, utilities)
    proposer.load_draft_model(target, set())
    assert seen["vllm_config"].quant_config is None
    assert seen["vllm_config"].load_config == "draft"
    assert seen["model_config"] is proposer.draft_model_config
    assert seen["target"] is target
    assert proposer.vllm_config.quant_config is target_quantizer
    assert proposer.vllm_config.load_config == "target"


@pytest.mark.parametrize("dtype", [torch.float32, torch.int8, torch.float8_e4m3fn])
def test_borrowed_io_gathers_float_shards_and_matches_draft_dtype(dtype):
    path = DIRECTORY.parent / "deepseek_mmuse_reference.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekMMuseReference"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "bind_target"
    )
    namespace = dict(
        torch=torch,
        get_tensor_model_parallel_world_size=lambda: 2,
        tensor_model_parallel_all_gather=lambda weight, dim: torch.cat(
            (weight, weight + 10), dim
        ),
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    owner = SimpleNamespace(
        config=SimpleNamespace(hidden_size=3, vocab_size=6),
        feature_storage=torch.empty(1, 4, 3, dtype=torch.bfloat16),
    )
    weight = torch.arange(9).reshape(3, 3).to(dtype)
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight)),
        lm_head=SimpleNamespace(weight=weight.clone()),
    )
    if dtype != torch.float32:
        with pytest.raises(ValueError, match="floating matrices"):
            namespace[method.name](owner, target)
    else:
        namespace[method.name](owner, target)
        expected = torch.cat((weight, weight + 10)).to(torch.bfloat16)
        torch.testing.assert_close(owner.embedding_weight, expected)
        torch.testing.assert_close(owner.output_weight, expected)


@pytest.mark.parametrize(
    "states",
    [[torch.zeros(2, 12)], [torch.zeros(1, 3)], [torch.zeros(2, 3), torch.zeros(2, 3)]],
)
def test_proposer_rejects_wrong_hc_width_rows_or_auxiliary_count(states):
    proposer = reference_proposer()
    batch = SimpleNamespace(num_reqs=1, positions=torch.arange(2))
    with pytest.raises(ValueError, match="auxiliary tensors"):
        proposer.propose(
            batch, {}, {}, None, states, None, None, None, None, None, None
        )
    assert proposer.model.history.requests == {}


@pytest.mark.parametrize(
    "names",
    [
        ("embed_tokens.weight",),
        ("lm_head.weight",),
        ("embed_tokens.weight", "lm_head.weight"),
    ],
)
def test_checkpoint_io_is_ignored_and_target_io_is_always_bound(names):
    path = DIRECTORY.parent / "deepseek_mmuse_reference.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekMMuseReference"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("load_weights", "bind_target")
    ]
    namespace = dict(torch=torch, get_tensor_model_parallel_world_size=lambda: 1)
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    owner = SimpleNamespace(
        config=SimpleNamespace(hidden_size=3, vocab_size=6),
        feature_storage=torch.empty(1, 4, 3, dtype=torch.bfloat16),
        named_parameters=lambda: [],
    )
    serialized = torch.arange(18).reshape(6, 3).float()
    assert (
        namespace["load_weights"](owner, [(name, serialized) for name in names])
        == set()
    )
    target_weight = torch.full((6, 3), -1, dtype=torch.bfloat16)
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=target_weight)),
        lm_head=SimpleNamespace(weight=target_weight),
    )
    namespace["bind_target"](owner, target)
    for buffer in ("embedding_weight", "output_weight"):
        torch.testing.assert_close(getattr(owner, buffer), target_weight)
    # Loading another checkpoint must not replace IO already bound to Target.
    namespace["load_weights"](owner, [(names[0], torch.zeros(5, 3))])
    torch.testing.assert_close(owner.embedding_weight, target_weight)
    torch.testing.assert_close(owner.output_weight, target_weight)

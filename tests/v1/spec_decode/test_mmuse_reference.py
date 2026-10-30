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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("window", [4, None])
@torch.inference_mode()
def test_cached_context_matches_recompute_after_window_rejection_and_slot_reuse(
    dtype, window
):
    decoder, contract = module("deepseek_decoder"), module("deepseek_contract")
    path = DIRECTORY.parent / "deepseek_mmuse_reference.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekMMuseReference"
    )
    forward = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    namespace = dict(torch=torch, F=torch.nn.functional)
    exec(
        compile(ast.Module(body=[forward], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    torch.manual_seed(51)
    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000.0},
    )
    layers = torch.nn.ModuleList(
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
            for _ in range(2)
        ]
    ).to(dtype=dtype)
    capacity = window or 16
    for layer in layers:
        for conv in (layer.attention_conv, layer.mlp_conv):
            torch.nn.init.normal_(conv.base_kernel, std=0.1)
    storage = torch.empty(2, 2, 2, capacity, 1, 4, dtype=dtype)
    owner = SimpleNamespace(
        hidden_norm=decoder.RMS(8, 1e-6).to(dtype=dtype),
        norm=decoder.RMS(8, 1e-6).to(dtype=dtype),
        embedding_weight=torch.randn(32, 8, dtype=dtype),
        layers=layers,
        runtime=SimpleNamespace(
            block_position_embedding=None, context_hidden_proj=None
        ),
        context_kv_storage=storage,
        context_kv_cache=contract.ContextKVCache(storage),
    )
    projected_rows = []
    handles = [
        layer.self_attn.k_proj.register_forward_pre_hook(
            lambda _module, args: projected_rows.append(args[0].shape[0])
        )
        for layer in layers
    ]
    features = torch.randn(10, 8, dtype=dtype)
    ids = torch.tensor([7, 30, 30])
    # Includes extension, window eviction, overlapping corrected suffix,
    # request reorder (slot 1) and a different request reusing slot 0.
    rounds = [
        (0, "a", 3, 0),
        (0, "a", 5, 3),
        (0, "a", 6, 5),
        (1, "b", 3, 0),
        (0, "a", 6, 4),
        (0, "new", 3, 0),
    ]
    reused_counts = []
    for slot, request, end, update in rounds:
        if update == 4:
            features[4:6] += 0.5
        start = max(0, end - capacity)
        context_pos = torch.arange(start, end)
        positions = torch.arange(end, end + 3)
        args = (ids, positions, context_pos, features[start:end])
        expected = namespace["forward"](owner, *args)
        projected_rows.clear()
        actual = namespace["forward"](
            owner, *args, cache_slot=slot, request_id=request, update_start=update
        )
        tolerance = 0.03 if dtype == torch.bfloat16 else 2e-6
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        # Per layer, only newly confirmed context rows and query rows project.
        reused_counts.append(end - start - (sum(projected_rows) // len(layers) - 3))
    assert reused_counts[0] == 0
    assert reused_counts[1] > 0
    assert reused_counts[3] == reused_counts[5] == 0
    assert owner.context_kv_cache.entries[0][0] == "new"
    for handle in handles:
        handle.remove()


def test_verified_prefix_cpu_metadata_matches_concatenation():
    storage = torch.zeros(2, 5, 3)
    history = module("context").VerifiedPrefix(5, storage)
    previous = None
    for request, start, end in [
        ("a", 0, 3),
        ("a", 3, 7),
        ("a", 5, 8),
        ("a", 8, 18),
        ("new", 0, 2),
    ]:
        positions = torch.arange(start, end)
        values = torch.randn(end - start, 3)
        if start == 0:
            expected = values[-5:].clone()
        else:
            old_positions, old_values = previous
            expected = torch.cat((old_values[old_positions < start], values))[-5:]
        actual_positions, actual = history.append(
            0, request, positions, values, positions_cpu=positions.tolist()
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual_positions, torch.arange(end - len(expected), end)
        )
        previous = actual_positions.clone(), actual.clone()


def test_context_kv_cache_failure_invalidates_previous_entry():
    cache = module("deepseek_contract").ContextKVCache(torch.zeros(1, 1, 2, 4, 1, 2))
    cache.commit(0, "request", 0, 4)
    assert cache.begin(0, "request", 1, 5, 4) == 3
    # Simulate a failure before all layer writes finish: begin removed metadata.
    assert 0 not in cache.entries
    assert cache.begin(0, "request", 1, 5, 4) == 0


@pytest.mark.parametrize(
    "bad", [None, "full", "prefix", "teacher", "order", "geometry", "contract"]
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
    if bad == "full":
        target.enforce_eager = False
        spec.enforce_eager = False
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
    if bad in (None, "full"):
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


def test_target_dispatch_keeps_deepseek_tp_and_sliding_attention_support(monkeypatch):
    path = DIRECTORY / "validation.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(torch=torch, nn=torch.nn)
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.mmuse.deepseek_contract",
        module("deepseek_contract"),
    )
    source = dict(
        transformer_layer_config=dict(
            model_type="qwen3",
            hidden_size=16,
            vocab_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        ),
        target_hidden_state_format="deepseek_v4_mean_hc_head",
        aux_hidden_state_layer_ids=[1, 3],
        block_size=7,
        enable_correction_head=True,
    )
    target = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v4"),
        enforce_eager=True,
        dtype=torch.bfloat16,
        get_total_num_hidden_layers=lambda: 5,
        get_hidden_size=lambda: 16,
        get_vocab_size=lambda: 32,
    )
    config = SimpleNamespace(
        model_config=target,
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    mmuse_source_config=source, layer_types=["sliding_attention"]
                ),
                quantization=None,
            ),
            enforce_eager=True,
            num_speculative_tokens=7,
            draft_sample_method="greedy",
            enable_adaptive_verification=False,
            use_local_argmax_reduction=False,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    namespace["validate_mmuse_config"](config)
    target.hf_config.model_type = "qwen3"
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        namespace["validate_mmuse_config"](config)
    # Common source validation still precedes target-specific dispatch.
    config.speculative_config.draft_model_config.hf_config.mmuse_source_config = None
    with pytest.raises(ValueError, match="Speculators-format"):
        namespace["validate_mmuse_config"](config)


def reference_io_helpers(namespace):
    path = DIRECTORY.parent / "deepseek_mmuse_reference.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace.setdefault("torch", torch)
    namespace.setdefault("get_tensor_model_parallel_world_size", lambda: 1)
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )


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
            assert anchors.shape == (hidden.shape[0],)
            return anchors[:, None] + torch.arange(1, 4), None

    proposer.model = Model()
    proposer.model.runtime = Runtime()
    return proposer


def test_probabilistic_reference_proposer_caches_final_logits_by_slot_and_position():
    proposer = reference_proposer()
    proposer.model.feature_storage = torch.empty(4, 4, 3)
    proposer.draft_logits = torch.full((4, 3, 5), -99.0, dtype=torch.bfloat16)
    proposer.use_fp64_gumbel = False
    batch = SimpleNamespace(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 2, 4]),
        positions=torch.tensor([0, 1, 0, 1]),
        idx_mapping_np=[2, 0],
        idx_mapping=torch.tensor([2, 0], dtype=torch.int32),
        req_ids=["a", "b"],
    )
    temperatures = torch.tensor([1.0, 0.0, 0.7, 1.5])
    seeds = torch.tensor([11, 12, 13, 14])
    observed = []

    def sample(logits, slots, temperature, actual_seeds, positions, **kwargs):
        assert temperature is temperatures and actual_seeds is seeds
        assert kwargs["apply_temperature"] and kwargs["is_drafting"]
        assert logits.dtype == proposer.draft_logits.dtype
        cols = kwargs["logits_cache_col"].long()
        kwargs["logits_cache"][slots.long(), cols] = logits
        observed.append((positions.clone(), logits.clone()))
        return torch.multinomial(
            torch.softmax(logits.float() / temperature[slots.long(), None], -1), 1
        )[:, 0]

    proposer.sample = sample

    def rollout(hidden, anchors, *, temperature, sample_token):
        assert temperature == 0.0  # Native callback owns per-request temperature.
        tokens = []
        for column in range(3):
            logits = torch.arange(10).reshape(2, 5).float() * (column + 1) / 7
            tokens.append(sample_token(logits, column))
        return torch.stack(tokens, 1), None

    proposer.model.runtime.rollout_correction = rollout
    output = proposer.propose(
        batch,
        {},
        {},
        None,
        [torch.randn(4, 3)],
        torch.tensor([1, 1]),
        torch.tensor([0, 0]),
        torch.tensor([[9], [8], [7], [6]]),
        torch.zeros(1, 4, dtype=torch.long),
        temperatures,
        seeds,
    )
    assert output.shape == (2, 3)
    for column, (positions, logits) in enumerate(observed):
        torch.testing.assert_close(positions, torch.tensor([2 + column, 2 + column]))
        torch.testing.assert_close(proposer.draft_logits[[2, 0], column], logits)
    assert (proposer.draft_logits[[1, 3]] == -99).all()


@pytest.mark.parametrize("column_tokens", [False, True])
def test_proposer_preserves_anchor_alignment_after_rejection_and_reordering(
    column_tokens,
):
    proposer = reference_proposer()
    batch = SimpleNamespace(
        num_reqs=2,
        num_draft_tokens=0,
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
        last_sampled=torch.tensor([[8], [9]])
        if column_tokens
        else torch.tensor([8, 9]),
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
    batch.num_draft_tokens = 6
    batch.num_draft_tokens_per_req = torch.tensor([3, 3])
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


@pytest.mark.parametrize("backend", ["reference", "ascend_fia", "unknown"])
def test_reference_platform_rejects_unavailable_attention_backends(backend):
    proposer = reference_proposer()
    proposer.vllm_config = SimpleNamespace(
        additional_config={"mmuse_attention_backend": backend}
    )
    if backend == "reference":
        proposer.configure_attention(object())
    else:
        with pytest.raises(ValueError, match="only reference"):
            proposer.configure_attention(object())


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
        return SimpleNamespace(
            bind_target=lambda value, rotation: seen.update(
                target=value, rotation=rotation
            )
        )

    loader.get_model = get_model
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setitem(sys.modules, utilities.__name__, utilities)
    proposer.load_draft_model(target, set())
    assert seen["vllm_config"].quant_config is None
    assert seen["vllm_config"].load_config == "draft"
    assert seen["model_config"] is proposer.draft_model_config
    assert seen["target"] is target
    assert seen["rotation"] is None
    assert proposer.vllm_config.quant_config is target_quantizer
    assert proposer.vllm_config.load_config == "target"


def test_framework_cache_hook_uses_scheduler_batch_rows_and_actual_kernel_pages():
    proposer = reference_proposer()
    calls = []
    paged = SimpleNamespace(
        requires_eager=False,
        cache_owner=SimpleNamespace(layer_name="draft_model.layers.0.self_attn.attn"),
        set_page_size=lambda size: calls.append(("page", size)),
        begin_request=lambda table, start, end: calls.append(
            ("request", table.clone(), start, end)
        ),
        reset_request=lambda: calls.append(("reset",)),
    )
    proposer.model = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(paged_attention_impl=paged))]
    )
    table = torch.tensor([[4, 8], [3, 6]], dtype=torch.int32)
    blocks = SimpleNamespace(
        kernel_block_sizes=[64],
        input_block_tables=[table],
        num_blocks=SimpleNamespace(np=torch.tensor([[2, 1]])),
        blocks_per_kv_block=[1],
    )
    cache = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=[paged.cache_owner.layer_name])]
    )
    proposer.set_attn(None, cache, blocks, None, [])
    proposer.prepare_request_attention(1, 65, 99, request_slot=1)
    proposer.prepare_request_attention(None)
    assert calls[0] == ("page", 64)
    torch.testing.assert_close(calls[1][1], table[1, :1])
    assert calls[1][2:] == (65, 99)
    assert calls[2] == ("reset",)


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
    reference_io_helpers(namespace)
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
def test_unrotated_target_binding_keeps_legacy_io_policy(names):
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
    reference_io_helpers(namespace)
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
    # A malformed teacher matrix must not silently enter a rotated model.
    with pytest.raises(ValueError, match="teacher IO"):
        namespace["load_weights"](owner, [(names[0], torch.zeros(5, 3))])
    torch.testing.assert_close(owner.embedding_weight, target_weight)
    torch.testing.assert_close(owner.output_weight, target_weight)


def test_quarot_uses_teacher_io_and_unrotates_each_layer_before_fusion():
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
        and node.name in ("load_weights", "bind_target", "project_features")
    ]
    namespace = dict(torch=torch)
    reference_io_helpers(namespace)
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    torch.manual_seed(93)
    rotation = torch.linalg.qr(torch.randn(4, 4)).Q
    teacher_embedding, teacher_head = torch.randn(6, 4), torch.randn(6, 4)
    owner = SimpleNamespace(
        config=SimpleNamespace(hidden_size=4, vocab_size=6),
        feature_storage=torch.empty(1, 3, 4),
        named_parameters=lambda: [],
    )
    namespace["load_weights"](
        owner,
        [
            ("embed_tokens.weight", teacher_embedding),
            ("lm_head.weight", teacher_head),
        ],
    )
    # No target access or TP gather is permitted: its head has fused norm.
    namespace["bind_target"](owner, object(), rotation=rotation)
    torch.testing.assert_close(owner.embedding_weight, teacher_embedding)
    torch.testing.assert_close(owner.output_weight, teacher_head)
    raw = torch.randn(3, 2, 4)
    seen = []
    owner.runtime = SimpleNamespace(
        _prepare_target_hidden=lambda x: (seen.append(x.clone()) or x, None)
    )
    namespace["project_features"](owner, (raw @ rotation).reshape(3, 8))
    torch.testing.assert_close(seen[0], raw.reshape(3, 8))
    # The recovered teacher head reproduces teacher logits, unlike the fused
    # deployed head, even after its inverse rotation.
    norm = torch.tensor([0.2, 0.4, 0.6, 0.8])
    deployed_head = (teacher_head * norm) @ rotation
    hidden = torch.randn(2, 4)
    expected = hidden @ teacher_head.T
    torch.testing.assert_close(hidden @ owner.output_weight.T, expected)
    assert not torch.allclose(hidden @ (deployed_head @ rotation.T).T, expected)
    owner.checkpoint_head = None
    with pytest.raises(ValueError, match="original training teacher"):
        namespace["bind_target"](owner, object(), rotation=rotation)


@pytest.mark.parametrize(
    "name", ["layer_fusion_gate", "context_hidden_gate", "correction_markov_scale"]
)
@pytest.mark.parametrize("source_shape,target_shape", [((), (1,)), ((1,), ())])
def test_checkpoint_scalar_gate_layout_preserves_value(
    name, source_shape, target_shape
):
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
        if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
    )
    namespace = dict(torch=torch)
    reference_io_helpers(namespace)
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    parameter = torch.nn.Parameter(torch.zeros(target_shape))
    destination = "runtime." + name
    owner = SimpleNamespace(named_parameters=lambda: [(destination, parameter)])
    value = torch.tensor(-0.625).reshape(source_shape)
    loaded = namespace["load_weights"](owner, [(name, value)])
    assert loaded == {destination}
    assert parameter.shape == target_shape
    torch.testing.assert_close(parameter, value.reshape(target_shape))
    # Equal numel alone does not authorize arbitrary shape conversion.
    with pytest.raises(ValueError, match="Unexpected MMuse"):
        namespace["load_weights"](owner, [(name, value.reshape(1, 1))])
    with pytest.raises(ValueError, match="Unexpected MMuse"):
        namespace["load_weights"](owner, [(name, torch.zeros(2))])
    other = SimpleNamespace(named_parameters=lambda: [("norm.weight", parameter)])
    with pytest.raises(ValueError, match="Unexpected MMuse"):
        namespace["load_weights"](other, [("norm.weight", value)])
    with pytest.raises(ValueError, match="Missing trained"):
        namespace["load_weights"](owner, [])


def test_feature_projection_hook_preserves_validation_and_eager_default():
    proposer = reference_proposer()
    batch = SimpleNamespace(positions=torch.arange(2))
    states = [torch.randn(2, 3)]
    calls = []
    hook = lambda tensors: calls.append(tensors) or "graph"
    assert (
        proposer._project_target_features(batch, states, False, projector=hook)
        == "graph"
    )
    assert calls == [states]
    torch.testing.assert_close(
        proposer._project_target_features(batch, states, False), states[0]
    )
    with pytest.raises(ValueError, match="auxiliary tensors"):
        proposer._project_target_features(
            batch, [torch.zeros(2, 12)], False, projector=hook
        )
    assert len(calls) == 1
    assert proposer._project_target_features(batch, None, True, projector=hook) is None

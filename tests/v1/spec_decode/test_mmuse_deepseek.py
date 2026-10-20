# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the dense DeepSeek-V4 MMuse adapter, not device serving."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


def load_module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration(tp=1):
    source = {
        "speculators_model_type": "mmuse",
        "transformer_layer_config": {"model_type": "qwen3"},
        "target_hidden_state_format": "deepseek_v4_mean_hc_head",
        "target_hidden_size": 16,
        "draft_vocab_size": 32,
        "aux_hidden_state_layer_ids": [1, 3],
        "block_size": 7,
        "sample_from_anchor": True,
    }
    hf = NS(
        mmuse_source_config=source,
        hidden_size=16,
        vocab_size=32,
        layer_types=["sliding_attention", "full_attention"],
    )

    return NS(
        model_config=NS(
            hf_config=NS(model_type="deepseek_v4"),
            enforce_eager=True,
            dtype=torch.bfloat16,
            get_hidden_size=lambda: 16,
            get_vocab_size=lambda: 32,
            get_total_num_hidden_layers=lambda: 5,
        ),
        speculative_config=NS(
            draft_model_config=NS(hf_config=hf, quantization=None),
            draft_parallel_config=NS(tensor_parallel_size=tp),
            enforce_eager=True,
            draft_sample_method="greedy",
            num_speculative_tokens=7,
            enable_adaptive_verification=False,
            kv_cache_dtype=None,
        ),
        parallel_config=NS(tensor_parallel_size=tp),
        watermark_config=None,
    )


def test_dense_draft_attention_config_is_isolated_from_quantized_target():
    module = load_module("vllm/model_executor/models/mmuse_runtime.py", "mmuse_cache")
    config = configuration()
    config.cache_config = NS(cache_dtype="fp8_ds_mla")
    config.quant_config = object()

    class Parent:
        @property
        def attn_vllm_config(self):
            return self.vllm_config

        def load_draft_model(self, target, names):
            return target, names, self.speculative_config.kv_cache_dtype

    class Adapter(module.MMuseGreedyRolloutMixin, Parent):
        pass

    adapter = Adapter()
    adapter.vllm_config = config
    adapter.speculative_config = config.speculative_config
    draft = adapter.attn_vllm_config
    assert draft.model_config is config.speculative_config.draft_model_config
    assert draft.quant_config is None
    assert draft.cache_config.cache_dtype == "auto"
    assert config.cache_config.cache_dtype == "fp8_ds_mla"
    assert config.quant_config is not None
    assert adapter.load_draft_model("target", ["layer"]) == (
        "target",
        ["layer"],
        "auto",
    )


def test_anchor_mask_keeps_fixed_context_and_causal_query_block():
    module = load_module(
        "vllm/model_executor/models/mmuse/anchor_attention.py", "anchor_mask_test"
    )
    actual = module.anchor_mask(torch.arange(8), torch.arange(8, 15), 4, True)
    assert not actual[:, :4].any()
    assert actual[:, 4:8].all()
    assert torch.equal(actual[:, 8:], torch.ones(7, 7, dtype=torch.bool).tril())


def test_anchor_cache_reserved_storage_and_request_slot_reuse():
    module = load_module(
        "vllm/model_executor/models/mmuse/anchor_attention.py", "anchor_cache_test"
    )
    storage = torch.empty(2, 4, 16)
    cache = module.AnchorContextCache(4, storage)
    features = torch.randn(8, 16)
    cache.update(0, "a", torch.arange(5), features[:5])
    positions, stored = cache.update(0, "a", torch.arange(5, 8), features[5:])
    assert positions.tolist() == [4, 5, 6, 7]
    assert stored.data_ptr() == storage[0].data_ptr()
    torch.testing.assert_close(stored, features[4:])
    cache.update(1, "b", torch.arange(2), features[:2])
    cache.update(0, "c", torch.arange(1), features[:1])
    assert cache.entries[1][0] == "b"
    assert cache.entries[0][0] == "c"
    with pytest.raises(ValueError, match="missing.*prefix"):
        cache.update(0, "d", torch.arange(9, 11), features[:2])


def test_anchor_context_hook_excludes_rejected_target_suffix(monkeypatch):
    import sys

    helper = load_module(
        "vllm/model_executor/models/mmuse/anchor_attention.py", "anchor_hook_test"
    )
    runtime = load_module("vllm/model_executor/models/mmuse_runtime.py", "anchor_hook")
    cache = helper.AnchorContextCache(4, torch.empty(3, 4, 16))
    states = torch.randn(6, 16)
    model = NS(
        model=NS(context_anchor=torch.zeros(2, 16), hidden_norm=lambda x: x),
        anchor_context_cache=cache,
    )
    obj = NS(
        model=model,
        _mmuse_anchor_active=True,
        hidden_states=states,
        target_input_buffers=NS(positions=torch.tensor([0, 1, 2, 0, 1, 2])),
    )
    batch = NS(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 3, 6]),
        idx_mapping_np=[2, 0],
        req_ids=["a", "b"],
    )
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.mmuse.anchor_attention", helper
    )
    runtime.MMuseGreedyRolloutMixin.prepare_context_anchor(
        obj, batch, torch.tensor([1, 0])
    )
    assert cache.entries[2][1].tolist() == [0, 1]
    torch.testing.assert_close(cache.entries[2][2], states[:2])
    torch.testing.assert_close(cache.entries[0][2], states[3:])
    assert len(model._anchor_contexts) == 2


def test_anchor_attention_rejects_prefix_cache_without_hidden_history():
    runtime = load_module("vllm/model_executor/models/mmuse_runtime.py", "prefix_check")
    config = configuration()
    config.cache_config = NS(enable_prefix_caching=True)
    with pytest.raises(ValueError, match="enable_prefix_caching=False"):
        runtime.validate_mmuse_config(config)


@pytest.mark.parametrize("dummy", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_anchor_proposal_context_is_scoped_and_dummy_uses_native(dummy, fail):
    runtime = load_module("vllm/model_executor/models/mmuse_runtime.py", "anchor_scope")
    seen = []

    class Parent:
        def propose(self, *args, **kwargs):
            seen.append(self._mmuse_anchor_active)
            self.model._anchor_contexts = object()
            if fail:
                raise RuntimeError("proposal failed")
            return "tokens"

    class Adapter(runtime.MMuseGreedyRolloutMixin, Parent):
        pass

    adapter = Adapter()
    adapter.draft_model_config = NS(
        hf_config=NS(
            target_hidden_state_format="deepseek_v4_mean_hc_head",
            layer_types=["sliding_attention"],
        )
    )
    adapter.model = NS()
    if fail:
        with pytest.raises(RuntimeError, match="proposal failed"):
            adapter.propose(dummy_run=dummy)
    else:
        assert adapter.propose(dummy_run=dummy) == "tokens"
    assert seen == [not dummy]
    assert adapter.model._anchor_contexts is None
    assert not adapter._mmuse_anchor_active


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_deepseek_dense_contract_accepts_native_tp_and_mixed_attention(tp):
    module = load_module(
        "vllm/model_executor/models/mmuse_runtime.py", "mmuse_validation"
    )
    module.validate_mmuse_config(configuration(tp))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("target_hidden_state_format", "standard", "hidden_state_format"),
        ("target_hidden_size", 8, "target_hidden_size"),
        ("draft_vocab_size", 16, "vocabulary"),
        ("aux_hidden_state_layer_ids", [1, 5], "teacher"),
        ("sample_from_anchor", False, "sample_from_anchor"),
        ("target_training_contract", {"format": "wrong"}, "training_contract"),
    ],
)
def test_deepseek_rejects_incompatible_training_contract(field, value, message):
    cfg = configuration()
    cfg.speculative_config.draft_model_config.hf_config.mmuse_source_config[field] = (
        value
    )
    module = load_module(
        "vllm/model_executor/models/mmuse_runtime.py", "mmuse_validation"
    )
    with pytest.raises(ValueError, match=message):
        module.validate_mmuse_config(cfg)


def test_config_conversion_selects_independent_deepseek_entry():
    algos = load_module(
        "vllm/transformers_utils/configs/speculators/algos.py", "mmuse_algos"
    )
    hf = configuration().speculative_config.draft_model_config.hf_config
    source = hf.mmuse_source_config
    converted = dict(source["transformer_layer_config"])
    algos.update_mmuse(source, converted)
    assert converted["architectures"] == ["DeepseekV4MMuseSpeculator"]
    assert converted["speculators_model_type"] == "mmuse"
    assert converted["mmuse_source_config"] == source
    assert converted["eagle_aux_hidden_state_layer_ids"] == [1, 3]


@pytest.mark.parametrize("mmuse_heads", [True, False])
def test_dspark_labelled_export_preserves_baseline_or_selects_mmuse(mmuse_heads):
    algos = load_module(
        "vllm/transformers_utils/configs/speculators/algos.py", "mmuse_legacy_algos"
    )
    source = dict(
        configuration().speculative_config.draft_model_config.hf_config.mmuse_source_config
    )
    source.update(
        speculators_model_type="dspark",
        architectures=["DSparkDraftModel"],
        enable_correction_head=mmuse_heads,
        dflash2_dynamic_conv=True,
        dflash2_candidate_selector=True,
    )
    converted = dict(source["transformer_layer_config"])
    algos.update_dspark(source, converted)
    assert source["speculators_model_type"] == "dspark"
    if mmuse_heads:
        assert converted["architectures"] == ["DeepseekV4MMuseSpeculator"]
        assert converted["mmuse_source_config"]["speculators_model_type"] == "dspark"
        assert converted["speculators_model_type"] == "mmuse"
    else:
        assert converted["architectures"] == ["Qwen3DSparkModel"]
        assert "mmuse_source_config" not in converted


def test_dspark_labelled_runtime_schema_uses_checkpoint_head_dim(monkeypatch):
    directory = ROOT / "vllm/model_executor/models/mmuse"
    package = ModuleType("mmuse_checkpoint_schema_test")
    package.__path__ = [str(directory)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    algos = load_module(
        "vllm/transformers_utils/configs/speculators/algos.py", "legacy_mmuse_algos"
    )
    monkeypatch.setitem(
        sys.modules, "vllm.transformers_utils.configs.speculators.algos", algos
    )
    module = load_module(
        "vllm/model_executor/models/mmuse/config.py",
        package.__name__ + ".config",
    )
    source = {
        "speculators_model_type": "dspark",
        "architectures": ["DSparkDraftModel"],
        "target_hidden_state_format": "deepseek_v4_mean_hc_head",
        "enable_correction_head": True,
        "dflash2_dynamic_conv": True,
        "dflash2_candidate_selector": True,
        "block_size": 7,
        "draft_vocab_size": 129280,
        "target_hidden_size": None,
        "aux_hidden_state_layer_ids": [1, 11, 21, 30, 40],
        "sample_from_anchor": True,
        "correction_hidden_size": 768,
        "correction_num_heads": 8,
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 4096,
            "intermediate_size": 9728,
            "head_dim": 256,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 5,
            "vocab_size": 129280,
            "layer_types": ["sliding_attention"] * 5,
            "sliding_window": 2048,
        },
    }
    parsed = module.MMuseRuntimeConfig.from_checkpoint(source)
    assert parsed.speculators_model_type == "mmuse"
    assert parsed.transformer_layer_config.head_dim == 256
    assert parsed.block_size == 7
    assert source["speculators_model_type"] == "dspark"


@pytest.mark.parametrize("raw", [True, False])
def test_raw_export_with_decoder_model_type_reaches_speculators_conversion(raw):
    path = ROOT / "vllm/transformers_utils/config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and any(
            isinstance(item, ast.Constant) and item.value == "mmuse"
            for item in ast.walk(node.test)
        )
        and any(
            isinstance(item, ast.Name) and item.id == "model_type"
            for child in node.body
            for item in ast.walk(child)
        )
    )
    config = {"model_type": "qwen3"}
    if raw:
        config.update(
            speculators_model_type="mmuse",
            transformer_layer_config={"model_type": "qwen3"},
        )
    namespace = {"config_dict": config, "model_type": "qwen3"}
    exec(
        compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    assert namespace["model_type"] == ("speculators" if raw else "qwen3")


@pytest.mark.parametrize("noncausal", [False, True])
def test_mmuse_causal_mask_follows_training_per_layer(noncausal):
    path = ROOT / "vllm/model_executor/models/qwen3_dflash.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_dflash_layer_causal"
    )
    namespace = {"Qwen3Config": object, "_SLIDING_ATTENTION": "sliding_attention"}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    cfg = NS(
        speculators_model_type="mmuse",
        layer_types=["sliding_attention", "full_attention"],
        sliding_window_non_causal=noncausal,
    )
    assert namespace[fn.name](cfg, 0) is (not noncausal)
    assert namespace[fn.name](cfg, 1) is False


@pytest.mark.parametrize("tp", [1, 2])
def test_target_io_binding_gathers_once_and_preserves_full_logits(tp):
    path = ROOT / "vllm/model_executor/models/deepseek_v4_mmuse.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("bind_target_weights", "compute_logits")
    ]
    full_embedding = torch.randn(32, 16)
    full_head = torch.randn(32, 16)
    local_embedding, local_head = full_embedding[: 32 // tp], full_head[: 32 // tp]
    gathered = []

    def gather(weight, dim):
        gathered.append(weight)
        return full_embedding if weight is local_embedding else full_head

    namespace = {
        "torch": torch,
        "nn": nn,
        "F": torch.nn.functional,
        "get_tensor_model_parallel_world_size": lambda: tp,
        "tensor_model_parallel_all_gather": gather,
    }
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    obj = NS(config=NS(vocab_size=32, hidden_size=16), model=NS())
    target = NS(
        model=NS(embed_tokens=NS(weight=local_embedding)), lm_head=NS(weight=local_head)
    )
    namespace["bind_target_weights"](obj, target)
    hidden = torch.randn(3, 16)
    torch.testing.assert_close(
        namespace["compute_logits"](obj, hidden), hidden @ full_head.T
    )
    assert len(gathered) == (2 if tp > 1 else 0)
    assert obj._target_weights_bound


def test_target_io_binding_rejects_packed_quantized_head():
    path = ROOT / "vllm/model_executor/models/deepseek_v4_mmuse.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "bind_target_weights"
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "get_tensor_model_parallel_world_size": lambda: 1,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    obj = NS(config=NS(vocab_size=32, hidden_size=16), model=NS())
    target = NS(
        model=NS(embed_tokens=NS(weight=torch.randn(32, 16))),
        lm_head=NS(weight=torch.zeros(32, 16, dtype=torch.int8)),
    )
    with pytest.raises(ValueError, match="floating lm_head"):
        namespace[method.name](obj, target)


@pytest.mark.parametrize("mmuse", [False, True])
def test_runner_does_not_request_mtp_residual_for_mmuse(mmuse):
    path = ROOT / "vllm/v1/worker/gpu/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and any(
            isinstance(item, ast.Constant)
            and item.value == "get_mtp_target_hidden_states"
            for item in ast.walk(node.test)
        )
    ]
    assert len(branches) == 2  # dummy warmup and real execution
    calls = []
    wide = torch.randn(3, 64)

    def get_mtp():
        calls.append(True)
        return wide

    for branch in branches:
        hidden = torch.randn(3, 16)
        namespace = {
            "self": NS(
                speculative_config=NS(use_mmuse=lambda: mmuse),
                model=NS(get_mtp_target_hidden_states=get_mtp),
            ),
            "hidden_states": hidden,
            "spec_hidden_states": hidden,
            "draft_hidden_states": hidden,
        }
        exec(
            compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        torch.testing.assert_close(
            namespace["spec_hidden_states"], hidden if mmuse else wide
        )
    assert len(calls) == (0 if mmuse else 2)

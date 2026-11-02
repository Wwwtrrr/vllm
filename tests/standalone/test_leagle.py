# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts; run with --confcutdir=tests/standalone without CUDA imports.

Set LEAGLE_TRAINING_ROOT to the looped_eagle checkout for real training parity.
"""

import copy
import importlib.util
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
        "vllm.v1.worker.gpu.spec_decode.speculator": {"DraftModelSpeculator": object},
    }
    for name, attributes in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return load_file(
        "leagle_runtime", "vllm/v1/worker/gpu/spec_decode/leagle/speculator.py"
    )


def test_proposer_uses_verified_prefix_and_bonus_anchor(runtime):
    proposer = runtime.LEagleSpeculator.__new__(runtime.LEagleSpeculator)
    proposer.model = make_model(architecture())
    proposer.device = torch.device("cpu")
    proposer.num_speculative_steps = 3
    proposer.max_model_len = 32
    proposer.draft_tokens = torch.empty((2, 3), dtype=torch.long)
    proposer.context = context_module.LEagleContext(
        2, 32, 16, dtype=torch.float32, device="cpu"
    )
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
        ("model_config.enforce_eager", False, "enforce-eager"),
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
def test_logits_match_real_training_code(
    layers, loops, norm, window, reduced, dtype, monkeypatch
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
    features, anchor = torch.randn(1, 9, 32, dtype=dtype), torch.tensor([[5]])
    expected, logits = training.propose(features, anchor, num_tokens=7)
    actual_logits = []
    handle = inference.lm_head.register_forward_hook(
        lambda _module, _args, output: actual_logits.append(output)
    )
    actual = inference.propose_context(inference.project_context(features), anchor, 7)
    handle.remove()
    torch.testing.assert_close(actual, expected)
    tolerance = {"atol": 2e-6, "rtol": 2e-5} if dtype == torch.float32 else {}
    torch.testing.assert_close(torch.cat(actual_logits, 1), logits, **tolerance)

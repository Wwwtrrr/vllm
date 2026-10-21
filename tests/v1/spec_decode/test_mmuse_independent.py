# SPDX-License-Identifier: Apache-2.0
"""CPU request-lifecycle contracts for independent DeepSeek MMuse proposals."""

import ast
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_cache():
    path = ROOT / "vllm/model_executor/models/mmuse/anchor_attention.py"
    spec = importlib.util.spec_from_file_location("independent_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AnchorContextCache


def independent_class():
    path = ROOT / "vllm/v1/worker/gpu/spec_decode/mmuse/deepseek.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))

    class Base:
        def _prepare_eplb_forward(self, count):
            self.last_query_count = count

    namespace = {"torch": torch, "DraftModelSpeculator": Base}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[cls.name]


def proposer(probabilistic=False):
    cls = independent_class()
    obj = cls.__new__(cls)
    obj.device, obj.dtype = torch.device("cpu"), torch.float32
    obj.hidden_size, obj.max_model_len, obj.num_query_per_req = 4, 16, 3
    obj.input_buffers = NS(
        input_ids=torch.zeros(12, dtype=torch.long),
        positions=torch.zeros(12, dtype=torch.long),
    )
    obj.draft_tokens = torch.zeros(4, 3, dtype=torch.long)
    obj.draft_logits = torch.zeros(4, 3, 32) if probabilistic else None
    obj.use_fp64_gumbel = False
    storage = torch.empty(4, 4, 4)

    class Model:
        config = NS(mask_token_id=31)
        _anchor_feature_storage = storage
        anchor_context_cache = load_cache()(4, storage)
        fail = False

        def combine_hidden_states(self, states):
            self.combined = states.clone()
            return states

        def __call__(self, ids, positions):
            if self.fail:
                raise RuntimeError("decoder failed")
            self.query_ids = ids.clone()
            self.query_positions = positions.clone()
            return ids[:, None].float().expand(-1, 4)

        def rollout_mmuse(self, hidden, anchors, sample_token=None):
            if sample_token is None:
                return anchors[:, None] + torch.arange(1, 4)[None], None
            result = []
            for position in range(3):
                logits = torch.zeros(hidden.shape[0], 32)
                logits[:, position + 4] = 10
                result.append(sample_token(logits, position))
            return torch.stack(result, 1), None

    obj.model = Model()
    obj.sample_calls = []

    def sample(logits, slots, temperature, seeds, positions, **kwargs):
        obj.sample_calls.append(
            (slots.clone(), positions.clone(), temperature[slots].clone())
        )
        kwargs["logits_cache"][slots, kwargs["logits_cache_col"]] = logits
        return logits.argmax(-1)

    obj._gumbel_sample = sample
    return obj


def inputs():
    return dict(
        input_batch=NS(
            num_reqs=2,
            req_ids=["a", "b"],
            idx_mapping_np=[1, 3],
            idx_mapping=torch.tensor([1, 3]),
            query_start_loc=torch.tensor([0, 3, 6]),
            positions=torch.tensor([0, 1, 2, 0, 1, 2]),
        ),
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=None,
        aux_hidden_states=[torch.arange(12).reshape(6, 2).float()] * 2,
        num_sampled=torch.tensor([1, 0]),
        num_rejected=torch.tensor([1, 0]),
        last_sampled=torch.tensor([90, 11, 92, 13]),
        next_prefill_tokens=torch.tensor([[20, 21, 22, 23], [24, 25, 26, 27]]),
        temperature=torch.tensor([0.0, 0.5, 0.0, 1.0]),
        seeds=torch.arange(4),
    )


def test_independent_blocks_align_anchor_mask_and_exclude_rejected_rows():
    obj = proposer()
    args = inputs()
    actual = obj.propose(**args)
    assert actual.tolist() == [[12, 13, 14], [24, 25, 26]]
    assert obj.model.query_ids.tolist() == [11, 31, 31, 23, 31, 31]
    assert obj.model.query_positions.tolist() == [2, 3, 4, 3, 4, 5]
    assert obj.model.anchor_context_cache.entries[1][1].tolist() == [0, 1]
    assert obj.model.anchor_context_cache.entries[3][1].tolist() == [0, 1, 2]
    assert obj.model._anchor_contexts is None
    # Next round reorders requests; histories stay with persistent slots/IDs.
    args["input_batch"] = NS(
        num_reqs=2,
        req_ids=["b", "a"],
        idx_mapping_np=[3, 1],
        idx_mapping=torch.tensor([3, 1]),
        query_start_loc=torch.tensor([0, 4, 8]),
        positions=torch.tensor([3, 4, 5, 6, 2, 3, 4, 5]),
    )
    args["aux_hidden_states"] = [torch.randn(8, 2), torch.randn(8, 2)]
    args["num_rejected"] = torch.tensor([2, 3])
    obj.propose(**args)
    assert obj.model.anchor_context_cache.entries[3][1].tolist() == [1, 2, 3, 4]
    assert obj.model.anchor_context_cache.entries[1][1].tolist() == [0, 1, 2]


def test_independent_probability_cache_uses_persistent_slots_and_source_positions():
    obj = proposer(probabilistic=True)
    actual = obj.propose(**inputs())
    assert actual.tolist() == [[4, 5, 6], [4, 5, 6]]
    for position, (slots, positions, temperatures) in enumerate(obj.sample_calls):
        assert slots.tolist() == [1, 3]
        assert positions.tolist() == [2 + position, 3 + position]
        assert temperatures.tolist() == [0.5, 1.0]
        assert (
            obj.draft_logits[[1, 3], position].argmax(-1).tolist() == [4 + position] * 2
        )
    assert not obj.draft_logits[0].any()


def test_independent_dummy_and_exception_do_not_leave_query_context():
    obj = proposer()
    args = inputs()
    obj.propose(**args, dummy_run=True)
    assert obj.model.anchor_context_cache.entries == {}
    assert obj.model._anchor_contexts is None
    obj.model.fail = True
    with pytest.raises(RuntimeError, match="decoder failed"):
        obj.propose(**args)
    assert obj.model._anchor_contexts is None


@pytest.mark.parametrize("ascend", [False, True])
@pytest.mark.parametrize("target", ["deepseek_v4", "qwen3"])
def test_method_mmuse_dispatches_independent_deepseek_only(monkeypatch, ascend, target):
    if ascend:
        path = ROOT.parent / "vllm-ascend/vllm_ascend/worker/v2/spec_decode/__init__.py"
        base = "vllm_ascend.worker.v2.spec_decode.mmuse"
        new_name, old_name = "AscendDeepseekV4MMuseSpeculator", "AscendMMuseSpeculator"
    else:
        path = ROOT / "vllm/v1/worker/gpu/spec_decode/__init__.py"
        base = "vllm.v1.worker.gpu.spec_decode.mmuse"
        new_name, old_name = "DeepseekV4MMuseSpeculator", "MMuseSpeculator"
    for suffix, name, label in (
        ("deepseek", new_name, "independent"),
        ("speculator", old_name, "legacy"),
    ):
        module = ModuleType(f"{base}.{suffix}")
        setattr(module, name, lambda config, device, result=label: result)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "init_speculator"
    )
    namespace = {"torch": torch, "VllmConfig": object, "copy": copy}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    config = NS(
        speculative_config=NS(use_mmuse=lambda: True, enforce_eager=False),
        model_config=NS(hf_config=NS(model_type=target)),
    )
    assert namespace["init_speculator"](config, torch.device("cpu")) == (
        "independent" if target == "deepseek_v4" else "legacy"
    )

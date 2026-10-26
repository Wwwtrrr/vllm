# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch contracts, without importing CUDA/NPU runtime dependencies."""

import ast
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import groupby, product
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def graph_api():
    root = Path(__file__).resolve().parents[3] / "vllm"
    config = ast.parse((root / "config/compilation.py").read_text(encoding="utf-8"))
    source = ast.parse(
        (root / "v1/worker/gpu/cudagraph_utils.py").read_text(encoding="utf-8")
    )
    enum_node = next(
        n
        for n in config.body
        if isinstance(n, ast.ClassDef) and n.name == "CUDAGraphMode"
    )
    descriptor = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "BatchExecutionDescriptor"
    )
    manager = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "CudaGraphManager"
    )
    compatible = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "_is_compatible"
    )
    methods = [
        n
        for n in manager.body
        if isinstance(n, ast.FunctionDef) and n.name in ("_init_candidates", "dispatch")
    ]
    minimal = ast.ClassDef(
        name="Manager", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    namespace = {
        "enum": NS(Enum=Enum),
        "dataclass": dataclass,
        "defaultdict": defaultdict,
        "product": product,
        "groupby": groupby,
        "round_up": lambda value, multiple: (
            (value + multiple - 1) // multiple * multiple
        ),
    }
    module = ast.Module(
        body=[enum_node, descriptor, compatible, minimal], type_ignores=[]
    )
    module.body.insert(
        0,
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
    )
    exec(
        compile(ast.fix_missing_locations(module), "target_graph_contract", "exec"),
        namespace,
    )
    return NS(**namespace)


def make_manager(api, sizes, query_len, max_reqs, specialize=True):
    manager = api.Manager()
    manager.cudagraph_mode = api.CUDAGraphMode.FULL
    manager.specialize_full_decode = specialize
    manager.decode_query_len = query_len
    manager.max_num_reqs = max_reqs
    manager.varlen_decode = False
    manager.lora_capture_cases = [0]
    manager._lora_dispatch_map = {0: 0}
    manager._max_lora_case = 0
    manager._resolve_effective_loras = lambda count: count
    manager._maybe_ubatch_twin = lambda desc: None
    manager.compilation_config = NS(
        cudagraph_capture_sizes=sizes, max_cudagraph_capture_size=max(sizes)
    )
    manager.vllm_config = NS(
        speculative_config=None, cache_config=NS(use_kda_recoverssm=False)
    )
    manager._capture_descs = {}
    manager._candidates = {}
    manager._graphs_captured = True
    manager._init_candidates()
    return manager


@pytest.mark.parametrize("query_len,sizes", [(1, [1, 2, 4, 8]), (8, [8, 16, 32])])
def test_full_decode_selects_uniform_graph_and_prefill_keeps_mixed_graph(
    graph_api, query_len, sizes
):
    manager = make_manager(graph_api, sizes, query_len, 4)
    decode = manager.dispatch(1, query_len, query_len, 0, max_query_len=query_len)
    assert decode.cg_mode == graph_api.CUDAGraphMode.FULL
    assert decode.uniform_token_count == query_len
    assert decode.num_reqs == 1

    # Same total size, but a prefill must not replay the decode-only kernel.
    prefill = manager.dispatch(1, query_len, None, 0, max_query_len=query_len)
    assert prefill.cg_mode == graph_api.CUDAGraphMode.FULL
    assert prefill.uniform_token_count is None


def test_full_decode_pads_requests_without_using_a_different_query_length(graph_api):
    manager = make_manager(graph_api, [8, 16, 32], 8, 4)
    decode = manager.dispatch(3, 24, 8, 0, max_query_len=8)
    assert (decode.num_tokens, decode.num_reqs, decode.uniform_token_count) == (
        32,
        4,
        8,
    )


def test_default_backend_retains_mixed_full_capture(graph_api):
    manager = make_manager(graph_api, [1, 2, 4, 8], 1, 4, specialize=False)
    assert all(
        d.uniform_token_count is None
        for d in manager._capture_descs[graph_api.CUDAGraphMode.FULL]
    )


def test_out_of_range_batch_still_falls_back(graph_api):
    manager = make_manager(graph_api, [8, 16, 32], 8, 4)
    assert (
        manager.dispatch(5, 40, 8, 0, max_query_len=8).cg_mode
        == graph_api.CUDAGraphMode.NONE
    )

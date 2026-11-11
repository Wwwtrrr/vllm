# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed buffers and native graph replay for LEAGLE's complete draft rollout."""

from bisect import bisect_left
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager


def capture_buckets(limit: int, minimum: int = 1) -> tuple[int, ...]:
    if limit < 1 or minimum < 1:
        raise ValueError("Graph buffer capacities must be positive")
    sizes, size = [], min(minimum, limit)
    while size < limit:
        sizes.append(size)
        size *= 2
    return (*sizes, limit)


@dataclass
class LEagleGraphInputs:
    context: torch.Tensor
    lengths: torch.Tensor
    anchor: torch.Tensor
    features: torch.Tensor
    projected: torch.Tensor

    @classmethod
    def allocate(cls, model, max_length, max_tokens, *, dtype, device):
        model.validate_static_rollout()
        return cls(
            torch.zeros(
                (1, max_length, model.fc.out_features), dtype=dtype, device=device
            ),
            torch.ones(1, dtype=torch.long, device=device),
            torch.zeros((1, 1), dtype=torch.long, device=device),
            torch.zeros((max_tokens, model.fc.in_features), dtype=dtype, device=device),
            torch.zeros(
                (max_tokens, model.fc.out_features), dtype=dtype, device=device
            ),
        )


class NativeLEagleGraphBackend:
    """Own a private pool and capture stream, independent of target graphs."""

    def __init__(self, api, device, graph_class):
        self.api, self.device, self.graph_class = api, device, graph_class
        self.pool = api.graph_pool_handle()
        self.stream = api.Stream(device=device)
        self.tp_group = None

    def capture(self, forward):
        current = self.api.current_stream(self.device)
        self.stream.wait_stream(current)
        communicator_capture = nullcontext()
        if self.tp_group is not None and self.device.type == "cuda":
            from vllm.distributed.parallel_state import GraphCaptureContext

            communicator_capture = self.tp_group.graph_capture(
                GraphCaptureContext(self.stream)
            )
        with self.api.stream(self.stream), communicator_capture:
            # Warm up lazy operator workspaces outside the capture region.
            for _ in range(2):
                forward()
            graph = self.graph_class()
            with self.api.graph(graph, pool=self.pool, stream=self.stream):
                output = forward()
        current.wait_stream(self.stream)
        return graph, output


class LEagleGraphRunner(CudaGraphManager):
    """Capture projection and rollout buckets; never lazily capture at inference.

    Inherits the runner's graph-manager lifecycle marker so memory profiling
    drops these managers/pools and recreates them before real capture. Inputs
    are owned by the speculator and allocated before memory profiling.
    """

    def __init__(self, model, inputs, num_tokens, backend):
        self.model, self.inputs = model, inputs
        self.num_tokens, self.backend = num_tokens, backend
        self.context_buckets = capture_buckets(inputs.context.shape[1], minimum=128)
        self.feature_buckets = capture_buckets(inputs.features.shape[0])
        self.rollout_graphs = {}
        self.projection_graphs = {}
        self.proposal_replays = 0
        self.projection_replays = 0

    def needs_capture(self):
        return len(self.rollout_graphs) != len(self.context_buckets) or len(
            self.projection_graphs
        ) != len(self.feature_buckets)

    @torch.inference_mode()
    def capture(self):
        if not self.needs_capture():
            return
        self.inputs.context.zero_()
        self.inputs.features.zero_()
        self.inputs.anchor.zero_()
        for size in reversed(self.feature_buckets):
            self.projection_graphs[size] = self.backend.capture(
                lambda size=size: self.model.project_context(
                    self.inputs.features[:size]
                )
            )
        for size in reversed(self.context_buckets):
            self.inputs.lengths.fill_(size)
            self.rollout_graphs[size] = self.backend.capture(
                lambda size=size: self.model.propose_static(
                    self.inputs.context[:, :size],
                    self.inputs.lengths,
                    self.inputs.anchor,
                    self.num_tokens,
                )
            )

    @staticmethod
    def _bucket(sizes, count):
        if count < 1 or count > sizes[-1]:
            raise ValueError("LEAGLE FULL input exceeds captured capacity")
        return sizes[bisect_left(sizes, count)]

    def _require_capture(self):
        if self.needs_capture():
            raise RuntimeError("LEAGLE FULL graphs were not captured before inference")

    @torch.inference_mode()
    def project_context(self, features):
        self._require_capture()
        count = features.shape[0]
        size = self._bucket(self.feature_buckets, count)
        self.inputs.features[:count].copy_(features)
        graph, output = self.projection_graphs[size]
        graph.replay()
        self.projection_replays += 1
        # Keep all request rows alive while rollout graphs reuse the private
        # graph pool. No caller may retain an aliased graph-pool output.
        self.inputs.projected[:count].copy_(output[:count])
        return self.inputs.projected[:count]

    @torch.inference_mode()
    def propose(self, context, anchor, num_tokens):
        self._require_capture()
        if not 0 < num_tokens <= self.num_tokens:
            raise ValueError("LEAGLE draft length exceeds captured proposal length")
        length = context.shape[1]
        size = self._bucket(self.context_buckets, length)
        self.inputs.context[:, :length].copy_(context)
        self.inputs.lengths.fill_(length)
        self.inputs.anchor.copy_(anchor)
        graph, output = self.rollout_graphs[size]
        graph.replay()
        self.proposal_replays += 1
        return output[:, :num_tokens]

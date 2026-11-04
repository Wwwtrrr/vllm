# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Incremental target KV and batched eager/FULL LEAGLE execution."""

from dataclasses import dataclass

import torch

from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.leagle.graph import capture_buckets


@dataclass(frozen=True)
class LEagleExecutionOptions:
    reuse_context_kv: bool = True
    static_draft_kv: bool = True
    batch_drafts: bool = True
    fused_projections: bool = True
    fused_rmsnorm: bool = True

    @classmethod
    def from_config(cls, config):
        return cls(
            **{
                name: getattr(config, "leagle_" + name, True)
                for name in cls.__dataclass_fields__
            }
        )

    @property
    def optimized_executor(self):
        return self.reuse_context_kv or self.static_draft_kv or self.batch_drafts


class LEagleExecution:
    def __init__(self, model, context, max_tokens, tokens, options, *, dtype, device):
        self.model, self.context, self.options = model, context, options
        self.tokens = tokens
        requests, self.capacity, width = context.values.shape
        self.batch_capacity = requests if options.batch_drafts else 1
        self.features = torch.zeros(
            max_tokens, model.fc.in_features, dtype=dtype, device=device
        )
        self.positions = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.projected = torch.zeros(max_tokens, width, dtype=dtype, device=device)
        self.lengths = torch.ones(self.batch_capacity, dtype=torch.long, device=device)
        self.anchor = torch.zeros(
            self.batch_capacity, 1, dtype=torch.long, device=device
        )
        self.slots = torch.zeros(self.batch_capacity, dtype=torch.long, device=device)

        def allocate_kv(batch, capacity):
            return tuple(
                tuple(
                    torch.zeros(
                        batch,
                        layer.self_attn.num_kv_heads,
                        capacity,
                        layer.self_attn.head_dim,
                        dtype=dtype,
                        device=device,
                    )
                    for _ in range(2)
                )
                for layer in model.layers
            )

        self.prefix_kv = (
            allocate_kv(requests, self.capacity) if options.reuse_context_kv else ()
        )
        self.new_kv = allocate_kv(1, max_tokens) if options.reuse_context_kv else ()
        self.workspace = (
            model.allocate_static_kv(
                self.batch_capacity, self.capacity, tokens, dtype=dtype, device=device
            )
            if options.static_draft_kv
            else None
        )
        if options.reuse_context_kv:
            model.validate_static_rollout()
        self.context_buckets = capture_buckets(self.capacity, minimum=128)
        self.batch_buckets = capture_buckets(self.batch_capacity)
        self.feature_buckets = capture_buckets(max_tokens)

    def projection_forward(self, size):
        projected = self.model.project_context(self.features[:size])
        kv = (
            self.model.prepare_context_kv(projected[None], self.positions[None, :size])
            if self.options.reuse_context_kv
            else ()
        )
        return projected, kv

    @torch.inference_mode()
    def project(self, features, positions, graphs=None):
        count = features.shape[0]
        if not 0 < count <= self.features.shape[0]:
            raise ValueError("LEAGLE projection exceeds token buffer capacity")
        self.features[:count].copy_(features)
        self.positions[:count].copy_(positions[:count])
        projected, kv = (
            graphs.project(count)
            if graphs is not None
            else self.projection_forward(count)
        )
        self.projected[:count].copy_(projected[:count])
        for destination, source in zip(self.new_kv, kv):
            for storage, value in zip(destination, source):
                storage[:, :, :count].copy_(value[:, :, :count])
        return self.projected[:count]

    @torch.inference_mode()
    def update(self, slot, request_id, start, begin, end):
        context = self.context.update(
            slot, request_id, start, self.projected[begin:end]
        )
        length = end - begin
        for target, source in zip(self.prefix_kv, self.new_kv):
            for storage, values in zip(target, source):
                storage[slot, :, start : start + length].copy_(values[0, :, begin:end])
        return context

    def rollout_forward(self, batch, capacity):
        lengths = self.lengths[:batch]
        slots = self.slots[:batch]
        state = self.context.values[slots, lengths - 1].unsqueeze(1)
        if self.options.reuse_context_kv:
            kv = tuple(
                tuple(t[:, :, :capacity].index_select(0, slots) for t in pair)
                for pair in self.prefix_kv
            )
        else:
            context = self.context.values[:, :capacity].index_select(0, slots)
            positions = torch.arange(capacity, device=context.device)[None].expand(
                batch, -1
            )
            kv = self.model.prepare_context_kv(context, positions)
        workspace = (
            self.workspace.view(batch, capacity, self.tokens)
            if self.workspace is not None
            else None
        )
        return self.model.propose_cached(
            state, kv, lengths, self.anchor[:batch], self.tokens, workspace
        )

    @torch.inference_mode()
    def stage(self, slots, anchors, batch, capacity):
        device = self.slots.device
        padded_slots = slots + [0] * (batch - len(slots))
        lengths = [self.context.lengths[slot] for slot in slots] + [1] * (
            batch - len(slots)
        )
        self.slots[:batch].copy_(torch.tensor(padded_slots, device=device))
        self.lengths[:batch].copy_(torch.tensor(lengths, device=device))
        self.anchor[:batch].copy_(anchors.index_select(0, self.slots[:batch]))

    @torch.inference_mode()
    def propose(self, active, anchors, result, count, graphs=None):
        if not active or count == 0:
            return result
        if not 0 < count <= self.tokens:
            raise ValueError("LEAGLE proposal exceeds configured draft length")
        size = self.batch_capacity
        for begin in range(0, len(active), size):
            group = active[begin : begin + size]
            slots = [slot for _, slot in group]
            batch = LEagleOptimizedGraphRunner.bucket(self.batch_buckets, len(slots))
            capacity = LEagleOptimizedGraphRunner.bucket(
                self.context_buckets, max(self.context.lengths[slot] for slot in slots)
            )
            self.stage(slots, anchors, batch, capacity)
            tokens = (
                graphs.rollout(batch, capacity)
                if graphs is not None
                else self.rollout_forward(batch, capacity)
            )
            for index, (row, _) in enumerate(group):
                result[row, :count].copy_(tokens[index, :count])
        return result

    @torch.inference_mode()
    def profile(self):
        projected, kv = self.projection_forward(self.features.shape[0])
        self.rollout_forward(self.batch_capacity, self.capacity)
        del projected, kv


class LEagleOptimizedGraphRunner(CudaGraphManager):
    """Framework lifecycle marker releases profiling graphs, retaining inputs."""

    def __init__(self, execution, backend):
        self.execution, self.backend = execution, backend
        self.projections, self.rollouts = {}, {}
        self.projection_replays = self.proposal_replays = 0

    @staticmethod
    def bucket(sizes, count):
        if count < 1 or count > sizes[-1]:
            raise ValueError("LEAGLE input exceeds captured buffer capacity")
        return next(size for size in sizes if size >= count)

    def needs_capture(self):
        state = self.execution
        return len(self.projections) != len(state.feature_buckets) or len(
            self.rollouts
        ) != (len(state.batch_buckets) * len(state.context_buckets))

    @torch.inference_mode()
    def capture(self):
        if not self.needs_capture():
            return
        state = self.execution
        for size in reversed(state.feature_buckets):
            self.projections[size] = self.backend.capture(
                lambda size=size: state.projection_forward(size)
            )
        for batch in reversed(state.batch_buckets):
            for capacity in reversed(state.context_buckets):
                state.lengths[:batch].fill_(capacity)
                self.rollouts[batch, capacity] = self.backend.capture(
                    lambda batch=batch, capacity=capacity: state.rollout_forward(
                        batch, capacity
                    )
                )

    def _ready(self):
        if self.needs_capture():
            raise RuntimeError("Optimized LEAGLE graphs were not captured")

    def project(self, count):
        self._ready()
        size = self.bucket(self.execution.feature_buckets, count)
        graph, output = self.projections[size]
        graph.replay()
        self.projection_replays += 1
        return output

    def rollout(self, batch, capacity):
        self._ready()
        graph, output = self.rollouts[batch, capacity]
        graph.replay()
        self.proposal_replays += 1
        return output

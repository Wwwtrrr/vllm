# SPDX-License-Identifier: Apache-2.0
"""Capture row-local context normalization, K/V projection and rotary application.

History/cache decisions and committing valid rows stay outside this preparation
graph. Persistent outputs are staging buffers, never a per-request KV history.
"""

import torch

from .deepseek_features import DeepseekMMuseFeatureGraphs


class DeepseekMMuseContextGraphs(DeepseekMMuseFeatureGraphs):
    def __init__(self, model, capacity, backend):
        self.model, self.backend = model, backend
        storage = model.feature_storage
        self.inputs = storage.new_zeros(capacity, storage.shape[-1])
        self.positions = torch.zeros(capacity, dtype=torch.int64, device=storage.device)
        self.output = torch.zeros_like(self.inputs)
        self.kv = tuple(
            storage.new_zeros(
                2, capacity, layer.self_attn.kv_heads, layer.self_attn.dim
            )
            for layer in model.layers
        )
        sizes, size = [], 1
        while size < capacity:
            sizes.append(size)
            size *= 2
        self.sizes = (*sizes, capacity)
        self.graphs = {}
        self.pool = None

    def forward(self, size):
        self.output[:size].copy_(self.model.hidden_norm(self.inputs[:size]))
        for layer, storage in zip(self.model.layers, self.kv):
            key, value = layer.self_attn.project_kv(
                self.output[:size], self.positions[:size]
            )
            storage[0, :size].copy_(key)
            storage[1, :size].copy_(value)

    @torch.inference_mode()
    def project(self, features, positions, graphs):
        rows = features.shape[0]
        if (
            not graphs
            or not 0 < rows <= self.inputs.shape[0]
            or features.shape != (rows, self.inputs.shape[1])
            or features.dtype != self.inputs.dtype
            or features.device != self.inputs.device
            or positions.shape != (rows,)
            or positions.dtype != self.positions.dtype
            or positions.device != self.positions.device
        ):
            return None
        size = next(size for size in self.sizes if size >= rows)
        self.inputs[:rows].copy_(features)
        self.inputs[rows:size].zero_()
        self.positions[:rows].copy_(positions)
        self.positions[rows:size].fill_(-1)
        graphs[size].replay()
        return self.output[:rows], tuple(storage[:, :rows] for storage in self.kv)

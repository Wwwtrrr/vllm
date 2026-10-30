# SPDX-License-Identifier: Apache-2.0
"""Row-bucketed native graphs for DeepSeek MMuse teacher feature projection.

Only row-local feature alignment/fusion is captured. Request history and context
K/V ownership remain outside. Input/output buffers exist before memory profiling;
graph handles are transferred to the runner-owned graph manager after capture.
"""

import torch


class DeepseekMMuseFeatureGraphs:
    def __init__(self, model, num_layers, hidden_size, capacity, backend):
        self.model, self.backend = model, backend
        storage = model.feature_storage
        self.inputs = storage.new_zeros(num_layers, capacity, hidden_size)
        self.output = storage.new_zeros(capacity, model.config.hidden_size)
        sizes, size = [], 1
        while size < capacity:
            sizes.append(size)
            size *= 2
        self.sizes = (*sizes, capacity)
        self.graphs = {}
        self.pool = None

    def forward(self, size):
        features = torch.cat(tuple(self.inputs[:, :size].unbind(0)), -1)
        self.output[:size].copy_(self.model.project_features(features))

    @torch.inference_mode()
    def capture(self):
        if self.graphs:
            raise RuntimeError("MMuse feature graphs were already captured")
        self.inputs.zero_()
        self.pool = self.backend.graph_pool_handle()
        stream = self.backend.Stream()
        stream.wait_stream(self.backend.current_stream())
        with self.backend.stream(stream):
            for size in reversed(self.sizes):
                for _ in range(3):
                    self.forward(size)
                self.backend.synchronize()
                graph = (
                    self.backend.CUDAGraph()
                    if hasattr(self.backend, "CUDAGraph")
                    else self.backend.NPUGraph()
                )
                with self.backend.graph(graph, pool=self.pool):
                    self.forward(size)
                self.graphs[size] = graph
        self.backend.current_stream().wait_stream(stream)
        self.backend.synchronize()

    @torch.inference_mode()
    def project(self, states, graphs):
        if not graphs or len(states) != self.inputs.shape[0]:
            return None
        rows = states[0].shape[0]
        if not 0 < rows <= self.inputs.shape[1] or any(
            state.shape != (rows, self.inputs.shape[2])
            or state.dtype != self.inputs.dtype
            or state.device != self.inputs.device
            for state in states
        ):
            return None
        size = next(size for size in self.sizes if size >= rows)
        for index, state in enumerate(states):
            self.inputs[index, :rows].copy_(state)
        self.inputs[:, rows:size].zero_()
        graphs[size].replay()
        # Callers consume/copy these rows before the next proposal overwrites
        # the shared output. Padded rows must never enter verified history.
        return self.output[:rows]

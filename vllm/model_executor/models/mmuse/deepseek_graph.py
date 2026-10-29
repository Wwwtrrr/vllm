# SPDX-License-Identifier: Apache-2.0
"""Fixed-context native graphs for the replicated DeepSeek MMuse decoder.

History ownership and context K/V projection stay outside capture. Each graph
contains one complete query block and rollout; requests replay serially.
"""

import torch


class DeepseekMMuseGraphs:
    def __init__(
        self,
        model,
        block_size,
        sample,
        logits_cache,
        temperature,
        seeds,
        use_fp64,
        backend,
    ):
        self.model, self.block_size = model, block_size
        self.sample, self.logits_cache = sample, logits_cache
        self.temperature, self.seeds = temperature, seeds
        self.use_fp64, self.backend = use_fp64, backend
        storage = model.feature_storage
        capacity, width = storage.shape[1:]
        device = storage.device
        self.context = storage.new_zeros(capacity, width)
        self.context_positions = torch.full(
            (capacity,), -1, dtype=torch.int64, device=device
        )
        self.positions = torch.zeros(block_size, dtype=torch.int64, device=device)
        self.input_ids = torch.full(
            (block_size,), model.config.mask_token_id, dtype=torch.int64, device=device
        )
        self.anchor_index = torch.zeros(1, dtype=torch.int64, device=device)
        self.request_slot = torch.full((1,), -1, dtype=torch.int32, device=device)
        self.columns = tuple(
            torch.full((1,), i, dtype=torch.int32, device=device)
            for i in range(block_size)
        )
        self.output = torch.zeros(block_size, dtype=torch.int64, device=device)
        self.context_kv = tuple(
            storage.new_zeros(
                2, capacity, layer.self_attn.kv_heads, layer.self_attn.dim
            )
            for layer in model.layers
        )
        sizes, size = [], min(128, capacity)
        while size < capacity:
            sizes.append(size)
            size *= 2
        self.sizes = (*sizes, capacity)
        self.graphs = {}
        self.pool = None
        self.context_entry = None
        self.query_offsets = torch.arange(block_size, device=device)

    def bucket(self, length):
        if not 0 < length <= self.context.shape[0]:
            raise ValueError("DeepSeek MMuse context exceeds its reserved capacity")
        return next(size for size in self.sizes if size >= length)

    def sample_token(self, logits, column):
        return self.sample(
            logits.to(self.logits_cache.dtype).contiguous(),
            self.request_slot,
            self.temperature,
            self.seeds,
            self.positions[column : column + 1],
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.logits_cache,
            logits_cache_col=self.columns[column],
            use_fp64=self.use_fp64,
        )

    def forward(self, size):
        hidden = self.model.forward_fixed_context(
            self.input_ids,
            self.positions,
            self.context_positions[:size],
            self.context[:size],
            tuple(kv[:, :size] for kv in self.context_kv),
            self.anchor_index,
        )
        tokens, _ = self.model.runtime.rollout_correction(
            hidden.unsqueeze(0),
            self.input_ids[:1],
            temperature=0.0,
            sample_token=self.sample_token if self.logits_cache is not None else None,
        )
        self.output.copy_(tokens[0])

    def prepare_platform_context(self, size, start=0):
        for layer, storage in zip(self.model.layers, self.context_kv):
            paged = layer.self_attn.paged_attention_impl
            if paged is not None:
                paged.prepare_context(storage[0, :size], storage[1, :size], start=start)

    @torch.inference_mode()
    def prepare(
        self,
        context_positions,
        features,
        anchor,
        slot,
        *,
        request_id=None,
        start=None,
        end=None,
        update_start=None,
    ):
        length = features.shape[0]
        size = self.bucket(length)
        known_span = (
            request_id is not None
            and start is not None
            and end is not None
            and update_start is not None
        )
        if known_span and (end - start != length or not start <= update_start <= end):
            raise ValueError("Invalid DeepSeek MMuse context update span")
        reused, shift = 0, 0
        previous = self.context_entry
        # Invalidate before launching writes; a failed prepare must never leave
        # cache metadata claiming a partially overwritten context is reusable.
        self.context_entry = None
        if known_span and previous is not None:
            old_slot, old_request, old_start, old_end = previous
            if (slot, request_id) == (
                old_slot,
                old_request,
            ) and old_start <= start < old_end:
                reused = max(0, min(old_end, update_start, end) - start)
                shift = start - old_start
        if reused and shift:
            self.context[:reused].copy_(self.context[shift : shift + reused].clone())
            for storage in self.context_kv:
                storage[:, :reused].copy_(storage[:, shift : shift + reused].clone())
        if reused < length:
            self.context[reused:length].copy_(self.model.hidden_norm(features[reused:]))
        self.context[length:size].zero_()
        self.context_positions[:size].fill_(-1)
        self.context_positions[:length].copy_(context_positions)
        self.anchor_index.fill_(length - 1)
        self.input_ids.fill_(self.model.config.mask_token_id)
        self.input_ids[:1].copy_(anchor.reshape(1))
        self.positions.copy_(context_positions[-1] + 1 + self.query_offsets)
        self.request_slot.fill_(slot)
        for layer, storage in zip(self.model.layers, self.context_kv):
            if reused < length:
                key, value = layer.self_attn.project_kv(
                    self.context[reused:length], self.context_positions[reused:length]
                )
                storage[0, reused:length].copy_(key)
                storage[1, reused:length].copy_(value)
            storage[:, length:size].zero_()
        self.prepare_platform_context(size, start=0 if shift else reused)
        if known_span:
            self.context_entry = (slot, request_id, start, end)
        return size

    @torch.inference_mode()
    def capture(self):
        if self.graphs:
            raise RuntimeError("DeepSeek MMuse graphs were already captured")
        self.pool = self.backend.graph_pool_handle()
        self.context_entry = None
        for layer in self.model.layers:
            prepare = getattr(
                layer.self_attn.paged_attention_impl, "prepare_for_capture", None
            )
            if prepare is not None:
                prepare()
        self.request_slot.fill_(-1)
        self.context.zero_()
        self.context_positions.fill_(-1)
        self.context_positions[0] = 0
        self.anchor_index.zero_()
        for kv in self.context_kv:
            kv.zero_()
        self.positions.copy_(
            torch.arange(1, self.block_size + 1, device=self.positions.device)
        )
        stream = self.backend.Stream()
        stream.wait_stream(self.backend.current_stream())
        with self.backend.stream(stream):
            for size in reversed(self.sizes):
                self.prepare_platform_context(size)
                # Warm kernels before entering native capture.
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

    def replay(self, size):
        self.graphs[size].replay()

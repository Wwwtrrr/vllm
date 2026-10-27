# SPDX-License-Identifier: Apache-2.0
"""Confirmed feature history and optional per-request context KV storage."""

import torch


class VerifiedPrefix:
    """Own projected confirmed rows; never store synthetic query states."""

    def __init__(self, capacity, storage):
        self.capacity = capacity
        self.storage = storage
        self.requests = {}
        self.spans = {}

    def append(self, slot, request_id, positions, values, *, positions_cpu=None):
        # FULL callers transfer all request metadata once and supply the CPU
        # positions from that same snapshot. Eager callers keep validation here.
        if positions_cpu is None:
            first, end = positions[0].item(), positions[-1].item() + 1
            contiguous = torch.equal(
                positions, torch.arange(first, end, device=positions.device)
            )
        else:
            if len(positions_cpu) != positions.numel() or not positions_cpu:
                raise ValueError("MMuse target positions must be nonempty")
            first, end = positions_cpu[0], positions_cpu[-1] + 1
            contiguous = positions_cpu == list(range(first, end))
        if not contiguous:
            raise ValueError("MMuse target positions must be contiguous")
        previous = self.requests.get(slot)
        keep_count = 0
        if first == 0:
            old_positions, old_values = positions[:0], values[:0]
        else:
            if previous is None or previous[0] != request_id:
                raise ValueError("MMuse request is missing its verified prefix")
            _, old_positions, old_values = previous
            _, old_start, old_end, _ = self.spans[slot]
            keep_count = min(old_end - old_start, max(0, first - old_start))
            if not keep_count or old_start + keep_count != first:
                raise ValueError("MMuse verified prefix has a gap")
        length = min(self.capacity, keep_count + values.shape[0])
        evict = max(0, keep_count + values.shape[0] - self.capacity)
        retained = max(0, keep_count - evict)
        incoming = values[max(0, evict - keep_count) :]
        if (
            incoming.untyped_storage().data_ptr()
            == self.storage.untyped_storage().data_ptr()
        ):
            incoming = incoming.clone()
        owned = self.storage[slot, :length]
        if retained and evict:
            owned[:retained].copy_(old_values[evict:keep_count].clone())
        owned[retained:].copy_(incoming)
        joined_positions = torch.arange(
            end - length, end, device=positions.device, dtype=positions.dtype
        )
        self.requests[slot] = (request_id, joined_positions, owned)
        self.spans[slot] = (request_id, end - length, end, first)
        return joined_positions, owned


class ContextKVCache:
    """Bounded per-slot context-only KV; synthetic query KV is never committed.

    Storage is [layers, request slots, K/V, capacity, KV heads, head dim].
    The caller supplies a contiguous confirmed prefix and the first position
    updated by the current Target step. An overwritten suffix is reprojected.
    """

    def __init__(self, storage):
        self.storage = storage
        self.entries = {}

    def begin(self, slot, request_id, start, end, update_start):
        capacity = self.storage.shape[3]
        if not 0 <= start < end or end - start > capacity:
            raise ValueError("MMuse context KV span exceeds its reserved capacity")
        previous = self.entries.pop(slot, None)
        reused = 0
        if previous is not None:
            old_request, old_start, old_end = previous
            if old_request == request_id and old_start <= start < old_end:
                reused = max(0, min(old_end, update_start, end) - start)
                shift = start - old_start
                if reused and shift:
                    # copy_ cannot safely move overlapping storage slices.
                    self.storage[:, slot, :, :reused].copy_(
                        self.storage[:, slot, :, shift : shift + reused].clone()
                    )
        # Metadata stays invalid until every layer has completed successfully.
        return reused

    def commit(self, slot, request_id, start, end):
        self.entries[slot] = (request_id, start, end)

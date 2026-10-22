# SPDX-License-Identifier: Apache-2.0
"""Confirmed feature history and optional per-request context KV storage."""

import torch


class VerifiedPrefix:
    """Own projected confirmed rows; never store synthetic query states."""

    def __init__(self, capacity, storage):
        self.capacity = capacity
        self.storage = storage
        self.requests = {}

    def append(self, slot, request_id, positions, values):
        first, end = positions[0].item(), positions[-1].item() + 1
        if not torch.equal(
            positions, torch.arange(first, end, device=positions.device)
        ):
            raise ValueError("MMuse target positions must be contiguous")
        previous = self.requests.get(slot)
        if first == 0:
            old_positions, old_values = positions[:0], values[:0]
        else:
            if previous is None or previous[0] != request_id:
                raise ValueError("MMuse request is missing its verified prefix")
            _, old_positions, old_values = previous
            keep = old_positions < first
            old_positions, old_values = old_positions[keep], old_values[keep]
            if not old_positions.numel() or old_positions[-1].item() != first - 1:
                raise ValueError("MMuse verified prefix has a gap")
        joined = torch.cat((old_values, values))[-self.capacity :]
        joined_positions = torch.cat((old_positions, positions))[-self.capacity :]
        owned = self.storage[slot, : joined.shape[0]]
        owned.copy_(joined)
        self.requests[slot] = (request_id, joined_positions, owned)
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

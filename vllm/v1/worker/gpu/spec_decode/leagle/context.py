# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded projected target prefixes, indexed by model-runner request slots."""

import torch


class LEagleContext:
    def __init__(self, max_requests, max_length, width, *, dtype, device):
        self.values = torch.zeros(
            max_requests, max_length, width, dtype=dtype, device=device
        )
        self.request_ids: list[str | None] = [None] * max_requests
        self.lengths = [0] * max_requests

    def update(self, slot, request_id, start, features):
        """Replace the verified suffix, discarding rejected positions."""
        if self.request_ids[slot] != request_id or start == 0:
            self.request_ids[slot] = request_id
            self.lengths[slot] = 0
        if start < 0 or start > self.lengths[slot]:
            raise RuntimeError(
                "LEAGLE is missing target context; disable prefix caching and "
                "remote KV loading and recompute the complete prefix"
            )
        end = start + features.shape[0]
        if end > self.values.shape[1]:
            raise ValueError("LEAGLE context exceeds max_model_len")
        self.values[slot, start:end].copy_(features)
        self.lengths[slot] = end
        return self.values[slot : slot + 1, :end]

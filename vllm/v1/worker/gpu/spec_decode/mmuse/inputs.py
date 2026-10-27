# SPDX-License-Identifier: Apache-2.0
"""Single-device MMuse query layout, independent of DFlash sampling buffers."""

import torch


def prepare_mmuse_inputs(
    buffers,
    batch,
    num_sampled,
    num_rejected,
    last_sampled,
    next_prefill_tokens,
    context_positions,
    context_slots,
    query_slots,
    block_table,
    block_size,
    block_length,
    mask_token_id,
    max_model_len,
):
    """Publish confirmed context and K queries per request into stable buffers.

    Rejected suffixes and null cache blocks must never write K/V. State slots
    are indexed by idx_mapping, whereas cache tables are in batch order.
    """
    count, total = batch.num_reqs, batch.num_tokens
    ends = batch.query_start_loc[1 : count + 1].long() - num_rejected[:count]
    state = batch.idx_mapping[:count].long()
    last_positions = batch.positions[(ends - 1).clamp_min(0).long()]
    offsets = torch.arange(block_length, device=last_positions.device)
    positions = last_positions[:, None] + 1 + offsets
    # RequestState stores last_sampled_tokens as [capacity, 1]. Flatten
    # before gathering, otherwise where([N], [N, 1], [N]) broadcasts to [N, N].
    sampled = last_sampled.reshape(-1)
    prefill = next_prefill_tokens.reshape(-1, sampled.numel())[0]
    anchors = torch.where(num_sampled[:count] > 0, sampled[state], prefill[state])
    ids = buffers.input_ids[: count * block_length].view(count, block_length)
    ids.fill_(mask_token_id)
    ids[:, 0].copy_(anchors)
    buffers.input_ids[count * block_length :].zero_()
    buffers.positions[count * block_length :].zero_()
    buffers.positions[: count * block_length].copy_(
        positions.clamp_max(max_model_len - 1).reshape(-1)
    )
    buffers.seq_lens.zero_()
    buffers.seq_lens[:count].copy_(
        (last_positions + 1 + block_length).clamp_max(max_model_len)
    )
    buffers.query_start_loc.fill_(count * block_length)
    buffers.query_start_loc[: count + 1].copy_(
        torch.arange(count + 1, device=positions.device) * block_length
    )

    rows = torch.arange(total, device=positions.device)
    owners = torch.bucketize(
        rows, batch.query_start_loc[1 : count + 1].long(), right=True
    )
    valid = rows < ends[owners]
    context_positions[:total].copy_(torch.where(valid, batch.positions[:total], 0))

    def slots(pos, request):
        blocks = (pos // block_size).clamp(0, block_table.shape[1] - 1).long()
        physical = block_table[request, blocks].long()
        return torch.where(physical != 0, physical * block_size + pos % block_size, -1)

    context_slots.fill_(-1)
    context_slots[:total].copy_(
        torch.where(valid, slots(context_positions[:total], owners), -1)
    )
    query_slots.fill_(-1)
    request = torch.arange(count, device=positions.device)[:, None]
    query_slots[: count * block_length].copy_(slots(positions, request).reshape(-1))

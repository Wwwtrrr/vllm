# SPDX-License-Identifier: Apache-2.0
"""Eager anchor-relative MMuse attention, independent of paged SWA kernels."""

import torch
from torch.nn import functional as F


class AnchorContextCache:
    """Keep only verified target features, keyed by persistent request slot + ID."""

    def __init__(self, window, storage=None):
        self.window = window
        self.storage = storage
        self.entries = {}

    def update(self, slot, request_id, positions, features):
        ids = positions.tolist()
        if not ids or ids != list(range(ids[0], ids[0] + len(ids))):
            raise ValueError(
                "MMuse anchor context requires contiguous target positions"
            )
        old = self.entries.get(slot)
        if old is None or old[0] != request_id or ids[0] == 0:
            if ids[0] != 0:
                raise ValueError(
                    "MMuse anchor context is missing the request prefix; "
                    "disable prefix caching and restart the request"
                )
            retained_positions, retained_features = positions[:0], features[:0]
        else:
            _, retained_positions, retained_features = old
            # Recomputed/overlapping rows replace existing rows. Never retain a
            # rejected query suffix, which is not a verified target context.
            keep = retained_positions < ids[0]
            retained_positions = retained_positions[keep]
            retained_features = retained_features[keep]
            if (
                not retained_positions.numel()
                or int(retained_positions[-1]) != ids[0] - 1
            ):
                raise ValueError("MMuse anchor context has a gap after scheduling")
        positions = torch.cat((retained_positions, positions))
        features = torch.cat((retained_features, features))
        if self.window is not None:
            positions = positions[-self.window :]
            features = features[-self.window :]
        # Own storage: input buffers are overwritten on the next target step.
        if self.storage is None:
            owned_features = features.clone()
        else:
            owned_features = self.storage[slot, : features.shape[0]]
            if owned_features.shape != features.shape:
                raise ValueError("MMuse verified context exceeds its reserved capacity")
            owned_features.copy_(features)
        entry = (request_id, positions.clone(), owned_features)
        self.entries[slot] = entry
        return entry[1:]


def anchor_mask(context_positions, query_positions, window, causal):
    anchor = query_positions[0]
    visible_context = context_positions < anchor
    if window is not None:
        visible_context &= context_positions >= anchor - window
    context = visible_context[None].expand(query_positions.numel(), -1)
    query = torch.ones(
        query_positions.numel(),
        query_positions.numel(),
        dtype=torch.bool,
        device=query_positions.device,
    )
    if causal:
        query = query.tril()
    return torch.cat((context, query), dim=-1)


def anchor_attention(attention, hidden, context, context_positions, query_positions):
    """Use native TP projections/RoPE, with the training anchor-block mask."""
    qkv, _ = attention.qkv_proj(hidden)
    q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], -1)
    context_qkv, _ = attention.qkv_proj(context)
    context_q, context_k, context_v = context_qkv.split(
        [attention.q_size, attention.kv_size, attention.kv_size], -1
    )
    head_dim = attention.head_dim
    q = attention.q_norm(q.reshape(q.shape[0], -1, head_dim)).flatten(1)
    k = attention.k_norm(k.reshape(k.shape[0], -1, head_dim)).flatten(1)
    context_k = attention.k_norm(
        context_k.reshape(context_k.shape[0], -1, head_dim)
    ).flatten(1)
    q, k = attention.rotary_emb(query_positions, q, k)
    _, context_k = attention.rotary_emb(context_positions, context_q, context_k)
    keys = torch.cat((context_k, k)).reshape(-1, attention.num_kv_heads, head_dim)
    values = torch.cat((context_v, v)).reshape(-1, attention.num_kv_heads, head_dim)
    if attention.v_scale is not None:
        values = values * attention.v_scale
    groups = attention.num_heads // attention.num_kv_heads
    keys = keys.transpose(0, 1).repeat_interleave(groups, 0)
    values = values.transpose(0, 1).repeat_interleave(groups, 0)
    queries = q.reshape(-1, attention.num_heads, head_dim).transpose(0, 1)
    mask = anchor_mask(
        context_positions, query_positions, attention.sliding_window, attention.causal
    )
    result = (
        F.scaled_dot_product_attention(
            queries[None],
            keys[None],
            values[None],
            attn_mask=mask[None, None],
            scale=attention.scaling,
        )[0]
        .transpose(0, 1)
        .reshape(hidden.shape[0], -1)
    )
    result, _ = attention.o_proj(result)
    return result


def forward_anchor_blocks(model, input_ids, positions, contexts):
    """Execute trained MMuse decoder order without paged draft-cache eviction."""
    runtime = model._runtime_ref()
    block = model.config.block_size
    output = []
    for row, (context_positions, projected_context) in enumerate(contexts):
        start = row * block
        query_positions = positions[start : start + block]
        hidden = model.embed_input_ids(input_ids[start : start + block])
        if runtime.block_position_embedding is not None:
            slots = torch.arange(block, device=hidden.device)
            hidden = hidden + runtime.block_position_embedding(slots)
        context = model.hidden_norm(projected_context)
        if runtime.context_hidden_proj is not None:
            residual = runtime.context_hidden_proj(context[-1:])
            hidden = hidden + torch.tanh(runtime.context_hidden_gate) * residual
        for layer in model.layers:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            coefficients = None
            if layer.attention_conv is not None:
                hidden, coefficients = layer.attention_conv.prepare(hidden)
            hidden = anchor_attention(
                layer.self_attn, hidden, context, context_positions, query_positions
            )
            if layer.attention_conv is not None:
                hidden = layer.attention_conv.finish(hidden, coefficients)
            hidden = residual + hidden
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            if layer.mlp_conv is not None:
                hidden, coefficients = layer.mlp_conv.prepare(hidden)
            hidden = layer.mlp(hidden)
            if layer.mlp_conv is not None:
                hidden = layer.mlp_conv.finish(hidden, coefficients)
            hidden = residual + hidden
        output.append(model.norm(hidden))
    result = torch.zeros(
        input_ids.shape[0],
        model.config.hidden_size,
        dtype=output[0].dtype,
        device=output[0].device,
    )
    result[: len(contexts) * block] = torch.cat(output)
    return result

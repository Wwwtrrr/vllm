# SPDX-License-Identifier: Apache-2.0
"""Replicated, stateless block decoder matching MMuse's offline evaluator.

No AttentionLayerBase modules or paged draft KV are registered. Target TP is
independent: every rank executes the complete dense drafter and its heads.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .dynamic_conv import DFlash2GroupedConv


class RMS(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)).to(
            dtype
        ) * self.weight


def rotate(x, positions, theta):
    dim = x.shape[-1]
    inverse = theta ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    angles = positions.float()[:, None] * inverse[None]
    angles = torch.cat((angles, angles), -1)[:, None]
    left, right = x.chunk(2, -1)
    return x * angles.cos().to(x.dtype) + torch.cat(
        (-right, left), -1
    ) * angles.sin().to(x.dtype)


class BlockAttention(nn.Module):
    def __init__(self, config, window, noncausal):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = config.head_dim or config.hidden_size // self.heads
        self.window = window
        self.causal = window is not None and not noncausal
        rope = getattr(config, "rope_parameters", None) or {}
        self.theta = rope.get("rope_theta", getattr(config, "rope_theta", 1000000.0))
        for name, width in (
            ("q_proj", self.heads * self.dim),
            ("k_proj", self.kv_heads * self.dim),
            ("v_proj", self.kv_heads * self.dim),
        ):
            setattr(
                self,
                name,
                nn.Linear(config.hidden_size, width, bias=config.attention_bias),
            )
        self.o_proj = nn.Linear(
            self.heads * self.dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = RMS(self.dim, config.rms_norm_eps)
        self.k_norm = RMS(self.dim, config.rms_norm_eps)

    def forward(self, query, context, positions, context_positions):
        q = self.q_norm(self.q_proj(query).reshape(-1, self.heads, self.dim))
        joined = torch.cat((context, query))
        key_positions = torch.cat((context_positions, positions))
        k = self.k_norm(self.k_proj(joined).reshape(-1, self.kv_heads, self.dim))
        v = self.v_proj(joined).reshape(-1, self.kv_heads, self.dim)
        q, k = rotate(q, positions, self.theta), rotate(k, key_positions, self.theta)
        groups = self.heads // self.kv_heads
        k = k.transpose(0, 1).repeat_interleave(groups, 0)
        v = v.transpose(0, 1).repeat_interleave(groups, 0)
        visible = context_positions < positions[0]
        if self.window is not None:
            visible &= context_positions >= positions[0] - self.window
        block = torch.ones(
            query.shape[0], query.shape[0], device=query.device, dtype=torch.bool
        )
        if self.causal:
            block = block.tril()
        mask = torch.cat((visible[None].expand(query.shape[0], -1), block), -1)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k[None], v[None], attn_mask=mask[None, None]
        )[0]
        return self.o_proj(out.transpose(0, 1).reshape(query.shape[0], -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BlockLayer(nn.Module):
    def __init__(self, config, source, window):
        super().__init__()
        self.input_layernorm = RMS(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMS(config.hidden_size, config.rms_norm_eps)
        self.self_attn = BlockAttention(
            config, window, source.get("sliding_window_non_causal", False)
        )
        self.mlp = MLP(config)
        for name in ("attention_conv", "mlp_conv"):
            conv = None
            if source.get("dflash2_dynamic_conv", False):
                conv = DFlash2GroupedConv(
                    config.hidden_size,
                    taps=source.get("dflash2_conv_kernel_size", 2),
                    group_size=source.get("dflash2_conv_group_size", 16),
                    block_size=source["block_size"],
                )
            setattr(self, name, conv)

    def forward(self, hidden, context, positions, context_positions):
        residual = hidden
        value = self.input_layernorm(hidden)
        coefficients = None
        if self.attention_conv is not None:
            value, coefficients = self.attention_conv.prepare(value)
        value = self.self_attn(value, context, positions, context_positions)
        if self.attention_conv is not None:
            value = self.attention_conv.finish(value, coefficients)
        hidden = residual + value
        value = self.post_attention_layernorm(hidden)
        if self.mlp_conv is not None:
            value, coefficients = self.mlp_conv.prepare(value)
        value = self.mlp(value)
        if self.mlp_conv is not None:
            value = self.mlp_conv.finish(value, coefficients)
        return hidden + value

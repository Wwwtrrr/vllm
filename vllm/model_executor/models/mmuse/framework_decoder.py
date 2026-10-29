# SPDX-License-Identifier: Apache-2.0
"""Framework primitives for the replicated MMuse decoder.

Binding happens after loading reference weights, before profiling or capture.
The trained parameter names and replicated TP ownership remain unchanged.
"""

import copy

import torch
from torch import nn


class CachedRotary(nn.Module):
    """Use framework rotary application with a cache including padding position -1."""

    def __init__(self, dim, theta, capacity, weight, apply):
        super().__init__()
        self.apply_rotary = apply
        inverse = theta ** (
            -torch.arange(0, dim, 2, device=weight.device).float() / dim
        )
        positions = torch.arange(-1, capacity, device=weight.device).float()
        angles = positions[:, None] * inverse[None]
        self.register_buffer("cos", angles.cos().to(weight.dtype), persistent=False)
        self.register_buffer("sin", angles.sin().to(weight.dtype), persistent=False)

    def forward(self, value, positions):
        indexes = positions + 1
        return self.apply_rotary(value, self.cos[indexes], self.sin[indexes])


@torch.inference_mode()
def bind_framework_decoder(model, config):
    if getattr(model, "framework_primitives_bound", False):
        return
    # Worker-only imports: reference tests do not initialize device backends.
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    draft_config = copy.copy(config)
    draft_config.quant_config = None

    def norm(old):
        result = RMSNorm(old.weight.numel(), eps=old.eps, dtype=old.weight.dtype)
        result.weight = old.weight
        return result

    def linear(old, prefix):
        result = ReplicatedLinear(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            params_dtype=old.weight.dtype,
            quant_config=None,
            prefix=prefix,
            return_bias=False,
            disable_tp=True,
        )
        result.weight.data = old.weight.data
        if old.bias is not None:
            result.bias.data = old.bias.data
        result.quant_method.process_weights_after_loading(result)
        return result

    with set_current_vllm_config(draft_config):
        model.fc = linear(model.fc, "draft_model.fc")
        model.runtime.fc = model.fc
        model.hidden_norm = norm(model.hidden_norm)
        model.norm = norm(model.norm)
        for index, layer in enumerate(model.layers):
            prefix = f"draft_model.layers.{index}"
            layer.input_layernorm = norm(layer.input_layernorm)
            layer.post_attention_layernorm = norm(layer.post_attention_layernorm)
            attn = layer.self_attn
            attn.q_norm, attn.k_norm = norm(attn.q_norm), norm(attn.k_norm)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(
                    attn,
                    name,
                    linear(getattr(attn, name), f"{prefix}.self_attn.{name}"),
                )
            attn.rotary_impl = CachedRotary(
                attn.dim,
                attn.theta,
                config.model_config.max_model_len
                + config.speculative_config.num_speculative_tokens,
                attn.q_proj.weight,
                ApplyRotaryEmb(is_neox_style=True),
            )
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(
                    layer.mlp,
                    name,
                    linear(getattr(layer.mlp, name), f"{prefix}.mlp.{name}"),
                )
            layer.mlp.activation_impl = SiluAndMul()
    model.framework_primitives_bound = True
    model.eval()

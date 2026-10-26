# SPDX-License-Identifier: Apache-2.0
"""Eager, single-device Qwen3 MMuse adapter with standalone inference heads."""

import weakref
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.mmuse_runtime import (
    build_mmuse_runtime,
    validate_mmuse_config,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.utils import AutoWeightsLoader


class MMuseDecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self.attention_conv = None
        self.mlp_conv = None
        if getattr(config, "dflash2_dynamic_conv", False):
            from vllm.model_executor.models.mmuse.dynamic_conv import DFlash2GroupedConv

            for name in ("attention_conv", "mlp_conv"):
                setattr(
                    self,
                    name,
                    DFlash2GroupedConv(
                        config.hidden_size,
                        taps=config.dflash2_conv_kernel_size,
                        group_size=config.dflash2_conv_group_size,
                        block_size=config.block_size,
                    ),
                )

    def forward(self, positions, hidden_states, residual):
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        coefficients = None
        if self.attention_conv is not None:
            hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        if self.attention_conv is not None:
            hidden_states = self.attention_conv.finish(hidden_states, coefficients)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.mlp_conv is not None:
            hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.mlp_conv is not None:
            hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


class MMuseQwen3Model(DFlashQwen3Model):
    decoder_layer_cls = MMuseDecoderLayer

    def __init__(self, *, vllm_config, **kwargs):
        super().__init__(vllm_config=vllm_config, **kwargs)
        self.register_buffer(
            "context_anchor",
            torch.zeros(
                vllm_config.scheduler_config.max_num_seqs,
                self.config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            ),
            persistent=False,
        )

    def forward(self, input_ids, positions, input_embeds=None):
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        block = self.config.block_size
        if input_embeds.shape[0] % block:
            raise ValueError("MMuse query rows must contain complete trained blocks")
        count = input_embeds.shape[0] // block
        runtime = self._runtime_ref()
        hidden = input_embeds.reshape(count, block, -1)
        if runtime.block_position_embedding is not None:
            slots = torch.arange(block, device=hidden.device)
            hidden = hidden + runtime.block_position_embedding(slots)[None]
        if runtime.context_hidden_proj is not None:
            residual = runtime.context_hidden_proj(self.context_anchor[:count])
            hidden = hidden + (
                torch.tanh(runtime.context_hidden_gate) * residual[:, None]
            )
        return super().forward(input_ids, positions, hidden.flatten(0, 1))


class _TargetEmbedding(nn.Module):
    def __init__(self, owner):
        super().__init__()
        self._owner = weakref.ref(owner)

    @property
    def weight(self):
        return self._owner().model.embed_tokens.weight

    def forward(self, ids):
        return self._owner().model.embed_tokens(ids)


class _TargetHead(nn.Module):
    def __init__(self, owner):
        super().__init__()
        self._owner = weakref.ref(owner)
        self.out_features = owner.config.draft_vocab_size

    @property
    def weight(self):
        return self._owner().lm_head.weight[: self.out_features]

    def forward(self, hidden):
        return F.linear(hidden, self.weight)


class MMuseQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = MMuseQwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix=""):
        validate_mmuse_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.runtime = build_mmuse_runtime(
            self.config.mmuse_source_config, self.model.fc
        ).to(dtype=vllm_config.model_config.dtype)
        self.runtime.embed_tokens = _TargetEmbedding(self)
        self.runtime.lm_head = _TargetHead(self)
        self.model._runtime_ref = weakref.ref(self.runtime)
        self._target_weights_bound = False

    def combine_hidden_states(self, hidden_states):
        projection, _ = self.runtime._prepare_target_hidden(hidden_states)
        return projection

    def bind_target_weights(self, target_model):
        target = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        self.model.embed_tokens = target.model.embed_tokens
        self.lm_head = target.lm_head
        self._target_weights_bound = True

    @torch.inference_mode()
    def rollout_mmuse(self, hidden, anchor_ids, *, sample_token=None):
        if not self._target_weights_bound:
            raise RuntimeError("MMuse target embedding/head have not been bound")
        # The local rollout preserves Selector, cache, feedback and Markov order.
        return self.runtime.rollout_correction(
            hidden, anchor_ids, temperature=0.0, sample_token=sample_token
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        mapped = []
        for name, weight in weights:
            if name in ("t2d", "d2t") or name.startswith("verifier_"):
                continue
            if name.startswith(("embed_tokens.", "lm_head.")):
                # Full-vocabulary MMuse always borrows these from the target.
                continue
            if name.startswith(("layers.", "fc.", "hidden_norm.", "norm.")):
                name = "model." + name
            else:
                name = "runtime." + name
            mapped.append((name, weight))
        loaded = AutoWeightsLoader(self).load_weights(mapped)
        borrowed = {"model.embed_tokens.weight", "lm_head.weight"}
        initialized = {"model.mask_embedding"}
        missing = {name for name, _ in self.named_parameters()} - loaded
        missing -= borrowed | initialized
        if missing:
            raise ValueError(f"MMuse checkpoint is missing trained weights: {missing}")
        self.model._build_fused_kv_buffers()
        return loaded | borrowed | initialized

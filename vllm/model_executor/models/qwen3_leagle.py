# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager LEAGLE inference matching speculators.models.leagle checkpoints."""

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos.unsqueeze(1) + _rotate_half(x) * sin.unsqueeze(1)


class LEagleAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        width = config.hidden_size
        bias = config.attention_bias
        self.q_proj = nn.Linear(width, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, width, bias=bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def project_context(self, context, position_embeddings):
        shape = (*context.shape[:2], self.num_kv_heads, self.head_dim)
        key = self.k_norm(self.k_proj(context).view(shape)).transpose(1, 2)
        key = _rope(key, *position_embeddings)
        value = self.v_proj(context).view(shape).transpose(1, 2)
        return key, value

    def forward(self, hidden, context_kv, position_embeddings, past_kv, mask):
        shape = (*hidden.shape[:2], -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden).view(shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden).view(shape)).transpose(1, 2)
        query = _rope(query, *position_embeddings)
        key = _rope(key, *position_embeddings)
        value = self.v_proj(hidden).view(shape).transpose(1, 2)
        if past_kv is not None:
            key = torch.cat((past_kv[0], key), dim=-2)
            value = torch.cat((past_kv[1], value), dim=-2)
        present = (key, value)
        key = torch.cat((context_kv[0], key), dim=-2)
        value = torch.cat((context_kv[1], value), dim=-2)
        # Explicit repetition also supports backends without SDPA enable_gqa.
        groups = self.num_heads // self.num_kv_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        return self.o_proj(
            output.transpose(1, 2).reshape(*hidden.shape[:2], -1)
        ), present


class LEagleDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = LEagleAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden, context_kv, position_embeddings, past_kv, mask):
        output, present = self.self_attn(
            self.input_layernorm(hidden), context_kv, position_embeddings, past_kv, mask
        )
        hidden = hidden + output
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden, present


class LEagleDraftModel(nn.Module):
    """Single-device eager drafter; physical decoder weights are shared by loops."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        self._init_architecture(hf_config.leagle_config)

    def _init_architecture(self, config: dict) -> None:
        decoder = Qwen3Config(**config["transformer_layer_config"])
        self.target_layer_ids = config["aux_hidden_state_layer_ids"]
        self.num_loops = config["num_loops"]
        self.norm_output = config["norm_output"]
        self.draft_vocab_size = config["draft_vocab_size"]
        self.target_vocab_size = decoder.vocab_size
        self.use_draft_vocab = self.draft_vocab_size != decoder.vocab_size
        if decoder.num_hidden_layers < 1 or self.num_loops < 1:
            raise ValueError("LEAGLE needs positive physical depth and loop count")
        if not 0 < self.draft_vocab_size <= decoder.vocab_size:
            raise ValueError("LEAGLE draft vocabulary must fit the target vocabulary")
        width, eps = decoder.hidden_size, decoder.rms_norm_eps
        self.embed_tokens = nn.Embedding(
            decoder.vocab_size, width, padding_idx=decoder.pad_token_id
        )
        self.lm_head = nn.Linear(width, self.draft_vocab_size, bias=False)
        self.fc = nn.Linear(len(self.target_layer_ids) * width, width, bias=False)
        self.hidden_norm = Qwen3RMSNorm(width, eps=eps)
        self.embed_norm = Qwen3RMSNorm(width, eps=eps)
        self.state_norm = Qwen3RMSNorm(width, eps=eps)
        self.input_fc = nn.Linear(2 * width, width, bias=False)
        self.layers = nn.ModuleList(
            LEagleDecoderLayer(decoder, i) for i in range(decoder.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(width, eps=eps)
        rope = getattr(decoder, "rope_parameters", None)
        if rope and "sliding_attention" in rope:
            decoder.rope_parameters = rope["sliding_attention"]
        self.rotary_emb = Qwen3RotaryEmbedding(decoder)
        self.register_buffer(
            "d2t",
            torch.zeros(self.draft_vocab_size, dtype=torch.long)
            if self.use_draft_vocab
            else None,
        )

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def forward(self, input_ids, positions, **kwargs):
        raise RuntimeError("LEAGLE must run through LEagleSpeculator.propose")

    def project_context(self, features):
        return self.hidden_norm(self.fc(features))

    @torch.inference_mode()
    def propose_context(self, context, anchor_token, num_tokens: int):
        """Return greedy target-vocabulary tokens from one verified prefix."""
        if context.shape[1] == 0 or num_tokens < 1:
            raise ValueError(
                "LEAGLE requires a nonempty prefix and positive draft length"
            )
        positions = torch.arange(context.shape[1], device=context.device)[None]
        context_rope = self.rotary_emb(context, positions)
        # Context is immutable across tokens and loops; project it once per layer.
        context_kvs = [
            layer.self_attn.project_context(context, context_rope)
            for layer in self.layers
        ]
        caches = [None] * (len(self.layers) * self.num_loops)
        state, token = context[:, -1:], anchor_token.long()
        proposals = []
        for step in range(num_tokens):
            hidden = self.input_fc(
                torch.cat(
                    (self.embed_norm(self.embed_tokens(token)), self.state_norm(state)),
                    dim=-1,
                )
            )
            query_position = positions[:, -1:] + 1 + step
            rope = self.rotary_emb(hidden, query_position)
            key_positions = torch.arange(
                context.shape[1] + step + 1, device=context.device
            )
            for loop in range(self.num_loops):
                for index, layer in enumerate(self.layers):
                    depth = loop * len(self.layers) + index
                    window = layer.self_attn.sliding_window
                    mask = (
                        None
                        if window is None
                        else (
                            key_positions > query_position[..., None] - window
                        ).unsqueeze(1)
                    )
                    hidden, caches[depth] = layer(
                        hidden, context_kvs[index], rope, caches[depth], mask
                    )
            normalized = self.norm(hidden)
            token = self.lm_head(normalized).argmax(-1)
            if self.use_draft_vocab:
                token = token + self.d2t[token]
            state = normalized if self.norm_output else hidden
            proposals.append(token)
        return torch.cat(proposals, dim=1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load exact training names and reject incomplete inference checkpoints."""
        tensors = dict(self.named_parameters())
        if self.d2t is not None:
            tensors["d2t"] = self.d2t
        loaded = set()
        for name, value in weights:
            if name == "t2d" or name.startswith(
                ("verifier_norm.", "verifier_lm_head.")
            ):
                continue
            if name not in tensors:
                raise ValueError(f"Unexpected LEAGLE checkpoint tensor: {name}")
            target = tensors[name]
            if target.shape != value.shape:
                raise ValueError(f"LEAGLE shape mismatch for {name}: {value.shape}")
            with torch.no_grad():
                target.copy_(value)
            loaded.add(name)
        missing = tensors.keys() - loaded
        if missing:
            raise ValueError(f"Missing LEAGLE checkpoint tensors: {sorted(missing)}")
        if self.d2t is not None:
            ids = torch.arange(self.draft_vocab_size, device=self.d2t.device) + self.d2t
            if (
                (ids < 0).any()
                or (ids >= self.target_vocab_size).any()
                or ids.unique().numel() != self.draft_vocab_size
            ):
                raise ValueError("Invalid LEAGLE draft-to-target vocabulary mapping")
        return loaded & dict(self.named_parameters()).keys()

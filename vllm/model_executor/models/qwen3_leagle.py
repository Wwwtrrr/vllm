# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LEAGLE inference matching speculators.models.leagle checkpoints."""

import logging
import time
from collections.abc import Iterable
from typing import NamedTuple

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


class LEagleForwardOutput(NamedTuple):
    hidden_states: torch.Tensor
    recurrent_state: torch.Tensor
    past_draft_kv: tuple


class LEagleRMSNorm(Qwen3RMSNorm):
    """Preserve training weight names while allowing a device fused kernel."""

    implementation = None

    def forward(self, hidden_states):
        if self.implementation is None:
            return super().forward(hidden_states)
        return self.implementation(hidden_states, self.weight, self.variance_epsilon)


class LEagleCandidateHead(nn.Module):
    """Keep the trained low-rank retrieval parameter names unchanged."""

    def __init__(self, width, vocab, rank):
        super().__init__()
        self.down_proj = nn.Linear(width, rank, bias=False)
        self.up_proj = nn.Linear(rank, vocab, bias=False)

    def forward(self, hidden):
        return self.up_proj(self.down_proj(hidden))


class LEagleSVDHead(nn.Module):
    """Inference-only WQ/Q.T factors; match looped_eagle's CPU FP64 Gram method."""

    @torch.no_grad()
    def __init__(self, weight, rank):
        super().__init__()
        if weight.is_meta or weight.ndim != 2 or not weight.is_floating_point():
            raise ValueError("SVD requires a loaded floating-point LM head")
        if type(rank) is not int or not 1 <= rank <= min(weight.shape):
            raise ValueError("SVD rank exceeds hidden/vocabulary size")
        started = time.perf_counter()
        logging.getLogger(__name__).info(
            "Building LEAGLE SVD head on CPU (FP64 Gram/eigh), rank=%d; "
            "initialization may take time for large vocabularies",
            rank,
        )
        vocab, width = weight.shape
        chunk = 2048
        gram = torch.zeros(width, width, dtype=torch.float64, device="cpu")
        for start in range(0, vocab, chunk):
            rows = weight[start : start + chunk].detach().to("cpu", torch.float64)
            if not torch.isfinite(rows).all():
                raise ValueError("SVD requires finite LM head weights")
            gram.addmm_(rows.T, rows)
        if not torch.isfinite(gram).all():
            raise ValueError("SVD Gram matrix overflowed")
        _, vectors = torch.linalg.eigh(gram)
        basis = vectors[:, -rank:].flip(-1).contiguous()
        del gram, vectors, rows
        up = torch.empty(vocab, rank, dtype=weight.dtype, device="cpu")
        for start in range(0, vocab, chunk):
            rows = weight[start : start + chunk].detach().to("cpu", torch.float64)
            up[start : start + rows.shape[0]].copy_(rows @ basis)
        self.register_buffer("up_weight", up.to(weight.device), persistent=False)
        self.register_buffer(
            "down_weight",
            basis.T.to(device=weight.device, dtype=weight.dtype),
            persistent=False,
        )
        if not all(torch.isfinite(t).all() for t in (self.up_weight, self.down_weight)):
            raise ValueError("SVD factors are non-finite after dtype conversion")
        logging.getLogger(__name__).info(
            "LEAGLE SVD head initialized in %.2f seconds", time.perf_counter() - started
        )

    def forward(self, hidden):
        return F.linear(F.linear(hidden, self.down_weight), self.up_weight)


class LEagleStaticKV:
    def __init__(self, draft, attention):
        self.draft, self.attention = draft, attention

    @classmethod
    def allocate(cls, model, batch, capacity, tokens, *, device, dtype):
        attn = model.layers[0].self_attn
        shape = (batch, attn.num_kv_heads)
        draft = tuple(
            tuple(
                torch.zeros(*shape, tokens, attn.head_dim, device=device, dtype=dtype)
                for _ in range(2)
            )
            for _ in range(len(model.layers) * model.num_loops)
        )
        attention = tuple(
            tuple(
                torch.zeros(
                    *shape, capacity + tokens, attn.head_dim, device=device, dtype=dtype
                )
                for _ in range(2)
            )
            for _ in model.layers
        )
        return cls(draft, attention)

    def view(self, batch, capacity, tokens):
        return LEagleStaticKV(
            tuple(tuple(t[:batch, :, :tokens] for t in pair) for pair in self.draft),
            tuple(
                tuple(t[:batch, :, : capacity + tokens] for t in pair)
                for pair in self.attention
            ),
        )

    def begin_round(self, context_kvs):
        if torch.is_grad_enabled():
            raise ValueError("Static draft KV is inference-only")
        for target, source in zip(self.attention, context_kvs):
            for destination, value in zip(target, source):
                destination[:, :, : value.shape[-2]].copy_(value)


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
        self.q_norm = LEagleRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LEagleRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.register_buffer("_qkv_weight", None, persistent=False)
        self.register_buffer("_qkv_bias", None, persistent=False)
        self.register_buffer("_kv_weight", None, persistent=False)
        self.register_buffer("_kv_bias", None, persistent=False)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def project_context(self, context, position_embeddings):
        shape = (*context.shape[:2], self.num_kv_heads, self.head_dim)
        if self._kv_weight is None:
            key, value = self.k_proj(context), self.v_proj(context)
        else:
            key, value = F.linear(context, self._kv_weight, self._kv_bias).chunk(2, -1)
        key = self.k_norm(key.view(shape)).transpose(1, 2)
        key = _rope(key, *position_embeddings)
        value = value.view(shape).transpose(1, 2)
        return key, value

    def forward(
        self,
        hidden,
        context_kv,
        position_embeddings,
        past_kv,
        mask,
        static_kv=None,
        draft_step=0,
    ):
        shape = (*hidden.shape[:2], -1, self.head_dim)
        if self._qkv_weight is None:
            query, key, value = (
                self.q_proj(hidden),
                self.k_proj(hidden),
                self.v_proj(hidden),
            )
        else:
            query, key, value = F.linear(
                hidden, self._qkv_weight, self._qkv_bias
            ).split(
                (
                    self.num_heads * self.head_dim,
                    self.num_kv_heads * self.head_dim,
                    self.num_kv_heads * self.head_dim,
                ),
                -1,
            )
        query = self.q_norm(query.view(shape)).transpose(1, 2)
        key = self.k_norm(key.view(shape)).transpose(1, 2)
        query = _rope(query, *position_embeddings)
        key = _rope(key, *position_embeddings)
        value = value.view(shape).transpose(1, 2)
        if static_kv is not None:
            if torch.is_grad_enabled():
                raise ValueError("Static draft KV is inference-only")
            draft, workspace = static_kv
            capacity = context_kv[0].shape[-2]
            for storage, combined, current in zip(draft, workspace, (key, value)):
                storage[:, :, draft_step : draft_step + 1].copy_(current)
                combined[:, :, capacity : capacity + draft_step + 1].copy_(
                    storage[:, :, : draft_step + 1]
                )
            present = tuple(t[:, :, : draft_step + 1] for t in draft)
            key, value = (t[:, :, : capacity + draft_step + 1] for t in workspace)
        else:
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
        self.input_layernorm = LEagleRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LEagleRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.register_buffer("_gate_up_weight", None, persistent=False)

    def forward(
        self,
        hidden,
        context_kv,
        position_embeddings,
        past_kv,
        mask,
        static_kv=None,
        draft_step=0,
    ):
        output, present = self.self_attn(
            self.input_layernorm(hidden),
            context_kv,
            position_embeddings,
            past_kv,
            mask,
            static_kv,
            draft_step,
        )
        hidden = hidden + output
        normalized = self.post_attention_layernorm(hidden)
        if self._gate_up_weight is None:
            output = self.mlp(normalized)
        else:
            gate, up = F.linear(normalized, self._gate_up_weight).chunk(2, -1)
            output = self.mlp.down_proj(self.mlp.act_fn(gate) * up)
        hidden = hidden + output
        return hidden, present


class LEagleDraftModel(nn.Module):
    """Single-device drafter; physical decoder weights are shared by loops."""

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
        self.quantized_lm_head = None
        candidate = config.get("candidate_head", False)
        if type(candidate) is not bool:
            raise ValueError("LEAGLE candidate_head must be a boolean")
        self.candidate_head = None
        self.svd_candidate_head = None
        self.candidate_head_enabled = candidate
        self.candidate_rerank = config.get("candidate_rerank", True)
        if type(self.candidate_rerank) is not bool:
            raise ValueError("LEAGLE candidate_rerank must be a boolean")
        self.candidate_top_k = config.get("candidate_head_top_k", 128)
        if candidate:
            rank = config.get("candidate_head_rank", 256)
            if type(rank) is not int or not 1 <= rank <= min(
                width, self.draft_vocab_size
            ):
                raise ValueError("LEAGLE candidate rank exceeds hidden/vocabulary size")
            self.candidate_head = LEagleCandidateHead(
                width, self.draft_vocab_size, rank
            )
            self.configure_candidate_head()
        self.fc = nn.Linear(len(self.target_layer_ids) * width, width, bias=False)
        self.hidden_norm = LEagleRMSNorm(width, eps=eps)
        self.embed_norm = LEagleRMSNorm(width, eps=eps)
        self.state_norm = LEagleRMSNorm(width, eps=eps)
        self.input_fc = nn.Linear(2 * width, width, bias=False)
        self.layers = nn.ModuleList(
            LEagleDecoderLayer(decoder, i) for i in range(decoder.num_hidden_layers)
        )
        self.norm = LEagleRMSNorm(width, eps=eps)
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
        if self.candidate_head_enabled:
            if not self.candidate_rerank:
                return self._coarse_logits(hidden_states)
            candidates, scores = self._candidate_scores(hidden_states)
            logits = hidden_states.new_full(
                (*hidden_states.shape[:-1], self.draft_vocab_size), -torch.inf
            )
            return logits.scatter_(-1, candidates, scores)
        if self.quantized_lm_head is not None:
            return self.quantized_lm_head(hidden_states)
        return self.lm_head(hidden_states)

    def set_quantized_lm_head(self, head):
        if self.quantized_lm_head is not None:
            raise ValueError(
                "Replacing LEAGLE quantized head requires restart and recapture"
            )
        if self.candidate_head_enabled or self.svd_candidate_head is not None:
            raise ValueError(
                "LEAGLE quantized full head cannot combine with candidate/SVD scoring"
            )
        if (
            head.hidden_size != self.lm_head.in_features
            or head.vocab_size != self.draft_vocab_size
        ):
            raise ValueError("LEAGLE quantized head geometry mismatch")
        self.quantized_lm_head = head
        # Quantization is complete before profiling/capture. No dense fallback
        # or candidate reranking is allowed after installing this head.
        self.lm_head = None

    def configure_output_head(
        self, *, svd_rank=None, enabled=None, top_k=None, rerank=None
    ):
        if self.quantized_lm_head is not None:
            raise ValueError(
                "Changing LEAGLE quantized scoring requires restart and recapture"
            )
        if svd_rank is not None:
            if enabled is False:
                raise ValueError("SVD rank cannot disable the candidate head")
            if rerank is False and top_k is not None:
                raise ValueError("Candidate top_k requires candidate reranking")
            default_k = min(128, self.draft_vocab_size) if rerank is False else 128
            self.set_svd_candidate_head(
                svd_rank, top_k=default_k if top_k is None else top_k
            )
        self.configure_candidate_head(enabled, top_k, rerank)

    def set_svd_candidate_head(self, rank, *, top_k=128):
        if self.quantized_lm_head is not None:
            raise ValueError("LEAGLE SVD cannot combine with quantized full head")
        if self.training:
            raise ValueError("SVD head requires eval mode")
        if type(top_k) is not int or not 1 <= top_k <= self.draft_vocab_size:
            raise ValueError("LEAGLE candidate top_k must be in [1, draft_vocab_size]")
        self.svd_candidate_head = LEagleSVDHead(self.lm_head.weight, rank).eval()
        self.configure_candidate_head(enabled=True, top_k=top_k, rerank=True)

    def configure_candidate_head(self, enabled=None, top_k=None, rerank=None):
        if enabled and self.quantized_lm_head is not None:
            raise ValueError(
                "LEAGLE candidate scoring cannot combine with quantized full head"
            )
        selected_enabled = self.candidate_head_enabled if enabled is None else enabled
        selected_rerank = self.candidate_rerank if rerank is None else rerank
        if rerank is True and not selected_enabled:
            raise ValueError("Candidate reranking requires an enabled candidate head")
        if top_k is not None and selected_enabled and not selected_rerank:
            raise ValueError("Candidate top_k requires candidate reranking")
        if enabled is not None:
            if (
                enabled
                and self.candidate_head is None
                and self.svd_candidate_head is None
            ):
                raise ValueError("LEAGLE checkpoint has no trained candidate head")
            self.candidate_head_enabled = enabled
        if top_k is not None:
            if not self.candidate_head_enabled:
                raise ValueError(
                    "LEAGLE candidate top_k requires an enabled candidate head"
                )
            self.candidate_top_k = top_k
        self.candidate_rerank = selected_rerank
        if self.candidate_head_enabled and (
            type(self.candidate_top_k) is not int
            or not 1 <= self.candidate_top_k <= self.draft_vocab_size
        ):
            raise ValueError("LEAGLE candidate top_k must be in [1, draft_vocab_size]")

    def _coarse_logits(self, hidden_states):
        head = self.svd_candidate_head
        if head is None:
            head = self.candidate_head
        return head(hidden_states)

    def _candidate_scores(self, hidden_states):
        candidates = (
            self._coarse_logits(hidden_states)
            .topk(self.candidate_top_k, dim=-1, sorted=False)
            .indices
        )
        rows = F.embedding(candidates, self.lm_head.weight)
        scores = torch.matmul(rows, hidden_states.unsqueeze(-1)).squeeze(-1)
        return candidates, scores

    def select_greedy_token(self, hidden_states):
        if not self.candidate_head_enabled or not self.candidate_rerank:
            return self.compute_logits(hidden_states).argmax(-1)
        candidates, scores = self._candidate_scores(hidden_states)
        maximum = scores.max(dim=-1, keepdim=True).values
        winners = (scores == maximum) | (torch.isnan(scores) & torch.isnan(maximum))
        # Match argmax of dense scattered logits, including unsorted Top-K ties.
        token = torch.where(winners, candidates, self.draft_vocab_size).min(-1).values
        return torch.where(torch.isneginf(maximum.squeeze(-1)), 0, token)

    def forward(
        self,
        input_ids,
        positions,
        *,
        hidden_states,
        context_kvs,
        attention_masks,
        past_draft_kv=None,
        static_kv=None,
        draft_step=0,
    ) -> LEagleForwardOutput:
        """Advance one draft token through every shared decoder loop.

        Args:
            input_ids: Known target-vocabulary token IDs, shaped [B, 1].
            positions: Absolute token positions, shaped [B, 1].
            hidden_states: Previous recurrent state, shaped [B, 1, H].
            context_kvs: Immutable target K/V pair per physical layer.
            attention_masks: SDPA mask per attention window, including None
                for full attention. Padding and trajectory visibility are
                supplied by the caller rather than inferred from tensor values.
            past_draft_kv: Draft K/V per logical (loop, layer), or None on
                the first step. Inputs are never modified in place.
            static_kv: Optional mutable inference workspace, separate from
                the immutable tuple-cache API.
            draft_step: Current token offset within that inference workspace.

        Returns:
            Normalized hidden states for compute_logits, next recurrent state,
            and updated draft K/V. Sampling remains the caller's responsibility.

        """
        depth_count = len(self.layers) * self.num_loops
        if len(context_kvs) != len(self.layers):
            raise ValueError("LEAGLE requires one context K/V pair per physical layer")
        if past_draft_kv is None:
            past_draft_kv = (None,) * depth_count
        elif len(past_draft_kv) != depth_count:
            raise ValueError("LEAGLE draft K/V must index every (loop, layer)")
        hidden = self.input_fc(
            torch.cat(
                (
                    self.embed_norm(self.embed_input_ids(input_ids)),
                    self.state_norm(hidden_states),
                ),
                dim=-1,
            )
        )
        rope = self.rotary_emb(hidden, positions)
        next_kv = []
        for loop in range(self.num_loops):
            for index, layer in enumerate(self.layers):
                depth = loop * len(self.layers) + index
                hidden, present = layer(
                    hidden,
                    context_kvs[index],
                    rope,
                    past_draft_kv[depth],
                    attention_masks[layer.self_attn.sliding_window],
                    None
                    if static_kv is None
                    else (static_kv.draft[depth], static_kv.attention[index]),
                    draft_step,
                )
                next_kv.append(present)
        normalized = self.norm(hidden)
        return LEagleForwardOutput(
            normalized, normalized if self.norm_output else hidden, tuple(next_kv)
        )

    def project_context(self, features):
        return self.hidden_norm(self.fc(features))

    @torch.no_grad()
    def configure_inference(self, *, fused_projections=False, rms_norm=None):
        self._fused_projections = fused_projections
        for layer in self.layers:
            attention = layer.self_attn
            if fused_projections:
                projections = (attention.q_proj, attention.k_proj, attention.v_proj)
                attention._qkv_weight = torch.cat([p.weight for p in projections])
                attention._kv_weight = torch.cat([p.weight for p in projections[1:]])
                attention._qkv_bias = (
                    torch.cat([p.bias for p in projections])
                    if projections[0].bias is not None
                    else None
                )
                attention._kv_bias = (
                    torch.cat([p.bias for p in projections[1:]])
                    if projections[0].bias is not None
                    else None
                )
                layer._gate_up_weight = torch.cat(
                    (layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight)
                )
            else:
                attention._qkv_weight = attention._kv_weight = None
                attention._qkv_bias = attention._kv_bias = None
                layer._gate_up_weight = None
        for module in self.modules():
            if isinstance(module, LEagleRMSNorm):
                module.implementation = rms_norm

    def prepare_context_kv(self, context, positions):
        rope = self.rotary_emb(context, positions)
        return tuple(
            layer.self_attn.project_context(context, rope) for layer in self.layers
        )

    def allocate_static_kv(self, batch, capacity, tokens, *, dtype, device):
        return LEagleStaticKV.allocate(
            self, batch, capacity, tokens, dtype=dtype, device=device
        )

    @torch.inference_mode()
    def propose_cached(
        self, state, context_kvs, lengths, anchor, num_tokens, workspace=None
    ):
        """Batched rollout from projected target KV; synthetic KV is never promoted."""
        batch = state.shape[0]
        capacity = context_kvs[0][0].shape[-2]
        positions = torch.arange(capacity, device=state.device)[None]
        visible_context = positions < lengths[:, None]
        token, caches = anchor.long(), None
        proposals = []
        if workspace is not None:
            workspace.begin_round(context_kvs)
        for step in range(num_tokens):
            query = lengths[:, None] + step
            draft_positions = (
                lengths[:, None] + torch.arange(step + 1, device=state.device)[None]
            )
            key_positions = torch.cat(
                (positions.expand(batch, -1), draft_positions), -1
            )
            visible = torch.cat(
                (
                    visible_context,
                    torch.ones(batch, step + 1, device=state.device, dtype=torch.bool),
                ),
                -1,
            )
            masks = {}
            for layer in self.layers:
                window = layer.self_attn.sliding_window
                mask = (
                    visible
                    if window is None
                    else visible & (key_positions > query - window)
                )
                masks[window] = mask[:, None, None]
            output = self(
                token,
                query,
                hidden_states=state,
                context_kvs=context_kvs,
                attention_masks=masks,
                past_draft_kv=caches,
                static_kv=workspace,
                draft_step=step,
            )
            token = self.select_greedy_token(output.hidden_states)
            if self.use_draft_vocab:
                token = token + self.d2t[token]
            state, caches = output.recurrent_state, output.past_draft_kv
            proposals.append(token)
        return torch.cat(proposals, 1)

    def validate_static_rollout(self):
        rope_type = self.rotary_emb.rope_type
        if "dynamic" in rope_type or rope_type == "longrope":
            raise ValueError(
                "LEAGLE cached/FULL execution requires RoPE "
                "without runtime cache resizing"
            )

    @torch.inference_mode()
    def propose_static(self, context, context_lengths, anchor_token, num_tokens: int):
        """Fixed-shape rollout for FULL capture; lengths remain device tensors.

        Context padding may contain stale rejected features. It is excluded at
        every logical depth, including sliding attention. Draft KV is rebuilt
        inside each replay and never carries state between proposal rounds.
        """
        batch, capacity, width = context.shape
        positions = torch.arange(capacity, device=context.device)[None].expand(
            batch, -1
        )
        context_rope = self.rotary_emb(context, positions)
        context_kvs = [
            layer.self_attn.project_context(context, context_rope)
            for layer in self.layers
        ]
        state_index = (context_lengths - 1).view(batch, 1, 1).expand(-1, 1, width)
        state = context.gather(1, state_index)
        token = anchor_token.long()
        caches = [None] * (len(self.layers) * self.num_loops)
        proposals = []
        context_visible = positions < context_lengths[:, None]
        for step in range(num_tokens):
            query_position = context_lengths[:, None] + step
            draft_positions = (
                context_lengths[:, None]
                + torch.arange(step + 1, device=context.device)[None]
            )
            key_positions = torch.cat((positions, draft_positions), dim=-1)
            visible = torch.cat(
                (
                    context_visible,
                    torch.ones(
                        (batch, step + 1), dtype=torch.bool, device=context.device
                    ),
                ),
                dim=-1,
            )
            masks = {}
            for layer in self.layers:
                window = layer.self_attn.sliding_window
                mask = visible
                if window is not None:
                    mask = mask & (key_positions > query_position - window)
                masks[window] = mask[:, None, None, :]
            output = self(
                token,
                query_position,
                hidden_states=state,
                context_kvs=context_kvs,
                attention_masks=masks,
                past_draft_kv=caches,
            )
            token = self.select_greedy_token(output.hidden_states)
            if self.use_draft_vocab:
                token = token + self.d2t[token]
            state, caches = output.recurrent_state, output.past_draft_kv
            proposals.append(token)
        return torch.cat(proposals, dim=1)

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
            query_position = positions[:, -1:] + 1 + step
            key_positions = torch.arange(
                context.shape[1] + step + 1, device=context.device
            )
            masks = {}
            for layer in self.layers:
                window = layer.self_attn.sliding_window
                masks[window] = (
                    None
                    if window is None
                    else (key_positions > query_position[..., None] - window).unsqueeze(
                        1
                    )
                )
            output = self(
                token,
                query_position,
                hidden_states=state,
                context_kvs=context_kvs,
                attention_masks=masks,
                past_draft_kv=caches,
            )
            token = self.select_greedy_token(output.hidden_states)
            if self.use_draft_vocab:
                token = token + self.d2t[token]
            state, caches = output.recurrent_state, output.past_draft_kv
            proposals.append(token)
        return torch.cat(proposals, dim=1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load exact training names and reject incomplete inference checkpoints."""
        if self.quantized_lm_head is not None:
            raise ValueError(
                "Reloading a quantized LEAGLE head requires restart and recapture"
            )
        if self.svd_candidate_head is not None:
            raise ValueError(
                "Reloading an SVD head requires restart and graph recapture"
            )
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
        if self.candidate_head is not None:
            for parameter in self.candidate_head.parameters():
                if not torch.isfinite(parameter).all():
                    raise ValueError("LEAGLE candidate head weights must be finite")
        if self.d2t is not None:
            ids = torch.arange(self.draft_vocab_size, device=self.d2t.device) + self.d2t
            if (
                (ids < 0).any()
                or (ids >= self.target_vocab_size).any()
                or ids.unique().numel() != self.draft_vocab_size
            ):
                raise ValueError("Invalid LEAGLE draft-to-target vocabulary mapping")
        if getattr(self, "_fused_projections", False):
            implementation = self.norm.implementation
            self.configure_inference(fused_projections=True, rms_norm=implementation)
        return loaded & dict(self.named_parameters()).keys()

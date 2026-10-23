# SPDX-License-Identifier: Apache-2.0
# Adapted from the local MMuse/Speculators inference implementation.
# See NOTICE and source_manifest.json for provenance and modifications.
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from .selector import DFlash2CandidateSelector
from .selector_runtime import MMuseSelectorMixin


class MMuseBackboneMixin(MMuseSelectorMixin):
    def _init_layer_fusion(
        self, config, hidden_size: int, num_target_layers: int
    ) -> None:
        self.dflash_gated_layer_fusion = config.dflash_gated_layer_fusion
        self.layer_fusion_norms: nn.ModuleList | None = None
        self.layer_fusion_score: nn.Linear | None = None
        self.layer_fusion_proj: nn.Linear | None = None
        self.layer_fusion_gate: nn.Parameter | None = None
        if self.dflash_gated_layer_fusion:
            self.layer_fusion_norms = nn.ModuleList(
                [
                    Qwen3RMSNorm(
                        hidden_size,
                        eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
                    )
                    for _ in range(num_target_layers)
                ]
            )
            self.layer_fusion_score = nn.Linear(hidden_size, 1, bias=False)
            self.layer_fusion_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.layer_fusion_gate = nn.Parameter(torch.zeros(()))

    def _init_query_features(self, config, hidden_size: int) -> None:
        self.context_hidden_proj: nn.Linear | None = None
        self.context_hidden_gate: nn.Parameter | None = None
        if config.dflash_context_residual:
            self.context_hidden_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.context_hidden_gate = nn.Parameter(torch.zeros(()))

        self.block_position_embedding: nn.Embedding | None = None
        if config.dflash_block_position_embedding:
            self.block_position_embedding = nn.Embedding(self.block_size, hidden_size)

        self.candidate_selector: DFlash2CandidateSelector | None = None
        if config.dflash2_candidate_selector:
            self.candidate_selector = DFlash2CandidateSelector(
                hidden_size=hidden_size,
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                rank=config.dflash2_selector_rank,
                top_k=config.dflash2_selector_top_k,
            )

    def _prepare_target_hidden(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build the shared target projection and expose raw per-layer states."""
        baseline_projection = self.fc(hidden_states)
        if not self.dflash_gated_layer_fusion:
            return baseline_projection, None

        if (
            self.layer_fusion_norms is None
            or self.layer_fusion_score is None
            or self.layer_fusion_proj is None
            or self.layer_fusion_gate is None
        ):
            raise RuntimeError("Gated layer fusion modules were not initialized")
        num_layers = len(self.layer_fusion_norms)
        hidden_size = self.config.transformer_layer_config.hidden_size
        expected_size = num_layers * hidden_size
        if hidden_states.shape[-1] != expected_size:
            raise ValueError(
                "Expected concatenated verifier hidden size "
                f"{expected_size}, got {hidden_states.shape[-1]}"
            )

        layer_states = hidden_states.reshape(
            *hidden_states.shape[:-1], num_layers, hidden_size
        )
        normalized = torch.stack(
            [
                norm(layer_states[..., layer_idx, :])
                for layer_idx, norm in enumerate(self.layer_fusion_norms)
            ],
            dim=-2,
        )
        scores = self.layer_fusion_score(normalized).squeeze(-1)
        weights = torch.softmax(scores.float(), dim=-1).to(normalized.dtype)
        fused = (normalized * weights.unsqueeze(-1)).sum(dim=-2)
        fusion_residual = self.layer_fusion_proj(fused)
        projected = baseline_projection + (
            torch.tanh(self.layer_fusion_gate) * fusion_residual
        )
        return projected, layer_states

    def _draft_ids_to_verifier(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Map draft-vocabulary IDs to the verifier vocabulary."""
        if self.d2t is None:
            return draft_ids.long()
        draft_ids = draft_ids.long()
        return draft_ids + self.d2t[draft_ids]

    def _condition_noise_embedding(
        self,
        noise_embedding: torch.Tensor,
        fused_context: torch.Tensor,
        anchor_positions: torch.Tensor,
        document_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply opt-in, inference-safe DFlash block conditioning."""
        if self.block_position_embedding is not None:
            slot_ids = torch.arange(
                self.block_size,
                dtype=torch.long,
                device=noise_embedding.device,
            ).repeat(anchor_positions.numel())
            noise_embedding = noise_embedding + self.block_position_embedding(
                slot_ids
            ).unsqueeze(0).to(noise_embedding.dtype)

        if self.context_hidden_proj is not None:
            if self.context_hidden_gate is None:
                raise RuntimeError("Context residual gate was not initialized")
            context_positions = (anchor_positions - 1).clamp_min(0)
            last_context = fused_context[:, context_positions, :]
            residual = self.context_hidden_proj(last_context)
            residual = residual.repeat_interleave(self.block_size, dim=1)

            anchor_docs = document_ids[:, anchor_positions]
            context_docs = document_ids[:, context_positions]
            valid_context = (
                (anchor_positions.unsqueeze(0) > 0)
                & (anchor_docs == context_docs)
                & (anchor_docs != -1)
            )
            valid_context = valid_context.repeat_interleave(
                self.block_size, dim=1
            ).unsqueeze(-1)
            residual = residual * valid_context.to(residual.dtype)
            noise_embedding = noise_embedding + (
                torch.tanh(self.context_hidden_gate)
                * residual.to(noise_embedding.dtype)
            )

        return noise_embedding

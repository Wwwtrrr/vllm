# SPDX-License-Identifier: Apache-2.0
# Adapted from the local MMuse/Speculators inference implementation.
# See NOTICE and source_manifest.json for provenance and modifications.
"""Inference-only MMuse modules and causal rollout; no training dependency."""

from collections.abc import Callable

import torch
from torch import nn

from .backbone import MMuseBackboneMixin
from .config import MMuseRuntimeConfig, validate_mmuse_options
from .correction import CausalCorrectionHead, CorrectionCache
from .heads import ConfidenceHead, MarkovHead
from .rollout_state import RolloutFeedbackState
from .rollout_validation import (
    validate_selector_logit_features,
    validate_selector_token_ids,
)
from .runtime_types import (
    CorrectionRolloutOutput,
    CorrectionStepOutput,
    InitialLogitFeedback,
    SelectorCorrectionInputs,
)
from .selector_inputs import align_selector_token_ids, shift_selector_candidates

_BLOCK_HIDDEN_RANK = 3


class MMuseRuntime(MMuseBackboneMixin, nn.Module):
    def _init_sequential_heads(self, config: MMuseRuntimeConfig) -> None:
        validate_mmuse_options(vars(config))
        hidden_size = config.transformer_layer_config.hidden_size
        self.markov_head: MarkovHead | None = None
        self.correction_head: CausalCorrectionHead | None = None
        self.correction_markov_gate: torch.nn.Linear | None = None
        self.correction_markov_scale: torch.nn.Parameter | None = None
        if config.enable_correction_head:
            self.correction_head = CausalCorrectionHead(
                input_hidden_size=hidden_size,
                token_embedding_size=hidden_size,
                block_size=self.block_size,
                correction_hidden_size=config.correction_hidden_size,
                correction_rank=config.correction_rank,
                num_layers=config.correction_num_layers,
                num_heads=config.correction_num_heads,
                gate_bias=config.correction_gate_bias,
                output_mode=config.correction_output_mode,
                draft_vocab_size=self.draft_vocab_size,
                enable_hidden_auxiliary=(
                    config.correction_hidden_aux_loss
                    or config.correction_project_corrected_hidden
                ),
                enable_hidden_feedback=config.correction_hidden_feedback,
            )
            if config.correction_with_markov:
                self.markov_head = MarkovHead(
                    verifier_vocab_size=self.verifier_vocab_size,
                    draft_vocab_size=self.draft_vocab_size,
                    markov_rank=config.markov_rank,
                    hidden_size=hidden_size,
                    head_type=config.markov_head_type,
                    init_std=config.markov_init_std,
                )
                self.correction_markov_gate = torch.nn.Linear(
                    config.correction_hidden_size, 1
                )
                self.correction_markov_scale = torch.nn.Parameter(torch.zeros(()))
                torch.nn.init.zeros_(self.correction_markov_gate.weight)
                torch.nn.init.constant_(
                    self.correction_markov_gate.bias,
                    config.correction_markov_gate_bias,
                )
        elif config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
                init_std=config.markov_init_std,
            )

        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            sequential_dim = 0
            if config.confidence_head_with_markov:
                sequential_dim = (
                    config.correction_hidden_size
                    if self.correction_head is not None
                    else config.markov_rank
                )
            input_dim = hidden_size + sequential_dim
            self.confidence_head = ConfidenceHead(input_dim)

    def _apply_collaborative_markov(
        self,
        correction_logits: torch.Tensor,
        correction_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gate a low-rank Markov bias with Correction's causal state."""
        if (
            self.markov_head is None
            or self.correction_markov_gate is None
            or self.correction_markov_scale is None
        ):
            raise RuntimeError("Correction-Markov collaboration is not enabled")
        prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
        markov_bias = self.markov_head.block_bias(
            prev_token_ids=prev_token_ids,
            hidden_states=hidden_states,
            prev_emb=prev_emb,
        )
        gate_dtype = self.correction_markov_gate.weight.dtype
        local_gate = torch.sigmoid(
            self.correction_markov_gate(correction_states.to(gate_dtype))
        )
        gate = torch.tanh(self.correction_markov_scale) * local_gate
        collaborative_logits = correction_logits + (
            gate.to(markov_bias.dtype) * markov_bias
        )
        return collaborative_logits, gate, prev_emb

    @staticmethod
    def _replace_compact_feature_position(
        features: torch.Tensor | None,
        replacement: torch.Tensor | None,
        position: int,
    ) -> torch.Tensor | None:
        """Replace one compact sequence position without a vocabulary tensor."""
        if features is None:
            return None
        if replacement is None:
            raise RuntimeError("Replacement compact logit features are missing")
        return torch.cat(
            [features[:, :position], replacement, features[:, position + 1 :]],
            dim=1,
        )

    def _selector_correction_inputs(
        self,
        candidate_ids: torch.Tensor,
        selector_logits: torch.Tensor,
        selected_draft_ids: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        initial_previous_logits: torch.Tensor | None = None,
    ) -> SelectorCorrectionInputs:
        """Validate a static path, align its inputs, then encode compact features."""
        if self.correction_head is None:
            raise RuntimeError("Selector conditioning requires Correction")
        if candidate_ids.shape != selector_logits.shape:
            raise ValueError("Selector candidate IDs and logits must align")
        if selected_draft_ids.shape != candidate_ids.shape[:-1]:
            raise ValueError("Selector path must align with candidate block positions")
        if anchor_token_ids.shape != (candidate_ids.shape[0],):
            raise ValueError("Selector conditioning requires one anchor per block")

        # Greedy keeps its realized Top-K row. Global search is a deterministic
        # whole-path proposal, so its actual per-position q is the selected token's
        # one-hot row rather than the local edge argmax distribution.
        selector_logits = self._dflash2_proposal_logits(
            candidate_ids,
            selector_logits,
            selected_draft_ids,
        ).detach()

        num_blocks, block_size, _ = candidate_ids.shape
        start_position = 0 if self.config.sample_from_anchor else 1
        token_inputs = align_selector_token_ids(
            selected_draft_ids,
            anchor_token_ids,
            block_size=block_size,
            start_position=start_position,
            draft_to_verifier=self._draft_ids_to_verifier,
        )

        needs_logit_features = (
            getattr(self.correction_head, "output_mode", "hidden") == "logits"
        )
        if not needs_logit_features:
            return token_inputs

        previous_candidates = shift_selector_candidates(
            candidate_ids,
            selector_logits,
            sample_from_anchor=self.config.sample_from_anchor,
            initial_previous_logits=initial_previous_logits,
        )
        previous_rank = self.correction_head.encode_previous_distribution(
            previous_candidates.mask,
            candidate_ids=previous_candidates.candidate_ids,
            candidate_logits=previous_candidates.candidate_logits,
        )
        previous_logits_mask = previous_candidates.mask
        if not self.config.sample_from_anchor:
            if initial_previous_logits is None:
                raise ValueError(
                    "Logit-aware Selector Correction with sample_from_anchor=False "
                    "requires initial verifier logits"
                )
            expected = (num_blocks, self.draft_vocab_size)
            if initial_previous_logits.shape != expected:
                raise ValueError(
                    "Expected selector initial logits shape "
                    f"{expected}, got {tuple(initial_previous_logits.shape)}"
                )
            initial_mask = torch.ones(
                num_blocks,
                1,
                dtype=torch.bool,
                device=candidate_ids.device,
            )
            initial_rank = self.correction_head.encode_previous_distribution(
                initial_mask,
                previous_logits=initial_previous_logits.unsqueeze(1),
            )

            previous_rank = self._replace_compact_feature_position(
                previous_rank,
                initial_rank,
                start_position,
            )
            previous_logits_mask = previous_candidates.mask.clone()
            previous_logits_mask[:, start_position] = True
        return SelectorCorrectionInputs(
            current_token_ids=token_inputs.current_token_ids,
            previous_token_ids=token_inputs.previous_token_ids,
            previous_rank_features=previous_rank,
            previous_logits_mask=previous_logits_mask,
        )

    def _validate_rollout_inputs(
        self,
        dflash_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        precomputed_base_logits: torch.Tensor | None = None,
        conditioning_current_ids: torch.Tensor | None = None,
        conditioning_previous_ids: torch.Tensor | None = None,
        conditioning_previous_rank_features: torch.Tensor | None = None,
        conditioning_previous_logits_mask: torch.Tensor | None = None,
    ) -> tuple[bool, bool]:
        """Validate block, Selector IDs/features, then base logits in that order."""
        if self.correction_head is None:
            raise RuntimeError(
                "_rollout_correction_steps requires enable_correction_head=True"
            )
        if dflash_hidden.ndim != _BLOCK_HIDDEN_RANK:
            raise ValueError("dflash_hidden must be rank-3")
        if dflash_hidden.shape[1] != self.block_size:
            raise ValueError(
                f"Expected block_size={self.block_size}, got {dflash_hidden.shape[1]}"
            )
        if anchor_token_ids.shape != (dflash_hidden.shape[0],):
            raise ValueError(
                f"Expected anchor_token_ids shape {(dflash_hidden.shape[0],)}, "
                f"got {anchor_token_ids.shape}"
            )
        expected_block_shape = (dflash_hidden.shape[0], self.block_size)
        selector_feedback = getattr(
            self.config,
            "selector_correction_feedback",
            "static",
        )
        online_selector = (
            self.candidate_selector is not None and selector_feedback == "corrected"
        )
        validate_selector_token_ids(
            conditioning_current_ids,
            conditioning_previous_ids,
            expected_block_shape=expected_block_shape,
            online_selector=online_selector,
        )
        has_conditioning_features = validate_selector_logit_features(
            conditioning_previous_rank_features,
            conditioning_previous_logits_mask,
            expected_block_shape=expected_block_shape,
        )
        if precomputed_base_logits is not None:
            expected_base_shape = (
                dflash_hidden.shape[0],
                self.block_size,
                self.draft_vocab_size,
            )
            if precomputed_base_logits.shape != expected_base_shape:
                raise ValueError(
                    "Expected precomputed_base_logits shape "
                    f"{expected_base_shape}, got "
                    f"{tuple(precomputed_base_logits.shape)}"
                )
        if online_selector and precomputed_base_logits is None:
            raise ValueError("Corrected Selector feedback requires DFlash base logits")
        return online_selector, has_conditioning_features

    def _initial_rollout_logit_feedback(
        self,
        dflash_hidden: torch.Tensor,
        initial_previous_logits: torch.Tensor | None,
        *,
        logit_feedback_enabled: bool,
        has_conditioning_features: bool,
        online_selector: bool,
    ) -> InitialLogitFeedback:
        """Prepare dense or online compact logit feedback before the first slot."""
        previous_feedback_logits = None
        previous_feedback_mask = None
        online_previous_rank = None
        online_previous_mask = None
        if (
            logit_feedback_enabled
            and not has_conditioning_features
            and not online_selector
        ):
            draft_vocab_size = getattr(
                self, "draft_vocab_size", self.lm_head.out_features
            )
            expected_initial_shape = (
                dflash_hidden.shape[0],
                draft_vocab_size,
            )
            if self.config.sample_from_anchor:
                if initial_previous_logits is not None:
                    raise ValueError(
                        "initial_previous_logits is only valid when "
                        "sample_from_anchor=False"
                    )
                previous_feedback_logits = dflash_hidden.new_zeros(
                    dflash_hidden.shape[0],
                    1,
                    draft_vocab_size,
                    dtype=self.lm_head.weight.dtype,
                )
                previous_feedback_mask = torch.zeros(
                    dflash_hidden.shape[0],
                    1,
                    dtype=torch.bool,
                    device=dflash_hidden.device,
                )
            else:
                if initial_previous_logits is None:
                    raise ValueError(
                        "Logit-aware Correction with sample_from_anchor=False "
                        "requires verifier logits for the current anchor"
                    )
                if initial_previous_logits.shape != expected_initial_shape:
                    raise ValueError(
                        "Expected initial_previous_logits shape "
                        f"{expected_initial_shape}, got "
                        f"{tuple(initial_previous_logits.shape)}"
                    )
                previous_feedback_logits = (
                    initial_previous_logits.detach()
                    .to(
                        device=dflash_hidden.device,
                        dtype=self.lm_head.weight.dtype,
                    )
                    .unsqueeze(1)
                )
                previous_feedback_mask = torch.ones(
                    dflash_hidden.shape[0],
                    1,
                    dtype=torch.bool,
                    device=dflash_hidden.device,
                )
        elif logit_feedback_enabled and online_selector:
            if self.config.sample_from_anchor:
                if initial_previous_logits is not None:
                    raise ValueError(
                        "initial_previous_logits is only valid when "
                        "sample_from_anchor=False"
                    )
                online_previous_mask = torch.zeros(
                    dflash_hidden.shape[0],
                    1,
                    dtype=torch.bool,
                    device=dflash_hidden.device,
                )
                assert self.candidate_selector is not None  # noqa: S101
                dummy_ids = torch.zeros(
                    dflash_hidden.shape[0],
                    1,
                    self.candidate_selector.top_k,
                    dtype=torch.long,
                    device=dflash_hidden.device,
                )
                dummy_logits = dflash_hidden.new_zeros(
                    dummy_ids.shape,
                    dtype=self.lm_head.weight.dtype,
                )
                online_previous_rank = (
                    self.correction_head.encode_previous_distribution(
                        online_previous_mask,
                        candidate_ids=dummy_ids,
                        candidate_logits=dummy_logits,
                    )
                )
            else:
                if initial_previous_logits is None:
                    raise ValueError(
                        "Logit-aware Correction with sample_from_anchor=False "
                        "requires verifier logits for the current anchor"
                    )
                online_previous_mask = torch.ones(
                    dflash_hidden.shape[0],
                    1,
                    dtype=torch.bool,
                    device=dflash_hidden.device,
                )
                online_previous_rank = (
                    self.correction_head.encode_previous_distribution(
                        online_previous_mask,
                        previous_logits=initial_previous_logits.detach().unsqueeze(1),
                    )
                )
        return InitialLogitFeedback(
            dense_logits=previous_feedback_logits,
            dense_mask=previous_feedback_mask,
            online_rank_features=online_previous_rank,
            online_mask=online_previous_mask,
        )

    def _rollout_correction_step(
        self,
        previous_embeddings: torch.Tensor,
        current_hidden: torch.Tensor,
        correction_hidden: torch.Tensor,
        block_positions: torch.Tensor,
        *,
        position: int,
        cache: CorrectionCache | None,
        precomputed_base_logits: torch.Tensor | None = None,
        fused_base_logits: torch.Tensor | None = None,
        **correction_kwargs: torch.Tensor,
    ) -> CorrectionStepOutput:
        """Compute one Correction residual and its final vocabulary projection.

        The caller supplies independent residual/input hidden slices to preserve
        autograd accumulation order. Selector conditioning, Markov collaboration,
        sampling and feedback-state updates remain in the rollout loop.
        """
        correction_output_mode = getattr(self.correction_head, "output_mode", "hidden")
        hidden_auxiliary_enabled = getattr(
            self.config, "correction_hidden_aux_loss", False
        )
        hidden_feedback_enabled = getattr(
            self.config, "correction_hidden_feedback", False
        )
        project_corrected_hidden = getattr(
            self.config, "correction_project_corrected_hidden", False
        )
        residual, causal_states, next_cache = self.correction_head(
            previous_embeddings,
            correction_hidden,
            block_positions,
            cache=cache,
            use_cache=True,
            **correction_kwargs,
        )
        if correction_output_mode == "logits":
            if (
                project_corrected_hidden
                or hidden_feedback_enabled
                or (hidden_auxiliary_enabled and self.training)
            ):
                delta_hidden = self.correction_head.auxiliary_hidden_residual(
                    causal_states,
                )
                corrected_current_hidden = current_hidden + delta_hidden[:, 0].to(
                    current_hidden.dtype
                )
            else:
                corrected_current_hidden = current_hidden
            projection_hidden = (
                corrected_current_hidden if project_corrected_hidden else current_hidden
            )
            if project_corrected_hidden and fused_base_logits is not None:
                hidden_delta_logits = self.correction_head.fused_lm_head_residual(
                    causal_states,
                    self.lm_head.weight,
                )
                projected_logits = fused_base_logits[:, position] + (
                    hidden_delta_logits[:, 0].to(fused_base_logits.dtype)
                )
            elif precomputed_base_logits is not None and not project_corrected_hidden:
                projected_logits = precomputed_base_logits[:, position]
            else:
                projected_logits = self.lm_head(
                    projection_hidden.to(self.lm_head.weight.dtype)
                )
            final_logits = projected_logits + residual[:, 0].to(projected_logits.dtype)
        else:
            corrected_current_hidden = current_hidden + residual[:, 0].to(
                current_hidden.dtype
            )
            if fused_base_logits is not None:
                delta_logits = self.correction_head.fused_lm_head_residual(
                    causal_states,
                    self.lm_head.weight,
                )
                final_logits = fused_base_logits[:, position] + (
                    delta_logits[:, 0].to(fused_base_logits.dtype)
                )
            else:
                final_logits = self.lm_head(
                    corrected_current_hidden.to(self.lm_head.weight.dtype)
                )
        return CorrectionStepOutput(
            logits=final_logits,
            causal_states=causal_states[:, 0],
            corrected_hidden=corrected_current_hidden,
            cache=next_cache,
        )

    def _rollout_token_embeddings(
        self,
        step_previous_ids: torch.Tensor,
        dflash_hidden: torch.Tensor,
        *,
        position: int,
        online_selector: bool,
        precomputed_base_logits: torch.Tensor | None,
        conditioning_current_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Freeze previous/current embeddings and any online Selector proposal.

        The loop chooses previous IDs before this call. Keep the previous-token
        embedding first, and preserve the Selector's argmax tie breaking and
        draft-to-verifier mapping before looking up the current token.
        """
        with torch.no_grad():
            previous_emb = self.embed_tokens(step_previous_ids).unsqueeze(1)
            if online_selector:
                assert precomputed_base_logits is not None  # noqa: S101
                online_candidate_ids, online_candidate_logits = (
                    self.dflash2_select_candidates(
                        precomputed_base_logits[:, position : position + 1],
                        dflash_hidden[:, position : position + 1],
                        step_previous_ids.unsqueeze(1),
                    )
                )
                selected_indices = online_candidate_logits.argmax(dim=-1, keepdim=True)
                selected_draft_ids = online_candidate_ids.gather(
                    -1, selected_indices
                ).squeeze(-1)
                selected_verifier_ids = self._draft_ids_to_verifier(selected_draft_ids)
                current_emb = self.embed_tokens(selected_verifier_ids)
            else:
                current_emb = (
                    self.embed_tokens(conditioning_current_ids[:, position]).unsqueeze(
                        1
                    )
                    if conditioning_current_ids is not None
                    else None
                )
        return previous_emb, current_emb

    @torch.compiler.disable
    def _rollout_correction_steps(
        self,
        dflash_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        temperature: float = 0.0,
        sample_token: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        initial_previous_logits: torch.Tensor | None = None,
        precomputed_base_logits: torch.Tensor | None = None,
        conditioning_current_ids: torch.Tensor | None = None,
        conditioning_previous_ids: torch.Tensor | None = None,
        conditioning_previous_rank_features: torch.Tensor | None = None,
        conditioning_previous_logits_mask: torch.Tensor | None = None,
    ) -> CorrectionRolloutOutput:
        """Run sequential Correction for rollout and offline evaluation.

        The first input is always the real anchor. By default later inputs come
        from the current Correction model.
        Static Selector conditioning supplies a complete path. Corrected feedback
        instead scores the greedy Selector online and feeds each final Correction
        token into the next slot. Main train/validation metrics use a separate
        teacher-forced path.
        """
        online_selector, has_conditioning_features = self._validate_rollout_inputs(
            dflash_hidden,
            anchor_token_ids,
            precomputed_base_logits=precomputed_base_logits,
            conditioning_current_ids=conditioning_current_ids,
            conditioning_previous_ids=conditioning_previous_ids,
            conditioning_previous_rank_features=conditioning_previous_rank_features,
            conditioning_previous_logits_mask=conditioning_previous_logits_mask,
        )
        previous_ids = anchor_token_ids.long()
        cache = None
        output_tokens: list[torch.Tensor] = []
        output_logits: list[torch.Tensor] = []
        output_states: list[torch.Tensor] = []
        output_corrected_hidden: list[torch.Tensor] = []
        start_position = 0 if self.config.sample_from_anchor else 1
        correction_output_mode = getattr(self.correction_head, "output_mode", "hidden")
        hidden_feedback_enabled = getattr(
            self.config, "correction_hidden_feedback", False
        )
        logit_feedback_enabled = correction_output_mode == "logits"
        fused_lm_head_enabled = (
            bool(getattr(self.config, "correction_lm_head_fusion", False))
            and not torch.is_grad_enabled()
        )
        fused_base_logits = None
        if fused_lm_head_enabled:
            # Project the full parallel DFlash block once. Each sequential
            # Correction position adds an inference-cached rank-to-vocabulary
            # residual instead of invoking the full LM head again.
            fused_base_logits = precomputed_base_logits
            if fused_base_logits is None:
                fused_base_logits = self.lm_head(
                    dflash_hidden.to(self.lm_head.weight.dtype)
                )
        initial_feedback = self._initial_rollout_logit_feedback(
            dflash_hidden,
            initial_previous_logits,
            logit_feedback_enabled=logit_feedback_enabled,
            has_conditioning_features=has_conditioning_features,
            online_selector=online_selector,
        )
        feedback = RolloutFeedbackState.from_initial(
            initial_feedback,
            dflash_hidden,
            hidden_feedback_enabled=hidden_feedback_enabled,
        )
        del initial_feedback  # Do not retain superseded dense logits through rollout.

        for position in range(self.block_size):
            current_hidden = dflash_hidden[:, position]
            if position < start_position:
                corrected_current_hidden = current_hidden
                available_base_logits = (
                    fused_base_logits
                    if fused_base_logits is not None
                    else precomputed_base_logits
                )
                if available_base_logits is not None:
                    final_logits = available_base_logits[:, position]
                else:
                    final_logits = self.lm_head(
                        current_hidden.to(self.lm_head.weight.dtype)
                    )
                causal_states = current_hidden.new_zeros(
                    current_hidden.shape[0], self.config.correction_hidden_size
                )
            else:
                step_previous_ids = (
                    conditioning_previous_ids[:, position]
                    if conditioning_previous_ids is not None
                    else previous_ids
                )
                previous_emb, current_emb = self._rollout_token_embeddings(
                    step_previous_ids,
                    dflash_hidden,
                    position=position,
                    online_selector=online_selector,
                    precomputed_base_logits=precomputed_base_logits,
                    conditioning_current_ids=conditioning_current_ids,
                )
                current_token_kwargs = (
                    {}
                    if current_emb is None
                    else {"current_token_embeddings": current_emb}
                )
                block_positions = torch.full(
                    (dflash_hidden.shape[0], 1),
                    position,
                    dtype=torch.long,
                    device=dflash_hidden.device,
                )
                feedback_kwargs = feedback.correction_kwargs(
                    position,
                    hidden_feedback_enabled=hidden_feedback_enabled,
                    logit_feedback_enabled=logit_feedback_enabled,
                    has_conditioning_features=has_conditioning_features,
                    online_selector=online_selector,
                    conditioning_previous_rank_features=conditioning_previous_rank_features,
                    conditioning_previous_logits_mask=conditioning_previous_logits_mask,
                )
                step = self._rollout_correction_step(
                    previous_emb,
                    current_hidden,
                    dflash_hidden[:, position : position + 1],
                    block_positions,
                    position=position,
                    cache=cache,
                    precomputed_base_logits=precomputed_base_logits,
                    fused_base_logits=fused_base_logits,
                    **current_token_kwargs,
                    **feedback_kwargs,
                )
                final_logits = step.logits
                causal_states = step.causal_states
                corrected_current_hidden = step.corrected_hidden
                cache = step.cache
                del step  # Match the previous tuple-unpacking tensor lifetimes.
                if getattr(self, "markov_head", None) is not None:
                    final_logits, _, _ = self._apply_collaborative_markov(
                        final_logits.unsqueeze(1),
                        causal_states.unsqueeze(1),
                        step_previous_ids.unsqueeze(1),
                        dflash_hidden[:, position : position + 1],
                    )
                    final_logits = final_logits[:, 0]

            with torch.no_grad():
                if sample_token is not None:
                    sampled_indices = sample_token(final_logits, position)
                elif temperature > 0:
                    probabilities = torch.softmax(
                        final_logits.float() / temperature, dim=-1
                    )
                    sampled_indices = torch.multinomial(
                        probabilities, num_samples=1
                    ).squeeze(-1)
                else:
                    sampled_indices = torch.argmax(final_logits, dim=-1)
                draft_ids = sampled_indices
            output_tokens.append(draft_ids)
            output_logits.append(final_logits)
            output_states.append(causal_states)
            output_corrected_hidden.append(corrected_current_hidden)

            feedback.advance(
                final_logits,
                corrected_current_hidden,
                position=position,
                start_position=start_position,
                logit_feedback_enabled=logit_feedback_enabled,
                has_conditioning_features=has_conditioning_features,
                online_selector=online_selector,
                hidden_feedback_enabled=hidden_feedback_enabled,
                correction_head=self.correction_head,
            )
            if position < start_position:
                continue
            if conditioning_previous_ids is None:
                previous_ids = self._draft_ids_to_verifier(draft_ids)

        return CorrectionRolloutOutput(
            token_ids=torch.stack(output_tokens, dim=1),
            logits=torch.stack(output_logits, dim=1),
            causal_states=torch.stack(output_states, dim=1),
            corrected_hidden=torch.stack(output_corrected_hidden, dim=1),
        )

    @torch.compiler.disable
    @torch.no_grad()
    def rollout_correction(
        self,
        dflash_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        temperature: float = 0.0,
        sample_token: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        initial_previous_logits: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Correction using a static selector path or online feedback."""
        conditioning_previous_ids = None
        conditioning_current_ids = None
        conditioning_previous_rank_features = None
        conditioning_previous_logits_mask = None
        if self.candidate_selector is not None:
            if base_logits is None:
                base_logits = self.lm_head(dflash_hidden.to(self.lm_head.weight.dtype))
            if (
                getattr(self.config, "selector_correction_feedback", "static")
                == "static"
            ):
                candidate_ids, selector_logits, selected_ids = self.dflash2_select_path(
                    base_logits,
                    dflash_hidden,
                    anchor_token_ids,
                )
                selector_inputs = self._selector_correction_inputs(
                    candidate_ids,
                    selector_logits,
                    selected_ids,
                    anchor_token_ids,
                    initial_previous_logits=initial_previous_logits,
                )
                conditioning_current_ids = selector_inputs.current_token_ids
                conditioning_previous_ids = selector_inputs.previous_token_ids
                conditioning_previous_rank_features = (
                    selector_inputs.previous_rank_features
                )
                conditioning_previous_logits_mask = selector_inputs.previous_logits_mask
        rollout = self._rollout_correction_steps(
            dflash_hidden,
            anchor_token_ids,
            temperature=temperature,
            sample_token=sample_token,
            initial_previous_logits=initial_previous_logits,
            precomputed_base_logits=base_logits,
            conditioning_current_ids=conditioning_current_ids,
            conditioning_previous_ids=conditioning_previous_ids,
            conditioning_previous_rank_features=(conditioning_previous_rank_features),
            conditioning_previous_logits_mask=conditioning_previous_logits_mask,
        )
        return rollout.token_ids, rollout.logits

    def __init__(self, source_config: dict, fc: nn.Module):
        super().__init__()
        self.config = MMuseRuntimeConfig.from_checkpoint(source_config)
        config = self.config
        self.block_size = config.block_size
        self.hidden_size = config.transformer_layer_config.hidden_size
        self.verifier_vocab_size = config.transformer_layer_config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.use_draft_vocab = self.draft_vocab_size != self.verifier_vocab_size
        if self.use_draft_vocab:
            raise ValueError("MMuse runtime currently requires the full vocabulary")
        self.register_buffer("t2d", None)
        self.register_buffer("d2t", None)
        self.fc = fc
        self._init_layer_fusion(
            config, self.hidden_size, len(config.aux_hidden_state_layer_ids)
        )
        self._init_query_features(config, self.hidden_size)
        self._init_sequential_heads(config)

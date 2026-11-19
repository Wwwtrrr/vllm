# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.v1.worker.gpu.spec_decode.speculator_config import DraftModelSpeculatorConfig


class LEagleSpeculatorConfig(DraftModelSpeculatorConfig):
    base_method = "leagle"
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.leagle.speculator.LEagleSpeculator"
    )
    supported_options = frozenset({"leagle_lm_head_quantization"})
    uses_target_hidden_states = True

    @property
    def uses_aux_hidden_states(self) -> bool:
        return True

    @property
    def num_lookahead_tokens(self) -> int:
        return self.num_speculative_tokens

    def hash_factors(self) -> list[Any]:
        spec = self.spec_config
        factors = [
            spec.leagle_reuse_context_kv,
            spec.leagle_static_draft_kv,
            spec.leagle_batch_drafts,
            spec.leagle_fused_projections,
            spec.leagle_fused_rmsnorm,
            spec.leagle_use_candidate_head,
            spec.leagle_candidate_top_k,
            spec.leagle_candidate_rerank,
            spec.leagle_lm_head_quantization,
            spec.leagle_release_quantized_source,
        ]
        if spec.draft_model_config is not None:
            config = getattr(spec.draft_model_config.hf_config, "leagle_config", {})
            factors.extend(
                config.get(key)
                for key in (
                    "candidate_head",
                    "candidate_head_rank",
                    "candidate_head_top_k",
                    "candidate_rerank",
                )
            )
        return factors

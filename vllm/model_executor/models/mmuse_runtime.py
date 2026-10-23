# SPDX-License-Identifier: Apache-2.0
"""MMuse execution helpers shared by the GPU and Ascend speculators."""

import torch
from torch import nn

from vllm.model_executor.models.mmuse.validation import (
    validate_mmuse_config as validate_mmuse_config,
)


def build_mmuse_runtime(source_config: dict, fc: nn.Module) -> nn.Module:
    """Construct inference heads without a training package or second decoder."""
    from vllm.model_executor.models.mmuse.runtime import MMuseRuntime

    return MMuseRuntime(source_config, fc).eval()


class MMuseGreedyRolloutMixin:
    """Shared GPU/NPU rollout with greedy or framework probabilistic sampling."""

    def _mmuse_gumbel_sample(self, *args, **kwargs):
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        return gumbel_sample(*args, **kwargs)

    def _sample_mmuse_token(self, logits, position, num_reqs):
        if self.draft_logits is None:
            raise RuntimeError("MMuse probabilistic drafting requires a logits cache")
        # Sample the same precision that the verifier will read from the cache.
        logits = logits.to(self.draft_logits.dtype).contiguous()
        rows = slice(
            position, num_reqs * self.num_query_per_req, self.num_query_per_req
        )
        return self._mmuse_gumbel_sample(
            logits,
            self.sample_idx_mapping[rows],
            self.temperature,
            self.seeds,
            self.sample_pos[rows] - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self.sample_col[rows],
            use_fp64=self.use_fp64_gumbel,
        )

    def prepare_context_anchor(self, input_batch, num_rejected) -> None:
        count = input_batch.num_reqs
        starts = input_batch.query_start_loc[:count]
        ends = input_batch.query_start_loc[1 : count + 1] - num_rejected[:count]
        valid = ends > starts
        indices = (ends - 1).clamp_min(0).long()
        anchor = self.model.model.context_anchor
        anchor.zero_()
        anchor[:count].copy_(
            self.model.model.hidden_norm(self.hidden_states[indices]) * valid[:, None]
        )

    @torch.inference_mode()
    def _generate_draft(
        self,
        num_reqs,
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=None,
    ) -> None:
        from vllm.config.compilation import CUDAGraphMode

        hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            CUDAGraphMode.NONE,
        )
        if not num_reqs:
            return
        # Include the anchor: the source rollout requires the entire trained block.
        block = hidden[: num_reqs * self.num_query_per_req].reshape(
            num_reqs, self.num_query_per_req, -1
        )
        anchor_ids = self.input_buffers.input_ids[
            : num_reqs * self.num_query_per_req : self.num_query_per_req
        ]
        if self.draft_logits is None:
            tokens, _ = self.model.rollout_mmuse(block, anchor_ids)
        else:
            tokens, _ = self.model.rollout_mmuse(
                block,
                anchor_ids,
                sample_token=lambda logits, position: self._sample_mmuse_token(
                    logits, position, num_reqs
                ),
            )
        self.draft_tokens[:num_reqs].copy_(tokens)

# SPDX-License-Identifier: Apache-2.0
"""Graph-safe MMuse probability proposals with request-indexed logits caching."""

import torch

from vllm.v1.worker.gpu.spec_decode.mmuse.greedy import MMuseGreedySpeculator


class MMuseProbabilisticSampling:
    """Sampling extension shared by our CUDA and Ascend MMuse graph drivers.

    Runtime state is written outside capture into stable buffers. The captured
    callback reads it on every replay; padded rows have an inert state index.
    """

    draft_sample_method = "probabilistic"

    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        self.sample_positions = torch.zeros(
            self.max_num_reqs,
            self.num_query_per_req,
            dtype=torch.int64,
            device=device,
        )
        self.sample_columns = tuple(
            torch.full((self.max_num_reqs,), column, dtype=torch.int32, device=device)
            for column in range(self.num_query_per_req)
        )
        self.sample_offsets = torch.arange(self.num_query_per_req, device=device)
        self.reset_sampling()

    def reset_sampling(self):
        self.idx_mapping.fill_(-1)
        self.sample_positions.zero_()
        self.temperature.zero_()
        self.seeds.zero_()

    def prepare_sampling(self, batch, num_rejected, temperature, seeds, dummy_run):
        if dummy_run:
            self.reset_sampling()
            return
        self.idx_mapping.fill_(-1)
        self.sample_positions.zero_()
        count = batch.num_reqs
        self.idx_mapping[:count].copy_(batch.idx_mapping[:count])
        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        ends = batch.query_start_loc[1 : count + 1] - num_rejected[:count]
        # Each query at P predicts token P+1. Native sampling keys its random
        # draw by the preceding position P, not by the draft column alone.
        positions = batch.positions[(ends - 1).clamp_min(0).long()]
        self.sample_positions[:count].copy_(
            positions[:, None] + 1 + self.sample_offsets
        )

    def _gumbel_sample(self, *args, **kwargs):
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        return gumbel_sample(*args, **kwargs)

    def sample_token(self, logits, column, count):
        if self.draft_logits is None:
            raise RuntimeError("MMuse probabilistic proposals require a logits cache")
        # Quantize once to the precision the verifier reads, before either
        # sampling or caching. Cached scores remain before temperature scaling.
        logits = logits.to(self.draft_logits.dtype).contiguous()
        return self._gumbel_sample(
            logits,
            self.idx_mapping[:count],
            self.temperature,
            self.seeds,
            self.sample_positions[:count, column],
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self.sample_columns[column][:count],
            use_fp64=self.use_fp64_gumbel,
        )

    def rollout_queries(self, block, anchors):
        return self.model.rollout_mmuse(
            block,
            anchors,
            sample_token=lambda logits, column: self.sample_token(
                logits, column, block.shape[0]
            ),
        )


class MMuseProbabilisticSpeculator(MMuseProbabilisticSampling, MMuseGreedySpeculator):
    """Independent Qwen3 MMuse probability proposer for eager and FULL execution."""

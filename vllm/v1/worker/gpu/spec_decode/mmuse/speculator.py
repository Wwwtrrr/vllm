# SPDX-License-Identifier: Apache-2.0
"""Legacy DFlash-based MMuse entry point; not selected by the worker factory."""

from vllm.model_executor.models.mmuse_runtime import (
    MMuseGreedyRolloutMixin,
    validate_mmuse_config,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


class MMuseSpeculator(MMuseGreedyRolloutMixin, DFlashSpeculator):
    _speculator_name = "MMuse"

    def __init__(self, vllm_config, device):
        validate_mmuse_config(vllm_config)
        super().__init__(vllm_config, device)
        self.sample_from_anchor = True
        self.num_query_per_req = self.num_speculative_steps

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        model.bind_target_weights(target_model)
        return model

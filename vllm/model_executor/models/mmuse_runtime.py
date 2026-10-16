# SPDX-License-Identifier: Apache-2.0
"""MMuse execution helpers shared by the GPU and Ascend speculators."""

import torch
from torch import nn


def build_mmuse_runtime(source_config: dict, fc: nn.Module) -> nn.Module:
    """Construct inference heads without a training package or second decoder."""
    from vllm.model_executor.models.mmuse.runtime import MMuseRuntime

    return MMuseRuntime(source_config, fc).eval()


def validate_mmuse_config(config) -> None:
    """Reject execution modes not covered by the reference adapter."""
    spec = config.speculative_config
    hf = spec.draft_model_config.hf_config
    source = getattr(hf, "mmuse_source_config", None)
    if not source:
        raise ValueError("MMuse requires a Speculators-format MMuse checkpoint")
    if config.model_config.hf_config.model_type != "qwen3":
        raise ValueError("MMuse reference adapter requires a Qwen3 target")
    if source["transformer_layer_config"].get("model_type") != "qwen3":
        raise ValueError("MMuse reference adapter requires a Qwen3 draft decoder")
    parallel = config.parallel_config
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"MMuse reference adapter requires {name}=1")
    # if not config.model_config.enforce_eager or not spec.enforce_eager:
    #     raise ValueError("MMuse requires target and draft enforce_eager=True")
    # if spec.draft_sample_method != "greedy":
    #     raise ValueError("MMuse reference adapter only supports greedy drafting")
    # if spec.enable_adaptive_verification:
    #     raise ValueError("MMuse adaptive verification is not implemented")
    # if getattr(config, "watermark_config", None) is not None:
    #     raise ValueError("MMuse reference adapter does not support watermarking")
    # if config.model_config.quantization or spec.draft_model_config.quantization:
    #     raise ValueError("MMuse reference adapter requires unquantized weights")
    # if not source.get("sample_from_anchor", True):
    #     raise ValueError("MMuse reference adapter requires sample_from_anchor=True")
    # if spec.num_speculative_tokens != source["block_size"]:
    #     raise ValueError("MMuse num_speculative_tokens must equal trained block_size")
    # if not source.get("enable_correction_head", False):
    #     raise ValueError("MMuse reference adapter requires a Correction head")
    # if not source.get("use_aux_hidden_state", True):
    #     raise ValueError("MMuse reference adapter requires auxiliary hidden states")
    # vocab_size = config.model_config.get_vocab_size()
    # if (source.get("draft_vocab_size") or hf.vocab_size) != vocab_size:
    #     raise ValueError("MMuse reference adapter requires the full target vocabulary")
    # if hf.hidden_size != config.model_config.get_hidden_size():
    #     raise ValueError("MMuse target and draft hidden sizes must match")
    if any(
        layer_type != "full_attention"
        for layer_type in (getattr(hf, "layer_types", None) or [])
    ):
        raise ValueError("MMuse reference adapter requires full-attention draft layers")
    layer_ids = source["aux_hidden_state_layer_ids"]
    num_layers = config.model_config.get_total_num_hidden_layers()
    if not layer_ids or any(i < 1 or i > num_layers for i in layer_ids):
        raise ValueError("MMuse auxiliary layer IDs exceed the target's layer range")


class MMuseGreedyRolloutMixin:
    """Shared GPU/NPU proposal stage; feedback stays inside the source rollout."""

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
        tokens, _ = self.model.rollout_mmuse(block, anchor_ids)
        self.draft_tokens[:num_reqs].copy_(tokens)

# SPDX-License-Identifier: Apache-2.0
"""MMuse execution helpers shared by the GPU and Ascend speculators."""

import copy

import torch
from torch import nn


def build_mmuse_runtime(source_config: dict, fc: nn.Module) -> nn.Module:
    """Construct inference heads without a training package or second decoder."""
    from vllm.model_executor.models.mmuse.runtime import MMuseRuntime

    return MMuseRuntime(source_config, fc).eval()


def validate_mmuse_config(config) -> None:
    """Check required model interfaces and currently supported execution modes.

    This check does not validate checkpoint tensors or end-to-end correctness.
    Detailed MMuse option validation belongs to MMuseRuntimeConfig.
    """
    spec = config.speculative_config
    hf = spec.draft_model_config.hf_config
    source = getattr(hf, "mmuse_source_config", None)

    # Required configuration and interfaces used by the Qwen3 adapter.
    if not source:
        raise ValueError("MMuse requires a Speculators-format MMuse checkpoint")
    target_type = config.model_config.hf_config.model_type
    if target_type not in ("qwen3", "deepseek_v4"):
        raise ValueError("MMuse requires a Qwen3 or DeepSeek-V4 target")
    if source["transformer_layer_config"].get("model_type") != "qwen3":
        raise ValueError("MMuse reference adapter requires a Qwen3 draft decoder")
    layer_ids = source["aux_hidden_state_layer_ids"]
    num_layers = config.model_config.get_total_num_hidden_layers()
    if not layer_ids or any(i < 1 or i > num_layers for i in layer_ids):
        raise ValueError("MMuse auxiliary layer IDs exceed the target's layer range")
    if target_type == "deepseek_v4":
        validate_mmuse_deepseek_v4_config(config, source)

    # Execution modes not yet covered by this adapter.
    parallel = config.parallel_config
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if name == "tensor_parallel_size" and target_type == "deepseek_v4":
            continue
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"MMuse reference adapter requires {name}=1")
    if any(
        layer_type
        not in (
            ("full_attention", "sliding_attention")
            if target_type == "deepseek_v4"
            else ("full_attention",)
        )
        for layer_type in (getattr(hf, "layer_types", None) or [])
    ):
        raise ValueError("MMuse draft layer_types are unsupported for this target")

    # Additional contracts for the probabilistic sampler and its logits cache.
    if spec.draft_sample_method == "probabilistic":
        if not config.model_config.enforce_eager or not spec.enforce_eager:
            raise ValueError("MMuse probabilistic drafting requires enforce_eager=True")
        if getattr(config, "watermark_config", None) is not None:
            raise ValueError(
                "MMuse probabilistic drafting does not support watermarking"
            )
        if spec.enable_adaptive_verification:
            raise ValueError(
                "MMuse probabilistic adaptive verification is not supported"
            )
        if not source.get("sample_from_anchor", True):
            raise ValueError(
                "MMuse probabilistic drafting requires sample_from_anchor=True"
            )
        if spec.num_speculative_tokens != source["block_size"]:
            raise ValueError(
                "MMuse num_speculative_tokens must equal trained block_size"
            )


def validate_mmuse_deepseek_v4_config(config, source) -> None:
    """Validate the dense drafter contract used by speculators_dsv4 training."""
    spec = config.speculative_config
    target = config.model_config
    hf = spec.draft_model_config.hf_config
    layer_ids = source["aux_hidden_state_layer_ids"]
    if layer_ids != sorted(set(layer_ids)):
        raise ValueError(
            "DeepSeek-V4 MMuse auxiliary layer IDs must be unique and ascending"
        )
    target_tp = config.parallel_config.tensor_parallel_size
    draft_parallel = getattr(spec, "draft_parallel_config", None)
    if draft_parallel is not None and draft_parallel.tensor_parallel_size != target_tp:
        raise ValueError(
            "DeepSeek-V4 MMuse currently requires matching "
            "target/draft tensor parallel sizes"
        )
    if source.get("target_hidden_state_format") != "deepseek_v4_mean_hc_head":
        raise ValueError(
            "DeepSeek-V4 MMuse requires "
            "target_hidden_state_format=deepseek_v4_mean_hc_head"
        )
    if not target.enforce_eager or not spec.enforce_eager:
        raise ValueError(
            "DeepSeek-V4 MMuse initially requires target and draft enforce_eager=True"
        )
    if target.dtype != torch.bfloat16:
        raise ValueError(
            "DeepSeek-V4 MMuse requires BF16 activations, "
            "independently of target weight quantization"
        )
    if "sliding_attention" in (getattr(hf, "layer_types", None) or []) and getattr(
        getattr(config, "cache_config", None), "enable_prefix_caching", False
    ):
        raise ValueError(
            "DeepSeek-V4 MMuse anchor attention requires enable_prefix_caching=False "
            "to retain verified target features for every request"
        )
    if hf.hidden_size != target.get_hidden_size():
        raise ValueError(
            "DeepSeek-V4 MMuse borrowed embedding/head require "
            "equal target/draft hidden sizes"
        )
    if (source.get("target_hidden_size") or hf.hidden_size) != target.get_hidden_size():
        raise ValueError("MMuse target_hidden_size differs from the loaded target")
    if (source.get("draft_vocab_size") or hf.vocab_size) != target.get_vocab_size():
        raise ValueError("DeepSeek-V4 MMuse requires the full target vocabulary")
    if any(
        i >= target.get_total_num_hidden_layers()
        for i in source["aux_hidden_state_layer_ids"]
    ):
        raise ValueError(
            "DeepSeek-V4 MMuse auxiliary IDs must precede the "
            "final teacher HC-head slot"
        )
    if (
        not source.get("sample_from_anchor", True)
        or spec.num_speculative_tokens != source["block_size"]
    ):
        raise ValueError(
            "DeepSeek-V4 MMuse requires sample_from_anchor=True "
            "and the trained block_size"
        )
    if getattr(spec.draft_model_config, "quantization", None):
        raise ValueError(
            "DeepSeek-V4 MMuse requires an unquantized draft; "
            "target W4A8 is independent"
        )
    if getattr(spec, "kv_cache_dtype", None) not in (None, "auto", "bfloat16"):
        raise ValueError("DeepSeek-V4 MMuse requires a BF16 draft KV cache")
    contract = source.get("target_training_contract")
    if contract is not None:
        expected = {
            "format": "deepseek_v4_mean_hc_head",
            "hidden_size": target.get_hidden_size(),
            "auxiliary_hs_ids": source["aux_hidden_state_layer_ids"],
            "teacher_hs_id": target.get_total_num_hidden_layers(),
            "target_dtype": "bfloat16",
        }
        if not isinstance(contract, dict) or any(
            contract.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(
                "DeepSeek-V4 MMuse target_training_contract "
                "disagrees with the loaded model interfaces"
            )


class MMuseGreedyRolloutMixin:
    """Shared GPU/NPU rollout with greedy or framework probabilistic sampling."""

    def propose(self, *args, **kwargs):
        dummy_run = kwargs.get("dummy_run", args[12] if len(args) > 12 else False)
        source = self.draft_model_config.hf_config
        self._mmuse_anchor_active = (
            not dummy_run
            and getattr(source, "target_hidden_state_format", None)
            == "deepseek_v4_mean_hc_head"
            and "sliding_attention" in (getattr(source, "layer_types", None) or [])
        )
        try:
            return super().propose(*args, **kwargs)
        finally:
            self._mmuse_anchor_active = False
            if hasattr(self, "model"):
                self.model._anchor_contexts = None

    def _precompute_context_kv(self, start, end, dummy_run=False):
        if getattr(self, "_mmuse_anchor_active", False):
            # Anchor attention projects the verified feature history directly.
            return
        return super()._precompute_context_kv(start, end, dummy_run)

    @property
    def attn_vllm_config(self):
        config = super().attn_vllm_config
        if self.vllm_config.model_config.hf_config.model_type == "deepseek_v4":
            # Dense draft attention must not inherit the target's DSA or W4A8.
            config = copy.copy(config)
            config.model_config = self.speculative_config.draft_model_config
            config.quant_config = None
            config.cache_config = copy.copy(config.cache_config)
            config.cache_config.cache_dtype = (
                self.speculative_config.kv_cache_dtype or "auto"
            )
        return config

    def load_draft_model(self, target_model, target_attn_layer_names):
        if (
            self.vllm_config.model_config.hf_config.model_type == "deepseek_v4"
            and self.speculative_config.kv_cache_dtype is None
        ):
            self.speculative_config.kv_cache_dtype = "auto"
        return super().load_draft_model(target_model, target_attn_layer_names)

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
        if getattr(self, "_mmuse_anchor_active", False):
            if hasattr(self.model, "_anchor_feature_storage"):
                self.model.anchor_context_cache.storage = (
                    self.model._anchor_feature_storage
                )
            contexts = []
            for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
                if end <= start:
                    raise ValueError("MMuse anchor path requires verified context rows")
                contexts.append(
                    self.model.anchor_context_cache.update(
                        int(input_batch.idx_mapping_np[row]),
                        input_batch.req_ids[row],
                        self.target_input_buffers.positions[start:end],
                        self.hidden_states[start:end],
                    )
                )
            self.model._anchor_contexts = contexts

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

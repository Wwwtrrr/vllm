# SPDX-License-Identifier: Apache-2.0
"""Independent eager speculator for dense MMuse trained on DeepSeek-V4.

Block semantics follow dspark_offline_eval._single_anchor_backbone and MMuse's
rollout_correction. DFlash proposal kernels, query graphs and paged draft KV
updates are not part of this execution path.
"""

import torch

from vllm.config import replace
from vllm.model_executor.models.mmuse_runtime import validate_mmuse_config
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class DeepseekV4MMuseSpeculator(DraftModelSpeculator):
    _speculator_name = "DeepSeek-V4 MMuse"

    def __init__(self, vllm_config, device):
        validate_mmuse_config(vllm_config)
        if vllm_config.model_config.hf_config.model_type != "deepseek_v4":
            raise ValueError("DeepseekV4MMuseSpeculator requires a DeepSeek-V4 target")
        spec = vllm_config.speculative_config
        if not spec.draft_model_config.hf_config.mmuse_source_config.get(
            "enable_correction_head", False
        ):
            raise ValueError(
                "Independent DeepSeek MMuse requires a trained Correction head"
            )
        if spec.enable_adaptive_verification:
            raise ValueError(
                "Independent DeepSeek MMuse does not support adaptive verification"
            )
        if getattr(vllm_config, "watermark_config", None) is not None:
            raise ValueError("Independent DeepSeek MMuse does not support watermarking")
        if spec.use_local_argmax_reduction:
            raise ValueError(
                "Independent DeepSeek MMuse uses full-vocabulary Correction logits"
            )
        super().__init__(vllm_config, device)
        self.num_query_per_req = self.num_speculative_steps
        self.sample_from_anchor = True
        self.query_cudagraph_manager = None

    def load_draft_model(self, target_model, target_attn_layer_names):
        from vllm.compilation.backends import set_model_tag
        from vllm.model_executor.model_loader import get_model
        from vllm.model_executor.model_loader.utils import get_draft_load_config
        from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config

        draft_config = replace(
            self.vllm_config,
            cache_config=replace(
                self.vllm_config.cache_config,
                cache_dtype=self.speculative_config.kv_cache_dtype or "auto",
            ),
            load_config=get_pp_safe_draft_load_config(
                get_draft_load_config(self.vllm_config)
            ),
            attention_config=replace(
                self.vllm_config.attention_config,
                backend=self.speculative_config.attention_backend,
                use_non_causal=(
                    "full_attention"
                    in (
                        getattr(self.draft_model_config.hf_config, "layer_types", None)
                        or []
                    )
                    or getattr(
                        self.draft_model_config.hf_config,
                        "sliding_window_non_causal",
                        False,
                    )
                ),
            ),
        )
        with set_model_tag("mmuse_head"):
            model = get_model(
                vllm_config=draft_config, model_config=self.draft_model_config
            )
        model.bind_target_weights(target_model)
        return model

    def set_attn(
        self,
        model_state,
        kv_cache_config,
        block_tables,
        target_input_buffers,
        target_attn_groups,
    ):
        # The runner may reserve draft cache groups because the native decoder
        # modules register attention layers. This speculator reads only its
        # verified feature history, not those paged groups.
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.target_input_buffers = target_input_buffers
        self.target_attn_groups = target_attn_groups

    def init_cudagraph_manager(self, cudagraph_mode):
        self.query_cudagraph_manager = None

    def capture(self):
        return

    def _gumbel_sample(self, *args, **kwargs):
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        return gumbel_sample(*args, **kwargs)

    def _prepare_blocks(
        self,
        input_batch,
        projected,
        num_rejected,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
    ):
        count = input_batch.num_reqs
        block = self.num_query_per_req
        if count * block > self.input_buffers.input_ids.numel():
            raise ValueError("MMuse query block exceeds the allocated input buffer")
        starts = input_batch.query_start_loc[:count].tolist()
        ends = (
            input_batch.query_start_loc[1 : count + 1] - num_rejected[:count]
        ).tolist()
        # One transfer also makes the prefix boundary checks deterministic in
        # eager mode. It is not a graph-compatible input-preparation routine.
        sampled_counts = num_sampled[:count].tolist()
        contexts = []
        cache = self.model.anchor_context_cache
        cache.storage = self.model._anchor_feature_storage
        for row, (start, end) in enumerate(zip(starts, ends)):
            if end <= start:
                raise ValueError("MMuse requires a verified prefix before its anchor")
            positions = input_batch.positions[start:end]
            slot = int(input_batch.idx_mapping_np[row])
            context = cache.update(
                slot, input_batch.req_ids[row], positions, projected[start:end]
            )
            contexts.append(context)
            anchor_position = positions[-1] + 1
            ids = self.input_buffers.input_ids[row * block : (row + 1) * block]
            ids.fill_(self.model.config.mask_token_id)
            if sampled_counts[row] > 0:
                ids[0] = last_sampled[slot]
            else:
                # Prefill lookahead is [lookahead, persistent request slot].
                ids[0] = next_prefill_tokens[0, slot]
            self.input_buffers.positions[row * block : (row + 1) * block].copy_(
                anchor_position + torch.arange(block, device=self.device)
            )
        return contexts

    def _dummy_blocks(self, count):
        # Exercise the same attention/Correction implementation while profiling,
        # without committing fake rows into the real request history.
        window = self.model.anchor_context_cache.window
        length = min(
            window or self.model._anchor_feature_storage.shape[1], self.max_model_len
        )
        positions = torch.arange(length, device=self.device)
        features = torch.zeros(
            length, self.hidden_size, dtype=self.dtype, device=self.device
        )
        block = self.num_query_per_req
        self.input_buffers.input_ids[: count * block].fill_(
            self.model.config.mask_token_id
        )
        query_positions = length + torch.arange(block, device=self.device)
        for row in range(count):
            self.input_buffers.positions[row * block : (row + 1) * block].copy_(
                query_positions
            )
        return [(positions, features)] * count

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        count = input_batch.num_reqs
        if count == 0:
            return self.draft_tokens[:0]
        if dummy_run:
            if aux_hidden_states:
                # Include projection workspace in the worker's memory profile.
                self.model.combine_hidden_states(torch.cat(aux_hidden_states, -1))
            contexts = self._dummy_blocks(count)
        else:
            if not aux_hidden_states:
                raise ValueError("DeepSeek MMuse requires mean-HC auxiliary states")
            projected = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, -1)
            )
            contexts = self._prepare_blocks(
                input_batch,
                projected,
                num_rejected,
                num_sampled,
                last_sampled,
                next_prefill_tokens,
            )
        tokens = count * self.num_query_per_req
        self._prepare_eplb_forward(tokens)
        self.model._anchor_contexts = contexts
        try:
            hidden = self.model(
                self.input_buffers.input_ids[:tokens],
                self.input_buffers.positions[:tokens],
            ).reshape(count, self.num_query_per_req, -1)
            anchors = self.input_buffers.input_ids[: tokens : self.num_query_per_req]

            def sample(logits, position):
                # Proposal at query position p predicts token p+1. The native
                # sampler seeds by the source position p, as in ordinary drafting.
                slots = input_batch.idx_mapping[:count]
                return self._gumbel_sample(
                    logits.to(self.draft_logits.dtype).contiguous(),
                    slots,
                    temperature,
                    seeds,
                    self.input_buffers.positions[
                        position : tokens : self.num_query_per_req
                    ],
                    apply_temperature=True,
                    is_drafting=True,
                    logits_cache=self.draft_logits,
                    logits_cache_col=torch.full_like(slots, position),
                    use_fp64=self.use_fp64_gumbel,
                )

            proposed, _ = self.model.rollout_mmuse(
                hidden,
                anchors,
                sample_token=sample if self.draft_logits is not None else None,
            )
            self.draft_tokens[:count].copy_(proposed)
        finally:
            self.model._anchor_contexts = None
        return self.draft_tokens[:count]

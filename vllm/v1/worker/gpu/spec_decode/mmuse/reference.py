# SPDX-License-Identifier: Apache-2.0
"""Verified-prefix MMuse proposal loop, independent of DFlash scheduling kernels."""

import copy

import torch

from vllm.model_executor.models.mmuse_runtime import validate_mmuse_config
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class MMuseReferenceSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config, device):
        validate_mmuse_config(vllm_config)
        super().__init__(vllm_config, device)
        self.num_query_per_req = self.num_speculative_steps

    def load_draft_model(self, target_model, target_attn_layer_names):
        from vllm.model_executor.model_loader import get_model
        from vllm.model_executor.model_loader.utils import get_draft_load_config

        # Retain target metadata for the training contract, but never run the
        # target quantizer against the independently loaded dense draft.
        draft_context = copy.copy(self.vllm_config)
        draft_context.load_config = get_draft_load_config(self.vllm_config)
        draft_context.quant_config = None
        model = get_model(
            vllm_config=draft_context, model_config=self.draft_model_config
        )
        model.bind_target(target_model)
        return model

    def set_attn(
        self,
        model_state,
        kv_cache_config,
        block_tables,
        target_input_buffers,
        target_attn_groups,
    ):
        # No draft AttentionLayerBase objects are registered: target caches and
        # their Ascend allocator are unchanged by the replicated reference draft.
        self.target_input_buffers = target_input_buffers

    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def sample(self, *args, **kwargs):
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        return gumbel_sample(*args, **kwargs)

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
        count, block = input_batch.num_reqs, self.num_query_per_req
        if not count:
            return self.draft_tokens[:0]
        model = self.model
        if not dummy_run and not aux_hidden_states:
            raise ValueError(
                "MMuse requires the trained mean-HC auxiliary feature list"
            )
        if not dummy_run:
            expected = model.config.mmuse_source_config["aux_hidden_state_layer_ids"]
            if len(aux_hidden_states) != len(expected) or any(
                state.ndim != 2
                or state.shape[0] != input_batch.positions.shape[0]
                or state.shape[1] != self.hidden_size
                for state in aux_hidden_states
            ):
                raise ValueError(
                    "MMuse auxiliary tensors do not match "
                    "trained layers and target rows"
                )
        projected = (
            model.project_features(torch.cat(aux_hidden_states, -1))
            if aux_hidden_states
            else None
        )
        starts = input_batch.query_start_loc[:count].tolist()
        ends = (
            input_batch.query_start_loc[1 : count + 1] - num_rejected[:count]
        ).tolist()
        sampled = num_sampled[:count].tolist()
        hidden_blocks, anchor_ids, source_positions = [], [], []
        model.history.storage = model.feature_storage
        for row in range(count):
            slot = int(input_batch.idx_mapping_np[row])
            if dummy_run:
                length = model.feature_storage.shape[1]
                context_positions = torch.arange(length, device=self.device)
                features = torch.zeros_like(model.feature_storage[slot])
                anchor = torch.tensor(model.config.mask_token_id, device=self.device)
            else:
                start, end = starts[row], ends[row]
                if end <= start:
                    raise ValueError("MMuse requires a nonempty verified target span")
                context_positions, features = model.history.append(
                    slot,
                    input_batch.req_ids[row],
                    input_batch.positions[start:end],
                    projected[start:end],
                )
                anchor = (
                    last_sampled[slot] if sampled[row] else next_prefill_tokens[0, slot]
                )
            positions = (
                context_positions[-1] + 1 + torch.arange(block, device=self.device)
            )
            ids = torch.full(
                (block,),
                model.config.mask_token_id,
                device=self.device,
                dtype=torch.long,
            )
            ids[0] = anchor
            hidden_blocks.append(model(ids, positions, context_positions, features))
            anchor_ids.append(anchor)
            source_positions.append(positions)
        hidden = torch.stack(hidden_blocks)
        positions = torch.stack(source_positions)

        def sample_token(logits, position):
            slots = input_batch.idx_mapping[:count]
            return self.sample(
                logits.to(self.draft_logits.dtype).contiguous(),
                slots,
                temperature,
                seeds,
                positions[:, position],
                apply_temperature=True,
                is_drafting=True,
                logits_cache=self.draft_logits,
                logits_cache_col=torch.full_like(slots, position),
                use_fp64=self.use_fp64_gumbel,
            )

        proposals, _ = model.runtime.rollout_correction(
            hidden,
            torch.stack(anchor_ids),
            temperature=0.0,
            sample_token=sample_token if self.draft_logits is not None else None,
        )
        self.draft_tokens[:count].copy_(proposals)
        return self.draft_tokens[:count]

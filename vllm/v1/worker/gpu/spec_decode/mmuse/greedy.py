# SPDX-License-Identifier: Apache-2.0
"""Qwen3 MMuse execution and FULL graph lifecycle without a DFlash proposer."""

import copy
from contextlib import nullcontext

import torch

from vllm.compilation.backends import set_model_tag
from vllm.config import CompilationMode, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.model_executor.models.mmuse.validation import validate_mmuse_config
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.mmuse.inputs import prepare_mmuse_inputs
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import (
    get_parallel_drafting_token_id,
    get_pp_safe_draft_load_config,
)

logger = init_logger(__name__)


class MMuseGraphManager(CudaGraphManager):
    """Capture query decoder plus the complete MMuse rollout.

    Context K/V projection stays outside the graph, including for long prefills.
    Only persistent query buffers and the output token buffer cross replay.
    """

    def capture_mmuse(self, proposer):
        def prepare(desc, warmup):
            count = desc.num_reqs
            assert count is not None
            batch = InputBatch.make_dummy(
                count, desc.num_tokens, proposer.input_buffers
            )
            proposer.reset_sampling()
            tables = proposer.block_tables.get_dummy_block_tables(count)
            slots = proposer.block_tables.get_dummy_slot_mappings(desc.num_tokens)
            metadata = proposer.capture_metadata(
                attn_groups=proposer.attn_groups,
                num_reqs=count,
                num_tokens=desc.num_tokens,
                query_start_loc_gpu=batch.query_start_loc,
                query_start_loc_cpu=torch.from_numpy(batch.query_start_loc_np),
                max_query_len=proposer.num_query_per_req,
                seq_lens=batch.seq_lens,
                max_seq_len=proposer.max_model_len,
                block_tables=tables,
                slot_mappings=slots,
                kv_cache_config=proposer.kv_cache_config,
                for_cudagraph_capture=True,
                causal=False,
            )
            proposer.fix_query_lengths(metadata, count)
            by_layer = build_slot_mappings_by_layer(slots, proposer.kv_cache_config)
            return lambda mode: proposer.forward_queries(
                count, desc.num_tokens, metadata, by_layer
            )

        super().capture(prepare, "Capturing MMuse query and rollout")


class MMuseGreedySpeculator(DraftModelSpeculator):
    """Single-device, full-attention Qwen3 adapter; no DFlash runtime inheritance."""

    graph_manager_cls = MMuseGraphManager
    capture_metadata = staticmethod(build_attn_metadata)
    draft_sample_method = "greedy"

    def __init__(self, vllm_config, device):
        validate_mmuse_config(vllm_config)
        spec = vllm_config.speculative_config
        source = spec.draft_model_config.hf_config.mmuse_source_config
        if spec.draft_sample_method != self.draft_sample_method:
            raise ValueError(
                f"This MMuse proposer requires {self.draft_sample_method} drafting"
            )
        if spec.num_speculative_tokens != source["block_size"]:
            raise ValueError("MMuse query length must equal the trained block_size")
        if not source.get("sample_from_anchor", True):
            raise ValueError("MMuse requires sample_from_anchor=True")
        if spec.enable_adaptive_verification:
            raise ValueError("MMuse FULL does not support adaptive verification")
        if spec.uses_dynamic_speculative_decoding():
            raise ValueError("MMuse requires a fixed trained query block length")
        if getattr(vllm_config, "watermark_config", None) is not None:
            raise ValueError("MMuse does not support watermarking")
        # Device graph capture does not require Dynamo compilation. Do not
        # propagate this draft-only choice back into the Target configuration.
        config = copy.copy(vllm_config)
        config.compilation_config = copy.copy(vllm_config.compilation_config)
        config.compilation_config.mode = CompilationMode.NONE
        super().__init__(config, device)
        self.num_query_per_req = self.num_speculative_steps
        if self.max_num_reqs * self.num_query_per_req > self.max_num_tokens:
            raise ValueError("MMuse query capacity exceeds max_num_batched_tokens")
        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self.mask_token_id = get_parallel_drafting_token_id(
            self.draft_model_config.hf_config
        )
        self.query_cudagraph_manager = None

    @property
    def attn_vllm_config(self):
        config = copy.copy(self.vllm_config)
        config.attention_config = replace(
            config.attention_config,
            use_non_causal=True,
            backend=self.speculative_config.attention_backend,
        )
        return config

    def load_draft_model(self, target_model, target_attn_layer_names):
        config = self.attn_vllm_config
        config.load_config = get_pp_safe_draft_load_config(
            get_draft_load_config(self.vllm_config)
        )
        if self.speculative_config.kv_cache_dtype is not None:
            config.cache_config = replace(
                config.cache_config, cache_dtype=self.speculative_config.kv_cache_dtype
            )
        with set_model_tag("mmuse_draft"):
            model = get_model(vllm_config=config, model_config=self.draft_model_config)
        model.bind_target_weights(target_model)
        return model

    def set_attn(self, *args, **kwargs):
        super().set_attn(*args, **kwargs)
        self.group_ids = [gid for gid, groups in enumerate(self.attn_groups) if groups]
        name_to_group = {
            name: gid
            for gid, group in enumerate(self.kv_cache_config.kv_cache_groups)
            for name in group.layer_names
        }
        self.layer_groups = [
            name_to_group[name] for name in self.model.get_draft_kv_cache_layer_names()
        ]
        self.context_slots = {
            gid: torch.full(
                (self.max_num_tokens,), -1, dtype=torch.int64, device=self.device
            )
            for gid in self.group_ids
        }

    def init_cudagraph_manager(self, mode):
        full = mode.decode_mode() == CUDAGraphMode.FULL
        supported = (
            self.attn_cg_support.min_cg_support.value
            >= AttentionCGSupport.UNIFORM_BATCH.value
        )
        if full and not self.speculative_config.enforce_eager and not supported:
            raise ValueError(
                "MMuse FULL requires uniform-batch graph attention support"
            )
        mode = (
            CUDAGraphMode.FULL_DECODE_ONLY
            if full and not self.speculative_config.enforce_eager
            else CUDAGraphMode.NONE
        )
        self.query_cudagraph_manager = self.graph_manager_cls(
            self.vllm_config, self.device, mode, decode_query_len=self.num_query_per_req
        )
        if (
            mode != CUDAGraphMode.NONE
            and not self.query_cudagraph_manager.needs_capture()
        ):
            raise ValueError(
                "No complete MMuse block fits cudagraph_capture_sizes; include "
                "num_speculative_tokens in the capture sizes"
            )
        logger.info(
            "Independent MMuse draft: graph=%s, query length=%d, "
            "sampling=%s, torch.compile=NONE",
            mode.name,
            self.num_query_per_req,
            self.draft_sample_method,
        )

    def capture(self):
        self.model.model.context_anchor.zero_()
        self.query_cudagraph_manager.capture_mmuse(self)

    def fix_query_lengths(self, metadata, count):
        """Platform hook for attention's padded uniform query lengths."""

    def query_context(self):
        return nullcontext()

    def reset_sampling(self):
        """Platform-independent capture hook; greedy has no sampling state."""

    def prepare_sampling(self, batch, num_rejected, temperature, seeds, dummy_run):
        """Publish per-request state before replay; greedy needs none."""

    def rollout_queries(self, block, anchors):
        return self.model.rollout_mmuse(block, anchors)

    def build_query_metadata(self, batch, desc):
        metadata = self._build_uniform_attn_metadata(
            num_reqs=batch.num_reqs,
            batch_desc=desc,
            num_query_per_req=self.num_query_per_req,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
            step=self.num_query_per_req,
            causal=False,
        )
        self.fix_query_lengths(metadata, desc.num_reqs or batch.num_reqs)
        return metadata

    @torch.inference_mode()
    def forward_queries(self, count, total, metadata, slots):
        with (
            set_forward_context(
                metadata,
                self.vllm_config,
                num_tokens=total,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=slots,
                batch_descriptor=BatchDescriptor(num_tokens=total),
            ),
            self.query_context(),
        ):
            hidden = self.model(
                input_ids=self.input_buffers.input_ids[:total],
                positions=self.input_buffers.positions[:total],
                inputs_embeds=None,
            )
        block = hidden.reshape(count, self.num_query_per_req, -1)
        anchors = self.input_buffers.input_ids[: total : self.num_query_per_req]
        tokens, _ = self.rollout_queries(block, anchors)
        self.draft_tokens[:count].copy_(tokens)

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
        count, total = input_batch.num_reqs, input_batch.num_tokens
        if count == 0:
            return self.draft_tokens[:0]
        self.prepare_sampling(input_batch, num_rejected, temperature, seeds, dummy_run)
        query_total = count * self.num_query_per_req
        hidden = (
            self.model.combine_hidden_states(torch.cat(aux_hidden_states, -1))
            if aux_hidden_states
            else last_hidden_states
        )
        self.hidden_states[:total].copy_(hidden[:total])
        ends = input_batch.query_start_loc[1 : count + 1] - num_rejected[:count]
        valid = ends > input_batch.query_start_loc[:count]
        anchor = self.model.model.context_anchor
        anchor.zero_()
        anchor[:count].copy_(
            self.model.model.hidden_norm(
                self.hidden_states[(ends - 1).clamp_min(0).long()]
            )
            * valid[:, None]
        )
        if dummy_run and skip_attn_for_dummy_run:
            self.input_buffers.input_ids[:query_total].fill_(self.mask_token_id)
            self.input_buffers.positions[:query_total].zero_()
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:total], self.context_positions[:total]
            )
            self.forward_queries(count, query_total, None, None)
            return self.draft_tokens[:count]
        for gid in self.group_ids:
            prepare_mmuse_inputs(
                self.input_buffers,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                self.context_positions,
                self.context_slots[gid],
                self.block_tables.slot_mappings[gid],
                self.block_tables.input_block_tables[gid],
                self.block_tables.kernel_block_sizes[gid],
                self.num_query_per_req,
                self.mask_token_id,
                self.max_model_len,
            )
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:total],
            self.context_positions[:total],
            None
            if dummy_run
            else [self.context_slots[gid][:total] for gid in self.layer_groups],
        )
        if is_profile or self.query_cudagraph_manager is None:
            desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE, num_tokens=query_total, num_reqs=count
            )
        else:
            desc = self.query_cudagraph_manager.dispatch(
                count,
                query_total,
                uniform_token_count=self.num_query_per_req,
                num_active_loras=0,
            )
        self.draft_max_seq_len = min(
            int(input_batch.seq_lens_cpu_upper_bound[:count].max())
            + self.num_query_per_req,
            self.max_model_len,
        )
        metadata = self.build_query_metadata(input_batch, desc)
        slots = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, : desc.num_tokens], self.kv_cache_config
        )
        if desc.cg_mode == CUDAGraphMode.FULL:
            self.replay_metadata = metadata
            self.query_cudagraph_manager.run_fullgraph(desc)
        else:
            self.forward_queries(count, query_total, metadata, slots)
        return self.draft_tokens[:count]

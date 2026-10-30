# SPDX-License-Identifier: Apache-2.0
"""DeepSeek MMuse fixed-context query graphs with reference history ownership."""

import copy

import torch

from vllm.config import CompilationMode
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.models.mmuse.deepseek_context_graph import (
    DeepseekMMuseContextGraphs,
)
from vllm.model_executor.models.mmuse.deepseek_features import (
    DeepseekMMuseFeatureGraphs,
)
from vllm.model_executor.models.mmuse.deepseek_graph import DeepseekMMuseGraphs
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.mmuse.reference import MMuseReferenceSpeculator

logger = init_logger(__name__)


class DeepseekMMuseGraphManager(CudaGraphManager):
    """Expose native graph ownership to the runner's profiling cleanup."""

    def __init__(self, config, device, inputs):
        super().__init__(config, device, CUDAGraphMode.NONE, inputs.block_size)
        self.inputs = inputs
        self.context_graphs = {}
        self.feature_graphs = {}
        self.feature_pool = None
        self.context_prepare_graphs = {}
        self.context_prepare_pool = None

    def capture(self):
        features = getattr(self.inputs, "feature_inputs", None)
        if features is not None:
            features.capture()
            self.feature_graphs = features.graphs
            features.graphs = {}
            self.feature_pool = features.pool
        context = getattr(self.inputs, "context_inputs", None)
        if context is not None:
            context.capture()
            self.context_prepare_graphs = context.graphs
            context.graphs = {}
            self.context_prepare_pool = context.pool
        self.inputs.capture()
        # The runner drops CudaGraphManager instances after memory profiling.
        # Keep graph references here, not in the persistent input buffers.
        self.context_graphs = self.inputs.graphs
        self.inputs.graphs = {}
        self.pool = self.inputs.pool

    def project_features(self, states):
        return self.inputs.feature_inputs.project(states, self.feature_graphs)

    def prepare_context(self, features, positions):
        return self.inputs.context_inputs.project(
            features, positions, self.context_prepare_graphs
        )

    def replay(self, size):
        self.context_graphs[size].replay()


class DeepseekMMuseFullSpeculator(MMuseReferenceSpeculator):
    """Replicated per-request graphs; Target graph/cache facilities are unchanged."""

    def __init__(self, config, device):
        if (getattr(config, "additional_config", None) or {}).get(
            "mmuse_draft_kv_reuse", False
        ):
            raise ValueError(
                "DeepSeek MMuse FULL uses its own context buffers; "
                "disable mmuse_draft_kv_reuse or use an eager draft"
            )
        config = copy.copy(config)
        config.compilation_config = copy.copy(config.compilation_config)
        config.compilation_config.mode = CompilationMode.NONE
        super().__init__(config, device)
        self.full_enabled = False
        self.fixed_graphs = None
        self.query_cudagraph_manager = None

    def graph_backend(self):
        return torch.cuda

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        # Allocate before memory profiling, so graph context storage is counted.
        self.fixed_graphs = DeepseekMMuseGraphs(
            model,
            self.num_query_per_req,
            self.sample,
            self.draft_logits,
            self.temperature,
            self.seeds,
            self.use_fp64_gumbel,
            self.graph_backend(),
        )
        # Decode-sized row buckets cover the normal verifier batches without
        # reserving another max-prefill-size auxiliary-feature arena.
        capacity = min(
            self.max_num_tokens,
            max(
                self.max_num_reqs * (self.num_query_per_req + 1),
                max(self.vllm_config.compilation_config.cudagraph_capture_sizes or [1]),
            ),
        )
        self.fixed_graphs.feature_inputs = DeepseekMMuseFeatureGraphs(
            model,
            len(model.config.mmuse_source_config["aux_hidden_state_layer_ids"]),
            self.hidden_size,
            capacity,
            self.graph_backend(),
        )
        self.fixed_graphs.context_inputs = DeepseekMMuseContextGraphs(
            model, min(capacity, model.feature_storage.shape[1]), self.graph_backend()
        )
        return model

    def _project_captured_features(self, aux_hidden_states):
        manager = self.query_cudagraph_manager
        if self.full_enabled and manager is not None:
            projected = manager.project_features(aux_hidden_states)
            if projected is not None:
                return projected
        return super()._project_feature_tensors(aux_hidden_states)

    def init_cudagraph_manager(self, cudagraph_mode):
        self.query_cudagraph_manager = DeepseekMMuseGraphManager(
            self.vllm_config, self.device, self.fixed_graphs
        )
        self.full_enabled = (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and not self.speculative_config.enforce_eager
        )
        logger.info(
            "DeepSeek MMuse draft: FULL=%s, sampling=%s, context buckets=%s; "
            "history/KV commits outside graph; feature buckets=%s, "
            "context update buckets=%s",
            self.full_enabled,
            self.speculative_config.draft_sample_method,
            self.fixed_graphs.sizes,
            self.fixed_graphs.feature_inputs.sizes,
            self.fixed_graphs.context_inputs.sizes,
        )

    def capture(self):
        if self.full_enabled:
            self.query_cudagraph_manager.capture()

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
        if not self.full_enabled or dummy_run or is_profile:
            # Eager/profile calls may overwrite a shared paged arena. The next
            # FULL prepare must rebuild it rather than reuse stale span metadata.
            if self.fixed_graphs is not None:
                self.fixed_graphs.context_entry = None
            return super().propose(
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
                dp_sync,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile,
            )
        count = input_batch.num_reqs
        if not count:
            return self.draft_tokens[:0]
        projected = self._project_target_features(
            input_batch,
            aux_hidden_states,
            False,
            projector=self._project_captured_features,
        )
        # One device-to-host snapshot instead of per-field/per-request reads.
        token_count = (
            int(input_batch.query_start_loc_np[count])
            if hasattr(input_batch, "query_start_loc_np")
            else input_batch.positions.numel()
        )
        metadata = torch.cat(
            (
                input_batch.query_start_loc[:count].long(),
                (
                    input_batch.query_start_loc[1 : count + 1] - num_rejected[:count]
                ).long(),
                num_sampled[:count].long(),
                input_batch.positions[:token_count].long(),
            )
        ).tolist()
        starts, ends, sampled = (
            metadata[:count],
            metadata[count : 2 * count],
            metadata[2 * count : 3 * count],
        )
        positions_cpu = metadata[3 * count :]
        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        self.model.history.storage = self.model.feature_storage
        for row in range(count):
            slot = int(input_batch.idx_mapping_np[row])
            start, end = starts[row], ends[row]
            if end <= start:
                raise ValueError("MMuse requires a nonempty verified target span")
            positions, features = self.model.history.append(
                slot,
                input_batch.req_ids[row],
                input_batch.positions[start:end],
                projected[start:end],
                positions_cpu=positions_cpu[start:end],
            )
            anchor = (
                last_sampled[slot] if sampled[row] else next_prefill_tokens[0, slot]
            )
            request, context_start, context_end, update_start = (
                self.model.history.spans[slot]
            )
            self.prepare_request_attention(row, context_start, context_end, slot)
            size = self.fixed_graphs.prepare(
                positions,
                features,
                anchor,
                slot,
                request_id=request,
                start=context_start,
                end=context_end,
                update_start=max(context_start, update_start),
                context_projector=self.query_cudagraph_manager.prepare_context,
            )
            if getattr(self, "native_attention_requires_eager", False):
                # Near the scheduler length limit, a complete trained query may
                # not fit its lookahead pages. Compute it densely, without any
                # write to null/unallocated pages, and invalidate page reuse.
                self.fixed_graphs.forward(size)
                self.fixed_graphs.context_entry = None
            else:
                self.query_cudagraph_manager.replay(size)
            # Copy before the next request overwrites shared graph buffers.
            self.draft_tokens[row].copy_(self.fixed_graphs.output)
        return self.draft_tokens[:count]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LEAGLE proposer for Model Runner V2 (CUDA and Ascend)."""

import torch
from transformers import Qwen3Config

from vllm.compilation.backends import set_model_tag
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.leagle.context import LEagleContext
from vllm.v1.worker.gpu.spec_decode.leagle.execution import (
    LEagleExecution,
    LEagleExecutionOptions,
    LEagleOptimizedGraphRunner,
)
from vllm.v1.worker.gpu.spec_decode.leagle.graph import (
    LEagleGraphInputs,
    LEagleGraphRunner,
    NativeLEagleGraphBackend,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


def validate_leagle_runtime(config):
    spec = config.speculative_config
    parallel = config.parallel_config
    if not config.use_v2_model_runner:
        raise ValueError("LEAGLE requires Model Runner V2")
    if config.model_config.enforce_eager:
        if getattr(spec, "enforce_eager", None) is False:
            raise ValueError("LEAGLE draft FULL requires target FULL capture enabled")
    else:
        compilation = getattr(config, "compilation_config", None)
        mode = getattr(compilation, "cudagraph_mode", None)
        if mode is None or not mode.has_full_cudagraphs():
            raise ValueError("LEAGLE requires eager execution or a FULL graph mode")
    if any(
        getattr(parallel, name, 1) != 1
        for name in (
            "pipeline_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        raise ValueError("LEAGLE currently requires PP=DCP=PCP=1")
    if spec.draft_parallel_config.tensor_parallel_size != parallel.tensor_parallel_size:
        raise ValueError("LEAGLE requires draft TP to equal target TP")
    compilation = getattr(config, "compilation_config", None)
    passes = getattr(compilation, "pass_config", None)
    if getattr(passes, "enable_sp", False) or getattr(
        parallel, "use_sequence_parallel_moe", False
    ):
        raise ValueError("LEAGLE does not support sequence parallel target features")
    if config.model_config.hf_config.model_type != "qwen3":
        raise ValueError("LEAGLE currently supports Qwen3 targets only")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("LEAGLE requires --no-enable-prefix-caching")
    if getattr(config, "kv_transfer_config", None) is not None:
        raise ValueError("LEAGLE does not support remote KV transfer")
    if getattr(config, "lora_config", None) is not None:
        raise ValueError("LEAGLE does not support target LoRA adapters")
    if getattr(config, "watermark_config", None) is not None:
        raise ValueError("LEAGLE does not support watermarking")
    if spec.draft_sample_method != "greedy":
        raise ValueError("LEAGLE currently requires draft_sample_method='greedy'")
    if spec.parallel_drafting or spec.enable_adaptive_verification:
        raise ValueError(
            "LEAGLE does not support parallel drafting or adaptive verification"
        )
    if spec.use_local_argmax_reduction or spec.disable_padded_drafter_batch:
        raise ValueError("LEAGLE does not support local argmax or unpadded drafting")
    hf = spec.draft_model_config.hf_config
    if not hasattr(hf, "leagle_config"):
        raise ValueError("method='leagle' requires a LEAGLE speculators checkpoint")
    if spec.draft_model_config.quantization is not None:
        raise ValueError("LEAGLE draft quantization is not supported")
    decoder = Qwen3Config(**hf.leagle_config["transformer_layer_config"])
    target = config.model_config.hf_config
    if decoder.hidden_size != target.hidden_size:
        raise ValueError("LEAGLE draft and target hidden sizes must match")
    if decoder.vocab_size != target.vocab_size:
        raise ValueError("LEAGLE draft and target vocabularies must match")
    if max(hf.leagle_config["aux_hidden_state_layer_ids"]) >= target.num_hidden_layers:
        raise ValueError("LEAGLE auxiliary layers must precede the final target norm")


class LEagleSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config, device):
        validate_leagle_runtime(vllm_config)
        super().__init__(vllm_config, device)
        self.draft_attn_layer_names: set[str] = set()
        self.attn_groups = []
        self.full_draft = (
            not vllm_config.model_config.enforce_eager
            and self.speculative_config.enforce_eager is not True
        )
        self.graph_inputs = None
        self.graph_runner = None
        self.execution = None
        self.execution_options = LEagleExecutionOptions.from_config(
            self.speculative_config
        )
        self.context = LEagleContext(
            self.max_num_reqs,
            self.max_model_len,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )

    def load_draft_model(self, target_model, target_attn_layer_names):
        draft_config = self.speculative_config.apply_draft_overrides(self.vllm_config)
        with set_model_tag("leagle"):
            return get_model(
                vllm_config=draft_config, model_config=self.draft_model_config
            )

    def load_model(self, target_model):
        self.model = self.load_draft_model(target_model, set()).eval()
        self.model.configure_output_head(
            svd_rank=getattr(
                self.speculative_config, "leagle_candidate_svd_rank", None
            ),
            enabled=getattr(self.speculative_config, "leagle_use_candidate_head", None),
            top_k=getattr(self.speculative_config, "leagle_candidate_top_k", None),
            rerank=getattr(self.speculative_config, "leagle_candidate_rerank", None),
        )
        self.configure_lm_head_quantization()
        options = self.execution_options
        self.model.configure_inference(
            fused_projections=options.fused_projections,
            rms_norm=self.create_rmsnorm_impl() if options.fused_rmsnorm else None,
        )
        if options.optimized_executor:
            self.execution = LEagleExecution(
                self.model,
                self.context,
                self.max_num_tokens,
                self.num_speculative_steps,
                options,
                dtype=self.dtype,
                device=self.device,
            )
        elif self.full_draft:
            self.graph_inputs = LEagleGraphInputs.allocate(
                self.model,
                self.max_model_len,
                self.max_num_tokens,
                dtype=self.dtype,
                device=self.device,
            )

    def set_attn(
        self,
        model_state,
        kv_cache_config,
        block_tables,
        target_input_buffers,
        target_attn_groups,
    ):
        # LEAGLE owns dense per-round draft KV rather than paged attention layers.
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.target_input_buffers = target_input_buffers
        self.target_attn_groups = target_attn_groups

    def init_cudagraph_manager(self, cudagraph_mode):
        if not self.full_draft:
            self.graph_runner = None
            return
        if not cudagraph_mode.has_full_cudagraphs():
            raise ValueError("Target backend disabled FULL graphs required by LEAGLE")
        if getattr(self, "execution", None) is not None:
            self.graph_runner = LEagleOptimizedGraphRunner(
                self.execution, self.create_graph_backend()
            )
            return
        if self.graph_inputs is None:
            raise RuntimeError("LEAGLE graph inputs must be allocated before profiling")
        self.graph_runner = LEagleGraphRunner(
            self.model,
            self.graph_inputs,
            self.num_speculative_steps,
            self.create_graph_backend(),
        )

    def create_graph_backend(self):
        if self.device.type != "cuda":
            raise ValueError("Use the Ascend LEAGLE speculator for NPU graph capture")
        backend = NativeLEagleGraphBackend(
            torch.cuda, self.device, torch.cuda.CUDAGraph
        )
        backend.tp_group = self.model.tp.group
        return backend

    def create_rmsnorm_impl(self):
        # Worker-only import keeps the CPU reference model independent of CUDA.
        from vllm import _custom_ops

        def rms_norm(hidden, weight, epsilon):
            flattened = hidden.reshape(-1, hidden.shape[-1]).contiguous()
            output = torch.empty_like(flattened)
            _custom_ops.rms_norm(output, flattened, weight, epsilon)
            return output.view_as(hidden)

        return rms_norm

    def configure_lm_head_quantization(self):
        if (
            getattr(self.speculative_config, "leagle_lm_head_quantization", None)
            is not None
        ):
            raise ValueError("LEAGLE W8A8 full head requires the Ascend speculator")

    def capture(self):
        if self.graph_runner is not None:
            self.graph_runner.capture()

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
        num_speculative_tokens=None,
    ):
        count = (
            self.num_speculative_steps
            if num_speculative_tokens is None
            else num_speculative_tokens
        )
        self.draft_tokens.fill_(-1)
        result = self.draft_tokens[: input_batch.num_reqs]
        execution = getattr(self, "execution", None)
        if dummy_run:
            if (not skip_attn_for_dummy_run or is_profile) and count:
                if execution is not None:
                    execution.profile()
                elif getattr(self, "full_draft", False):
                    # Include both persistent inputs and graph-shaped operator
                    # workspace before the framework budgets target KV memory.
                    projected = self.model.project_context(self.graph_inputs.features)
                    self.model.propose_static(
                        self.graph_inputs.context,
                        self.graph_inputs.lengths,
                        self.graph_inputs.anchor,
                        self.num_speculative_steps,
                    )
                    del projected
                else:
                    length = max(1, self.max_model_len - count)
                    anchor = torch.zeros((1, 1), dtype=torch.long, device=self.device)
                    self.model.propose_context(
                        self.context.values[:1, :length], anchor, count
                    )
            return result
        if aux_hidden_states is None:
            raise ValueError("LEAGLE requires target auxiliary hidden states")
        layers = self.model.target_layer_ids
        capture_order = sorted(layers)
        if len(aux_hidden_states) != len(layers):
            raise ValueError("LEAGLE target feature count does not match checkpoint")
        features = torch.cat(
            [aux_hidden_states[capture_order.index(layer)] for layer in layers], dim=-1
        )
        full_draft = getattr(self, "full_draft", False)
        if execution is not None:
            if full_draft and self.graph_runner is None:
                raise RuntimeError("Optimized LEAGLE graph manager is not initialized")
            projected = execution.project(
                features,
                input_batch.positions,
                self.graph_runner if full_draft else None,
            )
        elif full_draft:
            if self.graph_runner is None:
                raise RuntimeError("LEAGLE FULL graph manager is not initialized")
            projected = self.graph_runner.project_context(features)
        else:
            projected = self.model.project_context(features)
        # Schedule/stage outside graphs: synchronize verdicts/positions once.
        rejected = num_rejected.cpu().tolist()
        sampled = num_sampled.cpu().tolist()
        positions = input_batch.positions.cpu().tolist()
        active = []
        updates = []
        for row, req_id in enumerate(input_batch.req_ids):
            begin, end = map(int, input_batch.query_start_loc_np[row : row + 2])
            end -= rejected[row]
            if end <= begin:
                continue
            start = positions[begin]
            if positions[begin:end] != list(range(start, start + end - begin)):
                raise ValueError("LEAGLE requires contiguous target positions")
            slot = int(input_batch.idx_mapping_np[row])
            if execution is not None:
                updates.append((slot, req_id, start, begin, end))
            else:
                context = self.context.update(slot, req_id, start, projected[begin:end])
            if sampled[row] and count:
                if execution is not None:
                    active.append((row, slot))
                    continue
                anchor = last_sampled[slot : slot + 1, :1]
                tokens = (
                    self.graph_runner.propose(context, anchor, count)
                    if full_draft
                    else self.model.propose_context(context, anchor, count)
                )
                result[row, :count].copy_(tokens[0])
        if execution is not None:
            execution.update_batch(updates, self.graph_runner if full_draft else None)
            return execution.propose(
                active,
                last_sampled,
                result,
                count,
                self.graph_runner if full_draft else None,
            )
        return result

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference LEAGLE proposer for Model Runner V2 (CUDA and Ascend)."""

import torch
from transformers import Qwen3Config

from vllm.compilation.backends import set_model_tag
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.leagle.context import LEagleContext
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


def validate_leagle_runtime(config):
    spec = config.speculative_config
    parallel = config.parallel_config
    if not config.use_v2_model_runner:
        raise ValueError("LEAGLE requires Model Runner V2")
    if (
        not config.model_config.enforce_eager
        or getattr(spec, "enforce_eager", None) is False
    ):
        raise ValueError("LEAGLE currently requires --enforce-eager")
    if any(
        getattr(parallel, name, 1) != 1
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        raise ValueError("LEAGLE currently requires TP=PP=DCP=PCP=1")
    if spec.draft_parallel_config.tensor_parallel_size != 1:
        raise ValueError("LEAGLE currently requires draft TP=1")
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
        pass

    def capture(self):
        pass

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
        if dummy_run:
            if not skip_attn_for_dummy_run and count:
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
        projected = self.model.project_context(features)
        # Eager reference path: synchronize verdicts and positions once per batch.
        rejected = num_rejected.cpu().tolist()
        sampled = num_sampled.cpu().tolist()
        positions = input_batch.positions.cpu().tolist()
        for row, req_id in enumerate(input_batch.req_ids):
            begin, end = map(int, input_batch.query_start_loc_np[row : row + 2])
            end -= rejected[row]
            if end <= begin:
                continue
            start = positions[begin]
            if positions[begin:end] != list(range(start, start + end - begin)):
                raise ValueError("LEAGLE requires contiguous target positions")
            slot = int(input_batch.idx_mapping_np[row])
            context = self.context.update(slot, req_id, start, projected[begin:end])
            if sampled[row] and count:
                anchor = last_sampled[slot : slot + 1, :1]
                tokens = self.model.propose_context(context, anchor, count)
                result[row, :count].copy_(tokens[0])
        return result

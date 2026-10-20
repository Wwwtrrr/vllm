# SPDX-License-Identifier: Apache-2.0
"""Dense MMuse drafter trained on DeepSeek-V4 mean-HC auxiliary features.

This is a separate target adapter, not a DeepSeek MLA/DSpark draft architecture.
Its Qwen3 decoder and MMuse heads follow the speculators_dsv4 training export.
"""

import weakref
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.mmuse_runtime import (
    build_mmuse_runtime,
    validate_mmuse_config,
)
from vllm.model_executor.models.qwen3_mmuse import (
    MMuseQwen3Model,
    _TargetEmbedding,
    _TargetHead,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

logger = init_logger(__name__)


class DeepseekV4MMuseForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        validate_mmuse_config(vllm_config)
        if vllm_config.model_config.hf_config.model_type != "deepseek_v4":
            raise ValueError("DeepseekV4MMuseSpeculator requires a DeepSeek-V4 target")
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config.draft_vocab_size = self.config.vocab_size
        self.model = MMuseQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=vllm_config.model_config.get_total_num_hidden_layers(),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.runtime = build_mmuse_runtime(
            self.config.mmuse_source_config, self.model.fc
        ).to(dtype=vllm_config.model_config.dtype)
        self.runtime.embed_tokens = _TargetEmbedding(self)
        self.runtime.lm_head = _TargetHead(self)
        self.model._runtime_ref = weakref.ref(self.runtime)
        self._target_weights_bound = False
        layer_types = getattr(self.config, "layer_types", None) or []
        if "sliding_attention" in layer_types:
            from vllm.model_executor.models.mmuse.anchor_attention import (
                AnchorContextCache,
            )

            window = (
                self.config.sliding_window
                if all(t == "sliding_attention" for t in layer_types)
                else None
            )
            history_size = (
                window
                or vllm_config.speculative_config.draft_model_config.max_model_len
            )
            # Allocate before the worker's memory profiling, not during serving.
            self.register_buffer(
                "_anchor_feature_storage",
                torch.empty(
                    vllm_config.scheduler_config.max_num_seqs,
                    history_size,
                    self.config.hidden_size,
                    dtype=vllm_config.model_config.dtype,
                ),
                persistent=False,
            )
            self.anchor_context_cache = AnchorContextCache(
                window, self._anchor_feature_storage
            )

    def bind_target_weights(self, target_model):
        target = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        embedding = target.model.embed_tokens
        head = target.lm_head
        # Backbone W4A8 does not authorize interpreting packed head weights as
        # floating matrices. The trainer borrows ordinary BF16/FP16/FP32 IO.
        expected = (self.config.vocab_size, self.config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        for name, module in (("embedding", embedding), ("lm_head", head)):
            weight = getattr(module, "weight", None)
            if (
                weight is None
                or weight.ndim != 2
                or weight.shape[1] != expected[1]
                or weight.shape[0] * tp_size < expected[0]
                or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            ):
                raise ValueError(
                    f"DeepSeek-V4 MMuse requires full floating "
                    f"{name} weights of shape {expected}"
                )
        if tp_size > 1:
            # Correction consumes full-vocabulary logits, not TP-local shards.
            # Gather borrowed IO once, outside the proposal loop; decoder TP
            # remains native. This costs replicated embedding/head memory.
            embed_weight = tensor_model_parallel_all_gather(embedding.weight, dim=0)[
                : expected[0]
            ].contiguous()
            head_weight = tensor_model_parallel_all_gather(head.weight, dim=0)[
                : expected[0]
            ].contiguous()
            embedding = nn.Embedding.from_pretrained(embed_weight, freeze=True)
            head = nn.Linear(
                expected[1],
                expected[0],
                bias=False,
                device="meta",
                dtype=head_weight.dtype,
            )
            head.weight = nn.Parameter(head_weight, requires_grad=False)
        self.model.embed_tokens = embedding
        self.lm_head = head
        self._target_weights_bound = True

    def embed_input_ids(self, input_ids, **kwargs):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, inputs_embeds=None):
        contexts = getattr(self, "_anchor_contexts", None)
        if contexts is not None:
            from vllm.model_executor.models.mmuse.anchor_attention import (
                forward_anchor_blocks,
            )

            logger.info_once(
                "MMuse DeepSeek draft uses eager anchor-relative attention; "
                "verified feature history replaces paged draft SWA reads."
            )
            return forward_anchor_blocks(self.model, input_ids, positions, contexts)
        return self.model(input_ids, positions, inputs_embeds)

    def get_draft_kv_cache_layer_names(self):
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def get_draft_attn_causal(self):
        return [layer.self_attn.causal for layer in self.model.layers]

    def precompute_and_store_context_kv(self, states, positions, slot_mapping=None):
        self.model.precompute_and_store_context_kv(states, positions, slot_mapping)

    def combine_hidden_states(self, hidden_states):
        expected = self.model.fc.input_size
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                f"DeepSeek-V4 MMuse requires {expected} concatenated mean-HC features"
            )
        return self.runtime._prepare_target_hidden(hidden_states)[0]

    def compute_logits(self, hidden_states):
        return F.linear(hidden_states, self.lm_head.weight[: self.config.vocab_size])

    @torch.inference_mode()
    def rollout_mmuse(self, hidden, anchor_ids, *, sample_token=None):
        if not self._target_weights_bound:
            raise RuntimeError(
                "DeepSeek-V4 MMuse target IO weights have not been bound"
            )
        return self.runtime.rollout_correction(
            hidden, anchor_ids, temperature=0.0, sample_token=sample_token
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        mapped = []
        for name, value in weights:
            if name in ("t2d", "d2t") or name.startswith(
                ("verifier_", "embed_tokens.", "lm_head.")
            ):
                continue
            if name.startswith(("layers.", "fc.", "hidden_norm.", "norm.")):
                name = "model." + name
            else:
                name = "runtime." + name
            mapped.append((name, value))
        loaded = AutoWeightsLoader(self).load_weights(mapped)
        borrowed = {"model.embed_tokens.weight", "lm_head.weight"}
        initialized = {"model.mask_embedding"}
        missing = (
            {name for name, _ in self.named_parameters()}
            - loaded
            - borrowed
            - initialized
        )
        if missing:
            raise ValueError(
                f"DeepSeek-V4 MMuse checkpoint is missing trained "
                f"weights: {sorted(missing)}"
            )
        self.model._build_fused_kv_buffers()
        return loaded | borrowed | initialized

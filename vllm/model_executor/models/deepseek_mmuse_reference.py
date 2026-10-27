# SPDX-License-Identifier: Apache-2.0
"""Dense MMuse reference model with checkpoint-local, unpacked tensor names."""

import copy
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.models.mmuse.deepseek_contract import (
    ContextKVCache,
    VerifiedPrefix,
)
from vllm.model_executor.models.mmuse.deepseek_decoder import RMS, BlockLayer
from vllm.model_executor.models.mmuse.runtime import MMuseRuntime
from vllm.model_executor.models.mmuse_runtime import validate_mmuse_config


def _bind_teacher_io(model, rotation):
    h = model.config.hidden_size
    if rotation.shape != (h, h) or not torch.isfinite(rotation).all():
        raise ValueError("MMuse requires a finite hidden-size QuaRot matrix")
    embedding = getattr(model, "checkpoint_embedding", None)
    head = getattr(model, "checkpoint_head", None)
    if embedding is None or head is None:
        raise ValueError(
            "QuaRot MMuse requires embed_tokens.weight and lm_head.weight "
            "from the original training teacher in the draft checkpoint. "
            "The deployed Target head has fused normalization and cannot "
            "be borrowed or recovered by inverse rotation alone."
        )
    model.target_rotation = rotation.to(
        device=model.feature_storage.device, dtype=torch.float32
    ).contiguous()
    model.embedding_weight = embedding
    model.output_weight = head


def _bind_deployed_io(model, target):
    if hasattr(target, "get_language_model"):
        target = target.get_language_model()
    tp = get_tensor_model_parallel_world_size()
    for name, module in (
        ("embedding_weight", target.model.embed_tokens),
        ("output_weight", target.lm_head),
    ):
        weight = module.weight.detach()
        if (
            weight.ndim != 2
            or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or weight.shape[1] != model.config.hidden_size
        ):
            raise ValueError("MMuse borrowed IO must be ordinary floating matrices")
        if tp > 1:
            weight = tensor_model_parallel_all_gather(weight, dim=0)
        if weight.shape[0] < model.config.vocab_size:
            raise ValueError("MMuse borrowed IO does not cover the target vocabulary")
        setattr(
            model,
            name,
            weight[: model.config.vocab_size]
            .to(dtype=model.feature_storage.dtype)
            .contiguous(),
        )


def _load_teacher_io(model, name, value):
    if value.shape != (
        model.config.vocab_size,
        model.config.hidden_size,
    ) or value.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Invalid MMuse teacher IO: {name}")
    buffer = (
        "checkpoint_embedding" if name == "embed_tokens.weight" else "checkpoint_head"
    )
    setattr(
        model,
        buffer,
        value.detach()
        .to(
            device=model.feature_storage.device,
            dtype=model.feature_storage.dtype,
        )
        .clone(),
    )


class BorrowedIO(nn.Module):
    def __init__(self, owner, head=False):
        super().__init__()
        self.owner = weakref.ref(owner)
        self.head = head
        self.out_features = owner.config.vocab_size

    @property
    def weight(self):
        owner = self.owner()
        return owner.output_weight if self.head else owner.embedding_weight

    def forward(self, value):
        return (
            F.linear(value, self.weight)
            if self.head
            else F.embedding(value, self.weight)
        )


class DeepseekMMuseReference(nn.Module):
    """Replicate the draft on each target TP rank; no draft paged-KV allocation."""

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        validate_mmuse_config(vllm_config)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        source = copy.deepcopy(self.config.mmuse_source_config)
        source["speculators_model_type"] = "mmuse"
        source["architectures"] = ["MMuseSpeculator"]
        h = self.config.hidden_size
        self.fc = nn.Linear(
            len(source["aux_hidden_state_layer_ids"]) * h, h, bias=False
        )
        self.hidden_norm = RMS(h, self.config.rms_norm_eps)
        self.norm = RMS(h, self.config.rms_norm_eps)
        types = (
            self.config.layer_types
            or ["full_attention"] * self.config.num_hidden_layers
        )
        self.layers = nn.ModuleList(
            [
                BlockLayer(
                    self.config,
                    source,
                    self.config.sliding_window if kind == "sliding_attention" else None,
                )
                for kind in types
            ]
        )
        self.runtime = MMuseRuntime(source, self.fc).eval()
        self.runtime.embed_tokens = BorrowedIO(self)
        self.runtime.lm_head = BorrowedIO(self, head=True)
        self.register_buffer("embedding_weight", None, persistent=False)
        self.register_buffer("output_weight", None, persistent=False)
        self.register_buffer("checkpoint_embedding", None, persistent=False)
        self.register_buffer("checkpoint_head", None, persistent=False)
        self.register_buffer("target_rotation", None, persistent=False)
        self.window = (
            self.config.sliding_window
            if all(kind == "sliding_attention" for kind in types)
            else None
        )
        capacity = self.window or vllm_config.model_config.max_model_len
        self.register_buffer(
            "feature_storage",
            torch.empty(
                vllm_config.scheduler_config.max_num_seqs,
                capacity,
                h,
                dtype=vllm_config.model_config.dtype,
            ),
            persistent=False,
        )
        self.history = VerifiedPrefix(capacity, self.feature_storage)
        reuse = (getattr(vllm_config, "additional_config", None) or {}).get(
            "mmuse_draft_kv_reuse", False
        )
        if type(reuse) is not bool:
            raise ValueError("mmuse_draft_kv_reuse must be a boolean")
        dim = self.config.head_dim or h // self.config.num_attention_heads
        self.register_buffer(
            "context_kv_storage",
            torch.empty(
                len(self.layers),
                vllm_config.scheduler_config.max_num_seqs,
                2,
                capacity,
                self.config.num_key_value_heads,
                dim,
                dtype=self.feature_storage.dtype,
                device=self.feature_storage.device,
            )
            if reuse
            else None,
            persistent=False,
        )
        self.context_kv_cache = (
            ContextKVCache(self.context_kv_storage) if reuse else None
        )

    def bind_target(self, target, rotation=None):
        """Use serialized teacher IO for QuaRot, deployed IO otherwise."""
        if rotation is not None:
            _bind_teacher_io(self, rotation)
        else:
            _bind_deployed_io(self, target)

    def project_features(self, features):
        rotation = getattr(self, "target_rotation", None)
        if rotation is not None:
            # Undo each layer's residual-stream rotation BEFORE learned norms
            # and gated layer fusion; rotating only fc would miss those paths.
            shape = features.shape
            features = (
                (
                    features.reshape(*shape[:-1], -1, self.config.hidden_size).float()
                    @ rotation.T
                )
                .to(features.dtype)
                .reshape(shape)
            )
        return self.runtime._prepare_target_hidden(features)[0]

    def forward(
        self,
        input_ids,
        positions,
        context_positions,
        projected_context,
        *,
        cache_slot=None,
        request_id=None,
        update_start=None,
    ):
        context = self.hidden_norm(projected_context)
        cache = getattr(self, "context_kv_cache", None)
        use_cache = cache is not None and cache_slot is not None
        reused = 0
        if use_cache:
            if torch.is_grad_enabled():
                raise ValueError("MMuse context KV reuse is inference-only")
            if request_id is None or update_start is None:
                raise ValueError(
                    "MMuse context KV requires request identity and update span"
                )
            start, end = context_positions[0].item(), context_positions[-1].item() + 1
            cache.storage = self.context_kv_storage
            reused = cache.begin(cache_slot, request_id, start, end, update_start)
        hidden = F.embedding(input_ids, self.embedding_weight)
        if self.runtime.block_position_embedding is not None:
            hidden = hidden + self.runtime.block_position_embedding(
                torch.arange(input_ids.numel(), device=input_ids.device)
            )
        if self.runtime.context_hidden_proj is not None:
            hidden = hidden + torch.tanh(
                self.runtime.context_hidden_gate
            ) * self.runtime.context_hidden_proj(context[-1:])
        for index, layer in enumerate(self.layers):
            context_kv = None
            if use_cache:
                storage = self.context_kv_storage[index, cache_slot]
                if reused < context.shape[0]:
                    key, value = layer.self_attn.project_kv(
                        context[reused:], context_positions[reused:]
                    )
                    storage[0, reused : context.shape[0]].copy_(key)
                    storage[1, reused : context.shape[0]].copy_(value)
                context_kv = storage[:, : context.shape[0]]
            hidden = layer(
                hidden, context, positions, context_positions, context_kv=context_kv
            )
        output = self.norm(hidden)
        if use_cache:
            cache.commit(cache_slot, request_id, start, end)
        return output

    def forward_fixed_context(
        self,
        input_ids,
        positions,
        context_positions,
        normalized_context,
        context_kv,
        anchor_index,
    ):
        """Query-only forward over padded, immutable-shape context buffers.

        The caller normalizes/projects confirmed context outside capture.
        Padding positions are -1 and are masked by BlockAttention.
        """
        hidden = F.embedding(input_ids, self.embedding_weight)
        if self.runtime.block_position_embedding is not None:
            hidden = hidden + self.runtime.block_position_embedding(
                torch.arange(input_ids.numel(), device=input_ids.device)
            )
        if self.runtime.context_hidden_proj is not None:
            anchor = normalized_context.index_select(0, anchor_index)
            hidden = hidden + torch.tanh(
                self.runtime.context_hidden_gate
            ) * self.runtime.context_hidden_proj(anchor)
        for layer, kv in zip(self.layers, context_kv):
            hidden = layer(
                hidden,
                normalized_context,
                positions,
                context_positions,
                context_kv=kv,
            )
        return self.norm(hidden)

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            if name.startswith("verifier_") or name in ("t2d", "d2t"):
                continue
            if name in ("embed_tokens.weight", "lm_head.weight"):
                _load_teacher_io(self, name, value)
                continue
            destination = (
                name
                if name.startswith(("layers.", "fc.", "norm.", "hidden_norm."))
                else "runtime." + name
            )
            if (
                name
                in (
                    "layer_fusion_gate",
                    "context_hidden_gate",
                    "correction_markov_scale",
                )
                and destination in parameters
                and value.shape in ((), (1,))
                and parameters[destination].shape in ((), (1,))
            ):
                # FSDP2 exports these scalar gates as [1]; older checkpoints
                # use []. Only change their layout, never the learned value.
                value = value.reshape(parameters[destination].shape)
            if (
                destination not in parameters
                or parameters[destination].shape != value.shape
            ):
                raise ValueError(
                    f"Unexpected MMuse checkpoint tensor: {name} {tuple(value.shape)}"
                )
            with torch.no_grad():
                parameters[destination].copy_(value)
            loaded.add(destination)
        missing = parameters.keys() - loaded
        if missing:
            raise ValueError(f"Missing trained MMuse tensors: {sorted(missing)}")
        return loaded

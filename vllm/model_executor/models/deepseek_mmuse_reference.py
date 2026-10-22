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
from vllm.model_executor.models.mmuse.deepseek_contract import VerifiedPrefix
from vllm.model_executor.models.mmuse.deepseek_decoder import RMS, BlockLayer
from vllm.model_executor.models.mmuse.runtime import MMuseRuntime
from vllm.model_executor.models.mmuse_runtime import validate_mmuse_config


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
    """Replicate the small draft on each target TP rank; no draft KV allocation."""

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

    def bind_target(self, target, rotation=None):
        if rotation is not None:
            h = self.config.hidden_size
            if rotation.shape != (h, h) or not torch.isfinite(rotation).all():
                raise ValueError("MMuse requires a finite hidden-size QuaRot matrix")
            embedding = getattr(self, "checkpoint_embedding", None)
            head = getattr(self, "checkpoint_head", None)
            if embedding is None or head is None:
                raise ValueError(
                    "QuaRot MMuse requires embed_tokens.weight and lm_head.weight "
                    "from the original training teacher in the draft checkpoint. "
                    "The deployed Target head has fused normalization and cannot "
                    "be borrowed or recovered by inverse rotation alone."
                )
            self.target_rotation = rotation.to(
                device=self.feature_storage.device, dtype=torch.float32
            ).contiguous()
            self.embedding_weight = embedding
            self.output_weight = head
            return
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
                or weight.shape[1] != self.config.hidden_size
            ):
                raise ValueError("MMuse borrowed IO must be ordinary floating matrices")
            if tp > 1:
                weight = tensor_model_parallel_all_gather(weight, dim=0)
            if weight.shape[0] < self.config.vocab_size:
                raise ValueError(
                    "MMuse borrowed IO does not cover the target vocabulary"
                )
            setattr(
                self,
                name,
                weight[: self.config.vocab_size]
                .to(dtype=self.feature_storage.dtype)
                .contiguous(),
            )

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

    def forward(self, input_ids, positions, context_positions, projected_context):
        context = self.hidden_norm(projected_context)
        hidden = F.embedding(input_ids, self.embedding_weight)
        if self.runtime.block_position_embedding is not None:
            hidden = hidden + self.runtime.block_position_embedding(
                torch.arange(input_ids.numel(), device=input_ids.device)
            )
        if self.runtime.context_hidden_proj is not None:
            hidden = hidden + torch.tanh(
                self.runtime.context_hidden_gate
            ) * self.runtime.context_hidden_proj(context[-1:])
        for layer in self.layers:
            hidden = layer(hidden, context, positions, context_positions)
        return self.norm(hidden)

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            if name.startswith("verifier_") or name in ("t2d", "d2t"):
                continue
            if name in ("embed_tokens.weight", "lm_head.weight"):
                if value.shape != (
                    self.config.vocab_size,
                    self.config.hidden_size,
                ) or value.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise ValueError(f"Invalid MMuse teacher IO: {name}")
                buffer = (
                    "checkpoint_embedding"
                    if name == "embed_tokens.weight"
                    else "checkpoint_head"
                )
                setattr(
                    self,
                    buffer,
                    value.detach()
                    .to(
                        device=self.feature_storage.device,
                        dtype=self.feature_storage.dtype,
                    )
                    .clone(),
                )
                continue
            destination = (
                name
                if name.startswith(("layers.", "fc.", "norm.", "hidden_norm."))
                else "runtime." + name
            )
            if (
                name in (
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

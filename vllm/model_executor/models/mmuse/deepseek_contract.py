# SPDX-License-Identifier: Apache-2.0
"""Training export and feature-history contracts for the reference DSV4 drafter."""

import torch


def validate_deepseek_export(config, source):
    spec = config.speculative_config
    target = config.model_config
    decoder = source.get("transformer_layer_config", {})
    ids = source.get("aux_hidden_state_layer_ids", [])
    if decoder.get("model_type") != "qwen3":
        raise ValueError("DeepSeek MMuse requires a trained Qwen3 dense decoder")
    if source.get("target_hidden_state_format") != "deepseek_v4_mean_hc_head":
        raise ValueError("DeepSeek MMuse requires deepseek_v4_mean_hc_head features")
    if not ids or any(
        type(i) is not int or not 1 <= i < target.get_total_num_hidden_layers()
        for i in ids
    ):
        raise ValueError("DeepSeek MMuse auxiliary IDs must precede the teacher slot")
    if ids != sorted(set(ids)):
        raise ValueError("DeepSeek MMuse auxiliary IDs must be unique and ascending")
    if not target.enforce_eager or not spec.enforce_eager:
        raise ValueError(
            "DeepSeek MMuse reference execution requires both models eager"
        )
    if target.dtype != torch.bfloat16 or spec.draft_model_config.quantization:
        raise ValueError(
            "DeepSeek MMuse requires BF16 activations and an unquantized draft"
        )
    if (
        decoder.get("hidden_size") != target.get_hidden_size()
        or (source.get("target_hidden_size") or target.get_hidden_size())
        != target.get_hidden_size()
        or (source.get("draft_vocab_size") or decoder.get("vocab_size"))
        != target.get_vocab_size()
    ):
        raise ValueError(
            "DeepSeek MMuse requires matching target hidden size and full vocabulary"
        )
    layers = decoder.get("layer_types")
    if layers and len(layers) != decoder.get("num_hidden_layers"):
        raise ValueError("MMuse layer_types must describe every trained decoder layer")
    heads, kv_heads = (
        decoder.get("num_attention_heads", 0),
        decoder.get("num_key_value_heads", 0),
    )
    dim = decoder.get("head_dim") or decoder.get("hidden_size", 0) // max(heads, 1)
    if not heads or not kv_heads or heads % kv_heads or not dim or dim % 2:
        raise ValueError(
            "MMuse requires valid grouped-query heads and even RoPE head_dim"
        )
    if not source.get(
        "sample_from_anchor", True
    ) or spec.num_speculative_tokens != source.get("block_size"):
        raise ValueError(
            "DeepSeek MMuse requires sample_from_anchor and trained block_size"
        )
    if not source.get("enable_correction_head", False):
        raise ValueError(
            "DeepSeek MMuse reference path requires a trained Correction head"
        )
    rope = decoder.get("rope_parameters") or decoder.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError(
            "DeepSeek MMuse reference decoder currently supports default RoPE"
        )
    if decoder.get("hidden_act", "silu") != "silu":
        raise ValueError("DeepSeek MMuse reference decoder requires SiLU")
    if any(
        t not in ("full_attention", "sliding_attention")
        for t in decoder.get("layer_types", [])
    ):
        raise ValueError("Unsupported DeepSeek MMuse attention type")
    if config.cache_config.enable_prefix_caching:
        raise ValueError(
            "DeepSeek MMuse requires enable_prefix_caching=False for complete features"
        )
    for name in (
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(config.parallel_config, name, 1) != 1:
            raise ValueError(f"DeepSeek MMuse reference execution requires {name}=1")
    if spec.enable_adaptive_verification or getattr(config, "watermark_config", None):
        raise ValueError(
            "DeepSeek MMuse reference execution excludes "
            "adaptive verification and watermarking"
        )
    if spec.use_local_argmax_reduction:
        raise ValueError("MMuse Correction requires full vocabulary logits")
    contract = source.get("target_training_contract")
    expected = dict(
        format="deepseek_v4_mean_hc_head",
        hidden_size=target.get_hidden_size(),
        auxiliary_hs_ids=ids,
        teacher_hs_id=target.get_total_num_hidden_layers(),
        target_dtype="bfloat16",
    )
    if contract is not None and (
        not isinstance(contract, dict)
        or any(contract.get(k) != v for k, v in expected.items())
    ):
        raise ValueError(
            "DeepSeek MMuse training contract disagrees with target interfaces"
        )


class VerifiedPrefix:
    """Own projected confirmed rows; never store synthetic query states."""

    def __init__(self, capacity, storage):
        self.capacity = capacity
        self.storage = storage
        self.requests = {}

    def append(self, slot, request_id, positions, values):
        first, end = positions[0].item(), positions[-1].item() + 1
        if not torch.equal(
            positions, torch.arange(first, end, device=positions.device)
        ):
            raise ValueError("MMuse target positions must be contiguous")
        previous = self.requests.get(slot)
        if first == 0:
            old_positions, old_values = positions[:0], values[:0]
        else:
            if previous is None or previous[0] != request_id:
                raise ValueError("MMuse request is missing its verified prefix")
            _, old_positions, old_values = previous
            keep = old_positions < first
            old_positions, old_values = old_positions[keep], old_values[keep]
            if not old_positions.numel() or old_positions[-1].item() != first - 1:
                raise ValueError("MMuse verified prefix has a gap")
        joined = torch.cat((old_values, values))[-self.capacity :]
        joined_positions = torch.cat((old_positions, positions))[-self.capacity :]
        owned = self.storage[slot, : joined.shape[0]]
        owned.copy_(joined)
        self.requests[slot] = (request_id, joined_positions, owned)
        return joined_positions, owned


class ContextKVCache:
    """Bounded per-slot context-only KV; synthetic query KV is never committed.

    Storage is [layers, request slots, K/V, capacity, KV heads, head dim].
    The caller supplies a contiguous confirmed prefix and the first position
    updated by the current Target step. An overwritten suffix is reprojected.
    """

    def __init__(self, storage):
        self.storage = storage
        self.entries = {}

    def begin(self, slot, request_id, start, end, update_start):
        capacity = self.storage.shape[3]
        if not 0 <= start < end or end - start > capacity:
            raise ValueError("MMuse context KV span exceeds its reserved capacity")
        previous = self.entries.pop(slot, None)
        reused = 0
        if previous is not None:
            old_request, old_start, old_end = previous
            if old_request == request_id and old_start <= start < old_end:
                reused = max(0, min(old_end, update_start, end) - start)
                shift = start - old_start
                if reused and shift:
                    # copy_ cannot safely move overlapping storage slices.
                    self.storage[:, slot, :, :reused].copy_(
                        self.storage[:, slot, :, shift : shift + reused].clone()
                    )
        # Metadata stays invalid until every layer has completed successfully.
        return reused

    def commit(self, slot, request_id, start, end):
        self.entries[slot] = (request_id, start, end)

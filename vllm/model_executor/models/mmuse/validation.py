# SPDX-License-Identifier: Apache-2.0
"""Model-specific execution contracts for MMuse adapters."""


def validate_mmuse_config(config) -> None:
    """Check required model interfaces and currently supported execution modes.

    This check does not validate checkpoint tensors or end-to-end correctness.
    Detailed MMuse option validation belongs to MMuseRuntimeConfig.
    """
    spec = config.speculative_config
    hf = spec.draft_model_config.hf_config
    source = getattr(hf, "mmuse_source_config", None)

    if not source:
        raise ValueError("MMuse requires a Speculators-format MMuse checkpoint")

    validators = {
        "qwen3": _validate_qwen3_mmuse,
        "deepseek_v4": _validate_deepseek_mmuse,
    }
    validator = validators.get(config.model_config.hf_config.model_type)
    if validator is None:
        raise ValueError("MMuse adapter requires a Qwen3 or DeepSeek V4 target")
    validator(config, source)


def _validate_deepseek_mmuse(config, source) -> None:
    """Keep DeepSeek's independent execution contract and lazy imports."""
    from vllm.model_executor.models.mmuse.deepseek_contract import (
        validate_deepseek_export,
    )

    validate_deepseek_export(config, source)


def _validate_qwen3_mmuse(config, source) -> None:
    _validate_qwen3_interfaces(config, source)
    _validate_qwen3_execution(config)
    _validate_qwen3_sampling(config, source)


def _validate_qwen3_interfaces(config, source) -> None:
    """Check the trained decoder and target auxiliary-layer interface."""
    if source["transformer_layer_config"].get("model_type") != "qwen3":
        raise ValueError("MMuse reference adapter requires a Qwen3 draft decoder")
    layer_ids = source["aux_hidden_state_layer_ids"]
    num_layers = config.model_config.get_total_num_hidden_layers()
    if not layer_ids or any(i < 1 or i > num_layers for i in layer_ids):
        raise ValueError("MMuse auxiliary layer IDs exceed the target's layer range")


def _validate_qwen3_execution(config) -> None:
    """Check parallel and attention modes covered by the Qwen3 adapter."""
    hf = config.speculative_config.draft_model_config.hf_config
    parallel = config.parallel_config
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"MMuse reference adapter requires {name}=1")
    if any(
        layer_type != "full_attention"
        for layer_type in (getattr(hf, "layer_types", None) or [])
    ):
        raise ValueError("MMuse reference adapter requires full-attention draft layers")


def _validate_qwen3_sampling(config, source) -> None:
    """Check the fixed query-block contract shared by both sampling modes."""
    spec = config.speculative_config
    if spec.num_speculative_tokens != source["block_size"]:
        raise ValueError("MMuse num_speculative_tokens must equal trained block_size")
    if not source.get("sample_from_anchor", True):
        raise ValueError("MMuse requires sample_from_anchor=True")
    if spec.enable_adaptive_verification:
        raise ValueError("MMuse does not support adaptive verification")
    if spec.uses_dynamic_speculative_decoding():
        raise ValueError("MMuse requires a fixed trained query block length")
    if getattr(config, "watermark_config", None) is not None:
        raise ValueError("MMuse does not support watermarking")

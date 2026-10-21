# SPDX-License-Identifier: Apache-2.0
"""Opt-in, eager-only MMuse checkpoint and verification diagnostics."""

import json
import math
from pathlib import Path

import torch


def checkpoint_parameter(model, name, value, rank, tp_size):
    """Resolve a training tensor to its loaded TP/fused parameter slice."""
    if name in ("t2d", "d2t") or name.startswith("verifier_"):
        return None
    parameters = dict(model.named_parameters())
    if name in ("embed_tokens.weight", "lm_head.weight"):
        destination = "model." + name if name.startswith("embed_tokens.") else name
        return destination, parameters[destination][: value.shape[0]], value
    destination = (
        "model." + name
        if name.startswith(("layers.", "fc.", "hidden_norm.", "norm."))
        else "runtime." + name
    )
    for projection in ("q", "k", "v"):
        marker = f".self_attn.{projection}_proj."
        if marker in name:
            layer_index = int(name.split(".")[1])
            attention = model.model.layers[layer_index].self_attn
            destination = destination.replace(marker, ".self_attn.qkv_proj.")
            width = attention.q_size if projection == "q" else attention.kv_size
            offset = {
                "q": 0,
                "k": attention.q_size,
                "v": attention.q_size + attention.kv_size,
            }[projection]
            # KV heads are replicated when TP exceeds the KV-head count.
            shard_count = value.shape[0] // width
            shard_rank = rank // (tp_size // shard_count)
            expected = value.narrow(0, shard_rank * width, width)
            return (
                destination,
                parameters[destination].narrow(0, offset, width),
                expected,
            )
    for projection, part in (("gate", 0), ("up", 1)):
        marker = f".mlp.{projection}_proj."
        if marker in name:
            destination = destination.replace(marker, ".mlp.gate_up_proj.")
            width = parameters[destination].shape[0] // 2
            return (
                destination,
                parameters[destination].narrow(0, part * width, width),
                value.narrow(0, rank * width, width),
            )
    actual = parameters[destination]
    if ".mlp.down_proj.weight" in name:
        value = value.narrow(1, rank * actual.shape[1], actual.shape[1])
    return destination, actual, value


def audit_checkpoint(model, checkpoint, rank, tp_size):
    from safetensors import safe_open

    root = Path(checkpoint)
    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        files = sorted(path.name for path in root.glob("*.safetensors"))
    if not files:
        raise ValueError("MMuse diagnostics requires local safetensors draft weights")
    report = {"checked": 0, "skipped_borrowed": 0, "mismatches": []}
    for file in files:
        with safe_open(root / file, framework="pt", device="cpu") as weights:
            for name in weights.keys():  # noqa: SIM118 - safe_open is not a dict
                mapped = checkpoint_parameter(
                    model, name, weights.get_tensor(name), rank, tp_size
                )
                if mapped is None:
                    report["skipped_borrowed"] += 1
                    continue
                destination, actual, expected = mapped
                actual = actual.detach().cpu()
                expected = expected.to(actual.dtype)
                report["checked"] += 1
                if actual.shape != expected.shape or not torch.equal(actual, expected):
                    report["mismatches"].append(
                        {
                            "source": name,
                            "destination": destination,
                            "actual_shape": list(actual.shape),
                            "expected_shape": list(expected.shape),
                            "max_abs_error": (
                                float((actual.float() - expected.float()).abs().max())
                                if actual.shape == expected.shape
                                else None
                            ),
                        }
                    )
    return report


def verification_rows(logits, draft_sampled, positions, cumulative, slots, num_sampled):
    """Mirror verifier indexing: target row i checks draft input row i+1."""
    rows = []
    boundaries = cumulative.tolist()
    for request, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if end - start <= 1:
            continue
        draft = draft_sampled[start + 1 : end].long()
        target = logits[start : end - 1]
        valid = (draft >= 0) & (draft < target.shape[-1])
        predicted = target.argmax(-1)
        matches = (predicted == draft) & valid
        matched_prefix = 0
        for matched in matches.tolist():
            if not matched:
                break
            matched_prefix += 1
        log_probs = target.float().log_softmax(-1)
        probabilities = (
            log_probs.gather(-1, draft.clamp(0, target.shape[-1] - 1)[:, None])
            .exp()
            .flatten()
        )
        rows.append(
            {
                "slot": int(slots[request]),
                "anchor_input_id": int(draft_sampled[start]),
                "target_input_positions": positions[start : end - 1].tolist(),
                "predicted_token_positions": (positions[start : end - 1] + 1).tolist(),
                "draft_input_positions": positions[start + 1 : end].tolist(),
                "draft_ids": draft.tolist(),
                "target_argmax_ids": predicted.tolist(),
                "valid_draft": valid.tolist(),
                "matches": matches.tolist(),
                "greedy_matched_prefix": matched_prefix,
                "target_probability_of_draft": probabilities.tolist(),
                "num_sampled_before_postprocess": int(num_sampled[request]),
                "finite_target_logits": bool(torch.isfinite(target).all()),
                "has_nan_target_logits": bool(torch.isnan(target).any()),
                "has_posinf_target_logits": bool(torch.isposinf(target).any()),
            }
        )
    return rows


@torch.inference_mode()
def install_diagnostics(worker, directory, checkpoint, max_steps=3):
    """Install bounded diagnostic callbacks via LLM.collective_rpc, before generate."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    if max_steps < 1:
        raise ValueError("MMuse diagnostic steps must be positive")
    runner = worker.model_runner
    if not runner.vllm_config.model_config.enforce_eager:
        raise ValueError("MMuse diagnostics must run eagerly")
    proposer = runner.speculator
    if not runner.speculative_config.use_mmuse():
        raise ValueError("MMuse diagnostics requires method=mmuse")
    rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"rank-{rank}.jsonl"
    if path.exists():
        raise FileExistsError(f"Use an empty diagnostic directory: {path}")

    def json_finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_finite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_finite(item) for item in value]
        return value

    def write(event, **values):
        with path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(json_finite({"event": event, **values}), allow_nan=False)
                + "\n"
            )

    model = proposer.model
    target = runner.model
    target = (
        target.get_language_model() if hasattr(target, "get_language_model") else target
    )
    actual_auxiliary_ids = getattr(target.model, "aux_hidden_state_layers", None)
    write(
        "configuration",
        auxiliary_ids=model.config.mmuse_source_config["aux_hidden_state_layer_ids"],
        actual_target_auxiliary_ids=(
            list(actual_auxiliary_ids) if actual_auxiliary_ids is not None else None
        ),
        block_size=model.config.block_size,
        tp_size=tp_size,
        draft_sample_method=runner.speculative_config.draft_sample_method,
        rejection_sample_method=runner.speculative_config.rejection_sample_method,
        synthetic_acceptance_rates=runner.speculative_config.synthetic_acceptance_rates,
        target_quantization=runner.vllm_config.model_config.quantization,
        training_contract=model.config.mmuse_source_config.get(
            "target_training_contract"
        ),
    )
    report = audit_checkpoint(model, checkpoint, rank, tp_size)
    write("checkpoint_audit", **report)
    if report["mismatches"]:
        raise ValueError(
            f"Loaded MMuse draft weights differ from checkpoint; see {path}"
        )

    # Exercise the actual target IO modules, including their TP gather/order.
    ids = torch.tensor(
        [0, 1, model.config.mask_token_id, model.config.vocab_size - 1],
        device=model.lm_head.weight.device,
    )
    expected_embed = target.model.embed_tokens(ids)
    actual_embed = model.embed_input_ids(ids)
    hidden = expected_embed
    expected_logits = target.compute_logits(hidden)
    actual_logits = model.compute_logits(hidden)
    write(
        "borrowed_io_probe",
        embedding_max_abs_error=float((expected_embed - actual_embed).abs().max()),
        head_max_abs_error=float((expected_logits - actual_logits).abs().max()),
        head_argmax_equal=bool(
            torch.equal(expected_logits.argmax(-1), actual_logits.argmax(-1))
        ),
    )

    original_combine = model.combine_hidden_states
    counts = {"fusion": 0, "proposal": 0, "verification": 0}
    active = {"slots": []}
    pending = {}
    original_propose = proposer.propose

    def propose(*args, **kwargs):
        batch = args[0] if args else kwargs["input_batch"]
        active["slots"] = batch.idx_mapping_np.tolist()
        try:
            return original_propose(*args, **kwargs)
        finally:
            active["slots"] = []

    proposer.propose = propose

    def combine(hidden):
        result = original_combine(hidden)
        if counts["fusion"] < max_steps:
            counts["fusion"] += 1
            write(
                "features",
                step=counts["fusion"],
                raw_shape=list(hidden.shape),
                projected_shape=list(result.shape),
                finite_raw=bool(torch.isfinite(hidden).all()),
                finite_projected=bool(torch.isfinite(result).all()),
                raw_rms=float(hidden.float().square().mean().sqrt()),
                projected_rms=float(result.float().square().mean().sqrt()),
            )
        return result

    original_rollout = model.rollout_mmuse

    def layer_observer(name):
        def observe(module, inputs, output):
            if counts["proposal"] >= max_steps:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            write(
                "layer",
                proposal_step=counts["proposal"] + 1,
                name=name,
                shape=list(tensor.shape),
                finite=bool(torch.isfinite(tensor).all()),
                rms=float(tensor.float().square().mean().sqrt()),
            )

        return observe

    for index, layer in enumerate(model.model.layers):
        layer.self_attn.qkv_proj.register_forward_hook(
            layer_observer(f"layers.{index}.qkv")
        )
        layer.mlp.register_forward_hook(layer_observer(f"layers.{index}.mlp"))

    def rollout(hidden, anchor_ids, **kwargs):
        result = original_rollout(hidden, anchor_ids, **kwargs)
        if counts["proposal"] < max_steps:
            counts["proposal"] += 1
            tokens, logits = result
            for row, slot in enumerate(active["slots"]):
                pending[slot] = {
                    "anchor": int(anchor_ids[row]),
                    "tokens": tokens[row].tolist(),
                }
            finite = bool(torch.isfinite(logits).all())
            write(
                "proposal",
                step=counts["proposal"],
                anchors=anchor_ids.tolist(),
                slots=active["slots"],
                tokens=tokens.tolist(),
                finite_logits=finite,
                finite_hidden=bool(torch.isfinite(hidden).all()),
            )
        return result

    verifier = runner.rejection_sampler
    original_verify = verifier._verify

    def verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        if counts["verification"] < max_steps:
            # Signature: logits, draft_logits, draft_sampled, pos, cu_num_logits,
            # idx_mapping, ... . The framework calls this with positional args.
            rows = verification_rows(
                result[0], args[2], args[3], args[4], args[5], result[2]
            )
            if rows:
                for row in rows:
                    previous = pending.get(row["slot"])
                    row["previous_proposal_found"] = previous is not None
                    if previous is not None:
                        row["anchor_matches_previous_proposal"] = (
                            row["anchor_input_id"] == previous["anchor"]
                        )
                        row["draft_ids_match_previous_proposal"] = (
                            row["draft_ids"]
                            == previous["tokens"][: len(row["draft_ids"])]
                        )
                counts["verification"] += 1
                write(
                    "verification",
                    step=counts["verification"],
                    rows=rows,
                    temperature=verifier.sampler.sampling_states.temperature.gpu[
                        args[5]
                    ].tolist(),
                    has_draft_logits=args[1] is not None,
                )
        return result

    model.combine_hidden_states = combine
    model.rollout_mmuse = rollout
    verifier._verify = verify
    return {"rank": rank, "path": str(path), "checked_weights": report["checked"]}

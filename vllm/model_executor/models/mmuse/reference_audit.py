# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded diagnostics for zero-acceptance reference MMuse runs."""

import inspect
import json
import os
from pathlib import Path

import torch


def tensor_summary(value):
    # IO matrices can exceed a GiB. Never materialize a full FP32 copy just to
    # log a summary; these are explicitly sampled statistics, not weight audits.
    x = value.detach().reshape(-1)[:65536].float()
    finite = torch.isfinite(x)
    safe = x[finite]
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        sample_numel=x.numel(),
        nonfinite=int((~finite).sum()),
        rms=float(safe.square().mean().sqrt()) if safe.numel() else None,
    )


def verification_rows(logits, batch, draft_sampled, previous, limit):
    """Use actual rejection-sampler candidates, excluding its bonus row."""
    records = []
    cu = batch.cu_num_logits_np
    indices = batch.logits_indices.tolist()
    candidates = draft_sampled.tolist()
    positions = batch.positions.tolist()
    for row, request in enumerate(batch.req_ids[:limit]):
        start, end = int(cu[row]), int(cu[row + 1])
        old = previous.get(request)
        comparisons = []
        # The logit at row j predicts the candidate input at row j+1.
        for j in range(start, end - 1):
            candidate = int(candidates[j + 1])
            if candidate < 0:
                continue  # Chunked-prefill draft placeholders are not proposals.
            source_pos = int(positions[indices[j]])
            token_pos = int(positions[indices[j + 1]])
            x = logits[j].detach().float()
            predicted = int(x.argmax())
            offset = token_pos - old["anchor_position"] - 1 if old else -1
            expected = (
                old["tokens"][offset]
                if old and 0 <= offset < len(old["tokens"])
                else None
            )
            comparisons.append(
                dict(
                    source_position=source_pos,
                    candidate_position=token_pos,
                    candidate=candidate,
                    target_raw_argmax=predicted,
                    raw_match=candidate == predicted,
                    target_raw_probability=float(x.softmax(-1)[candidate]),
                    previous_proposal=expected,
                    transport_matches=None
                    if expected is None
                    else candidate == expected,
                )
            )
        records.append(dict(request_id=request, comparisons=comparisons))
    return records


@torch.inference_mode()
def install_reference_audit(worker, directory, steps=4, requests=2):
    """Install on every TP worker via LLM.collective_rpc; no extra collectives."""
    if steps < 2 or requests < 1:
        raise ValueError("Audit needs at least two steps and one request")
    runner = worker.model_runner
    proposer = runner.speculator
    if getattr(proposer, "_reference_audit_installed", False):
        raise ValueError("Reference audit is already installed in this worker")
    model = proposer.model
    if not hasattr(model, "history") or not hasattr(model, "runtime"):
        raise ValueError("Audit requires the DeepSeek stateless MMuse reference model")
    verifier = runner.rejection_sampler
    if verifier is None or not hasattr(verifier, "_verify_in_chunks"):
        raise ValueError("This worker does not expose the V2 rejection sampler")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"mmuse-{os.getpid()}-rank{getattr(worker, 'rank', 0)}.jsonl"
    state = dict(step=0, active=False, batch=None, anchors=[], previous={})

    def write(kind, **data):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    dict(kind=kind, step=state["step"], **data), ensure_ascii=False
                )
                + "\n"
            )

    source = model.config.mmuse_source_config
    inner = runner.model.model
    write(
        "interfaces",
        expected_aux_ids=source["aux_hidden_state_layer_ids"],
        target_aux_ids=list(getattr(inner, "aux_hidden_state_layers", [])),
        block_size=proposer.num_query_per_req,
        embedding=tensor_summary(model.embedding_weight),
        output_head=tensor_summary(model.output_weight),
        note=(
            "raw logits precede sampling filters; callbacks synchronize devices; "
            "not a benchmark"
        ),
    )
    original_propose = proposer.propose
    signature = inspect.signature(original_propose)
    original_rollout = model.runtime.rollout_correction
    original_verify = verifier._verify_in_chunks

    def before_forward(_model, args):
        if not state["active"]:
            return
        ids, positions, context_positions, features = args
        state["anchors"].append(int(positions[0]))
        row = len(state["anchors"]) - 1
        if row < requests:
            write(
                "block",
                request_id=state["batch"].req_ids[row],
                input_ids=ids.tolist(),
                positions=positions.tolist(),
                context_first=int(context_positions[0]),
                context_last=int(context_positions[-1]),
                context=tensor_summary(features),
            )

    handle = model.register_forward_pre_hook(before_forward)

    def rollout(hidden, anchors, **kwargs):
        result = original_rollout(hidden, anchors, **kwargs)
        if state["active"]:
            tokens, logits = result
            for row, request in enumerate(state["batch"].req_ids):
                proposal = dict(
                    anchor_position=state["anchors"][row], tokens=tokens[row].tolist()
                )
                state["previous"][request] = proposal
                if row < requests:
                    write(
                        "proposal",
                        request_id=request,
                        anchor=int(anchors[row]),
                        hidden=tensor_summary(hidden[row]),
                        **proposal,
                    )
        return result

    def verify(logits, batch, draft_logits, draft_sampled, *args, **kwargs):
        write(
            "verification",
            rows=verification_rows(
                logits, batch, draft_sampled, state["previous"], requests
            ),
        )
        return original_verify(
            logits, batch, draft_logits, draft_sampled, *args, **kwargs
        )

    def propose(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if values["dummy_run"]:
            return original_propose(*args, **kwargs)
        state["step"] += 1
        state["active"], state["batch"], state["anchors"] = (
            True,
            values["input_batch"],
            [],
        )
        aux = values["aux_hidden_states"]
        batch = state["batch"]
        write(
            "target_result",
            request_ids=batch.req_ids[:requests],
            num_sampled=values["num_sampled"][:requests].tolist(),
            num_rejected=values["num_rejected"][:requests].tolist(),
            scheduled_drafts=None
            if batch.num_draft_tokens_per_req is None
            else batch.num_draft_tokens_per_req[:requests].tolist(),
            aux=[] if aux is None else [tensor_summary(x) for x in aux],
        )
        try:
            return original_propose(*args, **kwargs)
        finally:
            state["active"] = False
            if state["step"] >= steps:
                proposer.propose = original_propose
                model.runtime.rollout_correction = original_rollout
                verifier._verify_in_chunks = original_verify
                handle.remove()
                write("audit_completed")

    proposer._reference_audit_installed = True
    proposer.propose = propose
    model.runtime.rollout_correction = rollout
    verifier._verify_in_chunks = verify
    return str(path)

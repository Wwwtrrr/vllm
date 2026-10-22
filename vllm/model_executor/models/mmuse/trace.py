# SPDX-License-Identifier: Apache-2.0
"""Bounded opt-in proposal/verification logs; no model or sampler changes."""

import json
import logging
import os

logger = logging.getLogger(__name__)


class MMuseTrace:
    def __init__(self, steps):
        if type(steps) is not int or not 0 <= steps <= 32:
            raise ValueError("mmuse_trace_steps must be an integer between 0 and 32")
        self.limit = steps
        self.proposals = 0
        self.verifications = 0
        self.results = 0
        self.previous = {}

    def emit(self, kind, **fields):
        logger.info(
            "MMUSE_TRACE %s", json.dumps(dict(kind=kind, pid=os.getpid(), **fields))
        )

    def proposal(self, batch, hidden, anchors, positions, tokens):
        if self.proposals >= self.limit:
            return
        self.proposals += 1
        anchor_ids = anchors.tolist()
        query_positions = positions.tolist()
        token_ids = tokens.tolist()
        rows = []
        for row, request in enumerate(batch.req_ids):
            prediction_positions = [position + 1 for position in query_positions[row]]
            self.previous[request] = dict(
                tokens=token_ids[row],
                positions=prediction_positions,
                proposal_batch=self.proposals,
            )
            rows.append(
                dict(
                    request_id=request,
                    slot=int(batch.idx_mapping_np[row]),
                    anchor_id=anchor_ids[row],
                    query_positions=query_positions[row],
                    prediction_positions=prediction_positions,
                    proposal_ids=token_ids[row],
                )
            )
        self.emit(
            "proposal", batch=self.proposals, hidden_shape=list(hidden.shape), rows=rows
        )

    def target(self, logits, batch):
        if self.verifications >= self.limit or not batch.num_draft_tokens:
            return
        self.verifications += 1
        indices = batch.logits_indices
        positions = batch.positions[indices].tolist()
        input_ids = batch.input_ids[indices].tolist()
        predictions = logits.argmax(-1).tolist()
        top_values, top_ids = logits.topk(min(5, logits.shape[-1]), dim=-1)
        top_values = top_values.tolist()
        top_ids = top_ids.tolist()
        # Only within-request successor scores are used below; boundary rows
        # have no candidate comparison.
        candidate_ids = batch.input_ids[indices].roll(-1).long()
        candidate_scores = (
            logits.gather(1, candidate_ids.unsqueeze(1)).squeeze(1).tolist()
        )
        cu = batch.cu_num_logits_np.tolist()
        prefill_lengths = batch.prefill_len_np.tolist()
        rows = []
        for row, request in enumerate(batch.req_ids):
            start, end = cu[row], cu[row + 1]
            previous = self.previous.get(request)
            comparisons = []
            for index in range(start, end - 1):
                candidate_position = positions[index + 1]
                # Same placeholder rule as gather_draft_sampled: a verification
                # span starting inside prefill cannot contain valid proposals.
                placeholder = (
                    candidate_position - (index + 1 - start) < prefill_lengths[row]
                )
                expected = None
                if previous and candidate_position in previous["positions"]:
                    offset = previous["positions"].index(candidate_position)
                    expected = previous["tokens"][offset]
                candidate = input_ids[index + 1]
                comparisons.append(
                    dict(
                        logit_position=positions[index],
                        candidate_position=candidate_position,
                        candidate_id=candidate,
                        placeholder=placeholder,
                        previous_proposal_id=expected,
                        transport_matches=None
                        if placeholder or expected is None
                        else candidate == expected,
                        target_argmax=predictions[index],
                        candidate_logit=candidate_scores[index],
                        target_top1_minus_candidate=(
                            top_values[index][0] - candidate_scores[index]
                        ),
                        argmax_matches=None
                        if placeholder
                        else candidate == predictions[index],
                    )
                )
            rows.append(
                dict(
                    request_id=request,
                    slot=int(batch.idx_mapping_np[row]),
                    previous_proposal_batch=None
                    if previous is None
                    else previous["proposal_batch"],
                    input_positions=positions[start:end],
                    input_ids=input_ids[start:end],
                    target_argmax=predictions[start:end],
                    target_top_ids=top_ids[start:end],
                    target_top_logits=top_values[start:end],
                    comparisons=comparisons,
                )
            )
        self.emit(
            "verify",
            batch=self.verifications,
            logits_stage="after_grammar_before_sampling_params",
            rows=rows,
        )

    def result(self, batch, num_sampled, num_rejected):
        if self.results >= self.limit or not batch.num_draft_tokens:
            return
        self.results += 1
        self.emit(
            "result",
            batch=self.results,
            request_ids=batch.req_ids,
            num_sampled=num_sampled[: batch.num_reqs].tolist(),
            num_rejected=num_rejected[: batch.num_reqs].tolist(),
            scheduled_drafts=None
            if batch.num_draft_tokens_per_req is None
            else batch.num_draft_tokens_per_req.tolist(),
        )

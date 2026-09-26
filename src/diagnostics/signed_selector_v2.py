"""Training, inference, and post-hoc helpers for experimental signed-selector v2."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.cbwdm_score import build_local_effects, one_hot
from src.diagnostics.signed_teacher_v1 import build_signed_training_groups
from src.diagnostics.signed_v2_model import SignedV2DualHeadSelector
from src.io_utils import read_jsonl
from src.selection_schema import make_selection_row, normalize_selected_doc
from src.selector_cross_encoder import build_selector_input, cbwdm_multitask_loss


@dataclass
class SignedV2TrainingGroup:
    """A signed-v1 teacher state augmented with authoritative gate targets."""

    example_id: str
    gold: str
    step: int
    query: str
    current_doc_ids: list[str]
    selected_docs: list[dict[str, Any]]
    candidate_doc_ids: list[str]
    candidate_docs: list[dict[str, Any]]
    effective_gains: list[float]
    supervision_classes: list[str]
    alignments: list[float]
    gate_targets: list[float]
    alignment_eps: float
    is_terminal_state: bool
    teacher_action: str


def gate_targets_from_alignments(
    alignments: Iterable[float], alignment_eps: float = 0.0
) -> list[float]:
    """Return 1 iff authoritative alignment is strictly above alignment_eps."""
    epsilon = float(alignment_eps)
    if epsilon < 0:
        raise ValueError("alignment_eps must be non-negative")
    return [1.0 if float(value) > epsilon else 0.0 for value in alignments]


def build_signed_v2_training_groups(
    *,
    teacher_path: str | Path,
    posteriors_path: str | Path,
    retrieval_path: str | Path | None = None,
    max_train_groups: int | None = None,
) -> list[SignedV2TrainingGroup]:
    """Reuse v1 text groups and attach alignment-only gate supervision."""
    base_groups = build_signed_training_groups(
        teacher_path=teacher_path,
        posteriors_path=posteriors_path,
        retrieval_path=retrieval_path,
        max_train_groups=max_train_groups,
    )
    posterior_rows = {str(row["id"]): row for row in read_jsonl(posteriors_path)}
    alignment_by_key: dict[tuple[str, int, str], tuple[float, bool, float]] = {}
    for row in read_jsonl(teacher_path):
        example_id = str(row["id"])
        if example_id not in posterior_rows:
            raise KeyError(f"Teacher id missing from posterior: {example_id}")
        source = posterior_rows[example_id]
        candidates = list(source.get("candidates", []))
        parameters = row.get("parameters", {})
        alignment_eps = float(parameters.get("alignment_eps", 0.0))
        if candidates:
            X, d = build_local_effects(
                np.asarray(source["eta0"]),
                np.asarray([candidate["eta"] for candidate in candidates]),
                str(source["label"]),
                list(source["labels"]),
                parameters.get("l_type", "euclidean_posterior_shift"),
                float(parameters.get("eps_smooth", 0.0)),
                parameters.get("target_smoothing", "paper_mixture"),
            )
            authoritative = {
                str(candidate["doc_id"]): float(X[index] @ d)
                for index, candidate in enumerate(candidates)
            }
        else:
            authoritative = {}
        for state in row.get("states", []):
            step = int(state["step"])
            for item in state.get("remaining_candidates", []):
                doc_id = str(item["candidate_id"])
                if doc_id not in authoritative:
                    raise KeyError(
                        f"Teacher candidate missing from posterior: {(example_id, step, doc_id)}"
                    )
                alignment = authoritative[doc_id]
                stored_alignment = float(item["alignment"])
                if not math.isclose(
                    alignment, stored_alignment, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(
                        "Teacher alignment differs from build_local_effects reconstruction "
                        f"for id={example_id} step={step} doc={doc_id}"
                    )
                admissible = bool(item.get("admissible", alignment > alignment_eps))
                expected = alignment > alignment_eps
                if admissible != expected:
                    raise ValueError(
                        "Teacher admissibility disagrees with authoritative alignment for "
                        f"id={example_id} step={step} doc={item['candidate_id']}"
                    )
                alignment_by_key[(example_id, step, doc_id)] = (
                    alignment,
                    admissible,
                    alignment_eps,
                )

    groups: list[SignedV2TrainingGroup] = []
    for group in base_groups:
        records = []
        for doc_id in group.candidate_doc_ids:
            key = (group.example_id, group.step, str(doc_id))
            if key not in alignment_by_key:
                raise KeyError(f"Missing signed-teacher alignment for {key}")
            records.append(alignment_by_key[key])
        epsilons = {record[2] for record in records}
        if len(epsilons) != 1:
            raise ValueError("A teacher state contains inconsistent alignment_eps values")
        epsilon = epsilons.pop()
        alignments = [record[0] for record in records]
        targets = gate_targets_from_alignments(alignments, epsilon)
        for target, supervision_class in zip(
            targets, group.supervision_classes, strict=True
        ):
            harmful = supervision_class == "explicit_harmful_negative"
            if harmful == bool(target):
                raise ValueError(
                    "signed-v1 supervision class disagrees with alignment-only gate target"
                )
        groups.append(
            SignedV2TrainingGroup(
                example_id=group.example_id,
                gold=group.gold,
                step=group.step,
                query=group.query,
                current_doc_ids=group.current_doc_ids,
                selected_docs=group.selected_docs,
                candidate_doc_ids=group.candidate_doc_ids,
                candidate_docs=group.candidate_docs,
                effective_gains=group.effective_gains,
                supervision_classes=group.supervision_classes,
                alignments=alignments,
                gate_targets=targets,
                alignment_eps=epsilon,
                is_terminal_state=group.is_terminal_state,
                teacher_action=group.teacher_action,
            )
        )
    return groups


def signed_v2_loss(
    gate_logits: torch.Tensor,
    utility_logits: torch.Tensor,
    *,
    alignments: Iterable[float],
    effective_gains: Iterable[float],
    alignment_eps: float = 0.0,
    b_plus: float = 0.01,
    b_minus: float = 0.001,
    beta: float = 0.25,
    gamma: float = 1.0,
    neutral_sample_policy: str = "negative",
    lambda_gate: float = 1.0,
    lambda_utility: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Gate all candidates; train utility only on authoritative-admissible ones."""
    if gate_logits.ndim != 1 or utility_logits.ndim != 1:
        raise ValueError("gate_logits and utility_logits must be one-dimensional")
    if gate_logits.shape != utility_logits.shape:
        raise ValueError("gate_logits and utility_logits must have identical shape")
    if lambda_gate < 0 or lambda_utility < 0:
        raise ValueError("loss weights must be non-negative")
    alignment_values = list(float(value) for value in alignments)
    gain_values = list(float(value) for value in effective_gains)
    if len(alignment_values) != gate_logits.numel() or len(gain_values) != gate_logits.numel():
        raise ValueError("alignment/gain lengths must match logits")
    if not torch.isfinite(gate_logits).all() or not torch.isfinite(utility_logits).all():
        raise ValueError("dual-head logits must be finite")

    gate_targets = torch.tensor(
        gate_targets_from_alignments(alignment_values, alignment_eps),
        dtype=gate_logits.dtype,
        device=gate_logits.device,
    )
    gate_loss = F.binary_cross_entropy_with_logits(gate_logits, gate_targets)
    utility_mask = gate_targets.bool()
    if bool(utility_mask.any()):
        masked_gains = [
            gain for gain, include in zip(gain_values, utility_mask.tolist(), strict=True)
            if include
        ]
        utility_loss, utility_details = cbwdm_multitask_loss(
            utility_logits[utility_mask],
            masked_gains,
            b_plus=b_plus,
            b_minus=b_minus,
            gamma=gamma,
            beta=beta,
            neutral_sample_policy=neutral_sample_policy,
        )
    else:
        zero = utility_logits.sum() * 0.0
        utility_loss = zero
        utility_details = {
            "ce_loss": zero,
            "rank_loss": zero,
            "num_positive": 0,
            "num_negative": 0,
            "num_neutral": 0,
            "valid_ranking_group": False,
            "skipped_ranking_group": True,
        }
    total = float(lambda_gate) * gate_loss + float(lambda_utility) * utility_loss
    if not torch.isfinite(total):
        raise FloatingPointError("signed-v2 loss became NaN or Inf")
    return total, {
        "gate_loss": gate_loss,
        "utility_loss": utility_loss,
        "utility_ce_loss": utility_details["ce_loss"],
        "utility_rank_loss": utility_details["rank_loss"],
        "num_gate_positive": int(gate_targets.sum().item()),
        "num_gate_negative": int((~utility_mask).sum().item()),
        "num_utility_candidates": int(utility_mask.sum().item()),
        **{
            key: value
            for key, value in utility_details.items()
            if key not in {"ce_loss", "rank_loss"}
        },
    }


DualScoreFunction = Callable[
    [str, list[dict[str, Any]], list[dict[str, Any]]],
    tuple[Sequence[float], Sequence[float]],
]


def _rank_value(candidate: dict[str, Any]) -> int:
    return int(candidate.get("rank", 0) or 0)


def greedy_select_dual_without_gold(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    score_remaining: DualScoreFunction,
    top_m: int = 4,
    min_docs: int = 0,
    gate_threshold: float = 0.0,
    utility_threshold: float = 0.0,
) -> dict[str, Any]:
    """Apply learned gate first, then rank admissible candidates by utility."""
    if top_m < 0 or min_docs < 0:
        raise ValueError("top_m and min_docs must be non-negative")
    if min_docs != 0:
        raise ValueError("signed-v2 v1 contract fixes min_docs=0")
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    steps: list[dict[str, Any]] = []
    stop_reason = "top_m_reached" if top_m == 0 else "no_predicted_admissible_candidate"

    for step in range(top_m):
        remaining = [
            candidate
            for candidate in candidates
            if str(candidate.get("doc_id")) not in selected_ids
        ]
        if not remaining:
            stop_reason = "no_predicted_admissible_candidate"
            steps.append(
                {
                    "step": step,
                    "selected_doc_id": None,
                    "predicted_gate_score": None,
                    "predicted_utility_score": None,
                    "remaining_count": 0,
                    "stop": True,
                    "stop_reason": stop_reason,
                    "all_candidate_scores": [],
                }
            )
            break
        gate_values, utility_values = score_remaining(query, selected, remaining)
        gates = [float(value) for value in gate_values]
        utilities = [float(value) for value in utility_values]
        if len(gates) != len(remaining) or len(utilities) != len(remaining):
            raise ValueError("score_remaining returned the wrong number of scores")
        if not all(math.isfinite(value) for value in gates + utilities):
            raise ValueError("score_remaining returned NaN or Inf")
        scored = list(zip(gates, utilities, remaining))
        score_rows = [
            {
                "doc_id": str(candidate.get("doc_id")),
                "gate_score": gate,
                "utility_score": utility,
                "rank": candidate.get("rank"),
                "source_rank": candidate.get("source_rank", candidate.get("rank")),
                "predicted_admissible": gate >= float(gate_threshold),
            }
            for gate, utility, candidate in sorted(
                scored,
                key=lambda item: (item[1], -_rank_value(item[2])),
                reverse=True,
            )
        ]
        admissible = [
            item for item in scored if item[0] >= float(gate_threshold)
        ]
        if not admissible:
            best_gate, _, _ = max(
                scored, key=lambda item: (item[0], -_rank_value(item[2]))
            )
            stop_reason = "no_predicted_admissible_candidate"
            steps.append(
                {
                    "step": step,
                    "selected_doc_id": None,
                    "predicted_gate_score": best_gate,
                    "predicted_utility_score": None,
                    "remaining_count": len(remaining),
                    "stop": True,
                    "stop_reason": stop_reason,
                    "all_candidate_scores": score_rows,
                }
            )
            break
        best_gate, best_utility, best = max(
            admissible, key=lambda item: (item[1], -_rank_value(item[2]))
        )
        if best_utility < float(utility_threshold):
            stop_reason = "utility_below_threshold"
            steps.append(
                {
                    "step": step,
                    "selected_doc_id": None,
                    "predicted_gate_score": best_gate,
                    "predicted_utility_score": best_utility,
                    "remaining_count": len(remaining),
                    "stop": True,
                    "stop_reason": stop_reason,
                    "all_candidate_scores": score_rows,
                }
            )
            break
        selected.append(best)
        selected_ids.add(str(best["doc_id"]))
        steps.append(
            {
                "step": step,
                "selected_doc_id": str(best["doc_id"]),
                "predicted_gate_score": best_gate,
                "predicted_utility_score": best_utility,
                "remaining_count": len(remaining),
                "stop": False,
                "stop_reason": None,
                "all_candidate_scores": score_rows,
            }
        )
        if len(selected) >= top_m:
            stop_reason = "top_m_reached"

    return {
        "selected_docs": selected,
        "selection_steps": steps,
        "stop_reason": stop_reason,
    }


def select_row_without_gold(
    row: dict[str, Any],
    selector: SignedV2DualHeadSelector,
    *,
    method: str = "rag_cbwdm_signed_v2",
    top_m: int = 4,
    min_docs: int = 0,
    gate_threshold: float = 0.0,
    utility_threshold: float = 0.0,
    batch_size: int = 32,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    """Select from a posterior container without using any gold-bearing field."""
    candidates = list(row.get("candidates", []))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    def score(
        query: str,
        selected: list[dict[str, Any]],
        remaining: list[dict[str, Any]],
    ) -> tuple[list[float], list[float]]:
        texts = [build_selector_input(query, selected, candidate) for candidate in remaining]
        output = selector.score_texts(
            texts, batch_size=batch_size, requires_grad=False
        )
        return (
            [float(value) for value in output.gate_logit.detach().cpu().tolist()],
            [float(value) for value in output.utility_logit.detach().cpu().tolist()],
        )

    action = greedy_select_dual_without_gold(
        query=str(row.get("query") or ""),
        candidates=candidates,
        score_remaining=score,
        top_m=top_m,
        min_docs=min_docs,
        gate_threshold=gate_threshold,
        utility_threshold=utility_threshold,
    )
    selected_docs = []
    selected_steps = [item for item in action["selection_steps"] if not item["stop"]]
    for candidate, step in zip(action["selected_docs"], selected_steps, strict=True):
        normalized = normalize_selected_doc(
            candidate,
            selector_score=float(step["predicted_utility_score"]),
            selection_step=int(step["step"]),
        )
        normalized.update(
            {
                "gate_score": float(step["predicted_gate_score"]),
                "utility_score": float(step["predicted_utility_score"]),
                "selector_score_alias": "utility_score",
            }
        )
        selected_docs.append(normalized)
    return make_selection_row(
        row,
        method=method,
        selected_docs=selected_docs,
        selection_steps=action["selection_steps"],
        stop_reason=action["stop_reason"],
        diagnostic_only=False,
        max_docs=top_m,
        uses_gold_at_test=False,
        selection_metadata={
            "variant": "signed_selector_v2",
            "architecture": "rag_cbwdm_signed_v2_dual_head_v1",
            "experimental": True,
            "deployable_selector": True,
            "uses_gold_at_inference": False,
            "state_aware": True,
            "gate_score_quantity": "raw_gate_logit",
            "utility_score_quantity": "raw_utility_logit",
            "selector_score_alias": "utility_score",
            "gate_threshold": float(gate_threshold),
            "utility_threshold": float(utility_threshold),
            "min_docs": min_docs,
            "top_m": top_m,
        },
    )


def _auroc(scores: list[float], targets: list[int]) -> float | None:
    positives = sum(targets)
    negatives = len(targets) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, targets), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(target for _, target in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _alignment_maps(
    source: dict[str, Any], cbwdm: dict[str, Any]
) -> tuple[dict[str, float], dict[str, float]]:
    candidates = list(source.get("candidates", []))
    if not candidates:
        return {}, {}
    eta0 = np.asarray(source["eta0"], dtype=float)
    candidate_etas = np.asarray([item["eta"] for item in candidates], dtype=float)
    labels = list(source["labels"])
    label = str(source["label"])
    X, d = build_local_effects(
        eta0,
        candidate_etas,
        label,
        labels,
        cbwdm.get("L_type", "euclidean_posterior_shift"),
        float(cbwdm.get("eps_smooth", 0)),
        cbwdm.get("target_smoothing", "paper_mixture"),
    )
    raw_d = one_hot(label, labels) - eta0
    raw_X = candidate_etas - eta0
    authoritative = {
        str(item["doc_id"]): float(X[index] @ d)
        for index, item in enumerate(candidates)
    }
    raw = {
        str(item["doc_id"]): float(raw_X[index] @ raw_d)
        for index, item in enumerate(candidates)
    }
    return authoritative, raw


def gate_separability_audit(
    *,
    selection_rows: dict[str, dict[str, Any]],
    posterior_rows: dict[str, dict[str, Any]],
    cbwdm: dict[str, Any],
    alignment_eps: float = 0.0,
) -> dict[str, Any]:
    records: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for identifier, selection in selection_rows.items():
        source = posterior_rows[identifier]
        label = str(source["label"])
        authoritative, raw = _alignment_maps(source, cbwdm)
        for step in selection.get("selection_steps", []):
            step_index = int(step["step"])
            for item in step.get("all_candidate_scores", []):
                doc_id = str(item["doc_id"])
                if doc_id not in authoritative:
                    continue
                gate_score = float(item["gate_score"])
                if not math.isfinite(gate_score):
                    raise ValueError(
                        f"Non-finite gate score for id={identifier} doc={doc_id}"
                    )
                record = (gate_score, authoritative[doc_id], raw[doc_id])
                for key in (
                    "ALL",
                    f"gold={label}",
                    f"step={step_index}",
                    f"gold={label}/step={step_index}",
                ):
                    records[key].append(record)
    result = {}
    for key, values in sorted(records.items()):
        auth_scores = [item[0] for item in values]
        auth_targets = [int(item[1] > alignment_eps) for item in values]
        raw_pairs = [item for item in values if item[2] != 0.0]
        raw_scores = [item[0] for item in raw_pairs]
        raw_targets = [int(item[2] > 0.0) for item in raw_pairs]
        result[key] = {
            "authoritative_pos_vs_nonpos": {
                "count": len(values),
                "positive": sum(auth_targets),
                "nonpositive": len(auth_targets) - sum(auth_targets),
                "auroc": _auroc(auth_scores, auth_targets),
            },
            "raw_pos_vs_neg": {
                "count": len(raw_pairs),
                "positive": sum(raw_targets),
                "negative": len(raw_targets) - sum(raw_targets),
                "zero_excluded": len(values) - len(raw_pairs),
                "auroc": _auroc(raw_scores, raw_targets),
            },
        }
    return result


def harmful_selection_audit(
    *,
    selection_rows: dict[str, dict[str, Any]],
    posterior_rows: dict[str, dict[str, Any]],
    prediction_rows: dict[str, dict[str, Any]],
    cbwdm: dict[str, Any],
) -> dict[str, Any]:
    buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "queries": 0,
            "selected": 0,
            "authoritative_negative": 0,
            "raw_negative": 0,
            "first_selected": 0,
            "first_raw_negative": 0,
        }
    )
    for identifier, selection in selection_rows.items():
        source = posterior_rows[identifier]
        label = str(source["label"])
        prediction = prediction_rows.get(identifier, {})
        predicted = prediction.get(
            "pred", prediction.get("prediction", prediction.get("predicted_label"))
        )
        if predicted is None:
            raise ValueError(f"Prediction label is missing for id={identifier}")
        keys = ["ALL", f"gold={label}"]
        if label == "SUPPORTS":
            keys.append("gold=SUPPORTS/correct" if str(predicted) == label else "gold=SUPPORTS/wrong")
        authoritative, raw = _alignment_maps(source, cbwdm)
        selected_ids = [
            doc_id
            for doc_id in map(str, selection.get("selected_doc_ids", []))
            if doc_id in authoritative
        ]
        for key in keys:
            bucket = buckets[key]
            bucket["queries"] += 1
            bucket["selected"] += len(selected_ids)
            bucket["authoritative_negative"] += sum(
                authoritative[doc_id] < 0 for doc_id in selected_ids
            )
            bucket["raw_negative"] += sum(raw[doc_id] < 0 for doc_id in selected_ids)
            if selected_ids:
                bucket["first_selected"] += 1
                bucket["first_raw_negative"] += int(raw[selected_ids[0]] < 0)
    result = {}
    for key, bucket in sorted(buckets.items()):
        selected = bucket["selected"]
        first = bucket["first_selected"]
        result[key] = {
            **bucket,
            "selected_authoritative_negative_ratio": (
                bucket["authoritative_negative"] / selected if selected else None
            ),
            "selected_raw_negative_ratio": (
                bucket["raw_negative"] / selected if selected else None
            ),
            "first_selected_raw_negative_ratio": (
                bucket["first_raw_negative"] / first if first else None
            ),
        }
    return result


def stopping_and_budget_audit(
    selection_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = [int(row.get("num_docs", len(row.get("selected_doc_ids", [])))) for row in selection_rows.values()]
    stops = Counter(str(row.get("stop_reason")) for row in selection_rows.values())
    return {
        "query_count": len(counts),
        "stop_reason_counts": {
            name: stops[name]
            for name in (
                "no_predicted_admissible_candidate",
                "utility_below_threshold",
                "top_m_reached",
            )
        },
        "other_stop_reason_counts": {
            key: value
            for key, value in sorted(stops.items())
            if key
            not in {
                "no_predicted_admissible_candidate",
                "utility_below_threshold",
                "top_m_reached",
            }
        },
        "avg_docs": sum(counts) / len(counts) if counts else None,
        "zero_doc_ratio": sum(value == 0 for value in counts) / len(counts) if counts else None,
        "num_docs_histogram": dict(sorted(Counter(counts).items())),
    }

"""Clean FEVER adaptation of pointwise Document Information Gain supervision."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable

TEACHER_DEFINITION = "eta_gold_doc_minus_eta_gold_query"
TRAINING_ROLES = frozenset({"train", "train_core"})
VALIDATION_ROLES = frozenset({"dev", "validation"})
HELD_OUT_ROLES = frozenset({"test", "held_out_test"})
TEACHER_PURPOSES = frozenset({"training", "validation_diagnostic"})


def validate_probability_vector(values: Any, *, where: str, tolerance: float = 1e-3) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{where}: expected non-empty probability list")
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) and value >= 0.0 for value in vector):
        raise ValueError(f"{where}: probabilities must be finite and non-negative")
    if abs(sum(vector) - 1.0) > tolerance:
        raise ValueError(f"{where}: probabilities sum to {sum(vector)}, expected 1")
    return vector


def validate_teacher_roles(roles: Iterable[str], *, purpose: str) -> str:
    normalized = {str(role) for role in roles}
    if purpose not in TEACHER_PURPOSES:
        raise ValueError(
            f"Unsupported InfoGain teacher purpose={purpose!r}; "
            f"choices={sorted(TEACHER_PURPOSES)}"
        )
    forbidden = normalized & HELD_OUT_ROLES
    if forbidden:
        raise ValueError(
            "InfoGain teacher must never use held-out roles: "
            f"{sorted(forbidden)}"
        )
    allowed = TRAINING_ROLES if purpose == "training" else VALIDATION_ROLES
    if len(normalized) != 1 or not normalized <= allowed:
        raise ValueError(
            f"InfoGain {purpose} teacher requires exactly one role from "
            f"{sorted(allowed)}, got {sorted(normalized)}; "
            "validation roles require purpose='validation_diagnostic'"
        )
    return next(iter(normalized))


def validate_teacher_rows_for_training(rows: Iterable[dict[str, Any]]) -> str:
    materialized = list(rows)
    return validate_teacher_roles(
        (str(row.get("split") or "") for row in materialized),
        purpose="training",
    )


def posterior_to_teacher_rows(
    row: dict[str, Any], *, purpose: str = "training"
) -> list[dict[str, Any]]:
    """Build pointwise teacher rows for one explicit training/diagnostic role."""
    split = str(row.get("split") or "")
    validate_teacher_roles({split}, purpose=purpose)
    labels = list(row.get("labels", []))
    gold = row.get("label", row.get("gold"))
    if gold not in labels:
        raise ValueError(f"Gold label {gold!r} is absent from posterior label order {labels!r}")
    eta0 = validate_probability_vector(row.get("eta0"), where=f"{row.get('id')}.eta0")
    if len(eta0) != len(labels):
        raise ValueError("eta0 length does not match labels")
    gold_index = labels.index(gold)
    result = []
    for candidate in row.get("candidates", []):
        eta = validate_probability_vector(
            candidate.get("eta"), where=f"{row.get('id')}.{candidate.get('doc_id')}.eta"
        )
        if len(eta) != len(labels):
            raise ValueError("candidate eta length does not match labels")
        dig = eta[gold_index] - eta0[gold_index]
        if not math.isfinite(dig):
            raise ValueError("DIG must be finite")
        result.append(
            {
                "schema_version": "rag_cbwdm_infogain_teacher.v1",
                "query_id": str(row.get("id")),
                "doc_id": str(candidate.get("doc_id")),
                "split": split,
                "query": row.get("query"),
                "title": candidate.get("title"),
                "text": candidate.get("text"),
                "gold_label": gold,
                "labels": labels,
                "gold_index": gold_index,
                "eta0_gold": eta0[gold_index],
                "eta_doc_gold": eta[gold_index],
                "dig": dig,
                "retrieval_rank": candidate.get("rank"),
                "retrieval_score": candidate.get(
                    "retrieval_score", candidate.get("score")
                ),
                "teacher_definition": TEACHER_DEFINITION,
                "teacher_purpose": purpose,
            }
        )
    return result


def resolve_thresholds(
    digs: Iterable[float],
    *,
    mode: str,
    b_pos: float | None = None,
    b_neg: float | None = None,
    positive_quantile: float = 0.75,
    negative_quantile: float = 0.25,
) -> dict[str, Any]:
    values = sorted(float(value) for value in digs)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Threshold calibration requires finite DIG values")
    if mode == "explicit":
        if b_pos is None or b_neg is None:
            raise ValueError("explicit threshold mode requires b_pos and b_neg")
    elif mode in {"train_quantile", "validation_calibrated"}:
        if not (0.0 <= negative_quantile <= positive_quantile <= 1.0):
            raise ValueError("Require 0 <= negative_quantile <= positive_quantile <= 1")

        def quantile(q: float) -> float:
            index = q * (len(values) - 1)
            lower = int(index)
            upper = min(lower + 1, len(values) - 1)
            weight = index - lower
            return values[lower] * (1.0 - weight) + values[upper] * weight

        b_neg, b_pos = quantile(negative_quantile), quantile(positive_quantile)
    else:
        raise ValueError(f"Unsupported threshold mode: {mode}")
    assert b_pos is not None and b_neg is not None
    if b_neg > b_pos:
        raise ValueError("b_neg cannot exceed b_pos")
    counts = Counter(label_dig(value, b_pos=b_pos, b_neg=b_neg) for value in values)
    if not counts["positive"] or not counts["negative"]:
        raise ValueError(
            "InfoGain thresholds produced no positive or negative examples; "
            "change thresholds using train/validation data."
        )
    return {
        "mode": mode,
        "b_pos": float(b_pos),
        "b_neg": float(b_neg),
        "positive_quantile": positive_quantile if mode != "explicit" else None,
        "negative_quantile": negative_quantile if mode != "explicit" else None,
        "label_distribution": dict(counts),
    }


def label_dig(dig: float, *, b_pos: float, b_neg: float) -> str:
    if dig >= b_pos:
        return "positive"
    if dig <= b_neg:
        return "negative"
    return "neutral"


def group_teacher_rows(rows: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["query_id"])].append(row)
    return list(groups.values())


def pointwise_input(query: str, title: str | None, text: str | None) -> str:
    evidence = f"{title}: {text}" if title and text else title or text or ""
    return (
        f"Claim:\n{query}\n\nCandidate evidence:\n{evidence}\n\n"
        "Task:\nScore this candidate's usefulness for verifying the claim."
    )


def infogain_multitask_loss(
    rank_scores: Any,
    filter_logits: Any,
    digs: list[float],
    *,
    b_pos: float,
    b_neg: float,
    beta: float,
) -> tuple[Any, dict[str, Any]]:
    """RankNet within a query plus positive/negative filtering; neutral CE ignored."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    import torch
    import torch.nn.functional as F

    if rank_scores.ndim != 1 or len(digs) != rank_scores.shape[0]:
        raise ValueError("rank_scores/digs shape mismatch")
    pair_losses = []
    for left in range(len(digs)):
        for right in range(left + 1, len(digs)):
            if digs[left] == digs[right]:
                continue
            high, low = (left, right) if digs[left] > digs[right] else (right, left)
            pair_losses.append(F.softplus(-(rank_scores[high] - rank_scores[low])))
    rank_loss = (
        torch.stack(pair_losses).mean()
        if pair_losses
        else rank_scores.sum() * 0.0
    )
    indices = []
    targets = []
    labels = []
    for index, dig in enumerate(digs):
        label = label_dig(dig, b_pos=b_pos, b_neg=b_neg)
        labels.append(label)
        if label != "neutral":
            indices.append(index)
            targets.append(1 if label == "positive" else 0)
    filter_loss = (
        F.cross_entropy(
            filter_logits[torch.tensor(indices, device=filter_logits.device)],
            torch.tensor(targets, device=filter_logits.device),
        )
        if indices
        else filter_logits.sum() * 0.0
    )
    total = beta * rank_loss + (1.0 - beta) * filter_loss
    return total, {
        "rank_loss": rank_loss,
        "filter_loss": filter_loss,
        "num_pairs": len(pair_losses),
        "num_filter": len(indices),
        "num_neutral": labels.count("neutral"),
    }

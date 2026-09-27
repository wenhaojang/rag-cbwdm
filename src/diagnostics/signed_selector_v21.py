"""Additive signed-ordinal training and v1-equivalent inference for signed-v2.1."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.cbwdm_score import build_local_effects, one_hot
from src.diagnostics.signed_selector_v1 import (
    greedy_select_without_gold as v1_greedy_select_without_gold,
)
from src.diagnostics.signed_selector_v1 import (
    select_row_without_gold as v1_select_row_without_gold,
)
from src.selector_cross_encoder import CrossEncoderSelector, cbwdm_multitask_loss


SIGNED_V21_METHOD = "rag_cbwdm_signed_v21"
SIGNED_V21_VARIANT = "signed_selector_v21"
SIGNED_V21_LOSS_TYPE = "cbwdm_multitask_plus_signed_ordinal"
SIGNED_V21_ARCHITECTURE = "pretrained_single_head_cross_encoder"


def signed_ordinal_rank_loss(
    scores: torch.Tensor,
    supervision_classes: Iterable[str],
    *,
    gamma_signed: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Rank ordinary admissible low-utility candidates above harmful candidates."""
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if gamma_signed <= 0:
        raise ValueError("gamma_signed must be positive")
    classes = [str(value) for value in supervision_classes]
    if len(classes) != scores.numel():
        raise ValueError("supervision_classes length does not match scores length")
    allowed = {"positive", "negative", "neutral", "explicit_harmful_negative"}
    unknown = sorted(set(classes) - allowed)
    if unknown:
        raise ValueError(f"Unknown signed supervision classes: {unknown}")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")

    low_mask = torch.tensor(
        [value in {"negative", "neutral"} for value in classes],
        dtype=torch.bool,
        device=scores.device,
    )
    harmful_mask = torch.tensor(
        [value == "explicit_harmful_negative" for value in classes],
        dtype=torch.bool,
        device=scores.device,
    )
    admissible_low_scores = scores[low_mask]
    harmful_scores = scores[harmful_mask]
    pair_count = int(admissible_low_scores.numel() * harmful_scores.numel())
    valid = pair_count > 0
    if valid:
        pairwise = float(gamma_signed) * (
            harmful_scores[:, None] - admissible_low_scores[None, :]
        )
        loss = F.softplus(pairwise).mean()
    else:
        loss = scores.sum() * 0.0
    if not torch.isfinite(loss):
        raise FloatingPointError("signed ordinal ranking loss became NaN or Inf")
    counts = Counter(classes)
    return loss, {
        "signed_valid_ranking_group": valid,
        "signed_skipped_ranking_group": not valid,
        "signed_pair_count": pair_count,
        "positive_candidate_count": counts["positive"],
        "ordinary_negative_count": counts["negative"],
        "neutral_count": counts["neutral"],
        "harmful_count": counts["explicit_harmful_negative"],
        "admissible_low_count": int(low_mask.sum().item()),
    }


def signed_v21_multitask_loss(
    scores: torch.Tensor,
    gains: Iterable[float],
    supervision_classes: Iterable[str],
    *,
    b_plus: float,
    b_minus: float,
    gamma: float = 1.0,
    beta: float = 0.25,
    neutral_sample_policy: str = "negative",
    lambda_signed: float = 0.25,
    gamma_signed: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the unchanged v1 base loss plus the signed ordinal auxiliary."""
    if lambda_signed < 0:
        raise ValueError("lambda_signed must be non-negative")
    gain_values = list(gains)
    class_values = list(supervision_classes)
    base_loss, base_details = cbwdm_multitask_loss(
        scores,
        gain_values,
        b_plus=b_plus,
        b_minus=b_minus,
        gamma=gamma,
        beta=beta,
        neutral_sample_policy=neutral_sample_policy,
    )
    signed_loss, signed_details = signed_ordinal_rank_loss(
        scores,
        class_values,
        gamma_signed=gamma_signed,
    )
    total = base_loss + float(lambda_signed) * signed_loss
    if not torch.isfinite(total):
        raise FloatingPointError("signed-v2.1 total loss became NaN or Inf")
    return total, {
        "base_loss": base_loss,
        "ce_loss": base_details["ce_loss"],
        "base_rank_loss": base_details["rank_loss"],
        "signed_rank_loss": signed_loss,
        "base_valid_ranking_group": base_details["valid_ranking_group"],
        "base_skipped_ranking_group": base_details["skipped_ranking_group"],
        "base_num_positive": base_details["num_positive"],
        "base_num_negative": base_details["num_negative"],
        "base_num_neutral": base_details["num_neutral"],
        **signed_details,
    }


ScalarScoreFunction = Callable[
    [str, list[dict[str, Any]], list[dict[str, Any]]], Sequence[float]
]


def greedy_select_without_gold(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    score_remaining: ScalarScoreFunction,
    top_m: int = 4,
    min_docs: int = 0,
    score_threshold: float | None = 0.0,
) -> dict[str, Any]:
    """Delegate exactly to signed-v1's scalar greedy inference policy."""
    return v1_greedy_select_without_gold(
        query=query,
        candidates=candidates,
        score_remaining=score_remaining,
        top_m=top_m,
        min_docs=min_docs,
        score_threshold=score_threshold,
    )


def select_row_without_gold(
    row: dict[str, Any],
    selector: CrossEncoderSelector,
    *,
    top_m: int,
    min_docs: int,
    score_threshold: float | None,
    batch_size: int,
    max_candidates: int | None,
) -> dict[str, Any]:
    """Reuse v1 feature construction and selection, changing metadata only."""
    result = v1_select_row_without_gold(
        row,
        selector,
        method=SIGNED_V21_METHOD,
        top_m=top_m,
        min_docs=min_docs,
        score_threshold=score_threshold,
        batch_size=batch_size,
        max_candidates=max_candidates,
    )
    result["selection_metadata"].update(
        {
            "variant": SIGNED_V21_VARIANT,
            "method": SIGNED_V21_METHOD,
            "architecture": SIGNED_V21_ARCHITECTURE,
        }
    )
    return result


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
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


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
    raw_X = candidate_etas - eta0
    raw_d = one_hot(label, labels) - eta0
    authoritative = {
        str(candidate["doc_id"]): float(X[index] @ d)
        for index, candidate in enumerate(candidates)
    }
    raw = {
        str(candidate["doc_id"]): float(raw_X[index] @ raw_d)
        for index, candidate in enumerate(candidates)
    }
    return authoritative, raw


def score_separability_audit(
    *,
    selection_rows: dict[str, dict[str, Any]],
    posterior_rows: dict[str, dict[str, Any]],
    cbwdm: dict[str, Any],
    alignment_eps: float = 0.0,
) -> dict[str, Any]:
    """Compare saved scalar logits against post-hoc raw/authoritative signs."""
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
                score = float(item["score"])
                if not math.isfinite(score):
                    raise ValueError(
                        f"Non-finite selector score for id={identifier} doc={doc_id}"
                    )
                record = (score, authoritative[doc_id], raw[doc_id])
                for key in (
                    "ALL",
                    f"gold={label}",
                    f"step={step_index}",
                    f"gold={label}/step={step_index}",
                ):
                    records[key].append(record)
    result = {}
    for key, values in sorted(records.items()):
        authoritative_pairs = [
            item
            for item in values
            if item[1] > alignment_eps or item[1] < -alignment_eps
        ]
        raw_pairs = [item for item in values if item[2] != 0.0]
        authoritative_targets = [
            int(item[1] > alignment_eps) for item in authoritative_pairs
        ]
        raw_targets = [int(item[2] > 0.0) for item in raw_pairs]
        result[key] = {
            "authoritative_pos_vs_neg": {
                "count": len(authoritative_pairs),
                "positive": sum(authoritative_targets),
                "negative": len(authoritative_targets) - sum(authoritative_targets),
                "zero_excluded": len(values) - len(authoritative_pairs),
                "auroc": _auroc(
                    [item[0] for item in authoritative_pairs], authoritative_targets
                ),
            },
            "raw_pos_vs_neg": {
                "count": len(raw_pairs),
                "positive": sum(raw_targets),
                "negative": len(raw_targets) - sum(raw_targets),
                "zero_excluded": len(values) - len(raw_pairs),
                "auroc": _auroc([item[0] for item in raw_pairs], raw_targets),
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
    """Summarize harmful selection and SUPPORTS error associations post hoc."""
    buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "queries": 0,
            "selected_docs": 0,
            "authoritative_negative_docs": 0,
            "raw_negative_docs": 0,
            "queries_with_any_raw_negative": 0,
            "queries_with_selected_docs": 0,
            "first_selected_raw_negative": 0,
        }
    )
    for identifier, selection in selection_rows.items():
        source = posterior_rows[identifier]
        label = str(source["label"])
        prediction = prediction_rows[identifier]
        predicted = prediction.get(
            "pred", prediction.get("prediction", prediction.get("predicted_label"))
        )
        if predicted is None:
            raise ValueError(f"Prediction label is missing for id={identifier}")
        keys = ["ALL", f"gold={label}"]
        if label == "SUPPORTS":
            keys.append(
                "gold=SUPPORTS/correct"
                if str(predicted) == label
                else "gold=SUPPORTS/wrong"
            )
        authoritative, raw = _alignment_maps(source, cbwdm)
        selected_ids = [
            doc_id
            for doc_id in map(str, selection.get("selected_doc_ids", []))
            if doc_id in authoritative
        ]
        any_raw_negative = any(raw[doc_id] < 0 for doc_id in selected_ids)
        first_raw_negative = bool(selected_ids and raw[selected_ids[0]] < 0)
        for key in keys:
            bucket = buckets[key]
            bucket["queries"] += 1
            bucket["selected_docs"] += len(selected_ids)
            bucket["authoritative_negative_docs"] += sum(
                authoritative[doc_id] < 0 for doc_id in selected_ids
            )
            bucket["raw_negative_docs"] += sum(
                raw[doc_id] < 0 for doc_id in selected_ids
            )
            bucket["queries_with_any_raw_negative"] += int(any_raw_negative)
            bucket["queries_with_selected_docs"] += int(bool(selected_ids))
            bucket["first_selected_raw_negative"] += int(first_raw_negative)
    result = {}
    for key, bucket in sorted(buckets.items()):
        query_count = bucket["queries"]
        selected_count = bucket["selected_docs"]
        first_denominator = bucket["queries_with_selected_docs"]
        result[key] = {
            **bucket,
            "selected_authoritative_negative_ratio": (
                bucket["authoritative_negative_docs"] / selected_count
                if selected_count
                else None
            ),
            "selected_raw_negative_ratio": (
                bucket["raw_negative_docs"] / selected_count
                if selected_count
                else None
            ),
            "query_any_raw_negative_ratio": (
                bucket["queries_with_any_raw_negative"] / query_count
                if query_count
                else None
            ),
            "first_selected_raw_negative_ratio": (
                bucket["first_selected_raw_negative"] / first_denominator
                if first_denominator
                else None
            ),
        }
    return result


def stopping_and_budget_audit(
    selection_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = [
        int(row.get("num_docs", len(row.get("selected_doc_ids", []))))
        for row in selection_rows.values()
    ]
    stops = Counter(str(row.get("stop_reason")) for row in selection_rows.values())
    return {
        "query_count": len(counts),
        "avg_docs": sum(counts) / len(counts) if counts else None,
        "zero_doc_ratio": (
            sum(value == 0 for value in counts) / len(counts) if counts else None
        ),
        "num_docs_histogram": dict(sorted(Counter(counts).items())),
        "stop_reason_counts": dict(sorted(stops.items())),
    }

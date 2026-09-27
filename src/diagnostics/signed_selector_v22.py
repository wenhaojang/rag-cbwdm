"""State-level top-of-list training and v1-equivalent inference for signed-v2.2."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

from src.diagnostics.signed_selector_v1 import (
    greedy_select_without_gold as v1_greedy_select_without_gold,
)
from src.diagnostics.signed_selector_v1 import (
    select_row_without_gold as v1_select_row_without_gold,
)
from src.selector_cross_encoder import CrossEncoderSelector, cbwdm_multitask_loss


SIGNED_V22_METHOD = "rag_cbwdm_signed_v22"
SIGNED_V22_VARIANT = "signed_selector_v22"
SIGNED_V22_LOSS_TYPE = "cbwdm_multitask_plus_state_top_ordinal"
SIGNED_V22_ARCHITECTURE = "pretrained_single_head_cross_encoder"

_ADMISSIBLE_CLASSES = {"positive", "negative", "neutral"}
_HARMFUL_CLASS = "explicit_harmful_negative"
_ALLOWED_CLASSES = _ADMISSIBLE_CLASSES | {_HARMFUL_CLASS}


def state_top_ordinal_loss(
    scores: torch.Tensor,
    supervision_classes: Iterable[str],
    *,
    gamma_top: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Order the highest-scoring admissible candidate above the highest harmful one."""
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if gamma_top <= 0:
        raise ValueError("gamma_top must be positive")
    classes = [str(value) for value in supervision_classes]
    if len(classes) != scores.numel():
        raise ValueError("supervision_classes length does not match scores length")
    unknown = sorted(set(classes) - _ALLOWED_CLASSES)
    if unknown:
        raise ValueError(f"Unknown signed supervision classes: {unknown}")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")

    admissible_mask = torch.tensor(
        [value in _ADMISSIBLE_CLASSES for value in classes],
        dtype=torch.bool,
        device=scores.device,
    )
    harmful_mask = torch.tensor(
        [value == _HARMFUL_CLASS for value in classes],
        dtype=torch.bool,
        device=scores.device,
    )
    admissible_scores = scores[admissible_mask]
    harmful_scores = scores[harmful_mask]
    valid = bool(admissible_scores.numel() and harmful_scores.numel())
    best_admissible: torch.Tensor | None = None
    best_harmful: torch.Tensor | None = None
    top_gap: torch.Tensor | None = None
    if valid:
        best_admissible = torch.max(admissible_scores)
        best_harmful = torch.max(harmful_scores)
        top_gap = best_admissible - best_harmful
        loss = F.softplus(float(gamma_top) * (best_harmful - best_admissible))
    else:
        loss = scores.sum() * 0.0
    if not torch.isfinite(loss):
        raise FloatingPointError("state top ordinal loss became NaN or Inf")

    counts = Counter(classes)
    gap_value = float(top_gap.detach().cpu()) if top_gap is not None else None
    return loss, {
        "top_valid_ranking_group": valid,
        "top_skipped_ranking_group": not valid,
        "top_competition_count": int(valid),
        "admissible_count": int(admissible_mask.sum().item()),
        "harmful_count": int(harmful_mask.sum().item()),
        "positive_candidate_count": counts["positive"],
        "ordinary_negative_count": counts["negative"],
        "neutral_count": counts["neutral"],
        "best_admissible_score": (
            float(best_admissible.detach().cpu())
            if best_admissible is not None
            else None
        ),
        "best_harmful_score": (
            float(best_harmful.detach().cpu()) if best_harmful is not None else None
        ),
        "top_gap": gap_value,
        "top_correct_order": bool(gap_value is not None and gap_value > 0.0),
    }


def signed_v22_multitask_loss(
    scores: torch.Tensor,
    gains: Iterable[float],
    supervision_classes: Iterable[str],
    *,
    b_plus: float,
    b_minus: float,
    gamma: float = 1.0,
    beta: float = 0.25,
    neutral_sample_policy: str = "negative",
    lambda_top: float = 0.25,
    gamma_top: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the unchanged v1 base loss plus the state-level top auxiliary."""
    if lambda_top < 0:
        raise ValueError("lambda_top must be non-negative")
    base_loss, base_details = cbwdm_multitask_loss(
        scores,
        list(gains),
        b_plus=b_plus,
        b_minus=b_minus,
        gamma=gamma,
        beta=beta,
        neutral_sample_policy=neutral_sample_policy,
    )
    top_loss, top_details = state_top_ordinal_loss(
        scores,
        list(supervision_classes),
        gamma_top=gamma_top,
    )
    total = base_loss + float(lambda_top) * top_loss
    if not torch.isfinite(total):
        raise FloatingPointError("signed-v2.2 total loss became NaN or Inf")
    return total, {
        "base_loss": base_loss,
        "ce_loss": base_details["ce_loss"],
        "base_rank_loss": base_details["rank_loss"],
        "top_rank_loss": top_loss,
        "base_valid_ranking_group": base_details["valid_ranking_group"],
        "base_skipped_ranking_group": base_details["skipped_ranking_group"],
        "base_num_positive": base_details["num_positive"],
        "base_num_negative": base_details["num_negative"],
        "base_num_neutral": base_details["num_neutral"],
        **top_details,
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
        method=SIGNED_V22_METHOD,
        top_m=top_m,
        min_docs=min_docs,
        score_threshold=score_threshold,
        batch_size=batch_size,
        max_candidates=max_candidates,
    )
    result["selection_metadata"].update(
        {
            "variant": SIGNED_V22_VARIANT,
            "method": SIGNED_V22_METHOD,
            "architecture": SIGNED_V22_ARCHITECTURE,
        }
    )
    return result

"""Deterministic adapted-InfoGain evidence selection protocol.

The filter head probability is ``softmax(filter_logits)[1]``.  It is distinct
from the unconstrained rank-head score: rank scores order candidates, while
filter probabilities are compared with the probability threshold.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


FINAL_PROTOCOL_ID = "adapted_infogain_rank_top4_then_filter02_min2_v1"


def _finite_float(value: Any, *, field: str, index: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}[{index}] must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field}[{index}] must be a finite number")
    return result


def rank_and_select(
    candidates: Sequence[dict[str, Any]],
    rank_scores: Sequence[Any],
    filter_probabilities: Sequence[Any],
    *,
    top_m: int,
    filter_threshold: float | None,
    min_docs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the full ranking trace and selected entries.

    Ranking is performed over the complete pool, with source-list position as
    the stable tie-break.  Filtering and minimum-document fallback are then
    restricted to the fixed top-``top_m`` window; lower-ranked candidates can
    never backfill that window.
    """
    if top_m < 0 or min_docs < 0:
        raise ValueError("top_m and min_docs must be non-negative")
    if min_docs > top_m:
        raise ValueError("min_docs cannot exceed top_m")
    if len(rank_scores) != len(candidates):
        raise ValueError("rank_scores length must match candidates")
    if len(filter_probabilities) != len(candidates):
        raise ValueError("filter_probabilities length must match candidates")
    threshold = None
    if filter_threshold is not None:
        threshold = _finite_float(filter_threshold, field="filter_threshold", index=0)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("filter_threshold must be in [0, 1]")

    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate[{index}] must be an object")
        doc_id_value = candidate.get("doc_id")
        if doc_id_value is None or str(doc_id_value) == "":
            raise ValueError(f"candidate[{index}] lacks doc_id")
        doc_id = str(doc_id_value)
        if doc_id in seen:
            raise ValueError(f"duplicate candidate doc_id: {doc_id}")
        seen.add(doc_id)
        probability = _finite_float(
            filter_probabilities[index], field="filter_probabilities", index=index
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"filter_probabilities[{index}] must be in [0, 1]")
        entries.append(
            {
                "candidate": candidate,
                "doc_id": doc_id,
                "source_position": index,
                "rank_score": _finite_float(
                    rank_scores[index], field="rank_scores", index=index
                ),
                "filter_probability": probability,
            }
        )

    ranked = sorted(entries, key=lambda item: (-item["rank_score"], item["source_position"]))
    for rank_index, item in enumerate(ranked):
        item["rerank_position"] = rank_index
        item["within_top_m"] = rank_index < top_m
        item["passes_filter"] = threshold is None or item["filter_probability"] >= threshold
        item["min_docs_fallback"] = False

    window = ranked[:top_m]
    selected = [item for item in window if item["passes_filter"]]
    target = min(min_docs, len(window))
    if len(selected) < target:
        selected_ids = {item["doc_id"] for item in selected}
        for item in window:
            if len(selected) >= target:
                break
            if item["doc_id"] not in selected_ids:
                item["min_docs_fallback"] = True
                selected.append(item)
                selected_ids.add(item["doc_id"])
    selected.sort(key=lambda item: item["rerank_position"])
    return ranked, selected


def stop_reason_for_selection(
    *,
    candidate_count: int,
    selected_count: int,
    top_m: int,
    min_docs: int,
    filter_threshold: float | None,
    used_fallback: bool,
) -> str:
    window_size = min(candidate_count, top_m)
    if selected_count == top_m:
        return "top_m_reached"
    if filter_threshold is None:
        return "threshold_or_candidates_exhausted"
    if window_size < min_docs or (
        candidate_count < top_m and selected_count == candidate_count
    ):
        return "candidates_exhausted"
    if used_fallback:
        return "min_docs_fallback_completed"
    return "filter_threshold_applied"

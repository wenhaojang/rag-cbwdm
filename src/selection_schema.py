"""Unified selection-output schema helpers."""

from __future__ import annotations

from typing import Any

REQUIRED_SELECTION_KEYS = {
    "schema_version",
    "id",
    "query",
    "label",
    "split",
    "method",
    "selected_doc_ids",
    "selected_docs",
    "num_docs",
    "max_docs",
    "selection_metadata",
    "selection_steps",
    "stop_reason",
}


def normalize_selected_doc(
    candidate: dict[str, Any],
    *,
    selector_score: float | None,
    selection_step: int,
) -> dict[str, Any]:
    result = dict(candidate)
    result.update(
        {
            "doc_id": candidate.get("doc_id"),
            "title": candidate.get("title"),
            "text": candidate.get("text"),
            "rank": candidate.get("rank"),
            "retrieval_score": candidate.get("retrieval_score", candidate.get("score")),
            "source_rank": candidate.get("source_rank", candidate.get("rank")),
            "source_score": candidate.get(
                "source_score", candidate.get("retrieval_score", candidate.get("score"))
            ),
            "selector_score": selector_score,
            "selection_step": selection_step,
        }
    )
    return result


def make_selection_row(
    source: dict[str, Any],
    *,
    method: str,
    selected_docs: list[dict[str, Any]],
    selection_steps: list[dict[str, Any]],
    stop_reason: str,
    diagnostic_only: bool = False,
    max_docs: int | None = None,
    selection_metadata: dict[str, Any] | None = None,
    uses_gold_at_test: bool = False,
) -> dict[str, Any]:
    resolved_max_docs = len(selected_docs) if max_docs is None else int(max_docs)
    return {
        "schema_version": "rag_cbwdm_selection.v2",
        "id": source.get("id"),
        "query": source.get("query"),
        "label": source.get("label", source.get("gold")),
        "gold": source.get("label", source.get("gold")),
        "split": source.get("split"),
        "method": method,
        "selected_doc_ids": [str(doc.get("doc_id")) for doc in selected_docs],
        "selected_docs": selected_docs,
        "num_docs": len(selected_docs),
        "max_docs": resolved_max_docs,
        "selection_metadata": dict(selection_metadata or {}),
        "selection_steps": selection_steps,
        "stop_reason": stop_reason,
        "diagnostic_only": bool(diagnostic_only),
        "deployable": not diagnostic_only,
        "uses_gold_at_test": bool(uses_gold_at_test),
    }


def validate_selection_row(row: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_SELECTION_KEYS - row.keys())
    if missing:
        raise ValueError(f"Selection row is missing keys: {', '.join(missing)}")
    if row["num_docs"] != len(row["selected_docs"]):
        raise ValueError("num_docs does not match selected_docs length")
    if row["selected_doc_ids"] != [str(doc.get("doc_id")) for doc in row["selected_docs"]]:
        raise ValueError("selected_doc_ids does not match selected_docs order")
    if int(row["max_docs"]) < int(row["num_docs"]):
        raise ValueError("max_docs cannot be smaller than num_docs")
    for doc in row["selected_docs"]:
        if "source_rank" not in doc or "source_score" not in doc:
            raise ValueError("selected_docs must preserve source_rank and source_score")

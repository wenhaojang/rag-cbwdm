"""BGE-compatible pointwise reranking with delayed optional model imports."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

from src.selection_schema import make_selection_row, normalize_selected_doc

INPUT_TEMPLATE_VERSION = "fever_claim_title_text.v1"


def format_pair(query: str, candidate: dict[str, Any]) -> tuple[str, str]:
    title = str(candidate.get("title") or "").strip()
    text = str(candidate.get("text") or "").strip()
    document = f"{title}: {text}" if title and text else title or text
    return str(query), document


def _model_imports() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "BGE reranking requires torch and transformers. Install the baseline "
            "environment with: pip install -r requirements-baselines.txt"
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer


class HuggingFaceReranker:
    """Sequence-classification reranker; larger scalar means more relevant."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        max_length: int = 512,
        normalize_score: bool = False,
        local_files_only: bool = False,
    ) -> None:
        torch, AutoModelForSequenceClassification, AutoTokenizer = _model_imports()
        self.torch = torch
        self.max_length = int(max_length)
        self.normalize_score = bool(normalize_score)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        tokenizer_kwargs: dict[str, Any] = {
            "revision": revision,
            "local_files_only": local_files_only,
        }
        model_kwargs = dict(tokenizer_kwargs)
        dtype_value = getattr(torch, dtype, None) if dtype != "auto" else None
        if dtype_value is not None:
            model_kwargs["torch_dtype"] = dtype_value
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, **tokenizer_kwargs
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, **model_kwargs
        ).to(self.device)
        self.model.eval()

    def score_pairs(
        self, pairs: list[tuple[str, str]], *, batch_size: int
    ) -> list[float]:
        scores: list[float] = []
        try:
            with self.torch.inference_mode():
                for start in range(0, len(pairs), batch_size):
                    batch = pairs[start : start + batch_size]
                    encoded = self.tokenizer(
                        [item[0] for item in batch],
                        [item[1] for item in batch],
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(self.device) for key, value in encoded.items()}
                    logits = self.model(**encoded).logits.float()
                    values = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
                    if self.normalize_score:
                        values = values.sigmoid()
                    scores.extend(float(value) for value in values.cpu().tolist())
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "BGE reranker ran out of memory. Lower --batch-size or "
                    "--max-length; the runner does not silently change experiment settings."
                ) from exc
            raise
        if len(scores) != len(pairs) or not all(math.isfinite(value) for value in scores):
            raise ValueError("BGE scorer returned wrong-length or non-finite scores")
        return scores


def score_rows(
    rows: Iterable[dict[str, Any]],
    score_pairs: Callable[[list[tuple[str, str]]], list[float]],
) -> list[dict[str, Any]]:
    """Score all query-document pairs while preserving exact input correspondence."""
    result: list[dict[str, Any]] = []
    for row in rows:
        candidates = list(row.get("candidates", []))
        pairs = [format_pair(str(row.get("query") or ""), candidate) for candidate in candidates]
        scores = score_pairs(pairs)
        if len(scores) != len(candidates):
            raise ValueError("Reranker score count does not match candidate count")
        result.append(
            {
                "id": row.get("id"),
                "scores": [
                    {"doc_id": str(candidate.get("doc_id")), "score": float(score)}
                    for candidate, score in zip(candidates, scores)
                ],
            }
        )
    return result


def choose_scored_candidates(
    candidates: list[dict[str, Any]],
    scores: list[float],
    *,
    top_m: int,
    score_threshold: float | None,
    min_docs: int,
) -> list[tuple[dict[str, Any], float, bool]]:
    if len(candidates) != len(scores):
        raise ValueError("candidates/scores length mismatch")
    if top_m < 0 or min_docs < 0 or min_docs > top_m:
        raise ValueError("Require 0 <= min_docs <= top_m")
    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: (-float(item[1]), int(item[0].get("rank") or 10**9)),
    )
    selected: list[tuple[dict[str, Any], float, bool]] = []
    for candidate, score in ranked[:top_m]:
        passes = score_threshold is None or float(score) >= score_threshold
        fallback = not passes and len(selected) < min_docs
        if passes or fallback:
            selected.append((candidate, float(score), fallback))
    return selected


def make_bge_selection(
    row: dict[str, Any],
    scores_by_doc_id: dict[str, float],
    *,
    method: str,
    top_m: int,
    score_threshold: float | None,
    min_docs: int,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    candidates = list(row.get("candidates", []))
    try:
        scores = [float(scores_by_doc_id[str(item.get("doc_id"))]) for item in candidates]
    except KeyError as exc:
        raise KeyError(f"Missing cached BGE score for doc_id={exc.args[0]!r}") from exc
    chosen = choose_scored_candidates(
        candidates,
        scores,
        top_m=top_m,
        score_threshold=score_threshold,
        min_docs=min_docs,
    )
    docs = []
    steps = []
    for step, (candidate, score, fallback) in enumerate(chosen):
        doc = normalize_selected_doc(
            candidate, selector_score=score, selection_step=step
        )
        doc["min_docs_fallback"] = fallback
        docs.append(doc)
        steps.append(
            {
                "step": step,
                "selected_doc_id": str(candidate.get("doc_id")),
                "predicted_score": score,
                "min_docs_fallback": fallback,
                "stop": False,
            }
        )
    return make_selection_row(
        row,
        method=method,
        selected_docs=docs,
        selection_steps=steps,
        stop_reason="top_m_reached" if len(docs) == top_m else "threshold_or_candidates_exhausted",
        max_docs=top_m,
        selection_metadata={
            **model_metadata,
            "state_aware": False,
            "uses_gold_at_test": False,
            "score_direction": "higher_is_more_relevant",
            "score_threshold": score_threshold,
            "min_docs": min_docs,
        },
    )

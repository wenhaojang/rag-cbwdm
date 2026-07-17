"""State-aware cross-encoder selector utilities."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from src.io_utils import ensure_dir, read_jsonl, require_keys


def _import_transformers() -> tuple[Any, Any]:
    """Import Transformers classes with a clear installation hint."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Cross-encoder selector requires transformers and torch. "
            "Install them with: pip install -r requirements.txt"
        ) from exc
    return AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class CrossEncoderGroup:
    """One listwise training group for a query and selected-state pair."""

    example_id: str
    step: int
    query: str
    current_doc_ids: list[str]
    candidate_doc_ids: list[str]
    candidate_texts: list[str]
    gains: list[float]
    selected_docs: list[dict[str, Any]]
    candidate_docs: list[dict[str, Any]]


def _format_doc(doc: dict[str, Any]) -> str:
    title = str(doc.get("title") or "").strip()
    text = str(doc.get("text") or "").strip()
    if title and text:
        return f"{title}: {text}"
    return title or text


def build_selector_input(
    query: str,
    selected_docs: list[dict[str, Any]],
    candidate_doc: dict[str, Any],
    max_selected_docs: int = 4,
    max_query_chars: int = 2000,
    max_candidate_chars: int = 5000,
    max_selected_chars: int = 5000,
) -> str:
    """Build state-aware input without gold labels/gains.

    Claim and candidate appear before the bounded selected-state block so ordinary
    tokenizer right truncation cannot discard them in favor of long history.
    """
    selected_docs = list(selected_docs)[-max_selected_docs:]
    selected_lines: list[str] = []
    for idx, doc in enumerate(selected_docs, start=1):
        selected_lines.append(f"[{idx}] {_format_doc(doc)}")
    selected_block = ("\n".join(selected_lines) if selected_lines else "None")[
        :max_selected_chars
    ]
    query = str(query)[:max_query_chars]
    candidate_text = _format_doc(candidate_doc)[:max_candidate_chars]

    return (
        "Claim:\n"
        f"{query}\n\n"
        "Candidate evidence:\n"
        f"{candidate_text}\n\n"
        "Already selected evidence:\n"
        f"{selected_block}\n\n"
        "Task:\n"
        "Score how useful the candidate evidence is as an additional evidence item "
        "for verifying the claim, considering the already selected evidence."
    )


def load_posterior_map(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load posterior JSONL rows keyed by sample id."""
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        require_keys(row, ["id", "query", "candidates"], "posterior row")
        rows[str(row["id"])] = row
    return rows


def load_retrieval_doc_map(path: str | Path | None) -> dict[str, dict[str, dict[str, Any]]]:
    """Load retrieval candidates keyed by sample id and doc id."""
    if path is None:
        return {}
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("id"))
        rows[sample_id] = {
            str(candidate.get("doc_id")): candidate
            for candidate in row.get("candidates", [])
            if candidate.get("doc_id") is not None
        }
    return rows


def _posterior_doc_map(
    posterior_row: dict[str, Any],
    retrieval_docs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    docs = {
        str(candidate.get("doc_id")): dict(candidate)
        for candidate in posterior_row.get("candidates", [])
        if candidate.get("doc_id") is not None
    }
    if retrieval_docs:
        for doc_id, retrieval_doc in retrieval_docs.items():
            current = docs.get(doc_id)
            if current is None:
                docs[doc_id] = dict(retrieval_doc)
                continue
            for key in ("title", "text", "rank", "score", "retrieval_score"):
                if current.get(key) in (None, "") and retrieval_doc.get(key) not in (None, ""):
                    current[key] = retrieval_doc[key]
    return docs


def _candidate_gain_items(step: dict[str, Any], max_candidates: int | None) -> list[dict[str, Any]]:
    items = list(step.get("candidate_gains", []))
    if max_candidates is not None:
        if max_candidates < 1:
            raise ValueError(f"max_candidates_per_group must be positive, got {max_candidates}")
        items = items[:max_candidates]
    return items


def build_cross_encoder_groups(
    posterior_path: str | Path,
    teacher_path: str | Path,
    retrieval_path: str | Path | None = None,
    max_train_groups: int | None = None,
    max_candidates_per_group: int | None = None,
) -> list[CrossEncoderGroup]:
    """Build listwise groups from aligned posterior and CBWDM teacher JSONL files."""
    posterior_map = load_posterior_map(posterior_path)
    retrieval_map = load_retrieval_doc_map(retrieval_path)
    groups: list[CrossEncoderGroup] = []
    skipped_missing = 0
    skipped_small = 0

    for teacher_row in read_jsonl(teacher_path):
        require_keys(teacher_row, ["id", "steps"], "teacher row")
        sample_id = str(teacher_row["id"])
        if sample_id not in posterior_map:
            raise KeyError(f"Teacher row id '{sample_id}' not found in posterior file")

        posterior_row = posterior_map[sample_id]
        doc_map = _posterior_doc_map(posterior_row, retrieval_map.get(sample_id))
        query = str(posterior_row.get("query") or teacher_row.get("query") or "")

        for step in teacher_row.get("steps", []):
            if max_train_groups is not None and len(groups) >= max_train_groups:
                break
            current_doc_ids = [str(doc_id) for doc_id in step.get("current_doc_ids", [])]
            selected_docs = [doc_map[doc_id] for doc_id in current_doc_ids if doc_id in doc_map]
            candidate_docs: list[dict[str, Any]] = []
            candidate_doc_ids: list[str] = []
            candidate_texts: list[str] = []
            gains: list[float] = []

            for item in _candidate_gain_items(step, max_candidates_per_group):
                doc_id = str(item.get("doc_id"))
                doc = doc_map.get(doc_id)
                if doc is None or not str(doc.get("text") or doc.get("title") or "").strip():
                    skipped_missing += 1
                    print(
                        f"[cross_encoder_groups][warning] missing text for id={sample_id} doc_id={doc_id}",
                        file=sys.stderr,
                    )
                    continue
                candidate_doc_ids.append(doc_id)
                candidate_docs.append(doc)
                candidate_texts.append(_format_doc(doc))
                gains.append(float(item.get("gain", 0.0)))

            if len(candidate_docs) < 2:
                skipped_small += 1
                continue
            groups.append(
                CrossEncoderGroup(
                    example_id=sample_id,
                    step=int(step.get("step", len(groups))),
                    query=query,
                    current_doc_ids=current_doc_ids,
                    candidate_doc_ids=candidate_doc_ids,
                    candidate_texts=candidate_texts,
                    gains=gains,
                    selected_docs=selected_docs,
                    candidate_docs=candidate_docs,
                )
            )
        if max_train_groups is not None and len(groups) >= max_train_groups:
            break

    if not groups:
        raise ValueError(
            "No cross-encoder groups were built. Check posterior/teacher alignment "
            "and candidate text availability."
        )
    print(
        f"[cross_encoder_groups] groups={len(groups)} skipped_missing={skipped_missing} "
        f"skipped_small={skipped_small}"
    )
    return groups


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve device string, using CUDA automatically when available."""
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrossEncoderSelector(torch.nn.Module):
    """HuggingFace sequence-classification cross-encoder with scalar scores."""

    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        device: str | None = None,
        load_from_checkpoint: bool = False,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
    ) -> None:
        super().__init__()
        if not model_name:
            raise ValueError("model_name must be non-empty")
        AutoModelForSequenceClassification, AutoTokenizer = _import_transformers()
        self.model_name = model_name
        self.max_length = int(max_length)
        self.device = resolve_device(device)
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision or revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=self.tokenizer_revision
        )
        kwargs: dict[str, Any] = {}
        if revision:
            kwargs["revision"] = revision
        if not load_from_checkpoint:
            kwargs.update(
                {
                    "num_labels": 1,
                    "problem_type": "regression",
                    "ignore_mismatched_sizes": True,
                }
            )
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        self.model.to(self.device)

    def _encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def score_texts(
        self,
        texts: list[str],
        batch_size: int = 8,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Return scalar scores with shape [n] for a list of cross-encoder inputs."""
        if not texts:
            return torch.empty(0, device=self.device)
        scores: list[torch.Tensor] = []
        context = torch.enable_grad() if requires_grad else torch.inference_mode()
        with context:
            for start in range(0, len(texts), max(int(batch_size), 1)):
                batch_texts = texts[start : start + max(int(batch_size), 1)]
                encoded = self._encode(batch_texts)
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits.squeeze(-1)
                scores.append(logits)
        return torch.cat(scores, dim=0)

    def save_checkpoint(self, checkpoint_dir: str | Path, extra_config: dict[str, Any] | None = None) -> None:
        """Save model, tokenizer, and selector metadata to a checkpoint directory."""
        checkpoint_path = ensure_dir(checkpoint_dir)
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
        }
        if extra_config:
            config.update(extra_config)
        with (checkpoint_path / "selector_config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        max_length: int | None = None,
        device: str | None = None,
    ) -> "CrossEncoderSelector":
        """Load a selector checkpoint saved by save_checkpoint."""
        checkpoint_path = Path(checkpoint_dir)
        config_path = checkpoint_path / "selector_config.json"
        config: dict[str, Any] = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
        selector = cls(
            model_name=str(checkpoint_path),
            max_length=max_length if max_length is not None else int(config.get("max_length", 512)),
            device=device,
            load_from_checkpoint=True,
        )
        selector.model_name = str(config.get("model_name", checkpoint_path))
        selector.revision = config.get("revision")
        selector.tokenizer_revision = config.get("tokenizer_revision")
        return selector


def listwise_distillation_loss(
    scores: torch.Tensor,
    gains: Iterable[float],
    teacher_temperature: float = 0.1,
) -> torch.Tensor:
    """Compute listwise softmax distillation loss for one group."""
    temperature = float(teacher_temperature)
    if temperature <= 0:
        raise ValueError(f"teacher_temperature must be positive, got {teacher_temperature}")
    gain_tensor = torch.tensor(list(gains), dtype=scores.dtype, device=scores.device)
    if gain_tensor.numel() != scores.numel():
        raise ValueError(f"gains length {gain_tensor.numel()} does not match scores length {scores.numel()}")
    shifted = (gain_tensor - torch.max(gain_tensor)) / temperature
    target_probs = torch.softmax(shifted, dim=0)
    log_probs = F.log_softmax(scores, dim=0)
    return -(target_probs * log_probs).sum()


def cbwdm_multitask_loss(
    scores: torch.Tensor,
    gains: Iterable[float],
    *,
    b_plus: float,
    b_minus: float,
    gamma: float = 1.0,
    beta: float = 0.5,
    neutral_sample_policy: str = "negative",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Manuscript state-aware CE + positive/negative LogSumExp ranking loss.

    Positive candidates satisfy ``gain > b_plus`` and negatives satisfy
    ``gain < b_minus``. Values in between are neutral. ``ignore`` removes
    neutrals from CE; ``negative`` includes them as CE-negative examples.
    """
    if not b_plus > b_minus > 0:
        raise ValueError("Expected b_plus > b_minus > 0")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if neutral_sample_policy not in {"ignore", "negative"}:
        raise ValueError("neutral_sample_policy must be 'ignore' or 'negative'")
    gain_tensor = torch.as_tensor(list(gains), dtype=scores.dtype, device=scores.device)
    if gain_tensor.numel() != scores.numel():
        raise ValueError("gains length does not match scores length")
    if not torch.isfinite(gain_tensor).all() or not torch.isfinite(scores).all():
        raise ValueError("scores and gains must be finite")

    positive = gain_tensor > b_plus
    negative = gain_tensor < b_minus
    neutral = ~(positive | negative)
    ce_mask = ~neutral if neutral_sample_policy == "ignore" else torch.ones_like(positive)
    if bool(ce_mask.any()):
        targets = positive[ce_mask].to(scores.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(scores[ce_mask], targets)
    else:
        ce_loss = scores.sum() * 0.0

    positive_scores = scores[positive]
    negative_scores = scores[negative]
    valid_rank = bool(positive_scores.numel() and negative_scores.numel())
    if valid_rank:
        pairwise = gamma * (
            negative_scores[:, None] - positive_scores[None, :]
        ).reshape(-1)
        # log(1 + sum exp(pairwise)), computed without overflow.
        rank_loss = torch.logaddexp(
            torch.zeros((), dtype=scores.dtype, device=scores.device),
            torch.logsumexp(pairwise, dim=0),
        )
    else:
        rank_loss = scores.sum() * 0.0
    total = beta * ce_loss + (1.0 - beta) * rank_loss
    if not torch.isfinite(total):
        raise FloatingPointError("CBWDM selector loss became NaN or Inf")
    return total, {
        "ce_loss": ce_loss,
        "rank_loss": rank_loss,
        "num_positive": int(positive.sum().item()),
        "num_negative": int(negative.sum().item()),
        "num_neutral": int(neutral.sum().item()),
        "valid_ranking_group": valid_rank,
        "skipped_ranking_group": not valid_rank,
    }


def shuffle_groups(groups: list[CrossEncoderGroup], seed: int) -> list[CrossEncoderGroup]:
    """Return a shuffled copy of training groups."""
    copied = list(groups)
    random.Random(seed).shuffle(copied)
    return copied

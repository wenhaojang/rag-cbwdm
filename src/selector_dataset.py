"""Feature construction utilities for training feature MLP selectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.io_utils import read_jsonl, require_keys


BASE_SCALAR_FEATURES = [
    "entropy_eta0",
    "entropy_eta_j",
    "entropy_drop",
    "l2_shift_norm",
    "retrieval_rank_inv",
    "retrieval_score",
    "step_index",
    "num_selected",
    "max_sim_to_selected",
    "sim_to_selected_mean",
]


@dataclass
class SelectorTrainingExample:
    """One candidate under a teacher state."""

    features: np.ndarray
    gain: float
    is_best: int
    group_id: str
    doc_id: str


def feature_names(num_labels: int) -> list[str]:
    """Return feature names for a label space with num_labels labels."""
    names: list[str] = []
    names.extend(f"eta0_{idx}" for idx in range(num_labels))
    names.extend(f"eta_j_{idx}" for idx in range(num_labels))
    names.extend(f"eta_shift_{idx}" for idx in range(num_labels))
    names.extend(f"abs_eta_shift_{idx}" for idx in range(num_labels))
    names.extend(BASE_SCALAR_FEATURES[:8])
    names.extend(f"selected_shift_mean_{idx}" for idx in range(num_labels))
    names.extend(BASE_SCALAR_FEATURES[8:])
    return names


FEATURE_NAMES: list[str] = []


def load_posterior_map(path: str | Path) -> dict[str, dict]:
    """Load posterior JSONL rows keyed by sample id."""
    posterior_map: dict[str, dict] = {}
    for row in read_jsonl(path):
        require_keys(row, ["id", "eta0", "candidates"], "posterior row")
        posterior_map[row["id"]] = row
    return posterior_map


def load_teacher_rows(path: str | Path) -> list[dict]:
    """Load teacher JSONL rows as a list."""
    rows: list[dict] = []
    for row in read_jsonl(path):
        require_keys(row, ["id", "steps"], "teacher row")
        rows.append(row)
    return rows


def _candidate_score(candidate: dict) -> float:
    score = candidate.get("retrieval_score", candidate.get("score", 0.0))
    return 0.0 if score is None else float(score)


def _candidate_map(posterior_row: dict) -> dict[str, dict]:
    return {candidate["doc_id"]: candidate for candidate in posterior_row.get("candidates", [])}


def _entropy(p: np.ndarray) -> float:
    clipped = np.clip(p.astype(float), 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_candidate_feature(
    posterior_row: dict,
    current_doc_ids: list[str],
    candidate_doc_id: str,
    step_index: int,
) -> np.ndarray:
    """Build fixed-order test-time features for one candidate under a selected state."""
    require_keys(posterior_row, ["eta0", "candidates"], "posterior row")
    candidates = _candidate_map(posterior_row)
    if candidate_doc_id not in candidates:
        raise KeyError(f"Candidate doc_id '{candidate_doc_id}' not found in posterior row {posterior_row.get('id')}")

    candidate = candidates[candidate_doc_id]
    eta0 = np.asarray(posterior_row["eta0"], dtype=float)
    eta_j = np.asarray(candidate["eta"], dtype=float)
    if eta0.shape != eta_j.shape:
        raise ValueError(f"eta0 shape {eta0.shape} does not match eta shape {eta_j.shape}")

    shift = eta_j - eta0
    selected_shifts = []
    for doc_id in current_doc_ids:
        selected = candidates.get(doc_id)
        if selected is None:
            continue
        selected_shifts.append(np.asarray(selected["eta"], dtype=float) - eta0)

    if selected_shifts:
        selected_matrix = np.vstack(selected_shifts)
        selected_shift_mean = np.mean(selected_matrix, axis=0)
        max_sim = max(_cosine(shift, selected_shift) for selected_shift in selected_shifts)
        mean_sim = _cosine(shift, selected_shift_mean)
    else:
        selected_shift_mean = np.zeros_like(eta0)
        max_sim = 0.0
        mean_sim = 0.0

    rank = float(candidate.get("rank", 0) or 0)
    rank_inv = 1.0 / rank if rank > 0 else 0.0
    scalars = np.asarray(
        [
            _entropy(eta0),
            _entropy(eta_j),
            _entropy(eta0) - _entropy(eta_j),
            float(np.linalg.norm(shift)),
            rank_inv,
            _candidate_score(candidate),
            float(step_index),
            float(len(current_doc_ids)),
        ],
        dtype=float,
    )
    tail_scalars = np.asarray([max_sim, mean_sim], dtype=float)
    return np.concatenate([eta0, eta_j, shift, np.abs(shift), scalars, selected_shift_mean, tail_scalars])


def _gain_map(step: dict) -> dict[str, float]:
    gains: dict[str, float] = {}
    for item in step.get("candidate_gains", []):
        if "doc_id" in item:
            gains[item["doc_id"]] = float(item.get("gain", 0.0))
    return gains


def build_training_examples(
    posterior_path: str | Path,
    teacher_path: str | Path,
    max_candidates_per_state: int | None = None,
    limit_states: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build pointwise training examples from aligned posterior and teacher JSONL files."""
    posterior_map = load_posterior_map(posterior_path)
    teacher_rows = load_teacher_rows(teacher_path)
    features: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict] = []
    states_seen = 0

    for teacher_row in teacher_rows:
        sample_id = teacher_row["id"]
        if sample_id not in posterior_map:
            raise KeyError(f"Teacher row id '{sample_id}' not found in posterior file")
        posterior_row = posterior_map[sample_id]
        all_doc_ids = [candidate["doc_id"] for candidate in posterior_row.get("candidates", [])]

        for step in teacher_row.get("steps", []):
            if limit_states is not None and states_seen >= limit_states:
                break
            current_doc_ids = list(step.get("current_doc_ids", []))
            used = set(current_doc_ids)
            candidate_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in used]
            if max_candidates_per_state is not None:
                candidate_doc_ids = candidate_doc_ids[:max_candidates_per_state]
            gains = _gain_map(step)
            best_doc_id = step.get("best_doc_id")
            group_id = f"{sample_id}::step_{step.get('step', states_seen)}"

            for doc_id in candidate_doc_ids:
                x = build_candidate_feature(posterior_row, current_doc_ids, doc_id, int(step.get("step", 0)))
                is_best = int(doc_id == best_doc_id)
                gain = float(gains.get(doc_id, 0.0))
                features.append(x)
                targets.append(float(is_best))
                metadata.append(
                    {
                        "id": sample_id,
                        "group_id": group_id,
                        "doc_id": doc_id,
                        "gain": gain,
                        "is_best": is_best,
                        "step": int(step.get("step", 0)),
                    }
                )
            states_seen += 1
        if limit_states is not None and states_seen >= limit_states:
            break

    if not features:
        raise ValueError("No selector training examples were built.")
    X = np.vstack(features).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32)
    return X, y, metadata

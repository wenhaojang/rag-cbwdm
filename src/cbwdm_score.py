"""Numerical utilities for the greedy RAG-CBWDM teacher.

The implementation stores candidate effects as rows ``[J, K]``.  This is the
transpose of the manuscript's ``X_S=[x_j:j in S]`` convention ``[K, |S|]``;
the resulting ``c``, ``G`` and ``Theta`` are algebraically identical.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

EUCLIDEAN_L_TYPE = "euclidean_posterior_shift"
TEACHER_SCHEMA_VERSION = "rag_cbwdm_teacher.v2"


def canonical_l_type(l_type: str) -> str:
    """Normalize the implemented Euclidean mode and reject unimplemented BW modes."""
    normalized = str(l_type).strip().lower()
    if normalized in {"identity", "euclidean", EUCLIDEAN_L_TYPE}:
        return EUCLIDEAN_L_TYPE
    if normalized in {"bw", "bw_local_hessian", "categorical_bw"}:
        raise NotImplementedError(
            "The manuscript defines H_i abstractly as a local BW Hessian but does not "
            "provide an unambiguous categorical closed form. Only "
            f"'{EUCLIDEAN_L_TYPE}' is implemented; no silent BW fallback is allowed."
        )
    raise ValueError(
        f"Unsupported L_type={l_type!r}. Use '{EUCLIDEAN_L_TYPE}' "
        "(or legacy alias 'identity')."
    )


def validate_probability_vector(p: np.ndarray, name: str, atol: float = 1e-5) -> None:
    """Validate that ``p`` is a finite, normalized one-dimensional probability vector."""
    p = np.asarray(p)
    if p.ndim != 1:
        raise ValueError(f"{name} must be a 1D probability vector, got shape {p.shape}")
    if p.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(p < -atol):
        raise ValueError(f"{name} contains negative probabilities: {p.tolist()}")
    total = float(np.sum(p))
    if abs(total - 1.0) > atol:
        raise ValueError(f"{name} must sum to 1 within {atol}, got {total}")


def smooth_probability_vector(p: np.ndarray, eps_smooth: float) -> np.ndarray:
    """Apply manuscript mixture smoothing: ``(1-eps)p + eps/K``."""
    p = np.asarray(p, dtype=float)
    validate_probability_vector(p, "probability vector")
    if not 0.0 <= eps_smooth < 1.0:
        raise ValueError(f"eps_smooth must be in [0, 1), got {eps_smooth}")
    if eps_smooth == 0:
        return p.copy()
    return (1.0 - eps_smooth) * p + eps_smooth / p.size


def _legacy_clip_probability_vector(p: np.ndarray, eps_smooth: float) -> np.ndarray:
    """Reproduce the pre-v2 clip-and-renormalize behavior for old experiments."""
    p = np.asarray(p, dtype=float)
    validate_probability_vector(p, "probability vector")
    if eps_smooth < 0:
        raise ValueError(f"eps_smooth must be non-negative, got {eps_smooth}")
    if eps_smooth == 0:
        return p.copy()
    result = np.clip(p, eps_smooth, None)
    return result / result.sum()


def one_hot(label: str, labels: list[str]) -> np.ndarray:
    """Return a one-hot vector using the supplied label order."""
    if label not in labels:
        raise ValueError(f"Label {label!r} is not in labels: {labels}")
    vec = np.zeros(len(labels), dtype=float)
    vec[labels.index(label)] = 1.0
    return vec


def build_local_effects(
    eta0: np.ndarray,
    candidate_etas: np.ndarray,
    label: str,
    labels: list[str],
    l_type: str = EUCLIDEAN_L_TYPE,
    eps_smooth: float = 0.0,
    target_smoothing: str = "paper_mixture",
) -> tuple[np.ndarray, np.ndarray]:
    """Return row-oriented candidate effects ``X_all[J,K]`` and gold direction ``d[K]``.

    ``paper_mixture`` smooths the query-only posterior and the one-hot target,
    exactly as stated in the manuscript. Candidate posteriors remain unchanged.
    ``legacy_clip_all`` preserves the old implementation for compatibility.
    """
    canonical_l_type(l_type)
    if not labels:
        raise ValueError("labels must not be empty")
    eta0 = np.asarray(eta0, dtype=float)
    candidate_etas = np.asarray(candidate_etas, dtype=float)
    if candidate_etas.ndim != 2:
        raise ValueError(
            "candidate_etas must have shape (num_candidates, num_labels), "
            f"got {candidate_etas.shape}"
        )
    if eta0.shape != (len(labels),):
        raise ValueError(f"eta0 shape {eta0.shape} does not match K={len(labels)}")
    if candidate_etas.shape[1] != len(labels):
        raise ValueError(
            f"candidate eta width {candidate_etas.shape[1]} does not match K={len(labels)}"
        )
    validate_probability_vector(eta0, "eta0")
    for idx, eta in enumerate(candidate_etas):
        validate_probability_vector(eta, f"candidate_etas[{idx}]")

    if target_smoothing == "paper_mixture":
        eta0_used = smooth_probability_vector(eta0, eps_smooth)
        target = smooth_probability_vector(one_hot(label, labels), eps_smooth)
        eta_matrix = candidate_etas.copy()
    elif target_smoothing == "legacy_clip_all":
        eta0_used = _legacy_clip_probability_vector(eta0, eps_smooth)
        target = one_hot(label, labels)
        eta_matrix = np.vstack(
            [_legacy_clip_probability_vector(eta, eps_smooth) for eta in candidate_etas]
        ) if len(candidate_etas) else np.empty((0, len(labels)), dtype=float)
    else:
        raise ValueError(
            "target_smoothing must be 'paper_mixture' or 'legacy_clip_all', "
            f"got {target_smoothing!r}"
        )
    return eta_matrix - eta0_used, target - eta0_used


def theta_for_indices(
    X_all: np.ndarray,
    d: np.ndarray,
    indices: list[int],
    ridge_lambda: float,
) -> float:
    """Compute ``Theta(S)=c^T(X X^T + lambda I)^{-1}c`` without explicit inversion."""
    if ridge_lambda <= 0:
        raise ValueError(f"ridge_lambda must be positive, got {ridge_lambda}")
    if not indices:
        return 0.0
    X_all = np.asarray(X_all, dtype=float)
    d = np.asarray(d, dtype=float)
    X = X_all[indices, :]
    c = X @ d
    G = X @ X.T + ridge_lambda * np.eye(len(indices), dtype=float)
    try:
        solution = np.linalg.solve(G, c)
    except np.linalg.LinAlgError:
        warnings.warn("Linear solve failed; using pseudo-inverse.", RuntimeWarning, stacklevel=2)
        solution = np.linalg.pinv(G) @ c
    value = float(c @ solution)
    if not np.isfinite(value):
        raise FloatingPointError("Theta is NaN or Inf")
    return value


def marginal_gain(
    X_all: np.ndarray,
    d: np.ndarray,
    current_indices: list[int],
    candidate_index: int,
    ridge_lambda: float,
) -> tuple[float, float]:
    """Return raw ``Delta(j|S)`` and ``Theta(S union {j})``."""
    if candidate_index in current_indices:
        raise ValueError(f"candidate_index {candidate_index} is already selected")
    before = theta_for_indices(X_all, d, current_indices, ridge_lambda)
    after = theta_for_indices(X_all, d, current_indices + [candidate_index], ridge_lambda)
    return float(after - before), float(after)


def greedy_teacher(
    X_all: np.ndarray,
    d: np.ndarray,
    top_m: int,
    ridge_lambda: float,
    stop_threshold: float = 0.0,
    store_all_gains: bool = True,
    gain_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Build a monotonic greedy trajectory with explicit numerical diagnostics."""
    if top_m < 0:
        raise ValueError(f"top_m must be non-negative, got {top_m}")
    if ridge_lambda <= 0:
        raise ValueError(f"ridge_lambda must be positive, got {ridge_lambda}")
    if gain_tolerance < 0:
        raise ValueError("gain_tolerance must be non-negative")
    X_all = np.asarray(X_all, dtype=float)
    d = np.asarray(d, dtype=float)
    if X_all.ndim != 2 or d.ndim != 1 or X_all.shape[1] != d.size:
        raise ValueError(f"Incompatible X_all {X_all.shape} and d {d.shape}")
    if not np.all(np.isfinite(X_all)) or not np.all(np.isfinite(d)):
        raise ValueError("X_all and d must be finite")

    selected: list[int] = []
    steps: list[dict[str, Any]] = []
    theta_current = 0.0
    stop_reason = "top_m_reached" if top_m == 0 else "no_candidates"
    terminal_decision: dict[str, Any] | None = None

    for step_idx in range(min(top_m, X_all.shape[0])):
        remaining = [idx for idx in range(X_all.shape[0]) if idx not in selected]
        if not remaining:
            stop_reason = "no_candidates"
            break
        before = theta_for_indices(X_all, d, selected, ridge_lambda)
        gains: list[dict[str, Any]] = []
        for idx in remaining:
            raw_gain, after = marginal_gain(X_all, d, selected, idx, ridge_lambda)
            if raw_gain < -gain_tolerance:
                raise FloatingPointError(
                    f"Non-monotonic gain {raw_gain} below tolerance {-gain_tolerance}; "
                    "inspect matrix orientation, smoothing, or numerical conditioning."
                )
            gain = 0.0 if abs(raw_gain) <= gain_tolerance else raw_gain
            gains.append(
                {
                    "index": idx,
                    "gain": float(gain),
                    "raw_gain": float(raw_gain),
                    "theta_after_add": float(after),
                }
            )
        best = max(gains, key=lambda item: (item["gain"], -item["index"]))
        if best["gain"] < stop_threshold:
            stop_reason = "gain_below_threshold"
            terminal_decision = {
                "step": step_idx,
                "selected": False,
                "best_index": int(best["index"]),
                "best_gain": float(best["gain"]),
                "threshold": float(stop_threshold),
            }
            break
        steps.append(
            {
                "step": step_idx,
                "current_indices": list(selected),
                "theta_before": float(before),
                "best_index": int(best["index"]),
                "best_gain": float(best["gain"]),
                "raw_best_gain": float(best["raw_gain"]),
                "theta_after": float(best["theta_after_add"]),
                "stop_decision": {"selected": True, "reason": "continue"},
                "candidate_gains": gains if store_all_gains else [],
            }
        )
        selected.append(int(best["index"]))
        theta_current = float(best["theta_after_add"])
        stop_reason = "top_m_reached" if len(selected) >= top_m else "no_candidates"

    return {
        "selected_indices": selected,
        "theta_empty": 0.0,
        "theta_final": theta_current,
        "steps": steps,
        "stop_reason": stop_reason,
        "terminal_stop_decision": terminal_decision,
        "gain_tolerance": float(gain_tolerance),
    }

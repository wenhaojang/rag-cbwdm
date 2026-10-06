"""Query-local RKHS projection utilities for experimental RAG-KCBWDM.

The current linear RAG-CBWDM code stores candidate posterior effects as rows
``X_all[J, K]``.  This module keeps that convention.  Its ``linear`` mode is a
compatibility implementation: it uses the same effects, absolute ridge,
singleton sign gate, numerical gain tolerance, stop comparison, and candidate
index tie-break as signed-v1.

This is a query-local candidate-atom projection.  It is not the n-by-n
cross-observation empirical KCBWDM estimator from the theory draft.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


KernelName = Literal["linear", "rbf"]
DEFAULT_RIDGE_LAMBDA = 0.01
BANDWIDTH_POLICY = "train_core_within_query_positive_distance_median"
BANDWIDTH_IMPLEMENTATION_VERSION = "kcbwdm_train_core_median_v1"


@dataclass(frozen=True)
class KernelMarginal:
    """Schur-complement diagnostics for adding one candidate to a set."""

    gain: float
    theta_after_add: float
    residual_target_alignment: float
    residual_self_information: float


@dataclass(frozen=True)
class BandwidthFit:
    """Immutable provenance for a train-core median bandwidth fit."""

    sigma: float
    policy: str
    positive_distance_count: int
    query_group_count: int
    nonempty_query_group_count: int
    source_split: str
    implementation_version: str
    subsampling: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sigma": self.sigma,
            "policy": self.policy,
            "positive_distance_count": self.positive_distance_count,
            "query_group_count": self.query_group_count,
            "nonempty_query_group_count": self.nonempty_query_group_count,
            "source_split": self.source_split,
            "implementation_version": self.implementation_version,
            "subsampling": self.subsampling,
        }


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_kernel(kernel: str, sigma: float | None) -> KernelName:
    name = str(kernel).strip().lower()
    if name not in {"linear", "rbf"}:
        raise ValueError(f"Unsupported kernel={kernel!r}; expected 'linear' or 'rbf'")
    if name == "rbf":
        if sigma is None or not np.isfinite(float(sigma)) or float(sigma) <= 0:
            raise ValueError("RBF kernel requires a finite positive sigma")
    return name  # type: ignore[return-value]


def linear_kernel(u: np.ndarray, v: np.ndarray) -> float:
    """Return the ordinary float64 linear kernel ``u.T @ v``."""
    left, right = _vector(u, "u"), _vector(v, "v")
    if left.shape != right.shape:
        raise ValueError(f"Kernel vectors have different shapes: {left.shape}, {right.shape}")
    return float(left @ right)


def rbf_kernel(u: np.ndarray, v: np.ndarray, *, sigma: float) -> float:
    """Return ``exp(-||u-v||^2 / (2 sigma^2))`` in float64."""
    left, right = _vector(u, "u"), _vector(v, "v")
    if left.shape != right.shape:
        raise ValueError(f"Kernel vectors have different shapes: {left.shape}, {right.shape}")
    sigma_value = float(sigma)
    if not np.isfinite(sigma_value) or sigma_value <= 0:
        raise ValueError("sigma must be finite and positive")
    difference = left - right
    return float(np.exp(-(difference @ difference) / (2.0 * sigma_value**2)))


def anchored_kernel(
    u: np.ndarray,
    v: np.ndarray,
    *,
    kernel: KernelName = "linear",
    sigma: float | None = None,
) -> float:
    """Return the zero-effect-anchored kernel ``<phi(u)-phi(0),phi(v)-phi(0)>``."""
    left, right = _vector(u, "u"), _vector(v, "v")
    if left.shape != right.shape:
        raise ValueError(f"Kernel vectors have different shapes: {left.shape}, {right.shape}")
    name = _validate_kernel(kernel, sigma)
    if name == "linear":
        # Preserve the exact signed-v1 inner-product path; anchoring a linear
        # kernel at zero is algebraically identical to the ordinary dot product.
        return linear_kernel(left, right)
    zero = np.zeros_like(left)
    assert sigma is not None
    return float(
        rbf_kernel(left, right, sigma=sigma)
        - rbf_kernel(left, zero, sigma=sigma)
        - rbf_kernel(zero, right, sigma=sigma)
        + 1.0
    )


def anchored_gram(
    left: np.ndarray,
    right: np.ndarray | None = None,
    *,
    kernel: KernelName = "linear",
    sigma: float | None = None,
) -> np.ndarray:
    """Return a float64 anchored Gram/cross-Gram matrix."""
    left_matrix = _matrix(left, "left")
    right_matrix = left_matrix if right is None else _matrix(right, "right")
    if left_matrix.shape[1] != right_matrix.shape[1]:
        raise ValueError("Gram inputs must have the same feature width")
    name = _validate_kernel(kernel, sigma)
    if name == "linear":
        return np.asarray(left_matrix @ right_matrix.T, dtype=np.float64)
    result = np.empty((len(left_matrix), len(right_matrix)), dtype=np.float64)
    for row_index, u in enumerate(left_matrix):
        for column_index, v in enumerate(right_matrix):
            result[row_index, column_index] = anchored_kernel(
                u, v, kernel=name, sigma=sigma
            )
    # Eliminate harmless asymmetric round-off when constructing a square Gram.
    if right is None:
        result = (result + result.T) / 2.0
    return result


def kernel_target_alignments(
    X_all: np.ndarray,
    d: np.ndarray,
    *,
    kernel: KernelName = "linear",
    sigma: float | None = None,
) -> np.ndarray:
    """Return static singleton target signals ``k0(x_j, d)``."""
    effects, target = _matrix(X_all, "X_all"), _vector(d, "d")
    if effects.shape[1] != target.size:
        raise ValueError(f"Incompatible X_all {effects.shape} and d {target.shape}")
    name = _validate_kernel(kernel, sigma)
    if name == "linear":
        return np.asarray(effects @ target, dtype=np.float64)
    return np.asarray(
        [anchored_kernel(row, target, kernel=name, sigma=sigma) for row in effects],
        dtype=np.float64,
    )


def _solve(system: np.ndarray, rhs: np.ndarray, *, context: str) -> np.ndarray:
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        warnings.warn(
            f"{context} linear solve failed; using pseudo-inverse.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.linalg.pinv(system) @ rhs


def kernel_set_score(
    X_all: np.ndarray,
    d: np.ndarray,
    indices: Sequence[int],
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    *,
    kernel: KernelName = "linear",
    sigma: float | None = None,
) -> float:
    """Compute ``r_S.T @ (K_S + lambda I)^-1 @ r_S``.

    Lambda is absolute.  The target is not normalized and the Gram is not
    centered within the selected set.
    """
    effects, target = _matrix(X_all, "X_all"), _vector(d, "d")
    if effects.shape[1] != target.size:
        raise ValueError(f"Incompatible X_all {effects.shape} and d {target.shape}")
    ridge = float(ridge_lambda)
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError(f"ridge_lambda must be finite and positive, got {ridge_lambda}")
    selected = [int(index) for index in indices]
    if len(selected) != len(set(selected)):
        raise ValueError("indices must not contain duplicates")
    if any(index < 0 or index >= len(effects) for index in selected):
        raise IndexError("selected index is out of range")
    name = _validate_kernel(kernel, sigma)
    if not selected:
        return 0.0
    subset = effects[selected, :]
    gram = anchored_gram(subset, kernel=name, sigma=sigma)
    relevance = kernel_target_alignments(subset, target, kernel=name, sigma=sigma)
    system = gram + ridge * np.eye(len(selected), dtype=np.float64)
    solution = _solve(system, relevance, context="KCBWDM set-score")
    value = float(relevance @ solution)
    if not np.isfinite(value):
        raise FloatingPointError("KCBWDM set score is NaN or Inf")
    return value


def kernel_marginal_gain(
    X_all: np.ndarray,
    d: np.ndarray,
    current_indices: Sequence[int],
    candidate_index: int,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    *,
    kernel: KernelName = "linear",
    sigma: float | None = None,
    numerical_tolerance: float = 1e-10,
) -> KernelMarginal:
    """Return the Schur-complement gain and its residual components."""
    effects, target = _matrix(X_all, "X_all"), _vector(d, "d")
    if effects.shape[1] != target.size:
        raise ValueError(f"Incompatible X_all {effects.shape} and d {target.shape}")
    selected = [int(index) for index in current_indices]
    candidate = int(candidate_index)
    if candidate in selected:
        raise ValueError(f"candidate_index {candidate} is already selected")
    if candidate < 0 or candidate >= len(effects):
        raise IndexError("candidate_index is out of range")
    ridge = float(ridge_lambda)
    tolerance = float(numerical_tolerance)
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError(f"ridge_lambda must be finite and positive, got {ridge_lambda}")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("numerical_tolerance must be finite and non-negative")
    name = _validate_kernel(kernel, sigma)
    all_alignments = kernel_target_alignments(effects, target, kernel=name, sigma=sigma)
    target_alignment = float(all_alignments[candidate])
    candidate_self = anchored_kernel(
        effects[candidate], effects[candidate], kernel=name, sigma=sigma
    )
    if selected:
        subset = effects[selected, :]
        system = anchored_gram(subset, kernel=name, sigma=sigma) + ridge * np.eye(
            len(selected), dtype=np.float64
        )
        relevance = all_alignments[selected]
        cross = anchored_gram(
            subset, effects[candidate : candidate + 1], kernel=name, sigma=sigma
        )[:, 0]
        solved = _solve(
            system,
            np.column_stack((relevance, cross)),
            context="KCBWDM Schur marginal",
        )
        residual_alignment = target_alignment - float(cross @ solved[:, 0])
        residual_information = candidate_self + ridge - float(cross @ solved[:, 1])
        before = float(relevance @ solved[:, 0])
    else:
        residual_alignment = target_alignment
        residual_information = candidate_self + ridge
        before = 0.0
    if not np.isfinite(residual_information) or residual_information <= 0:
        relation = "below tolerance" if residual_information < -tolerance else "non-positive"
        raise FloatingPointError(
            f"Schur residual self-information is {relation}: {residual_information}"
        )
    gain = float(residual_alignment**2 / residual_information)
    if not np.isfinite(gain) or gain < -tolerance:
        raise FloatingPointError(f"Invalid KCBWDM marginal gain: {gain}")
    if gain < 0:
        gain = 0.0
    return KernelMarginal(
        gain=gain,
        theta_after_add=float(before + gain),
        residual_target_alignment=float(residual_alignment),
        residual_self_information=float(residual_information),
    )


def kernel_signed_greedy(
    X_all: np.ndarray,
    d: np.ndarray,
    *,
    top_m: int,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    kernel: KernelName = "linear",
    sigma: float | None = None,
    stop_threshold: float = 0.0,
    alignment_eps: float = 0.0,
    gain_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Run signed-v1 greedy selection with a static kernel target signal."""
    effects, target = _matrix(X_all, "X_all"), _vector(d, "d")
    if effects.shape[1] != target.size:
        raise ValueError(f"Incompatible X_all {effects.shape} and d {target.shape}")
    if top_m < 0:
        raise ValueError("top_m must be non-negative")
    if alignment_eps < 0 or gain_tolerance < 0:
        raise ValueError("alignment_eps and gain_tolerance must be non-negative")
    name = _validate_kernel(kernel, sigma)
    alignments = kernel_target_alignments(effects, target, kernel=name, sigma=sigma)
    admissible = alignments > float(alignment_eps)
    selected: list[int] = []
    steps: list[dict[str, Any]] = []
    stop_reason = "top_m_reached" if top_m == 0 else "no_admissible_candidates"
    for step_index in range(top_m):
        remaining = [index for index in range(len(effects)) if admissible[index] and index not in selected]
        if not remaining:
            stop_reason = "no_admissible_candidates"
            break
        before = kernel_set_score(
            effects, target, selected, ridge_lambda, kernel=name, sigma=sigma
        )
        gains: list[dict[str, Any]] = []
        for index in remaining:
            marginal = kernel_marginal_gain(
                effects,
                target,
                selected,
                index,
                ridge_lambda,
                kernel=name,
                sigma=sigma,
                numerical_tolerance=gain_tolerance,
            )
            if name == "linear":
                # Preserve signed-v1's exact floating-point decision path.  The
                # Schur value remains available in ``marginal`` for diagnostics,
                # but current production ranks the before/after score difference.
                theta_after_add = kernel_set_score(
                    effects,
                    target,
                    selected + [index],
                    ridge_lambda,
                    kernel=name,
                    sigma=sigma,
                )
                raw_gain = float(theta_after_add - before)
            else:
                theta_after_add = float(marginal.theta_after_add)
                raw_gain = float(marginal.gain)
            if raw_gain < -gain_tolerance:
                raise FloatingPointError(f"Negative KCBWDM marginal gain: {raw_gain}")
            gain = 0.0 if abs(raw_gain) <= gain_tolerance else raw_gain
            gains.append(
                {
                    "index": index,
                    "gain": gain,
                    "raw_gain": raw_gain,
                    "theta_after_add": theta_after_add,
                    "alignment": float(alignments[index]),
                    "residual_target_alignment": float(
                        marginal.residual_target_alignment
                    ),
                    "residual_self_information": float(
                        marginal.residual_self_information
                    ),
                }
            )
        best = max(gains, key=lambda item: (item["gain"], -item["index"]))
        if best["gain"] < stop_threshold:
            stop_reason = "gain_below_threshold"
            break
        steps.append(
            {
                "step": step_index,
                "current_indices": list(selected),
                "theta_before": float(before),
                "candidate_gains": gains,
                "best_index": int(best["index"]),
                "best_gain": float(best["gain"]),
                "theta_after": float(best["theta_after_add"]),
            }
        )
        selected.append(int(best["index"]))
        stop_reason = "top_m_reached" if len(selected) >= top_m else "no_admissible_candidates"
    return {
        "selected_indices": selected,
        "steps": steps,
        "stop_reason": stop_reason,
        "alignments": alignments.tolist(),
        "admissible": admissible.tolist(),
        "theta_final": kernel_set_score(
            effects, target, selected, ridge_lambda, kernel=name, sigma=sigma
        ),
    }


def fit_train_core_bandwidth(
    effect_groups: Sequence[np.ndarray],
    *,
    split: str,
) -> BandwidthFit:
    """Fit deterministic RBF sigma from positive within-query distances.

    This helper accepts in-memory effects only and fails unless the caller
    explicitly declares the formal ``train_core`` split.  It therefore cannot
    silently inspect validation data.  Artifact fitting/provenance is deferred
    to a later phase.
    """
    if split != "train_core":
        raise ValueError("Kernel bandwidth may be fit only from split='train_core'")
    distances: list[float] = []
    nonempty_query_group_count = 0
    for group_index, group in enumerate(effect_groups):
        effects = _matrix(group, f"effect_groups[{group_index}]")
        nonempty_query_group_count += int(len(effects) > 0)
        for left in range(len(effects)):
            for right in range(left + 1, len(effects)):
                distance = float(np.linalg.norm(effects[left] - effects[right]))
                if distance > 0:
                    distances.append(distance)
    if not distances:
        raise ValueError("Cannot fit RBF bandwidth: train_core geometry has no positive distances")
    bandwidth = float(np.median(np.asarray(distances, dtype=np.float64)))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise FloatingPointError(f"Invalid median RBF bandwidth: {bandwidth}")
    return BandwidthFit(
        sigma=bandwidth,
        policy=BANDWIDTH_POLICY,
        positive_distance_count=len(distances),
        query_group_count=len(effect_groups),
        nonempty_query_group_count=nonempty_query_group_count,
        source_split="train_core",
        implementation_version=BANDWIDTH_IMPLEMENTATION_VERSION,
        subsampling="none",
    )


# Descriptive alias retained for callers that want the fitting statistic in the
# name; both entry points enforce the same train-core-only contract.
train_core_median_bandwidth = fit_train_core_bandwidth

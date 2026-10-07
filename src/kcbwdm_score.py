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
KERNEL_SCALE_NORMALIZATION_POLICY = "median_positive_diag_ratio_v1"
KERNEL_SCALE_IMPLEMENTATION_VERSION = "kcbwdm_kernel_scale_v1"
HYBRID_KERNEL_IMPLEMENTATION_VERSION = "scale_preserving_hybrid_kernel_v1"
RHO_KERNEL_IMPLEMENTATION_VERSION = "scale_controlled_rho_kernel_v1"
NORMALIZED_RHO_KERNEL_IMPLEMENTATION_VERSION = "median_diagonal_normalized_rho_v1"


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


@dataclass(frozen=True)
class KernelScaleFit:
    """Deterministic train-core scale fit for the anchored RBF component."""

    linear_diag_median_positive: float
    rbf_diag_median_positive: float
    kernel_scale_c: float
    candidate_count: int
    linear_diag_eligible_count: int
    rbf_diag_eligible_count: int
    linear_diag_near_zero_exclusion_count: int
    rbf_diag_near_zero_exclusion_count: int
    numerical_tolerance: float
    source_split: str
    normalization_policy: str
    implementation_version: str
    subsampling: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "linear_diag_median_positive": self.linear_diag_median_positive,
            "rbf_diag_median_positive": self.rbf_diag_median_positive,
            "kernel_scale_c": self.kernel_scale_c,
            "candidate_count": self.candidate_count,
            "eligible_counts": {
                "linear_diag": self.linear_diag_eligible_count,
                "rbf_diag": self.rbf_diag_eligible_count,
            },
            "near_zero_exclusion_counts": {
                "linear_diag": self.linear_diag_near_zero_exclusion_count,
                "rbf_diag": self.rbf_diag_near_zero_exclusion_count,
            },
            "numerical_tolerance": self.numerical_tolerance,
            "source_split": self.source_split,
            "normalization_policy": self.normalization_policy,
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


def linear_gate_kernel_signed_greedy(
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
    """Run RKHS greedy utility behind the signed-v1 linear directional gate.

    Admission is always decided by ``x_j.T @ d``.  The requested kernel is used
    only for the set score and Schur-complement marginal gain.  Consequently,
    ``kernel='linear'`` follows the exact signed-v1 decision path, while RBF
    mode cannot admit an effect whose linear target alignment is non-positive.
    """
    effects, target = _matrix(X_all, "X_all"), _vector(d, "d")
    if effects.shape[1] != target.size:
        raise ValueError(f"Incompatible X_all {effects.shape} and d {target.shape}")
    if top_m < 0:
        raise ValueError("top_m must be non-negative")
    if alignment_eps < 0 or gain_tolerance < 0:
        raise ValueError("alignment_eps and gain_tolerance must be non-negative")
    name = _validate_kernel(kernel, sigma)
    alignments = np.asarray(effects @ target, dtype=np.float64)
    admissible = alignments > float(alignment_eps)
    selected: list[int] = []
    steps: list[dict[str, Any]] = []
    stop_reason = "top_m_reached" if top_m == 0 else "no_admissible_candidates"
    for step_index in range(top_m):
        remaining = [
            index
            for index in range(len(effects))
            if admissible[index] and index not in selected
        ]
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
        stop_reason = (
            "top_m_reached"
            if len(selected) >= top_m
            else "no_admissible_candidates"
        )
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


def kernel_component_diagonals(
    effects: np.ndarray,
    *,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 linear and zero-anchored RBF candidate diagonals."""
    matrix = _matrix(effects, "effects")
    sigma_value = float(sigma)
    if not np.isfinite(sigma_value) or sigma_value <= 0:
        raise ValueError("sigma must be finite and positive")
    linear_diag = np.asarray(
        np.einsum("ij,ij->i", matrix, matrix), dtype=np.float64
    )
    rbf_diag = np.asarray(
        [
            anchored_kernel(row, row, kernel="rbf", sigma=sigma_value)
            for row in matrix
        ],
        dtype=np.float64,
    )
    return linear_diag, rbf_diag


def fit_kernel_scale_c(
    effects: np.ndarray,
    *,
    sigma: float,
    split: str,
) -> KernelScaleFit:
    """Fit ``c=median_positive(linear_diag)/median_positive(rbf_diag)``.

    Only candidate effects are accepted, so the fit has no label or target
    dependency.  The explicit split guard prevents accidental validation or
    held-out fitting.
    """
    if split != "train_core":
        raise ValueError("Kernel scale may be fit only from split='train_core'")
    linear_diag, rbf_diag = kernel_component_diagonals(effects, sigma=sigma)
    if not len(linear_diag):
        raise ValueError("Cannot fit kernel scale from zero candidate effects")
    combined = np.concatenate((linear_diag, rbf_diag)).astype(
        np.float64, copy=False
    )
    if not np.all(np.isfinite(combined)):
        raise FloatingPointError("Kernel component diagonals must be finite")
    maximum = float(np.max(np.abs(combined))) if combined.size else 0.0
    tolerance = max(
        1e-15,
        64.0 * np.finfo(np.float64).eps * max(1.0, maximum),
    )
    for name, values in (("linear", linear_diag), ("anchored RBF", rbf_diag)):
        if np.any(values < -tolerance):
            minimum = float(np.min(values))
            raise FloatingPointError(
                f"{name} diagonal is negative beyond numerical tolerance: {minimum}"
            )
    linear_positive = linear_diag[linear_diag > tolerance]
    rbf_positive = rbf_diag[rbf_diag > tolerance]
    if not len(linear_positive) or not len(rbf_positive):
        raise ValueError("Cannot fit kernel scale without positive component diagonals")
    linear_median = float(np.median(linear_positive, overwrite_input=False))
    rbf_median = float(np.median(rbf_positive, overwrite_input=False))
    if not np.isfinite(linear_median) or linear_median <= 0:
        raise FloatingPointError(
            f"Invalid positive linear diagonal median: {linear_median}"
        )
    if not np.isfinite(rbf_median) or rbf_median <= 0:
        raise FloatingPointError(
            f"Invalid positive anchored RBF diagonal median: {rbf_median}"
        )
    scale = float(linear_median / rbf_median)
    if not np.isfinite(scale) or scale <= 0:
        raise FloatingPointError(f"Invalid fitted kernel scale c: {scale}")
    return KernelScaleFit(
        linear_diag_median_positive=linear_median,
        rbf_diag_median_positive=rbf_median,
        kernel_scale_c=scale,
        candidate_count=len(linear_diag),
        linear_diag_eligible_count=len(linear_positive),
        rbf_diag_eligible_count=len(rbf_positive),
        linear_diag_near_zero_exclusion_count=int(
            np.count_nonzero(np.abs(linear_diag) <= tolerance)
        ),
        rbf_diag_near_zero_exclusion_count=int(
            np.count_nonzero(np.abs(rbf_diag) <= tolerance)
        ),
        numerical_tolerance=tolerance,
        source_split="train_core",
        normalization_policy=KERNEL_SCALE_NORMALIZATION_POLICY,
        implementation_version=KERNEL_SCALE_IMPLEMENTATION_VERSION,
        subsampling="none",
    )


def _validate_hybrid_parameters(
    *,
    sigma: float,
    alpha: float,
    kernel_scale_c: float,
) -> tuple[float, float, float]:
    sigma_value = float(sigma)
    alpha_value = float(alpha)
    scale_value = float(kernel_scale_c)
    if not np.isfinite(sigma_value) or sigma_value <= 0:
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(alpha_value) or alpha_value < 0:
        raise ValueError("alpha must be finite and non-negative")
    if not np.isfinite(scale_value) or scale_value <= 0:
        raise ValueError("kernel_scale_c must be finite and positive")
    return sigma_value, alpha_value, scale_value


def hybrid_kernel(
    u: np.ndarray,
    v: np.ndarray,
    *,
    sigma: float,
    alpha: float,
    kernel_scale_c: float,
) -> float:
    """Return the scale-controlled linear-plus-anchored-RBF kernel.

    Division by ``1 + alpha`` controls component scale but does not make the
    effective ridge exactly constant.
    """
    sigma_value, alpha_value, scale_value = _validate_hybrid_parameters(
        sigma=sigma, alpha=alpha, kernel_scale_c=kernel_scale_c
    )
    if alpha_value == 0.0:
        return linear_kernel(u, v)
    linear = linear_kernel(u, v)
    nonlinear = anchored_kernel(u, v, kernel="rbf", sigma=sigma_value)
    return float(
        (linear + alpha_value * scale_value * nonlinear) / (1.0 + alpha_value)
    )


def hybrid_gram(
    left: np.ndarray,
    right: np.ndarray | None = None,
    *,
    sigma: float,
    alpha: float,
    kernel_scale_c: float,
) -> np.ndarray:
    """Return a scale-controlled hybrid Gram or cross-Gram matrix."""
    sigma_value, alpha_value, scale_value = _validate_hybrid_parameters(
        sigma=sigma, alpha=alpha, kernel_scale_c=kernel_scale_c
    )
    if alpha_value == 0.0:
        return anchored_gram(left, right, kernel="linear")
    linear = anchored_gram(left, right, kernel="linear")
    nonlinear = anchored_gram(left, right, kernel="rbf", sigma=sigma_value)
    result = (linear + alpha_value * scale_value * nonlinear) / (
        1.0 + alpha_value
    )
    if right is None:
        result = (result + result.T) / 2.0
    return np.asarray(result, dtype=np.float64)


def hybrid_target_alignments(
    X_all: np.ndarray,
    d: np.ndarray,
    *,
    sigma: float,
    alpha: float,
    kernel_scale_c: float,
) -> np.ndarray:
    """Return ``k_alpha(x_j,d)`` for hybrid set utility diagnostics."""
    sigma_value, alpha_value, scale_value = _validate_hybrid_parameters(
        sigma=sigma, alpha=alpha, kernel_scale_c=kernel_scale_c
    )
    if alpha_value == 0.0:
        return kernel_target_alignments(X_all, d, kernel="linear")
    effects, target = _matrix(X_all, "X_all"), _vector(d, "d")
    if effects.shape[1] != target.size:
        raise ValueError(f"Incompatible X_all {effects.shape} and d {target.shape}")
    linear = kernel_target_alignments(effects, target, kernel="linear")
    nonlinear = kernel_target_alignments(
        effects, target, kernel="rbf", sigma=sigma_value
    )
    return np.asarray(
        (linear + alpha_value * scale_value * nonlinear) / (1.0 + alpha_value),
        dtype=np.float64,
    )


def hybrid_set_score(
    X_all: np.ndarray,
    d: np.ndarray,
    indices: Sequence[int],
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    *,
    sigma: float,
    alpha: float,
    kernel_scale_c: float,
) -> float:
    """Compute the projection score under the scale-controlled hybrid kernel."""
    sigma_value, alpha_value, scale_value = _validate_hybrid_parameters(
        sigma=sigma, alpha=alpha, kernel_scale_c=kernel_scale_c
    )
    if alpha_value == 0.0:
        return kernel_set_score(
            X_all, d, indices, ridge_lambda, kernel="linear"
        )
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
    if not selected:
        return 0.0
    subset = effects[selected, :]
    gram = hybrid_gram(
        subset,
        sigma=sigma_value,
        alpha=alpha_value,
        kernel_scale_c=scale_value,
    )
    relevance = hybrid_target_alignments(
        subset,
        target,
        sigma=sigma_value,
        alpha=alpha_value,
        kernel_scale_c=scale_value,
    )
    system = gram + ridge * np.eye(len(selected), dtype=np.float64)
    solution = _solve(system, relevance, context="hybrid KCBWDM set-score")
    value = float(relevance @ solution)
    if not np.isfinite(value):
        raise FloatingPointError("Hybrid KCBWDM set score is NaN or Inf")
    return value


def hybrid_marginal_gain(
    X_all: np.ndarray,
    d: np.ndarray,
    current_indices: Sequence[int],
    candidate_index: int,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    *,
    sigma: float,
    alpha: float,
    kernel_scale_c: float,
    numerical_tolerance: float = 1e-10,
) -> KernelMarginal:
    """Return the hybrid-kernel Schur-complement marginal gain."""
    sigma_value, alpha_value, scale_value = _validate_hybrid_parameters(
        sigma=sigma, alpha=alpha, kernel_scale_c=kernel_scale_c
    )
    if alpha_value == 0.0:
        return kernel_marginal_gain(
            X_all,
            d,
            current_indices,
            candidate_index,
            ridge_lambda,
            kernel="linear",
            numerical_tolerance=numerical_tolerance,
        )
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
    all_alignments = hybrid_target_alignments(
        effects,
        target,
        sigma=sigma_value,
        alpha=alpha_value,
        kernel_scale_c=scale_value,
    )
    target_alignment = float(all_alignments[candidate])
    candidate_self = hybrid_kernel(
        effects[candidate],
        effects[candidate],
        sigma=sigma_value,
        alpha=alpha_value,
        kernel_scale_c=scale_value,
    )
    if selected:
        subset = effects[selected, :]
        system = hybrid_gram(
            subset,
            sigma=sigma_value,
            alpha=alpha_value,
            kernel_scale_c=scale_value,
        ) + ridge * np.eye(len(selected), dtype=np.float64)
        relevance = all_alignments[selected]
        cross = hybrid_gram(
            subset,
            effects[candidate : candidate + 1],
            sigma=sigma_value,
            alpha=alpha_value,
            kernel_scale_c=scale_value,
        )[:, 0]
        solved = _solve(
            system,
            np.column_stack((relevance, cross)),
            context="hybrid KCBWDM Schur marginal",
        )
        residual_alignment = target_alignment - float(cross @ solved[:, 0])
        residual_information = candidate_self + ridge - float(cross @ solved[:, 1])
        before = float(relevance @ solved[:, 0])
    else:
        residual_alignment = target_alignment
        residual_information = candidate_self + ridge
        before = 0.0
    if not np.isfinite(residual_information) or residual_information <= 0:
        relation = (
            "below tolerance"
            if residual_information < -tolerance
            else "non-positive"
        )
        raise FloatingPointError(
            f"Hybrid Schur residual self-information is {relation}: "
            f"{residual_information}"
        )
    gain = float(residual_alignment**2 / residual_information)
    if not np.isfinite(gain) or gain < -tolerance:
        raise FloatingPointError(f"Invalid hybrid KCBWDM marginal gain: {gain}")
    if gain < 0:
        gain = 0.0
    return KernelMarginal(
        gain=gain,
        theta_after_add=float(before + gain),
        residual_target_alignment=float(residual_alignment),
        residual_self_information=float(residual_information),
    )


def hybrid_linear_gate_signed_greedy(
    X_all: np.ndarray,
    d: np.ndarray,
    *,
    top_m: int,
    sigma: float,
    alpha: float,
    kernel_scale_c: float,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    stop_threshold: float = 0.0,
    alignment_eps: float = 0.0,
    gain_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Run hybrid RKHS utility behind the immutable linear directional gate."""
    sigma_value, alpha_value, scale_value = _validate_hybrid_parameters(
        sigma=sigma, alpha=alpha, kernel_scale_c=kernel_scale_c
    )
    if alpha_value == 0.0:
        return linear_gate_kernel_signed_greedy(
            X_all,
            d,
            top_m=top_m,
            ridge_lambda=ridge_lambda,
            kernel="linear",
            stop_threshold=stop_threshold,
            alignment_eps=alignment_eps,
            gain_tolerance=gain_tolerance,
        )
    effects, target = _matrix(X_all, "X_all"), _vector(d, "d")
    if effects.shape[1] != target.size:
        raise ValueError(f"Incompatible X_all {effects.shape} and d {target.shape}")
    if top_m < 0:
        raise ValueError("top_m must be non-negative")
    if alignment_eps < 0 or gain_tolerance < 0:
        raise ValueError("alignment_eps and gain_tolerance must be non-negative")
    alignments = np.asarray(effects @ target, dtype=np.float64)
    admissible = alignments > float(alignment_eps)
    selected: list[int] = []
    steps: list[dict[str, Any]] = []
    stop_reason = "top_m_reached" if top_m == 0 else "no_admissible_candidates"
    for step_index in range(top_m):
        remaining = [
            index
            for index in range(len(effects))
            if admissible[index] and index not in selected
        ]
        if not remaining:
            stop_reason = "no_admissible_candidates"
            break
        before = hybrid_set_score(
            effects,
            target,
            selected,
            ridge_lambda,
            sigma=sigma_value,
            alpha=alpha_value,
            kernel_scale_c=scale_value,
        )
        gains: list[dict[str, Any]] = []
        for index in remaining:
            marginal = hybrid_marginal_gain(
                effects,
                target,
                selected,
                index,
                ridge_lambda,
                sigma=sigma_value,
                alpha=alpha_value,
                kernel_scale_c=scale_value,
                numerical_tolerance=gain_tolerance,
            )
            raw_gain = float(marginal.gain)
            if raw_gain < -gain_tolerance:
                raise FloatingPointError(
                    f"Negative hybrid KCBWDM marginal gain: {raw_gain}"
                )
            gain = 0.0 if abs(raw_gain) <= gain_tolerance else raw_gain
            gains.append(
                {
                    "index": index,
                    "gain": gain,
                    "raw_gain": raw_gain,
                    "theta_after_add": float(marginal.theta_after_add),
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
        stop_reason = (
            "top_m_reached"
            if len(selected) >= top_m
            else "no_admissible_candidates"
        )
    return {
        "selected_indices": selected,
        "steps": steps,
        "stop_reason": stop_reason,
        "alignments": alignments.tolist(),
        "admissible": admissible.tolist(),
        "theta_final": hybrid_set_score(
            effects,
            target,
            selected,
            ridge_lambda,
            sigma=sigma_value,
            alpha=alpha_value,
            kernel_scale_c=scale_value,
        ),
    }


def _validate_rho_parameters(
    *,
    sigma: float,
    rho: float,
    kernel_scale_c: float,
) -> tuple[float, float, float]:
    sigma_value = float(sigma)
    rho_value = float(rho)
    scale_value = float(kernel_scale_c)
    if not np.isfinite(sigma_value) or sigma_value <= 0:
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(rho_value) or rho_value < 0 or rho_value > 1:
        raise ValueError("rho must be finite and in [0, 1]")
    if not np.isfinite(scale_value) or scale_value <= 0:
        raise ValueError("kernel_scale_c must be finite and positive")
    return sigma_value, rho_value, scale_value


def rho_kernel(
    u: np.ndarray,
    v: np.ndarray,
    *,
    sigma: float,
    rho: float,
    kernel_scale_c: float,
) -> float:
    """Return ``(1-rho) linear + rho c anchored-RBF``.

    This is a scale-controlled interpolation under a fixed absolute ridge; it
    does not claim that the effective ridge is exactly invariant in ``rho``.
    """
    sigma_value, rho_value, scale_value = _validate_rho_parameters(
        sigma=sigma, rho=rho, kernel_scale_c=kernel_scale_c
    )
    if rho_value == 0.0:
        return linear_kernel(u, v)
    nonlinear = anchored_kernel(u, v, kernel="rbf", sigma=sigma_value)
    if rho_value == 1.0:
        return float(scale_value * nonlinear)
    return float(
        (1.0 - rho_value) * linear_kernel(u, v)
        + rho_value * scale_value * nonlinear
    )


def rho_gram(
    left: np.ndarray,
    right: np.ndarray | None = None,
    *,
    sigma: float,
    rho: float,
    kernel_scale_c: float,
) -> np.ndarray:
    """Return the rho-parameterized Gram or cross-Gram matrix."""
    sigma_value, rho_value, scale_value = _validate_rho_parameters(
        sigma=sigma, rho=rho, kernel_scale_c=kernel_scale_c
    )
    if rho_value == 0.0:
        return anchored_gram(left, right, kernel="linear")
    nonlinear = anchored_gram(left, right, kernel="rbf", sigma=sigma_value)
    if rho_value == 1.0:
        result = scale_value * nonlinear
    else:
        linear = anchored_gram(left, right, kernel="linear")
        result = (1.0 - rho_value) * linear + rho_value * scale_value * nonlinear
    if right is None:
        result = (result + result.T) / 2.0
    return np.asarray(result, dtype=np.float64)


def rho_target_alignments(
    X_all: np.ndarray,
    d: np.ndarray,
    *,
    sigma: float,
    rho: float,
    kernel_scale_c: float,
) -> np.ndarray:
    """Return rho-kernel target signals for set utility diagnostics."""
    sigma_value, rho_value, scale_value = _validate_rho_parameters(
        sigma=sigma, rho=rho, kernel_scale_c=kernel_scale_c
    )
    if rho_value == 0.0:
        return kernel_target_alignments(X_all, d, kernel="linear")
    effects, target = _matrix(X_all, "X_all"), _vector(d, "d")
    if effects.shape[1] != target.size:
        raise ValueError(f"Incompatible X_all {effects.shape} and d {target.shape}")
    nonlinear = kernel_target_alignments(
        effects, target, kernel="rbf", sigma=sigma_value
    )
    if rho_value == 1.0:
        return np.asarray(scale_value * nonlinear, dtype=np.float64)
    linear = kernel_target_alignments(effects, target, kernel="linear")
    return np.asarray(
        (1.0 - rho_value) * linear + rho_value * scale_value * nonlinear,
        dtype=np.float64,
    )


def rho_set_score(
    X_all: np.ndarray,
    d: np.ndarray,
    indices: Sequence[int],
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    *,
    sigma: float,
    rho: float,
    kernel_scale_c: float,
) -> float:
    """Compute the projection score under the rho kernel."""
    sigma_value, rho_value, scale_value = _validate_rho_parameters(
        sigma=sigma, rho=rho, kernel_scale_c=kernel_scale_c
    )
    if rho_value == 0.0:
        return kernel_set_score(
            X_all, d, indices, ridge_lambda, kernel="linear"
        )
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
    if not selected:
        return 0.0
    subset = effects[selected, :]
    gram = rho_gram(
        subset,
        sigma=sigma_value,
        rho=rho_value,
        kernel_scale_c=scale_value,
    )
    relevance = rho_target_alignments(
        subset,
        target,
        sigma=sigma_value,
        rho=rho_value,
        kernel_scale_c=scale_value,
    )
    system = gram + ridge * np.eye(len(selected), dtype=np.float64)
    solution = _solve(system, relevance, context="rho KCBWDM set-score")
    value = float(relevance @ solution)
    if not np.isfinite(value):
        raise FloatingPointError("Rho KCBWDM set score is NaN or Inf")
    return value


def rho_marginal_gain(
    X_all: np.ndarray,
    d: np.ndarray,
    current_indices: Sequence[int],
    candidate_index: int,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    *,
    sigma: float,
    rho: float,
    kernel_scale_c: float,
    numerical_tolerance: float = 1e-10,
) -> KernelMarginal:
    """Return the rho-kernel Schur-complement marginal gain."""
    sigma_value, rho_value, scale_value = _validate_rho_parameters(
        sigma=sigma, rho=rho, kernel_scale_c=kernel_scale_c
    )
    if rho_value == 0.0:
        return kernel_marginal_gain(
            X_all,
            d,
            current_indices,
            candidate_index,
            ridge_lambda,
            kernel="linear",
            numerical_tolerance=numerical_tolerance,
        )
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
    all_alignments = rho_target_alignments(
        effects,
        target,
        sigma=sigma_value,
        rho=rho_value,
        kernel_scale_c=scale_value,
    )
    target_alignment = float(all_alignments[candidate])
    candidate_self = rho_kernel(
        effects[candidate],
        effects[candidate],
        sigma=sigma_value,
        rho=rho_value,
        kernel_scale_c=scale_value,
    )
    if selected:
        subset = effects[selected, :]
        system = rho_gram(
            subset,
            sigma=sigma_value,
            rho=rho_value,
            kernel_scale_c=scale_value,
        ) + ridge * np.eye(len(selected), dtype=np.float64)
        relevance = all_alignments[selected]
        cross = rho_gram(
            subset,
            effects[candidate : candidate + 1],
            sigma=sigma_value,
            rho=rho_value,
            kernel_scale_c=scale_value,
        )[:, 0]
        solved = _solve(
            system,
            np.column_stack((relevance, cross)),
            context="rho KCBWDM Schur marginal",
        )
        residual_alignment = target_alignment - float(cross @ solved[:, 0])
        residual_information = candidate_self + ridge - float(cross @ solved[:, 1])
        before = float(relevance @ solved[:, 0])
    else:
        residual_alignment = target_alignment
        residual_information = candidate_self + ridge
        before = 0.0
    if not np.isfinite(residual_information) or residual_information <= 0:
        relation = (
            "below tolerance"
            if residual_information < -tolerance
            else "non-positive"
        )
        raise FloatingPointError(
            f"Rho Schur residual self-information is {relation}: "
            f"{residual_information}"
        )
    gain = float(residual_alignment**2 / residual_information)
    if not np.isfinite(gain) or gain < -tolerance:
        raise FloatingPointError(f"Invalid rho KCBWDM marginal gain: {gain}")
    if gain < 0:
        gain = 0.0
    return KernelMarginal(
        gain=gain,
        theta_after_add=float(before + gain),
        residual_target_alignment=float(residual_alignment),
        residual_self_information=float(residual_information),
    )


def rho_linear_gate_signed_greedy(
    X_all: np.ndarray,
    d: np.ndarray,
    *,
    top_m: int,
    sigma: float,
    rho: float,
    kernel_scale_c: float,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    stop_threshold: float = 0.001,
    alignment_eps: float = 0.0,
    gain_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Run rho-kernel utility behind the immutable linear directional gate."""
    sigma_value, rho_value, scale_value = _validate_rho_parameters(
        sigma=sigma, rho=rho, kernel_scale_c=kernel_scale_c
    )
    if rho_value == 0.0:
        return linear_gate_kernel_signed_greedy(
            X_all,
            d,
            top_m=top_m,
            ridge_lambda=ridge_lambda,
            kernel="linear",
            stop_threshold=stop_threshold,
            alignment_eps=alignment_eps,
            gain_tolerance=gain_tolerance,
        )
    effects, target = _matrix(X_all, "X_all"), _vector(d, "d")
    if effects.shape[1] != target.size:
        raise ValueError(f"Incompatible X_all {effects.shape} and d {target.shape}")
    if top_m < 0:
        raise ValueError("top_m must be non-negative")
    if alignment_eps < 0 or gain_tolerance < 0:
        raise ValueError("alignment_eps and gain_tolerance must be non-negative")
    alignments = np.asarray(effects @ target, dtype=np.float64)
    admissible = alignments > float(alignment_eps)
    selected: list[int] = []
    steps: list[dict[str, Any]] = []
    stop_reason = "top_m_reached" if top_m == 0 else "no_admissible_candidates"
    for step_index in range(top_m):
        remaining = [
            index
            for index in range(len(effects))
            if admissible[index] and index not in selected
        ]
        if not remaining:
            stop_reason = "no_admissible_candidates"
            break
        before = rho_set_score(
            effects,
            target,
            selected,
            ridge_lambda,
            sigma=sigma_value,
            rho=rho_value,
            kernel_scale_c=scale_value,
        )
        gains: list[dict[str, Any]] = []
        for index in remaining:
            marginal = rho_marginal_gain(
                effects,
                target,
                selected,
                index,
                ridge_lambda,
                sigma=sigma_value,
                rho=rho_value,
                kernel_scale_c=scale_value,
                numerical_tolerance=gain_tolerance,
            )
            raw_gain = float(marginal.gain)
            if raw_gain < -gain_tolerance:
                raise FloatingPointError(
                    f"Negative rho KCBWDM marginal gain: {raw_gain}"
                )
            gain = 0.0 if abs(raw_gain) <= gain_tolerance else raw_gain
            gains.append(
                {
                    "index": index,
                    "gain": gain,
                    "raw_gain": raw_gain,
                    "theta_after_add": float(marginal.theta_after_add),
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
        stop_reason = (
            "top_m_reached"
            if len(selected) >= top_m
            else "no_admissible_candidates"
        )
    return {
        "selected_indices": selected,
        "steps": steps,
        "stop_reason": stop_reason,
        "alignments": alignments.tolist(),
        "admissible": admissible.tolist(),
        "theta_final": rho_set_score(
            effects,
            target,
            selected,
            ridge_lambda,
            sigma=sigma_value,
            rho=rho_value,
            kernel_scale_c=scale_value,
        ),
    }


def _validate_normalized_rho_parameters(
    *,
    sigma: float,
    rho: float,
    linear_diag_median_positive: float,
    rbf_diag_median_positive: float,
) -> tuple[float, float, float, float, float]:
    linear_scale = float(linear_diag_median_positive)
    rbf_scale = float(rbf_diag_median_positive)
    if not np.isfinite(linear_scale) or linear_scale <= 0:
        raise ValueError("linear_diag_median_positive must be finite and positive")
    if not np.isfinite(rbf_scale) or rbf_scale <= 0:
        raise ValueError("rbf_diag_median_positive must be finite and positive")
    kernel_scale_c = float(linear_scale / rbf_scale)
    sigma_value, rho_value, scale_value = _validate_rho_parameters(
        sigma=sigma,
        rho=rho,
        kernel_scale_c=kernel_scale_c,
    )
    return sigma_value, rho_value, linear_scale, rbf_scale, scale_value


def normalized_rho_kernel(
    u: np.ndarray,
    v: np.ndarray,
    *,
    sigma: float,
    rho: float,
    linear_diag_median_positive: float,
    rbf_diag_median_positive: float,
) -> float:
    """Return the median-diagonal normalized rho kernel.

    Both component scales must come from the existing label-free train-core
    ``fit_kernel_scale_c`` contract.  The normalization controls median
    diagonal scale but does not claim complete spectral invariance.
    """
    sigma_value, rho_value, linear_scale, rbf_scale, _ = (
        _validate_normalized_rho_parameters(
            sigma=sigma,
            rho=rho,
            linear_diag_median_positive=linear_diag_median_positive,
            rbf_diag_median_positive=rbf_diag_median_positive,
        )
    )
    if rho_value == 0.0:
        return float(linear_kernel(u, v) / linear_scale)
    nonlinear = anchored_kernel(u, v, kernel="rbf", sigma=sigma_value)
    if rho_value == 1.0:
        return float(nonlinear / rbf_scale)
    return float(
        (1.0 - rho_value) * linear_kernel(u, v) / linear_scale
        + rho_value * nonlinear / rbf_scale
    )


def normalized_rho_gram(
    left: np.ndarray,
    right: np.ndarray | None = None,
    *,
    sigma: float,
    rho: float,
    linear_diag_median_positive: float,
    rbf_diag_median_positive: float,
) -> np.ndarray:
    """Return a median-diagonal normalized rho Gram or cross-Gram matrix."""
    sigma_value, rho_value, linear_scale, rbf_scale, _ = (
        _validate_normalized_rho_parameters(
            sigma=sigma,
            rho=rho,
            linear_diag_median_positive=linear_diag_median_positive,
            rbf_diag_median_positive=rbf_diag_median_positive,
        )
    )
    if rho_value == 0.0:
        return np.asarray(
            anchored_gram(left, right, kernel="linear") / linear_scale,
            dtype=np.float64,
        )
    nonlinear = anchored_gram(left, right, kernel="rbf", sigma=sigma_value)
    if rho_value == 1.0:
        result = nonlinear / rbf_scale
    else:
        linear = anchored_gram(left, right, kernel="linear")
        result = (
            (1.0 - rho_value) * linear / linear_scale
            + rho_value * nonlinear / rbf_scale
        )
    if right is None:
        result = (result + result.T) / 2.0
    return np.asarray(result, dtype=np.float64)


def normalized_rho_target_alignments(
    X_all: np.ndarray,
    d: np.ndarray,
    *,
    sigma: float,
    rho: float,
    linear_diag_median_positive: float,
    rbf_diag_median_positive: float,
) -> np.ndarray:
    """Return normalized rho-kernel target signals for utility diagnostics."""
    sigma_value, rho_value, linear_scale, rbf_scale, _ = (
        _validate_normalized_rho_parameters(
            sigma=sigma,
            rho=rho,
            linear_diag_median_positive=linear_diag_median_positive,
            rbf_diag_median_positive=rbf_diag_median_positive,
        )
    )
    if rho_value == 0.0:
        return np.asarray(
            kernel_target_alignments(X_all, d, kernel="linear") / linear_scale,
            dtype=np.float64,
        )
    nonlinear = kernel_target_alignments(
        X_all, d, kernel="rbf", sigma=sigma_value
    )
    if rho_value == 1.0:
        return np.asarray(nonlinear / rbf_scale, dtype=np.float64)
    linear = kernel_target_alignments(X_all, d, kernel="linear")
    return np.asarray(
        (1.0 - rho_value) * linear / linear_scale
        + rho_value * nonlinear / rbf_scale,
        dtype=np.float64,
    )


def normalized_rho_set_score(
    X_all: np.ndarray,
    d: np.ndarray,
    indices: Sequence[int],
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    *,
    sigma: float,
    rho: float,
    linear_diag_median_positive: float,
    rbf_diag_median_positive: float,
) -> float:
    """Return the normalized score through its exact raw-policy identity."""
    sigma_value, rho_value, linear_scale, rbf_scale, scale_value = (
        _validate_normalized_rho_parameters(
            sigma=sigma,
            rho=rho,
            linear_diag_median_positive=linear_diag_median_positive,
            rbf_diag_median_positive=rbf_diag_median_positive,
        )
    )
    ridge = float(ridge_lambda)
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError(f"ridge_lambda must be finite and positive, got {ridge_lambda}")
    if rho_value == 0.0:
        raw_score = kernel_set_score(
            X_all,
            d,
            indices,
            ridge * linear_scale,
            kernel="linear",
        )
        return float(raw_score / linear_scale)
    if rho_value == 1.0:
        raw_score = kernel_set_score(
            X_all,
            d,
            indices,
            ridge * rbf_scale,
            kernel="rbf",
            sigma=sigma_value,
        )
        return float(raw_score / rbf_scale)
    raw_score = rho_set_score(
        X_all,
        d,
        indices,
        ridge * linear_scale,
        sigma=sigma_value,
        rho=rho_value,
        kernel_scale_c=scale_value,
    )
    return float(raw_score / linear_scale)


def normalized_rho_marginal_gain(
    X_all: np.ndarray,
    d: np.ndarray,
    current_indices: Sequence[int],
    candidate_index: int,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    *,
    sigma: float,
    rho: float,
    linear_diag_median_positive: float,
    rbf_diag_median_positive: float,
    numerical_tolerance: float = 1e-10,
) -> KernelMarginal:
    """Return a normalized marginal through the exact raw-policy identity."""
    sigma_value, rho_value, linear_scale, rbf_scale, scale_value = (
        _validate_normalized_rho_parameters(
            sigma=sigma,
            rho=rho,
            linear_diag_median_positive=linear_diag_median_positive,
            rbf_diag_median_positive=rbf_diag_median_positive,
        )
    )
    ridge = float(ridge_lambda)
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError(f"ridge_lambda must be finite and positive, got {ridge_lambda}")
    output_scale = linear_scale
    if rho_value == 0.0:
        raw = kernel_marginal_gain(
            X_all,
            d,
            current_indices,
            candidate_index,
            ridge * linear_scale,
            kernel="linear",
            numerical_tolerance=numerical_tolerance,
        )
    elif rho_value == 1.0:
        output_scale = rbf_scale
        raw = kernel_marginal_gain(
            X_all,
            d,
            current_indices,
            candidate_index,
            ridge * rbf_scale,
            kernel="rbf",
            sigma=sigma_value,
            numerical_tolerance=numerical_tolerance,
        )
    else:
        raw = rho_marginal_gain(
            X_all,
            d,
            current_indices,
            candidate_index,
            ridge * linear_scale,
            sigma=sigma_value,
            rho=rho_value,
            kernel_scale_c=scale_value,
            numerical_tolerance=numerical_tolerance,
        )
    return KernelMarginal(
        gain=float(raw.gain / output_scale),
        theta_after_add=float(raw.theta_after_add / output_scale),
        residual_target_alignment=float(
            raw.residual_target_alignment / output_scale
        ),
        residual_self_information=float(
            raw.residual_self_information / output_scale
        ),
    )


def normalized_rho_linear_gate_signed_greedy(
    X_all: np.ndarray,
    d: np.ndarray,
    *,
    top_m: int,
    sigma: float,
    rho: float,
    linear_diag_median_positive: float,
    rbf_diag_median_positive: float,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    stop_threshold: float = 0.001,
    alignment_eps: float = 0.0,
    gain_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Run normalized rho utility behind the immutable linear gate."""
    sigma_value, rho_value, linear_scale, rbf_scale, scale_value = (
        _validate_normalized_rho_parameters(
            sigma=sigma,
            rho=rho,
            linear_diag_median_positive=linear_diag_median_positive,
            rbf_diag_median_positive=rbf_diag_median_positive,
        )
    )
    ridge = float(ridge_lambda)
    threshold = float(stop_threshold)
    tolerance = float(gain_tolerance)
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError(f"ridge_lambda must be finite and positive, got {ridge_lambda}")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("stop_threshold must be finite and non-negative")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("gain_tolerance must be finite and non-negative")
    output_scale = linear_scale
    if rho_value == 0.0:
        raw = linear_gate_kernel_signed_greedy(
            X_all,
            d,
            top_m=top_m,
            kernel="linear",
            ridge_lambda=ridge * linear_scale,
            stop_threshold=threshold * linear_scale,
            alignment_eps=alignment_eps,
            gain_tolerance=tolerance,
        )
    elif rho_value == 1.0:
        output_scale = rbf_scale
        raw = linear_gate_kernel_signed_greedy(
            X_all,
            d,
            top_m=top_m,
            kernel="rbf",
            sigma=sigma_value,
            ridge_lambda=ridge * rbf_scale,
            stop_threshold=threshold * rbf_scale,
            alignment_eps=alignment_eps,
            gain_tolerance=tolerance,
        )
    else:
        raw = rho_linear_gate_signed_greedy(
            X_all,
            d,
            top_m=top_m,
            sigma=sigma_value,
            rho=rho_value,
            kernel_scale_c=scale_value,
            ridge_lambda=ridge * linear_scale,
            stop_threshold=threshold * linear_scale,
            alignment_eps=alignment_eps,
            gain_tolerance=tolerance,
        )
    steps: list[dict[str, Any]] = []
    for step in raw["steps"]:
        candidate_gains = []
        for candidate in step["candidate_gains"]:
            candidate_gains.append(
                {
                    **candidate,
                    "gain": float(candidate["gain"] / output_scale),
                    "raw_gain": float(candidate["raw_gain"] / output_scale),
                    "theta_after_add": float(
                        candidate["theta_after_add"] / output_scale
                    ),
                    "residual_target_alignment": float(
                        candidate["residual_target_alignment"] / output_scale
                    ),
                    "residual_self_information": float(
                        candidate["residual_self_information"] / output_scale
                    ),
                }
            )
        steps.append(
            {
                **step,
                "theta_before": float(step["theta_before"] / output_scale),
                "candidate_gains": candidate_gains,
                "best_gain": float(step["best_gain"] / output_scale),
                "theta_after": float(step["theta_after"] / output_scale),
            }
        )
    return {
        **raw,
        "steps": steps,
        "theta_final": float(raw["theta_final"] / output_scale),
    }


# Descriptive alias retained for callers that want the fitting statistic in the
# name; both entry points enforce the same train-core-only contract.
train_core_median_bandwidth = fit_train_core_bandwidth

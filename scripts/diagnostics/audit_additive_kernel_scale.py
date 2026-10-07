from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cbwdm_score import build_local_effects
from src.experiment_identity import FORMAL_V2_MODE, validate_posterior_provenance
from src.io_utils import load_yaml, read_jsonl
from src.kcbwdm_score import (
    BANDWIDTH_POLICY,
    HYBRID_KERNEL_IMPLEMENTATION_VERSION,
    KERNEL_SCALE_NORMALIZATION_POLICY,
    NORMALIZED_RHO_KERNEL_IMPLEMENTATION_VERSION,
    RHO_KERNEL_IMPLEMENTATION_VERSION,
    anchored_kernel,
    fit_kernel_scale_c,
    kernel_component_diagonals,
)
from src.run_manifest import atomic_write_json, git_state, sha256_file, utc_now


AUDIT_SCHEMA_VERSION = "rag_cbwdm_additive_kernel_scale_audit.v3"
AUDIT_IMPLEMENTATION_VERSION = "additive_kernel_scale_audit_v3"
NEAR_ZERO_POLICY = "float64_relative_denominator_tau_v1"
PERCENTILES = (10, 25, 50, 75, 90, 95, 99)
RIDGE_LAMBDA = 0.01
RHO_GRID_CANDIDATE = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def summarize_values(values: np.ndarray) -> dict[str, Any]:
    """Return JSON-safe float64 descriptive statistics without clipping."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    summary: dict[str, Any] = {
        "count": int(array.size),
        "positive_count": int(np.count_nonzero(array > 0)),
        "negative_count": int(np.count_nonzero(array < 0)),
        "nonzero_count": int(np.count_nonzero(array != 0)),
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(array))),
    }
    if not finite.size:
        summary.update(
            {
                "min": None,
                "mean": None,
                "std": None,
                **{f"p{percentile}": None for percentile in PERCENTILES},
                "max": None,
            }
        )
        return summary
    percentile_values = np.percentile(finite, PERCENTILES)
    summary.update(
        {
            "min": float(np.min(finite)),
            "mean": float(np.mean(finite, dtype=np.float64)),
            "std": float(np.std(finite, dtype=np.float64)),
            **{
                f"p{percentile}": float(value)
                for percentile, value in zip(PERCENTILES, percentile_values)
            },
            "max": float(np.max(finite)),
        }
    )
    return summary


def summarize_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> dict[str, Any]:
    """Summarize a ratio after explicit exact-zero and near-zero exclusions."""
    numerator_array = np.asarray(numerator, dtype=np.float64).reshape(-1)
    denominator_array = np.asarray(denominator, dtype=np.float64).reshape(-1)
    if numerator_array.shape != denominator_array.shape:
        raise ValueError("Ratio numerator and denominator must have the same shape")
    finite_denominator = denominator_array[np.isfinite(denominator_array)]
    max_abs_denominator = (
        float(np.max(np.abs(finite_denominator)))
        if finite_denominator.size
        else 0.0
    )
    tau = max(
        1e-15,
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, max_abs_denominator),
    )
    finite_pair = np.isfinite(numerator_array) & np.isfinite(denominator_array)
    exact_zero = finite_pair & (denominator_array == 0.0)
    near_zero = (
        finite_pair
        & (denominator_array != 0.0)
        & (np.abs(denominator_array) <= tau)
    )
    eligible = finite_pair & (np.abs(denominator_array) > tau)
    ratios = np.asarray(
        numerator_array[eligible] / denominator_array[eligible],
        dtype=np.float64,
    )
    result = summarize_values(ratios)
    result.update(
        {
            "ratio_eligible_count": int(np.count_nonzero(eligible)),
            "excluded_zero_count": int(np.count_nonzero(exact_zero)),
            "excluded_near_zero_count": int(np.count_nonzero(near_zero)),
            "excluded_nonfinite_count": int(np.count_nonzero(~finite_pair)),
            "tau": tau,
        }
    )
    return result


def _load_bandwidth_provenance(
    path: Path | None,
    *,
    sigma: float,
) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Bandwidth provenance does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Bandwidth provenance must be a JSON object")
    provenance = payload.get("bandwidth_provenance")
    if provenance is None:
        provenance = (
            payload.get("contract", {})
            .get("parameters", {})
            .get("bandwidth_provenance")
        )
    if not isinstance(provenance, dict):
        raise ValueError("Bandwidth provenance object is missing")
    recorded_sigma = provenance.get("sigma")
    if recorded_sigma is None or float(recorded_sigma) != float(sigma):
        raise ValueError("Bandwidth provenance sigma differs from --sigma")
    if provenance.get("source_split") != "train_core":
        raise ValueError("Bandwidth provenance must be fitted from train_core")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "record": provenance,
    }


def build_audit_payload(
    *,
    posterior_path: Path,
    posterior_manifest_path: Path,
    config_path: Path,
    sigma: float,
    generator_id: str,
    dataset_id: str,
    bandwidth_provenance_path: Path | None = None,
) -> dict[str, Any]:
    """Build a train-core-only kernel-scale audit without writing artifacts."""
    sigma_value = float(sigma)
    if not np.isfinite(sigma_value) or sigma_value <= 0:
        raise ValueError("sigma must be finite and positive")
    posterior = posterior_path.resolve()
    posterior_manifest = posterior_manifest_path.resolve()
    config_file = config_path.resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_file}")
    config = load_yaml(config_file)
    if config.get("dataset_id") != dataset_id:
        raise ValueError("Config dataset_id differs from --dataset-id")
    if config.get("generator", {}).get("generator_id") != generator_id:
        raise ValueError("Config generator_id differs from --generator-id")
    binding = validate_posterior_provenance(
        posterior,
        posterior_manifest,
        mode=FORMAL_V2_MODE,
        expected_dataset_id=dataset_id,
        expected_split="train_core",
        expected_generator_id=generator_id,
    )
    cbwdm = config.get("cbwdm", {})
    l_type = str(cbwdm.get("L_type", "euclidean_posterior_shift"))
    eps_smooth = float(cbwdm.get("eps_smooth", 0.0))
    target_smoothing = str(cbwdm.get("target_smoothing", "paper_mixture"))

    rows = list(read_jsonl(posterior))
    if not rows:
        raise ValueError("Posterior audit input must not be empty")
    identifiers = [str(row.get("id")) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Posterior audit input contains duplicate query IDs")
    if {row.get("split") for row in rows} != {"train_core"}:
        raise ValueError("Kernel scale audit accepts only train_core posterior rows")

    effect_groups: list[np.ndarray] = []
    linear_targets: list[np.ndarray] = []
    rbf_targets: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        candidates = list(row.get("candidates", []))
        labels = list(row["labels"])
        candidate_etas = (
            np.asarray([candidate["eta"] for candidate in candidates], dtype=np.float64)
            if candidates
            else np.empty((0, len(labels)), dtype=np.float64)
        )
        effects, target = build_local_effects(
            np.asarray(row["eta0"], dtype=np.float64),
            candidate_etas,
            str(row["label"]),
            labels,
            l_type=l_type,
            eps_smooth=eps_smooth,
            target_smoothing=target_smoothing,
        )
        effect_groups.append(np.asarray(effects, dtype=np.float64))
        linear_targets.append(np.asarray(effects @ target, dtype=np.float64))
        rbf_targets.append(
            np.asarray(
                [
                    anchored_kernel(
                        effect,
                        target,
                        kernel="rbf",
                        sigma=sigma_value,
                    )
                    for effect in effects
                ],
                dtype=np.float64,
            )
        )
        if len(effects) != len(candidates):
            raise AssertionError(f"Effect construction row mismatch at index {row_index}")

    effects = np.vstack(effect_groups)
    linear_target = np.concatenate(linear_targets)
    rbf_target = np.concatenate(rbf_targets)
    linear_diag, rbf_diag = kernel_component_diagonals(
        effects, sigma=sigma_value
    )
    scale_fit = fit_kernel_scale_c(
        effects, sigma=sigma_value, split="train_core"
    )
    scale = scale_fit.kernel_scale_c
    scaled_rbf_target = np.asarray(scale * rbf_target, dtype=np.float64)
    admissible = linear_target > 0.0
    linear_median = scale_fit.linear_diag_median_positive
    rbf_median = scale_fit.rbf_diag_median_positive
    scaled_rbf_median = float(scale * rbf_median)
    identity_tolerance = max(
        scale_fit.numerical_tolerance,
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(linear_median), abs(scaled_rbf_median)),
    )
    lambda_over_linear = float(RIDGE_LAMBDA / linear_median)
    lambda_over_raw_rbf = float(RIDGE_LAMBDA / rbf_median)
    lambda_over_scaled_rbf = float(RIDGE_LAMBDA / scaled_rbf_median)
    ridge_identity_tolerance = max(
        1e-15,
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(lambda_over_linear), abs(lambda_over_scaled_rbf)),
    )
    finite_rbf_diag = rbf_diag[np.isfinite(rbf_diag)]
    if not finite_rbf_diag.size:
        raise FloatingPointError("RBF diagonals contain no finite values")

    bandwidth_provenance = _load_bandwidth_provenance(
        bandwidth_provenance_path, sigma=sigma_value
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "implementation_version": AUDIT_IMPLEMENTATION_VERSION,
        "rho_kernel_implementation_version": RHO_KERNEL_IMPLEMENTATION_VERSION,
        "rho_parameterization": {
            "description": "scale-preserving / scale-controlled hybrid kernel",
            "formula": "(1 - rho) * u^T v + rho * c * k0_rbf(u,v)",
            "domain": "0 <= rho <= 1",
            "linear_endpoint": "rho=0",
            "scale_matched_rbf_endpoint": "rho=1",
            "directional_gate": "x_j^T d_i > alignment_eps",
            "alignment_eps": 0.0,
            "ridge_lambda": RIDGE_LAMBDA,
            "stop_threshold": 0.001,
            "effective_ridge_claim": "not_exactly_constant",
            "rho_grid_candidate": list(RHO_GRID_CANDIDATE),
            "best_rho_selected": False,
            "raw_rbf_v2a_comparator": "separate_not_scale_matched_endpoint",
        },
        "alpha_backward_compatibility": {
            "implementation_version": HYBRID_KERNEL_IMPLEMENTATION_VERSION,
            "relation": "rho = alpha / (1 + alpha)",
            "rho_one_relation": "alpha approaches positive infinity",
            "recommended_parameter": "rho",
        },
        "normalized_rho_kernel_implementation_version": (
            NORMALIZED_RHO_KERNEL_IMPLEMENTATION_VERSION
        ),
        "normalized_rho_kernel_formula": (
            "(1 - rho) * k_linear(u,v) / m_L + "
            "rho * k0_rbf(u,v) / m_R"
        ),
        "canonical_linear_scale": linear_median,
        "canonical_rbf_scale": rbf_median,
        "normalized_component_median_target": {"linear": 1.0, "rbf": 1.0},
        "canonical_ridge_lambda": RIDGE_LAMBDA,
        "canonical_stop_threshold": 0.001,
        "equivalent_raw_linear_ridge": float(RIDGE_LAMBDA * linear_median),
        "equivalent_raw_linear_stop_threshold": float(0.001 * linear_median),
        "equivalent_raw_rbf_ridge": float(RIDGE_LAMBDA * rbf_median),
        "equivalent_raw_rbf_stop_threshold": float(0.001 * rbf_median),
        "rho_grid_candidate": list(RHO_GRID_CANDIDATE),
        "best_rho_selected": False,
        "created_at": utc_now(),
        "diagnostic_only": True,
        "dataset_id": dataset_id,
        "split": "train_core",
        "generator_id": generator_id,
        "posterior": {
            "path": str(posterior),
            "sha256": binding["posterior_sha256"],
        },
        "posterior_manifest": {
            "path": str(posterior_manifest),
            "sha256": binding["manifest_sha256"],
            "fingerprint": binding.get("manifest_fingerprint"),
            "identity_fingerprint": binding.get("identity_fingerprint"),
        },
        "posterior_identity": {
            "dataset_identity": binding.get("dataset_identity"),
            "generator_identity": binding.get("generator_identity"),
            "retrieval_protocol_identity": binding.get(
                "retrieval_protocol_identity"
            ),
        },
        "config": {"path": str(config_file), "sha256": sha256_file(config_file)},
        "row_count": len(rows),
        "query_count": len(rows),
        "candidate_count": len(effects),
        "effect_construction_identity": {
            "helper": "src.cbwdm_score.build_local_effects",
            "l_type": l_type,
            "eps_smooth": eps_smooth,
            "target_smoothing": target_smoothing,
            "candidate_posterior_smoothing": False,
        },
        "sigma": sigma_value,
        "bandwidth_policy": BANDWIDTH_POLICY,
        "bandwidth_provenance": bandwidth_provenance,
        "subsampling": "none",
        "normalization_policy": KERNEL_SCALE_NORMALIZATION_POLICY,
        **scale_fit.to_dict(),
        "ridge_lambda": RIDGE_LAMBDA,
        "raw_rbf_over_linear_diag_median": float(rbf_median / linear_median),
        "lambda_over_linear_diag_median": lambda_over_linear,
        "lambda_over_raw_rbf_diag_median": lambda_over_raw_rbf,
        "lambda_over_scaled_rbf_diag_median": lambda_over_scaled_rbf,
        "raw_rbf_scale_confounded": True,
        "scale_matching_identity": {
            "scaled_rbf_diag_median": scaled_rbf_median,
            "linear_diag_median": linear_median,
            "absolute_difference": abs(scaled_rbf_median - linear_median),
            "tolerance": identity_tolerance,
            "within_tolerance": bool(
                abs(scaled_rbf_median - linear_median) <= identity_tolerance
            ),
        },
        "effective_ridge_diagnostic_identity": {
            "absolute_difference": abs(
                lambda_over_scaled_rbf - lambda_over_linear
            ),
            "tolerance": ridge_identity_tolerance,
            "within_tolerance": bool(
                abs(lambda_over_scaled_rbf - lambda_over_linear)
                <= ridge_identity_tolerance
            ),
        },
        "gate_conditional_diagnostics": {
            "used_for_scale_fitting": False,
            "condition": "linear_target = x^T d > alignment_eps",
            "alignment_eps": 0.0,
            "admissible_candidate_count": int(np.count_nonzero(admissible)),
            "statistics": {
                "linear_target_positive": summarize_values(
                    linear_target[admissible]
                ),
                "rbf_target_given_linear_positive": summarize_values(
                    rbf_target[admissible]
                ),
                "scaled_rbf_target_given_linear_positive": summarize_values(
                    scaled_rbf_target[admissible]
                ),
                "absolute_linear_target_given_positive": summarize_values(
                    np.abs(linear_target[admissible])
                ),
                "absolute_scaled_rbf_target_given_positive": summarize_values(
                    np.abs(scaled_rbf_target[admissible])
                ),
            },
            "abs_scaled_rbf_target_over_abs_linear_target": summarize_ratio(
                np.abs(scaled_rbf_target[admissible]),
                np.abs(linear_target[admissible]),
            ),
        },
        "rbf_diag_fraction_ge_1_9": float(
            np.mean(finite_rbf_diag >= 1.9, dtype=np.float64)
        ),
        "rbf_diag_fraction_ge_1_99": float(
            np.mean(finite_rbf_diag >= 1.99, dtype=np.float64)
        ),
        "rbf_diag_fraction_ge_1_999": float(
            np.mean(finite_rbf_diag >= 1.999, dtype=np.float64)
        ),
        "near_zero_policy": {
            "name": NEAR_ZERO_POLICY,
            "formula": (
                "max(1e-15, 64 * float64_eps * "
                "max(1, max_abs_denominator))"
            ),
            "epsilon_replacement": False,
            "winsorization": False,
        },
        "statistics": {
            "linear_diag": summarize_values(linear_diag),
            "rbf_diag": summarize_values(rbf_diag),
            "linear_target": summarize_values(linear_target),
            "rbf_target": summarize_values(rbf_target),
            "rbf_diag_over_linear_diag": summarize_ratio(rbf_diag, linear_diag),
            "abs_rbf_target_over_abs_linear_target": summarize_ratio(
                np.abs(rbf_target), np.abs(linear_target)
            ),
        },
        "git": git_state(PROJECT_ROOT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit train-core component scales for the diagnostic "
            "scale-preserving hybrid KCBWDM kernel."
        )
    )
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--posterior-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sigma", required=True, type=float)
    parser.add_argument("--generator-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--bandwidth-provenance")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Kernel-scale audit output already exists: {output}")
    payload = build_audit_payload(
        posterior_path=Path(args.posteriors),
        posterior_manifest_path=Path(args.posterior_manifest),
        config_path=Path(args.config),
        sigma=args.sigma,
        generator_id=args.generator_id,
        dataset_id=args.dataset_id,
        bandwidth_provenance_path=(
            Path(args.bandwidth_provenance)
            if args.bandwidth_provenance
            else None
        ),
    )
    atomic_write_json(output, payload)
    print(
        "[additive_kernel_scale_audit] "
        f"queries={payload['query_count']} candidates={payload['candidate_count']} "
        f"c={payload['kernel_scale_c']:.17g} output={output}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cbwdm_score import build_local_effects
from src.diagnostics.kcbwdm_signed_teacher_v1 import (
    KCBWDM_SIGNED_V1_METHOD,
    build_kcbwdm_signed_teacher_row,
    kcbwdm_teacher_statistics,
)
from src.diagnostics.kcbwdm_linear_gate_v2 import (
    KCBWDM_LINEAR_GATE_V2_METHOD,
    build_kcbwdm_linear_gate_v2_teacher_row,
    kcbwdm_linear_gate_v2_statistics,
)
from src.diagnostics.kcbwdm_normalized_rho_v1 import (
    KCBWDM_NORMALIZED_RHO_V1_METHOD,
    build_kcbwdm_normalized_rho_v1_teacher_row,
    kcbwdm_normalized_rho_v1_statistics,
)
from src.experiment_identity import (
    load_optional_posterior_binding,
    posterior_binding_contract,
    resolve_dataset_identity,
)
from src.io_utils import load_yaml, read_jsonl
from src.kcbwdm_score import fit_kernel_scale_c, fit_train_core_bandwidth
from src.preformal.registry import (
    KCBWDM_LINEAR_GATE_V2_CONTRACT,
    KCBWDM_SIGNED_V1_CONTRACT,
    KCBWDM_NORMALIZED_RHO_V1_CONTRACT,
    assert_frozen_kcbwdm_normalized_rho_v1_contract,
    assert_frozen_kcbwdm_linear_gate_v2_contract,
    assert_frozen_kcbwdm_contract,
    assert_no_held_out_reference,
)
from src.run_manifest import (
    atomic_write_json,
    git_state,
    sha256_file,
    stable_hash,
    utc_now,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _effect_groups(rows: list[dict], params: dict) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for row in rows:
        candidates = list(row.get("candidates", []))
        if not candidates:
            groups.append(np.empty((0, len(row["labels"])), dtype=np.float64))
            continue
        effects, _ = build_local_effects(
            np.asarray(row["eta0"]),
            np.asarray([candidate["eta"] for candidate in candidates]),
            str(row["label"]),
            list(row["labels"]),
            params["l_type"],
            params["eps_smooth"],
            params["target_smoothing"],
        )
        groups.append(effects)
    return groups


def _fit_and_verify_kernel_scale(
    effect_groups: list[np.ndarray], *, sigma: float, split: str
):
    nonempty = [group for group in effect_groups if len(group)]
    if not nonempty:
        raise ValueError("Cannot fit normalized-rho scale without candidate effects")
    effects = np.vstack(nonempty)
    fitted = fit_kernel_scale_c(effects, sigma=sigma, split=split)
    recomputed = fit_kernel_scale_c(effects, sigma=sigma, split=split)
    tolerance = max(fitted.numerical_tolerance, recomputed.numerical_tolerance)
    for field in (
        "linear_diag_median_positive",
        "rbf_diag_median_positive",
        "kernel_scale_c",
    ):
        if not np.isclose(
            float(getattr(fitted, field)),
            float(getattr(recomputed, field)),
            rtol=1e-12,
            atol=tolerance,
        ):
            raise FloatingPointError(
                f"Deterministic normalized-rho scale recomputation mismatch: {field}"
            )
    expected_c = (
        fitted.linear_diag_median_positive / fitted.rbf_diag_median_positive
    )
    if not np.isclose(
        fitted.kernel_scale_c,
        expected_c,
        rtol=1e-12,
        atol=tolerance,
    ):
        raise FloatingPointError(
            "Fitted kernel_scale_c disagrees with the recorded median ratio"
        )
    return fitted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize experimental query-local KCBWDM signed-v1 teacher"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--training-split", choices=["train_core"], default="train_core"
    )
    parser.add_argument("--kernel", choices=["linear", "rbf"], default="rbf")
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--posterior-manifest")
    parser.add_argument("--dataset-id")
    parser.add_argument("--generator-id")
    parser.add_argument("--retrieval-protocol-id")
    parser.add_argument(
        "--method-name",
        choices=[
            KCBWDM_SIGNED_V1_METHOD,
            KCBWDM_LINEAR_GATE_V2_METHOD,
            KCBWDM_NORMALIZED_RHO_V1_METHOD,
        ],
        default=KCBWDM_SIGNED_V1_METHOD,
    )
    parser.add_argument("--rho", type=float)
    parser.add_argument("--formal-v2-identity", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive")
    if args.sigma is not None:
        raise ValueError(
            f"{args.method_name} freezes train-core median bandwidth; explicit --sigma is forbidden"
        )
    if args.method_name == KCBWDM_NORMALIZED_RHO_V1_METHOD:
        if args.rho is None:
            raise ValueError("kcbwdm_normalized_rho_v1 requires explicit --rho")
        if args.max_rows is not None:
            raise ValueError(
                "kcbwdm_normalized_rho_v1 requires the full train_core; --max-rows is forbidden"
            )
    elif args.rho is not None:
        raise ValueError("--rho is valid only for kcbwdm_normalized_rho_v1")

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    posterior = Path(args.posteriors).resolve()
    retrieval = Path(args.retrieval).resolve()
    output = Path(args.output_dir).resolve()
    assert_no_held_out_reference(
        {"posteriors": str(posterior), "retrieval": str(retrieval)}
    )

    configured_dataset = resolve_dataset_identity(
        config["dataset"], explicit_dataset_id=args.dataset_id
    )
    posterior_binding = load_optional_posterior_binding(
        posterior,
        Path(args.posterior_manifest).resolve() if args.posterior_manifest else None,
        formal_v2=args.formal_v2_identity,
        expected_dataset_id=(
            configured_dataset.dataset_id
            if args.formal_v2_identity or args.dataset_id
            else None
        ),
        expected_split=args.training_split,
        expected_generator_id=args.generator_id,
        expected_retrieval_protocol_id=args.retrieval_protocol_id,
    )
    identity_opt_in = bool(
        args.formal_v2_identity
        or args.posterior_manifest
        or args.dataset_id
        or args.generator_id
        or args.retrieval_protocol_id
    )
    has_stable_identity = bool(
        posterior_binding
        and posterior_binding["dataset_identity"].get("dataset_id")
        and posterior_binding["generator_identity"].get("generator_id")
        and posterior_binding["retrieval_protocol_identity"].get(
            "retrieval_protocol_id"
        )
    )
    binding_contract = (
        posterior_binding_contract(posterior_binding)
        if posterior_binding is not None and (identity_opt_in or has_stable_identity)
        else None
    )
    if binding_contract is not None:
        retrieval_sha = binding_contract["retrieval_protocol_identity"].get(
            "source_artifact_sha256"
        )
        if retrieval_sha and retrieval_sha != sha256_file(retrieval):
            raise ValueError("KCBWDM retrieval SHA differs from posterior retrieval identity")

    if args.method_name == KCBWDM_SIGNED_V1_METHOD:
        frozen_contract = KCBWDM_SIGNED_V1_CONTRACT
        assert_frozen_contract = assert_frozen_kcbwdm_contract
        build_teacher_row = build_kcbwdm_signed_teacher_row
        teacher_statistics = kcbwdm_teacher_statistics
        teacher_schema = "rag_kcbwdm_preformal_signed_teacher.v1"
        trajectory_implementation = (
            "src.diagnostics.kcbwdm_signed_teacher_v1."
            "build_kcbwdm_signed_teacher_row"
        )
    elif args.method_name == KCBWDM_LINEAR_GATE_V2_METHOD:
        frozen_contract = KCBWDM_LINEAR_GATE_V2_CONTRACT
        assert_frozen_contract = assert_frozen_kcbwdm_linear_gate_v2_contract
        build_teacher_row = build_kcbwdm_linear_gate_v2_teacher_row
        teacher_statistics = kcbwdm_linear_gate_v2_statistics
        teacher_schema = "rag_kcbwdm_linear_gate_v2_teacher_manifest.v1"
        trajectory_implementation = (
            "src.diagnostics.kcbwdm_linear_gate_v2."
            "build_kcbwdm_linear_gate_v2_teacher_row"
        )
    else:
        frozen_contract = KCBWDM_NORMALIZED_RHO_V1_CONTRACT
        assert_frozen_contract = assert_frozen_kcbwdm_normalized_rho_v1_contract
        build_teacher_row = build_kcbwdm_normalized_rho_v1_teacher_row
        teacher_statistics = kcbwdm_normalized_rho_v1_statistics
        teacher_schema = "rag_kcbwdm_normalized_rho_v1_teacher_manifest.v1"
        trajectory_implementation = (
            "src.diagnostics.kcbwdm_normalized_rho_v1."
            "build_kcbwdm_normalized_rho_v1_teacher_row"
        )
    frozen = frozen_contract["teacher"]
    kernel_contract = frozen_contract["kernel"]
    params = {
        "top_m": frozen["top_m"],
        "stop_threshold": frozen["stop_threshold"],
        "alignment_eps": frozen["alignment_eps"],
        "b_plus": frozen["b_plus"],
        "b_minus": frozen["b_minus"],
        "neutral_sample_policy": frozen["neutral_sample_policy"],
        "ridge_lambda": float(kernel_contract["ridge_lambda"]),
        "eps_smooth": float(config["cbwdm"].get("eps_smooth", 0)),
        "l_type": config["cbwdm"].get(
            "L_type", "euclidean_posterior_shift"
        ),
        "target_smoothing": config["cbwdm"].get(
            "target_smoothing", "paper_mixture"
        ),
        "gain_tolerance": float(frozen["gain_tolerance"]),
        "kernel": args.kernel,
        "sigma": None,
        "kernel_anchor": kernel_contract.get("anchor", "zero_effect"),
        "sign_policy": frozen_contract["sign_policy"],
        "lambda_policy": kernel_contract["lambda_policy"],
        "target_normalization": kernel_contract["target_normalization"],
        "set_dependent_centering": kernel_contract["set_dependent_centering"],
    }
    if args.method_name == KCBWDM_NORMALIZED_RHO_V1_METHOD:
        params.update(
            {
                "rho": float(args.rho),
                "kernel_family": kernel_contract["family"],
                "kernel_formula_version": kernel_contract["formula_version"],
                "normalization_policy": kernel_contract["normalization_policy"],
                "fit_scope": kernel_contract["fit_scope"],
                "lambda_policy": kernel_contract["lambda_policy"],
            }
        )
    assert_frozen_contract(
        {
            "method": args.method_name,
            "top_m": params["top_m"],
            "teacher_stop_threshold": params["stop_threshold"],
            "alignment_eps": params["alignment_eps"],
            "b_plus": params["b_plus"],
            "b_minus": params["b_minus"],
            "neutral_sample_policy": params["neutral_sample_policy"],
            "gain_tolerance": params["gain_tolerance"],
            "base_kernel": params["kernel"],
            "anchor": params["kernel_anchor"],
            "bandwidth_policy": kernel_contract["bandwidth_policy"],
            "ridge_lambda": params["ridge_lambda"],
            "lambda_policy": params["lambda_policy"],
            "target_normalization": params["target_normalization"],
            "set_dependent_centering": params["set_dependent_centering"],
            "sign_policy": params["sign_policy"],
            "kernel_family": params.get("kernel_family"),
            "kernel_formula_version": params.get("kernel_formula_version"),
            "normalization_policy": params.get("normalization_policy"),
            "fit_scope": params.get("fit_scope"),
            "rho": params.get("rho"),
        }
    )
    if float(config["cbwdm"]["ridge_lambda"]) != params["ridge_lambda"]:
        raise ValueError("Generator config ridge_lambda conflicts with KCBWDM contract")
    if float(config["cbwdm"].get("gain_tolerance", 1e-10)) != params[
        "gain_tolerance"
    ]:
        raise ValueError("Generator config gain_tolerance conflicts with KCBWDM contract")

    all_source_rows = list(read_jsonl(posterior))
    if not all_source_rows or {row.get("split") for row in all_source_rows} != {
        args.training_split
    }:
        raise ValueError(
            f"KCBWDM teacher input must contain only {args.training_split} rows"
        )
    source_rows = (
        all_source_rows[: args.max_rows]
        if args.max_rows is not None
        else all_source_rows
    )
    effect_groups = _effect_groups(source_rows, params)
    bandwidth_fit = fit_train_core_bandwidth(effect_groups, split=args.training_split)
    bandwidth_metadata = {
        **bandwidth_fit.to_dict(),
        "input_posterior_row_count": len(all_source_rows),
        "fitted_row_count": len(source_rows),
        "source_row_limit": args.max_rows,
        "fit_scope": (
            "smoke_only_limited_train_core"
            if args.max_rows is not None
            else "full_train_core"
        ),
        "reusable_for_full_development": args.max_rows is None,
    }
    params["sigma"] = bandwidth_fit.sigma
    params["bandwidth_policy"] = bandwidth_fit.policy
    params["bandwidth_provenance"] = bandwidth_metadata
    scale_fit = None
    scale_metadata = None
    if args.method_name == KCBWDM_NORMALIZED_RHO_V1_METHOD:
        scale_fit = _fit_and_verify_kernel_scale(
            effect_groups, sigma=bandwidth_fit.sigma, split=args.training_split
        )
        scale_metadata = {
            **scale_fit.to_dict(),
            "fit_scope": "full_train_core",
            "input_posterior_row_count": len(all_source_rows),
            "fitted_row_count": len(source_rows),
            "source_row_limit": args.max_rows,
            "reusable_for_full_development": True,
        }
        params.update(
            {
                "linear_diag_median_positive": scale_fit.linear_diag_median_positive,
                "rbf_diag_median_positive": scale_fit.rbf_diag_median_positive,
                "kernel_scale_c": scale_fit.kernel_scale_c,
                "kernel_scale_provenance": scale_metadata,
            }
        )

    contract = {
        "method": args.method_name,
        "stage": "teacher_training_only",
        "split": args.training_split,
        "config_sha256": sha256_file(config_path),
        "posterior_sha256": sha256_file(posterior),
        "retrieval_sha256": sha256_file(retrieval),
        "parameters": params,
        "method_contract_version": frozen_contract["contract_version"],
        "uses_gold_for_teacher": True,
        "evaluation_eligible": False,
        "calibration_eligible": False,
    }
    if args.method_name == KCBWDM_NORMALIZED_RHO_V1_METHOD:
        contract["rho"] = params["rho"]
    if binding_contract is not None:
        contract["posterior_binding"] = binding_contract
    fingerprint = stable_hash(contract)
    teacher_path = output / "teacher.jsonl"
    stats_path = output / "statistics.json"
    manifest_path = output / "manifest.json"
    if args.resume and all(
        path.is_file() for path in (teacher_path, stats_path, manifest_path)
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("fingerprint") == fingerprint
            and manifest.get("teacher_sha256") == sha256_file(teacher_path)
        ):
            print(f"[{args.method_name}_teacher] reused=true output={output}")
            return
        raise ValueError("Cannot resume KCBWDM teacher: fingerprint/checksum mismatch")
    if any(path.exists() for path in (teacher_path, stats_path, manifest_path)):
        raise FileExistsError("KCBWDM teacher artifacts exist; use matching --resume")

    rows = [build_teacher_row(row, params) for row in source_rows]
    _write_jsonl(teacher_path, rows)
    atomic_write_json(stats_path, teacher_statistics(rows))
    teacher_manifest = {
        "schema_version": teacher_schema,
        "status": "completed",
        "completed": True,
        "fingerprint": fingerprint,
        "contract": contract,
        "method": args.method_name,
        "num_rows": len(rows),
        "teacher_sha256": sha256_file(teacher_path),
        "statistics_sha256": sha256_file(stats_path),
        "posterior_sha256": contract["posterior_sha256"],
        "fitted_sigma": bandwidth_fit.sigma,
        "bandwidth_provenance": bandwidth_metadata,
        "trajectory_implementation": trajectory_implementation,
        "git": git_state(PROJECT_ROOT),
        "completed_at": utc_now(),
    }
    if scale_fit is not None:
        teacher_manifest.update(
            {
                "method_contract_version": frozen_contract["contract_version"],
                "rho": params["rho"],
                "normalization_policy": scale_fit.normalization_policy,
                "linear_diag_median_positive": scale_fit.linear_diag_median_positive,
                "rbf_diag_median_positive": scale_fit.rbf_diag_median_positive,
                "kernel_scale_c": scale_fit.kernel_scale_c,
                "kernel_scale_provenance": scale_metadata,
                "kernel_formula_version": params["kernel_formula_version"],
            }
        )
    if binding_contract is not None:
        teacher_manifest.update(
            {
                "identity_mode": (
                    "formal_v2" if args.formal_v2_identity else "legacy_compatible"
                ),
                "posterior_binding": binding_contract,
                "dataset_identity": binding_contract["dataset_identity"],
                "generator_identity": binding_contract["generator_identity"],
                "retrieval_protocol_identity": binding_contract[
                    "retrieval_protocol_identity"
                ],
                "generator_id": binding_contract["generator_identity"].get(
                    "generator_id"
                ),
            }
        )
    atomic_write_json(manifest_path, teacher_manifest)
    print(f"[{args.method_name}_teacher] rows={len(rows)} output={output}")


if __name__ == "__main__":
    main()

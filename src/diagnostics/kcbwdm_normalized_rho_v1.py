"""Normalized-rho KCBWDM teacher with an immutable linear directional gate."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.cbwdm_score import build_local_effects
from src.diagnostics.signed_teacher_v1 import (
    SignedTrainingGroup,
    build_signed_training_groups,
    supervision_for_candidate,
    teacher_statistics,
)
from src.kcbwdm_score import (
    normalized_rho_linear_gate_signed_greedy,
    normalized_rho_marginal_gain,
    normalized_rho_set_score,
    normalized_rho_target_alignments,
)


KCBWDM_NORMALIZED_RHO_V1_METHOD = "kcbwdm_normalized_rho_v1"
KCBWDM_NORMALIZED_RHO_V1_TEACHER_SCHEMA = (
    "rag_kcbwdm_normalized_rho_v1_teacher.v1"
)


def kcbwdm_normalized_rho_v1_statistics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result = teacher_statistics(rows)
    result.pop("old_teacher_ranking_skipped_ratio_reference", None)
    result["method"] = KCBWDM_NORMALIZED_RHO_V1_METHOD
    result["rho_values"] = sorted({float(row["rho"]) for row in rows})
    return result


def _local_effects(row: dict[str, Any], params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    candidates = list(row.get("candidates", []))
    labels = list(row["labels"])
    candidate_etas = (
        np.asarray([candidate["eta"] for candidate in candidates])
        if candidates
        else np.empty((0, len(labels)), dtype=np.float64)
    )
    return build_local_effects(
        np.asarray(row["eta0"]),
        candidate_etas,
        str(row["label"]),
        labels,
        params["l_type"],
        params["eps_smooth"],
        params["target_smoothing"],
    )


def build_kcbwdm_normalized_rho_v1_teacher_row(
    row: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Build teacher states under normalized utility and the linear sign gate."""
    candidates = list(row.get("candidates", []))
    labels = list(row["labels"])
    X, d = _local_effects(row, params)
    rho = float(params["rho"])
    sigma = float(params["sigma"])
    linear_scale = float(params["linear_diag_median_positive"])
    rbf_scale = float(params["rbf_diag_median_positive"])
    alignments = np.asarray(X @ d, dtype=np.float64)
    kernel_alignments = normalized_rho_target_alignments(
        X,
        d,
        sigma=sigma,
        rho=rho,
        linear_diag_median_positive=linear_scale,
        rbf_diag_median_positive=rbf_scale,
    )
    admissible = alignments > float(params["alignment_eps"])
    selected: list[int] = []
    states: list[dict[str, Any]] = []
    stop_reason = (
        "top_m_reached" if int(params["top_m"]) == 0 else "no_admissible_candidates"
    )

    for step in range(int(params["top_m"])):
        remaining = [index for index in range(len(candidates)) if index not in selected]
        if not remaining:
            stop_reason = "no_admissible_candidates"
            break
        theta_before = normalized_rho_set_score(
            X,
            d,
            selected,
            float(params["ridge_lambda"]),
            sigma=sigma,
            rho=rho,
            linear_diag_median_positive=linear_scale,
            rbf_diag_median_positive=rbf_scale,
        )
        candidate_rows: list[dict[str, Any]] = []
        for index in remaining:
            marginal = normalized_rho_marginal_gain(
                X,
                d,
                selected,
                index,
                float(params["ridge_lambda"]),
                sigma=sigma,
                rho=rho,
                linear_diag_median_positive=linear_scale,
                rbf_diag_median_positive=rbf_scale,
                numerical_tolerance=float(params["gain_tolerance"]),
            )
            raw_gain = float(marginal.gain)
            if raw_gain < -float(params["gain_tolerance"]):
                raise FloatingPointError(
                    f"Negative normalized-rho KCBWDM marginal gain: {raw_gain}"
                )
            gain = (
                0.0
                if abs(raw_gain) <= float(params["gain_tolerance"])
                else raw_gain
            )
            supervision = supervision_for_candidate(
                alignment=float(alignments[index]),
                gain=gain,
                alignment_eps=float(params["alignment_eps"]),
                b_plus=float(params["b_plus"]),
                b_minus=float(params["b_minus"]),
                neutral_sample_policy=str(params["neutral_sample_policy"]),
            )
            candidate = candidates[index]
            candidate_rows.append(
                {
                    "candidate_id": str(candidate["doc_id"]),
                    "index": index,
                    "original_bm25_rank": candidate.get("rank"),
                    "eta_j": candidate["eta"],
                    "x_j": X[index].tolist(),
                    "alignment": float(alignments[index]),
                    "linear_alignment": float(alignments[index]),
                    "kernel_target_alignment": float(kernel_alignments[index]),
                    "alignment_sign": (
                        "positive"
                        if alignments[index] > params["alignment_eps"]
                        else "negative"
                        if alignments[index] < -params["alignment_eps"]
                        else "zero"
                    ),
                    "admissible": bool(admissible[index]),
                    "theta_marginal_gain": gain,
                    "raw_theta_marginal_gain": raw_gain,
                    "theta_after_add": float(marginal.theta_after_add),
                    "residual_target_alignment": float(
                        marginal.residual_target_alignment
                    ),
                    "residual_self_information": float(
                        marginal.residual_self_information
                    ),
                    **supervision,
                }
            )

        admissible_rows = [item for item in candidate_rows if item["admissible"]]
        selected_ids = [candidates[index]["doc_id"] for index in selected]
        if not admissible_rows:
            states.append(
                {
                    "step": step,
                    "ordered_selected_ids_before_state": selected_ids,
                    "selected_indices_before_state": list(selected),
                    "teacher_action": "STOP",
                    "selected_candidate_id": None,
                    "stop_reason": "no_admissible_candidates",
                    "theta_before": theta_before,
                    "theta_after": theta_before,
                    "best_gain": None,
                    "is_terminal_state": True,
                    "remaining_candidates": candidate_rows,
                }
            )
            stop_reason = "no_admissible_candidates"
            break

        best = max(
            admissible_rows,
            key=lambda item: (item["theta_marginal_gain"], -item["index"]),
        )
        if float(best["theta_marginal_gain"]) < float(params["stop_threshold"]):
            states.append(
                {
                    "step": step,
                    "ordered_selected_ids_before_state": selected_ids,
                    "selected_indices_before_state": list(selected),
                    "teacher_action": "STOP",
                    "selected_candidate_id": None,
                    "stop_reason": "gain_below_threshold",
                    "theta_before": theta_before,
                    "theta_after": theta_before,
                    "best_gain": float(best["theta_marginal_gain"]),
                    "best_candidate_id": best["candidate_id"],
                    "is_terminal_state": True,
                    "remaining_candidates": candidate_rows,
                }
            )
            stop_reason = "gain_below_threshold"
            break

        states.append(
            {
                "step": step,
                "ordered_selected_ids_before_state": selected_ids,
                "selected_indices_before_state": list(selected),
                "teacher_action": "SELECT",
                "selected_candidate_id": best["candidate_id"],
                "stop_reason": None,
                "theta_before": theta_before,
                "theta_after": best["theta_after_add"],
                "best_gain": best["theta_marginal_gain"],
                "is_terminal_state": False,
                "remaining_candidates": candidate_rows,
            }
        )
        selected.append(int(best["index"]))
        stop_reason = (
            "top_m_reached"
            if len(selected) >= int(params["top_m"])
            else "no_admissible_candidates"
        )

    result = {
        "schema_version": KCBWDM_NORMALIZED_RHO_V1_TEACHER_SCHEMA,
        "method": KCBWDM_NORMALIZED_RHO_V1_METHOD,
        "variant": "kcbwdm_normalized_rho_v1_teacher",
        "rho": rho,
        "diagnostic_only": True,
        "uses_gold_for_teacher": True,
        "deployable_teacher": False,
        "id": row["id"],
        "query": row["query"],
        "label": row["label"],
        "gold": row["label"],
        "split": row["split"],
        "labels": labels,
        "eta0": row["eta0"],
        "d_i": d.tolist(),
        "teacher_selected_doc_ids": [
            candidates[index]["doc_id"] for index in selected
        ],
        "teacher_selected_indices": selected,
        "states": states,
        "stop_reason": stop_reason,
        "theta_final": normalized_rho_set_score(
            X,
            d,
            selected,
            float(params["ridge_lambda"]),
            sigma=sigma,
            rho=rho,
            linear_diag_median_positive=linear_scale,
            rbf_diag_median_positive=rbf_scale,
        ),
        "parameters": dict(params),
    }
    oracle = normalized_rho_linear_gate_signed_greedy(
        X,
        d,
        top_m=int(params["top_m"]),
        sigma=sigma,
        rho=rho,
        linear_diag_median_positive=linear_scale,
        rbf_diag_median_positive=rbf_scale,
        ridge_lambda=float(params["ridge_lambda"]),
        stop_threshold=float(params["stop_threshold"]),
        alignment_eps=float(params["alignment_eps"]),
        gain_tolerance=float(params["gain_tolerance"]),
    )
    if selected != oracle["selected_indices"] or stop_reason != oracle["stop_reason"]:
        raise AssertionError(
            "Normalized-rho teacher drifted from its authoritative greedy helper"
        )
    return result


__all__ = [
    "KCBWDM_NORMALIZED_RHO_V1_METHOD",
    "KCBWDM_NORMALIZED_RHO_V1_TEACHER_SCHEMA",
    "SignedTrainingGroup",
    "build_kcbwdm_normalized_rho_v1_teacher_row",
    "build_signed_training_groups",
    "kcbwdm_normalized_rho_v1_statistics",
]

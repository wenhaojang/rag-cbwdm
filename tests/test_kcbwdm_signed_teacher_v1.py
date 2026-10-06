from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.diagnostics.kcbwdm_signed_teacher_v1 import (
    KCBWDM_SIGNED_TEACHER_SCHEMA,
    KCBWDM_SIGNED_V1_METHOD,
    build_kcbwdm_signed_teacher_row,
    build_signed_training_groups,
)
from src.diagnostics.signed_teacher_v1 import build_signed_teacher_row


ATOL = 1e-10
RTOL = 1e-9


def _params(**updates):
    values = {
        "top_m": 4,
        "stop_threshold": 0.001,
        "alignment_eps": 0.0,
        "b_plus": 0.01,
        "b_minus": 0.001,
        "neutral_sample_policy": "negative",
        "ridge_lambda": 0.01,
        "eps_smooth": 0.0,
        "l_type": "euclidean_posterior_shift",
        "target_smoothing": "paper_mixture",
        "gain_tolerance": 1e-10,
        "kernel": "linear",
        "sigma": None,
    }
    values.update(updates)
    return values


def _row(etas):
    return {
        "id": "q1",
        "query": "claim",
        "label": "SUPPORTS",
        "split": "train_core",
        "labels": ["SUPPORTS", "REFUTES"],
        "eta0": [0.5, 0.5],
        "candidates": [
            {
                "doc_id": f"d{index}",
                "rank": index + 1,
                "title": f"T{index}",
                "text": f"text {index}",
                "eta": eta,
            }
            for index, eta in enumerate(etas)
        ],
    }


def _assert_linear_teacher_equivalent(row, params) -> None:
    current = build_signed_teacher_row(row, params)
    kernel = build_kcbwdm_signed_teacher_row(row, params)

    assert kernel["teacher_selected_indices"] == current["teacher_selected_indices"]
    assert kernel["teacher_selected_doc_ids"] == current["teacher_selected_doc_ids"]
    assert kernel["stop_reason"] == current["stop_reason"]
    np.testing.assert_allclose(
        kernel["theta_final"], current["theta_final"], atol=ATOL, rtol=RTOL
    )
    assert len(kernel["states"]) == len(current["states"])

    for actual_state, expected_state in zip(kernel["states"], current["states"]):
        for field in (
            "step",
            "ordered_selected_ids_before_state",
            "selected_indices_before_state",
            "teacher_action",
            "selected_candidate_id",
            "stop_reason",
            "is_terminal_state",
        ):
            assert actual_state.get(field) == expected_state.get(field)
        for field in ("theta_before", "theta_after", "best_gain"):
            if expected_state.get(field) is None:
                assert actual_state.get(field) is None
            else:
                np.testing.assert_allclose(
                    actual_state[field], expected_state[field], atol=ATOL, rtol=RTOL
                )

        actual_candidates = actual_state["remaining_candidates"]
        expected_candidates = expected_state["remaining_candidates"]
        assert [item["candidate_id"] for item in actual_candidates] == [
            item["candidate_id"] for item in expected_candidates
        ]
        for actual, expected in zip(actual_candidates, expected_candidates):
            for field in (
                "index",
                "alignment_sign",
                "admissible",
                "signed_supervision_class",
                "effective_bce_target",
                "ranking_role",
            ):
                assert actual[field] == expected[field]
            for field in (
                "alignment",
                "theta_marginal_gain",
                "raw_theta_marginal_gain",
                "theta_after_add",
                "effective_training_gain",
            ):
                np.testing.assert_allclose(
                    actual[field], expected[field], atol=ATOL, rtol=RTOL
                )


@pytest.mark.parametrize(
    ("etas", "updates"),
    [
        (
            [[0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.7, 0.3]],
            {"stop_threshold": 0.0},
        ),
        ([[0.3, 0.7], [0.4, 0.6]], {}),
        ([[0.5001, 0.4999], [0.50005, 0.49995]], {"stop_threshold": 0.01}),
        ([[0.7, 0.3], [0.6, 0.4]], {"top_m": 1, "stop_threshold": 0.0}),
    ],
)
def test_linear_teacher_trajectory_and_supervision_match_signed_v1(etas, updates) -> None:
    _assert_linear_teacher_equivalent(_row(etas), _params(**updates))


def test_kcbwdm_teacher_has_independent_method_and_schema_identity() -> None:
    current = build_signed_teacher_row(_row([[0.7, 0.3]]), _params())
    kernel = build_kcbwdm_signed_teacher_row(_row([[0.7, 0.3]]), _params())
    assert kernel["method"] == KCBWDM_SIGNED_V1_METHOD == "kcbwdm_signed_v1"
    assert kernel["schema_version"] == KCBWDM_SIGNED_TEACHER_SCHEMA
    assert kernel["variant"] == "kcbwdm_signed_teacher_v1"
    assert kernel["method"] != current.get("method")
    assert kernel["schema_version"] != current["schema_version"]


def test_existing_group_builder_accepts_kcbwdm_teacher_payload(tmp_path: Path) -> None:
    row = _row([[0.3, 0.7]])
    teacher = build_kcbwdm_signed_teacher_row(row, _params())
    posterior_path = tmp_path / "posterior.jsonl"
    teacher_path = tmp_path / "teacher.jsonl"
    posterior_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    teacher_path.write_text(json.dumps(teacher) + "\n", encoding="utf-8")

    groups = build_signed_training_groups(
        teacher_path=teacher_path, posteriors_path=posterior_path
    )
    assert len(groups) == 1
    assert groups[0].is_terminal_state
    assert groups[0].supervision_classes == ["explicit_harmful_negative"]
    assert groups[0].effective_gains == [0.0]


def test_rbf_teacher_uses_anchored_signed_gate_and_reports_residuals() -> None:
    teacher = build_kcbwdm_signed_teacher_row(
        _row([[0.8, 0.2], [0.2, 0.8], [0.65, 0.35]]),
        _params(kernel="rbf", sigma=1.0, stop_threshold=0.0),
    )
    first_state = teacher["states"][0]
    candidates = first_state["remaining_candidates"]
    assert {item["alignment_sign"] for item in candidates} >= {
        "positive",
        "negative",
    }
    assert all(np.isfinite(item["residual_target_alignment"]) for item in candidates)
    assert all(item["residual_self_information"] > 0 for item in candidates)
    assert teacher["teacher_selected_indices"]

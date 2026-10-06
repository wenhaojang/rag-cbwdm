from __future__ import annotations

import numpy as np
import pytest

from src.diagnostics.kcbwdm_linear_gate_v2 import (
    KCBWDM_LINEAR_GATE_V2_METHOD,
    KCBWDM_LINEAR_GATE_V2_TEACHER_SCHEMA,
    build_kcbwdm_linear_gate_v2_teacher_row,
)
from src.diagnostics.method_failure import signed_gated_greedy
from src.diagnostics.signed_teacher_v1 import build_signed_teacher_row
from src.kcbwdm_score import (
    anchored_gram,
    fit_train_core_bandwidth,
    kernel_marginal_gain,
    kernel_set_score,
    kernel_target_alignments,
    linear_gate_kernel_signed_greedy,
)


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
def test_v2a_linear_teacher_exactly_matches_signed_v1(etas, updates) -> None:
    row = _row(etas)
    params = _params(**updates)
    expected = build_signed_teacher_row(row, params)
    actual = build_kcbwdm_linear_gate_v2_teacher_row(row, params)

    assert actual["teacher_selected_indices"] == expected["teacher_selected_indices"]
    assert actual["teacher_selected_doc_ids"] == expected["teacher_selected_doc_ids"]
    assert actual["stop_reason"] == expected["stop_reason"]
    np.testing.assert_allclose(
        actual["theta_final"], expected["theta_final"], atol=ATOL, rtol=RTOL
    )
    assert len(actual["states"]) == len(expected["states"])
    for actual_state, expected_state in zip(actual["states"], expected["states"]):
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
        for actual_candidate, expected_candidate in zip(
            actual_state["remaining_candidates"],
            expected_state["remaining_candidates"],
        ):
            for field in (
                "index",
                "alignment_sign",
                "admissible",
                "signed_supervision_class",
                "effective_bce_target",
                "ranking_role",
            ):
                assert actual_candidate[field] == expected_candidate[field]
            for field in (
                "alignment",
                "theta_marginal_gain",
                "raw_theta_marginal_gain",
                "theta_after_add",
                "effective_training_gain",
            ):
                np.testing.assert_allclose(
                    actual_candidate[field],
                    expected_candidate[field],
                    atol=ATOL,
                    rtol=RTOL,
                )


def test_v2a_linear_greedy_matches_current_random_trajectories() -> None:
    for seed in range(50):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(8, 3))
        d = rng.normal(size=3)
        expected = signed_gated_greedy(
            X, d, top_m=4, ridge_lambda=0.01, stop_threshold=0.001
        )
        actual = linear_gate_kernel_signed_greedy(
            X,
            d,
            top_m=4,
            ridge_lambda=0.01,
            kernel="linear",
            stop_threshold=0.001,
        )
        assert actual["selected_indices"] == expected["selected_indices"]
        assert actual["stop_reason"] == expected["stop_reason"]
        assert actual["admissible"] == expected["admissible"]
        assert [step["best_index"] for step in actual["steps"]] == [
            step["best_index"] for step in expected["steps"]
        ]
        np.testing.assert_allclose(
            actual["theta_final"], expected["theta_final"], atol=ATOL, rtol=RTOL
        )


def test_rbf_v2a_gate_rejects_kernel_positive_opposite_direction() -> None:
    X = np.array([[-0.3, 0.3], [0.3, -0.3], [0.0, 0.0]])
    d = np.array([0.5, -0.5])
    kernel_alignments = kernel_target_alignments(X, d, kernel="rbf", sigma=0.01)
    assert X[0] @ d <= 0
    assert kernel_alignments[0] > 0

    result = linear_gate_kernel_signed_greedy(
        X,
        d,
        top_m=4,
        ridge_lambda=0.01,
        kernel="rbf",
        sigma=0.01,
        stop_threshold=0.0,
    )
    assert result["admissible"] == [False, True, False]
    assert 0 not in result["selected_indices"]
    assert 1 in result["selected_indices"]


def test_rbf_v2a_gate_admits_positive_linear_direction_with_tiny_kernel_signal() -> None:
    X = np.array([[0.001, -0.001]])
    d = np.array([0.5, -0.5])
    kernel_alignment = kernel_target_alignments(
        X, d, kernel="rbf", sigma=1_000_000.0
    )[0]
    assert X[0] @ d > 0
    assert abs(kernel_alignment) < 1e-8

    result = linear_gate_kernel_signed_greedy(
        X,
        d,
        top_m=1,
        ridge_lambda=0.01,
        kernel="rbf",
        sigma=1_000_000.0,
        stop_threshold=0.0,
    )
    assert result["admissible"] == [True]


def test_rbf_v2a_teacher_records_linear_gate_and_kernel_relevance() -> None:
    teacher = build_kcbwdm_linear_gate_v2_teacher_row(
        _row([[0.2, 0.8], [0.8, 0.2], [0.5, 0.5]]),
        _params(kernel="rbf", sigma=0.01, stop_threshold=0.0),
    )
    candidates = teacher["states"][0]["remaining_candidates"]
    assert candidates[0]["linear_alignment"] <= 0
    assert candidates[0]["kernel_target_alignment"] > 0
    assert candidates[0]["admissible"] is False
    assert candidates[1]["linear_alignment"] > 0
    assert candidates[1]["admissible"] is True
    assert candidates[2]["linear_alignment"] == 0
    assert candidates[2]["admissible"] is False
    assert teacher["method"] == KCBWDM_LINEAR_GATE_V2_METHOD
    assert teacher["schema_version"] == KCBWDM_LINEAR_GATE_V2_TEACHER_SCHEMA


def test_rbf_v2a_set_score_uses_zero_anchored_kernel() -> None:
    X = np.array([[0.3, -0.3], [0.2, -0.2]])
    d = np.array([0.5, -0.5])
    sigma = 0.4
    ridge = 0.01
    gram = anchored_gram(X, kernel="rbf", sigma=sigma)
    relevance = kernel_target_alignments(X, d, kernel="rbf", sigma=sigma)
    expected = float(relevance @ np.linalg.solve(gram + ridge * np.eye(2), relevance))
    actual = kernel_set_score(
        X, d, [0, 1], ridge, kernel="rbf", sigma=sigma
    )
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)


def test_rbf_v2a_schur_matches_brute_force_with_duplicates() -> None:
    X = np.array([[0.3, -0.3], [0.3, -0.3], [0.2, -0.2]])
    d = np.array([0.5, -0.5])
    before = kernel_set_score(X, d, [0, 1], 0.01, kernel="rbf", sigma=0.4)
    after = kernel_set_score(X, d, [0, 1, 2], 0.01, kernel="rbf", sigma=0.4)
    marginal = kernel_marginal_gain(
        X, d, [0, 1], 2, 0.01, kernel="rbf", sigma=0.4
    )
    assert np.isfinite(before)
    assert np.isfinite(after)
    assert marginal.residual_self_information > 0
    np.testing.assert_allclose(marginal.gain, after - before, atol=ATOL, rtol=RTOL)


def test_zero_effect_is_never_admitted_or_selected() -> None:
    X = np.array([[0.0, 0.0], [0.3, -0.3]])
    d = np.array([0.5, -0.5])
    result = linear_gate_kernel_signed_greedy(
        X,
        d,
        top_m=4,
        ridge_lambda=0.01,
        kernel="rbf",
        sigma=0.4,
        stop_threshold=0.0,
    )
    assert result["admissible"][0] is False
    assert 0 not in result["selected_indices"]
    assert kernel_set_score(X, d, [0], 0.01, kernel="rbf", sigma=0.4) == 0.0


def test_v2a_bandwidth_fit_semantics_are_unchanged() -> None:
    groups = [np.array([[0.0], [2.0]]), np.array([[1.0], [4.0]])]
    fit = fit_train_core_bandwidth(groups, split="train_core")
    assert fit.sigma == 2.5
    assert fit.policy == "train_core_within_query_positive_distance_median"
    assert fit.subsampling == "none"
    with pytest.raises(ValueError, match="train_core"):
        fit_train_core_bandwidth(groups, split="validation")

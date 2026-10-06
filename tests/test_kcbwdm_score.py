from __future__ import annotations

import numpy as np
import pytest

from src.cbwdm_score import marginal_gain, theta_for_indices
from src.diagnostics.method_failure import signed_gated_greedy
from src.kcbwdm_score import (
    anchored_gram,
    anchored_kernel,
    fit_train_core_bandwidth,
    kernel_marginal_gain,
    kernel_set_score,
    kernel_signed_greedy,
    kernel_target_alignments,
    linear_kernel,
    rbf_kernel,
)


ATOL = 1e-10
RTOL = 1e-9


def assert_close(actual, expected) -> None:
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    ("X", "d", "indices"),
    [
        (np.array([[1.0, 0.0]]), np.array([0.5, -0.2]), []),
        (np.array([[1.0, 0.0]]), np.array([0.5, -0.2]), [0]),
        (
            np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.25]]),
            np.array([0.5, -0.2]),
            [0, 2],
        ),
        (
            np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.array([0.5, -0.2]),
            [0, 1, 2],
        ),
        (
            np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
            np.array([0.5, -0.2]),
            [0, 1, 2],
        ),
    ],
)
def test_linear_set_score_matches_current_theta(X, d, indices) -> None:
    assert_close(
        kernel_set_score(X, d, indices, 0.01, kernel="linear"),
        theta_for_indices(X, d, list(indices), 0.01),
    )


def test_linear_set_score_matches_current_theta_random_fixtures() -> None:
    for seed in range(25):
        rng = np.random.default_rng(seed)
        rows = int(rng.integers(1, 9))
        width = int(rng.integers(1, 6))
        X = rng.normal(size=(rows, width))
        d = rng.normal(size=width)
        for size in range(rows + 1):
            indices = list(range(size))
            assert_close(
                kernel_set_score(X, d, indices, 0.01, kernel="linear"),
                theta_for_indices(X, d, indices, 0.01),
            )


def test_linear_schur_gain_matches_brute_force_and_current_marginal() -> None:
    rng = np.random.default_rng(101)
    for _ in range(20):
        X = rng.normal(size=(7, 4))
        d = rng.normal(size=4)
        for size in range(6):
            selected = list(range(size))
            candidate = size
            schur = kernel_marginal_gain(
                X, d, selected, candidate, 0.01, kernel="linear"
            )
            before = kernel_set_score(X, d, selected, 0.01, kernel="linear")
            after = kernel_set_score(
                X, d, selected + [candidate], 0.01, kernel="linear"
            )
            current_gain, current_after = marginal_gain(
                X, d, selected, candidate, 0.01
            )
            assert_close(schur.gain, after - before)
            assert_close(schur.gain, current_gain)
            assert_close(schur.theta_after_add, after)
            assert_close(schur.theta_after_add, current_after)
            assert schur.residual_self_information > 0


def test_linear_static_alignment_matches_x_dot_d_with_all_signs() -> None:
    X = np.array([[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])
    d = np.array([1.0, 0.0])
    actual = kernel_target_alignments(X, d, kernel="linear")
    expected = X @ d
    np.testing.assert_array_equal(actual, expected)
    assert ["positive" if value > 0 else "negative" if value < 0 else "zero" for value in actual] == [
        "positive",
        "zero",
        "negative",
    ]
    assert linear_kernel(X[0], d) == expected[0]
    assert anchored_kernel(X[2], d, kernel="linear") == expected[2]


def test_linear_candidate_ranking_and_tie_break_match_current() -> None:
    X = np.array(
        [
            [0.8, -0.2],
            [0.8, -0.2],
            [0.2, -0.8],
            [-0.4, 0.4],
        ]
    )
    d = np.array([0.7, -0.3])
    selected: list[int] = []
    current = []
    kernel = []
    for index in range(len(X)):
        current_gain, _ = marginal_gain(X, d, selected, index, 0.01)
        current.append((current_gain, -index, index))
        kernel_gain = kernel_marginal_gain(
            X, d, selected, index, 0.01, kernel="linear"
        ).gain
        kernel.append((kernel_gain, -index, index))
    assert [item[2] for item in sorted(current, reverse=True)] == [
        item[2] for item in sorted(kernel, reverse=True)
    ]
    # Duplicate candidates tie; the production key chooses the smaller index.
    assert max(kernel)[2] == 0


@pytest.mark.parametrize(
    ("X", "d", "top_m", "stop_threshold"),
    [
        (np.array([[0.8, -0.2], [0.4, -0.4], [-0.2, 0.2]]), np.array([0.7, -0.3]), 4, 0.0),
        (np.array([[-0.8, 0.2], [-0.4, 0.4]]), np.array([0.7, -0.3]), 4, 0.0),
        (np.array([[0.001, -0.001], [0.002, -0.002]]), np.array([0.7, -0.3]), 4, 1.0),
        (np.array([[0.8, -0.2], [0.8, -0.2], [0.4, -0.4]]), np.array([0.7, -0.3]), 1, 0.0),
    ],
)
def test_linear_greedy_stop_and_trajectory_match_current(
    X, d, top_m, stop_threshold
) -> None:
    current = signed_gated_greedy(
        X,
        d,
        top_m=top_m,
        ridge_lambda=0.01,
        stop_threshold=stop_threshold,
        alignment_eps=0.0,
        gain_tolerance=1e-10,
    )
    kernel = kernel_signed_greedy(
        X,
        d,
        top_m=top_m,
        ridge_lambda=0.01,
        kernel="linear",
        stop_threshold=stop_threshold,
        alignment_eps=0.0,
        gain_tolerance=1e-10,
    )
    assert kernel["selected_indices"] == current["selected_indices"]
    assert kernel["stop_reason"] == current["stop_reason"]
    assert_close(kernel["alignments"], current["alignments"])
    assert kernel["admissible"] == current["admissible"]
    assert_close(kernel["theta_final"], current["theta_final"])
    assert [step["best_index"] for step in kernel["steps"]] == [
        step["best_index"] for step in current["steps"]
    ]


def test_linear_random_full_greedy_trajectory_matches_current() -> None:
    for seed in range(50):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(8, 3))
        d = rng.normal(size=3)
        current = signed_gated_greedy(
            X, d, top_m=4, ridge_lambda=0.01, stop_threshold=0.001
        )
        kernel = kernel_signed_greedy(
            X,
            d,
            top_m=4,
            ridge_lambda=0.01,
            kernel="linear",
            stop_threshold=0.001,
        )
        assert kernel["selected_indices"] == current["selected_indices"]
        assert kernel["stop_reason"] == current["stop_reason"]


def test_linear_all_candidate_rankings_match_current_at_every_state() -> None:
    for seed in range(25):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(7, 4))
        d = rng.normal(size=4)
        selected: list[int] = []
        for _ in range(4):
            remaining = [index for index in range(len(X)) if index not in selected]
            current = []
            kernel = []
            for index in remaining:
                current_gain, _ = marginal_gain(X, d, selected, index, 0.01)
                kernel_gain = kernel_marginal_gain(
                    X, d, selected, index, 0.01, kernel="linear"
                ).gain
                current.append((current_gain, -index, index))
                kernel.append((kernel_gain, -index, index))
            assert [item[2] for item in sorted(current, reverse=True)] == [
                item[2] for item in sorted(kernel, reverse=True)
            ]
            selected.append(max(current)[2])


def test_anchored_rbf_gram_is_symmetric_psd_and_zero_effect_is_zero_feature() -> None:
    X = np.array([[0.0], [1.0], [-1.0], [0.25]])
    gram = anchored_gram(X, kernel="rbf", sigma=1.0)
    assert_close(gram, gram.T)
    assert_close(gram[0], np.zeros(len(X)))
    assert_close(gram[:, 0], np.zeros(len(X)))
    assert np.linalg.eigvalsh(gram).min() >= -1e-12


def test_raw_rbf_is_positive_but_anchored_target_alignment_has_both_signs() -> None:
    target = np.array([1.0])
    effects = np.array([[1.0], [-1.0]])
    assert all(rbf_kernel(row, target, sigma=1.0) > 0 for row in effects)
    alignments = kernel_target_alignments(
        effects, target, kernel="rbf", sigma=1.0
    )
    assert alignments[0] > 0
    assert alignments[1] < 0


def test_rbf_schur_gain_matches_brute_force_and_duplicate_atoms_are_finite() -> None:
    X = np.array([[0.5, -0.5], [0.5, -0.5], [-0.25, 0.25], [0.9, -0.9]])
    d = np.array([0.7, -0.7])
    selected = [0, 1]
    candidate = 3
    marginal = kernel_marginal_gain(
        X, d, selected, candidate, 0.01, kernel="rbf", sigma=0.4
    )
    before = kernel_set_score(
        X, d, selected, 0.01, kernel="rbf", sigma=0.4
    )
    after = kernel_set_score(
        X, d, selected + [candidate], 0.01, kernel="rbf", sigma=0.4
    )
    assert np.isfinite(before)
    assert np.isfinite(after)
    assert np.isfinite(marginal.gain)
    assert_close(marginal.gain, after - before)


def test_train_core_median_bandwidth_is_deterministic_and_split_guarded() -> None:
    groups = [np.array([[0.0], [2.0]]), np.array([[1.0], [4.0]])]
    first = fit_train_core_bandwidth(groups, split="train_core")
    second = fit_train_core_bandwidth(groups, split="train_core")
    assert first == second
    assert first.sigma == 2.5
    assert first.policy == "train_core_within_query_positive_distance_median"
    assert first.positive_distance_count == 2
    assert first.query_group_count == 2
    assert first.nonempty_query_group_count == 2
    assert first.source_split == "train_core"
    assert first.implementation_version == "kcbwdm_train_core_median_v1"
    assert first.subsampling == "none"
    with pytest.raises(ValueError, match="train_core"):
        fit_train_core_bandwidth(groups, split="validation")
    with pytest.raises(ValueError, match="no positive distances"):
        fit_train_core_bandwidth(
            [np.zeros((3, 2)), np.zeros((1, 2))], split="train_core"
        )

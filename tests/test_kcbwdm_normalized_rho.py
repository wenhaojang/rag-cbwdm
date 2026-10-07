from __future__ import annotations

import numpy as np
import pytest

from src.kcbwdm_score import (
    anchored_kernel,
    fit_kernel_scale_c,
    kernel_set_score,
    linear_gate_kernel_signed_greedy,
    linear_kernel,
    normalized_rho_gram,
    normalized_rho_kernel,
    normalized_rho_linear_gate_signed_greedy,
    normalized_rho_marginal_gain,
    normalized_rho_set_score,
)


SIGMA = 0.7
M_L = 0.05
M_R = 1.4


@pytest.mark.parametrize(
    ("linear_scale", "rbf_scale"),
    [
        (0.0, M_R),
        (-1.0, M_R),
        (np.nan, M_R),
        (np.inf, M_R),
        (M_L, 0.0),
        (M_L, -1.0),
        (M_L, np.nan),
        (M_L, np.inf),
    ],
)
def test_invalid_normalized_component_scales_fail_closed(
    linear_scale: float, rbf_scale: float
) -> None:
    with pytest.raises(ValueError, match="median_positive"):
        normalized_rho_kernel(
            np.array([1.0]),
            np.array([1.0]),
            sigma=SIGMA,
            rho=0.5,
            linear_diag_median_positive=linear_scale,
            rbf_diag_median_positive=rbf_scale,
        )


@pytest.mark.parametrize("rho", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_normalized_rho_fails_closed(rho: float) -> None:
    with pytest.raises(ValueError, match="rho"):
        normalized_rho_kernel(
            np.array([1.0]),
            np.array([1.0]),
            sigma=SIGMA,
            rho=rho,
            linear_diag_median_positive=M_L,
            rbf_diag_median_positive=M_R,
        )


def test_normalized_rho_endpoint_and_intermediate_formulas() -> None:
    u = np.array([0.4, -0.2])
    v = np.array([0.6, -0.4])
    linear = linear_kernel(u, v)
    rbf = anchored_kernel(u, v, kernel="rbf", sigma=SIGMA)
    assert normalized_rho_kernel(
        u,
        v,
        sigma=SIGMA,
        rho=0.0,
        linear_diag_median_positive=M_L,
        rbf_diag_median_positive=M_R,
    ) == linear / M_L
    assert normalized_rho_kernel(
        u,
        v,
        sigma=SIGMA,
        rho=1.0,
        linear_diag_median_positive=M_L,
        rbf_diag_median_positive=M_R,
    ) == rbf / M_R
    rho = 0.25
    expected = (1.0 - rho) * linear / M_L + rho * rbf / M_R
    np.testing.assert_allclose(
        normalized_rho_kernel(
            u,
            v,
            sigma=SIGMA,
            rho=rho,
            linear_diag_median_positive=M_L,
            rbf_diag_median_positive=M_R,
        ),
        expected,
        atol=1e-15,
        rtol=0.0,
    )


def test_normalized_rho_random_grams_are_psd() -> None:
    rng = np.random.default_rng(144)
    effects = rng.normal(size=(24, 5))
    for rho in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        gram = normalized_rho_gram(
            effects,
            sigma=SIGMA,
            rho=rho,
            linear_diag_median_positive=M_L,
            rbf_diag_median_positive=M_R,
        )
        np.testing.assert_allclose(gram, gram.T, atol=1e-14, rtol=0.0)
        assert float(np.min(np.linalg.eigvalsh(gram))) >= -1e-10


def test_normalized_linear_matches_equivalent_raw_policy() -> None:
    for seed in range(25):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(8, 3))
        d = rng.normal(size=3)
        normalized = normalized_rho_linear_gate_signed_greedy(
            X,
            d,
            top_m=4,
            sigma=SIGMA,
            rho=0.0,
            linear_diag_median_positive=M_L,
            rbf_diag_median_positive=M_R,
            ridge_lambda=0.01,
            stop_threshold=0.001,
        )
        raw = linear_gate_kernel_signed_greedy(
            X,
            d,
            top_m=4,
            kernel="linear",
            ridge_lambda=0.01 * M_L,
            stop_threshold=0.001 * M_L,
        )
        assert normalized["selected_indices"] == raw["selected_indices"]
        assert normalized["stop_reason"] == raw["stop_reason"]
        assert normalized["admissible"] == raw["admissible"]
        assert [step["best_index"] for step in normalized["steps"]] == [
            step["best_index"] for step in raw["steps"]
        ]
        np.testing.assert_allclose(
            normalized["theta_final"], raw["theta_final"] / M_L
        )


def test_normalized_rbf_matches_equivalent_raw_policy() -> None:
    for seed in range(25):
        rng = np.random.default_rng(100 + seed)
        X = rng.normal(size=(8, 3))
        d = rng.normal(size=3)
        normalized = normalized_rho_linear_gate_signed_greedy(
            X,
            d,
            top_m=4,
            sigma=SIGMA,
            rho=1.0,
            linear_diag_median_positive=M_L,
            rbf_diag_median_positive=M_R,
            ridge_lambda=0.01,
            stop_threshold=0.001,
        )
        raw = linear_gate_kernel_signed_greedy(
            X,
            d,
            top_m=4,
            kernel="rbf",
            sigma=SIGMA,
            ridge_lambda=0.01 * M_R,
            stop_threshold=0.001 * M_R,
        )
        assert normalized["selected_indices"] == raw["selected_indices"]
        assert normalized["stop_reason"] == raw["stop_reason"]
        assert normalized["admissible"] == raw["admissible"]
        assert [step["best_index"] for step in normalized["steps"]] == [
            step["best_index"] for step in raw["steps"]
        ]
        np.testing.assert_allclose(
            normalized["theta_final"], raw["theta_final"] / M_R
        )


def test_normalized_endpoint_set_scores_and_marginals_obey_raw_identities() -> None:
    X = np.array([[0.5, -0.5], [0.2, -0.2], [0.7, -0.7]])
    d = np.array([0.6, -0.4])
    for rho, kernel, component_scale in (
        (0.0, "linear", M_L),
        (1.0, "rbf", M_R),
    ):
        kwargs = {"kernel": kernel}
        if kernel == "rbf":
            kwargs["sigma"] = SIGMA
        raw_before = kernel_set_score(
            X, d, [0], 0.01 * component_scale, **kwargs
        )
        raw_after = kernel_set_score(
            X, d, [0, 1], 0.01 * component_scale, **kwargs
        )
        normalized_before = normalized_rho_set_score(
            X,
            d,
            [0],
            sigma=SIGMA,
            rho=rho,
            linear_diag_median_positive=M_L,
            rbf_diag_median_positive=M_R,
        )
        marginal = normalized_rho_marginal_gain(
            X,
            d,
            [0],
            1,
            sigma=SIGMA,
            rho=rho,
            linear_diag_median_positive=M_L,
            rbf_diag_median_positive=M_R,
        )
        np.testing.assert_allclose(
            normalized_before, raw_before / component_scale
        )
        np.testing.assert_allclose(
            marginal.gain, (raw_after - raw_before) / component_scale
        )


def test_normalized_rho_gate_is_always_linear() -> None:
    X = np.array([[-0.3, 0.3], [0.3, -0.3], [0.0, 0.0]])
    d = np.array([0.5, -0.5])
    for rho in (0.0, 0.5, 1.0):
        result = normalized_rho_linear_gate_signed_greedy(
            X,
            d,
            top_m=3,
            sigma=0.01,
            rho=rho,
            linear_diag_median_positive=M_L,
            rbf_diag_median_positive=M_R,
            stop_threshold=0.0,
        )
        assert result["alignments"] == (X @ d).tolist()
        assert result["admissible"] == [False, True, False]


def test_existing_scale_fit_supplies_normalized_scales_without_label_or_target() -> None:
    effects = np.array([[0.0], [1.0], [2.0]])
    fit = fit_kernel_scale_c(effects, sigma=1.0, split="train_core")
    assert fit.kernel_scale_c == (
        fit.linear_diag_median_positive / fit.rbf_diag_median_positive
    )
    assert fit.source_split == "train_core"
    assert fit.subsampling == "none"
    assert fit.linear_diag_median_positive > 0
    assert fit.rbf_diag_median_positive > 0

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnostics import audit_additive_kernel_scale as audit
from src import kcbwdm_score
from src.diagnostics.method_failure import signed_gated_greedy
from src.io_utils import write_jsonl
from src.kcbwdm_score import (
    KERNEL_SCALE_NORMALIZATION_POLICY,
    anchored_kernel,
    fit_kernel_scale_c,
    hybrid_gram,
    hybrid_kernel,
    hybrid_linear_gate_signed_greedy,
    hybrid_marginal_gain,
    hybrid_set_score,
    kernel_marginal_gain,
    kernel_set_score,
    linear_gate_kernel_signed_greedy,
    linear_kernel,
)


def test_hybrid_gram_is_numerically_psd() -> None:
    rng = np.random.default_rng(812)
    for alpha in (0.0, 0.1, 0.5, 1.0, 3.0):
        effects = rng.normal(size=(20, 5))
        gram = hybrid_gram(
            effects,
            sigma=0.7,
            alpha=alpha,
            kernel_scale_c=0.3,
        )
        np.testing.assert_allclose(gram, gram.T, atol=1e-14, rtol=0.0)
        assert float(np.min(np.linalg.eigvalsh(gram))) >= -1e-10


def test_alpha_zero_kernel_and_set_score_dispatch_exactly_to_linear() -> None:
    X = np.array([[0.8, -0.2], [0.4, -0.4], [-0.2, 0.2]])
    d = np.array([0.7, -0.3])
    assert hybrid_kernel(
        X[0], d, sigma=0.4, alpha=0.0, kernel_scale_c=2.5
    ) == linear_kernel(X[0], d)
    assert hybrid_set_score(
        X,
        d,
        [0, 1],
        0.01,
        sigma=0.4,
        alpha=0.0,
        kernel_scale_c=2.5,
    ) == kernel_set_score(X, d, [0, 1], 0.01, kernel="linear")
    assert hybrid_marginal_gain(
        X,
        d,
        [0],
        1,
        0.01,
        sigma=0.4,
        alpha=0.0,
        kernel_scale_c=2.5,
    ) == kernel_marginal_gain(X, d, [0], 1, 0.01, kernel="linear")


def test_alpha_zero_dispatch_never_evaluates_rbf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_rbf(*args, **kwargs):
        raise AssertionError("alpha=0 must not evaluate the RBF component")

    monkeypatch.setattr(kcbwdm_score, "anchored_kernel", reject_rbf)
    assert kcbwdm_score.hybrid_kernel(
        np.array([1.0]),
        np.array([2.0]),
        sigma=0.4,
        alpha=0.0,
        kernel_scale_c=1.0,
    ) == 2.0


def test_alpha_zero_complete_teacher_trajectories_rankings_and_stops_are_exact() -> None:
    for seed in range(50):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(8, 3))
        d = rng.normal(size=3)
        expected = linear_gate_kernel_signed_greedy(
            X,
            d,
            top_m=4,
            ridge_lambda=0.01,
            kernel="linear",
            stop_threshold=0.001,
            alignment_eps=0.0,
            gain_tolerance=1e-10,
        )
        actual = hybrid_linear_gate_signed_greedy(
            X,
            d,
            top_m=4,
            ridge_lambda=0.01,
            sigma=0.4,
            alpha=0.0,
            kernel_scale_c=1.7,
            stop_threshold=0.001,
            alignment_eps=0.0,
            gain_tolerance=1e-10,
        )
        signed_v1 = signed_gated_greedy(
            X,
            d,
            top_m=4,
            ridge_lambda=0.01,
            stop_threshold=0.001,
            alignment_eps=0.0,
            gain_tolerance=1e-10,
        )
        assert actual == expected
        assert actual["selected_indices"] == signed_v1["selected_indices"]
        assert actual["stop_reason"] == signed_v1["stop_reason"]
        assert actual["admissible"] == signed_v1["admissible"]
        assert [step["best_index"] for step in actual["steps"]] == [
            step["best_index"] for step in signed_v1["steps"]
        ]
        assert [step["best_index"] for step in actual["steps"]] == [
            step["best_index"] for step in expected["steps"]
        ]
        assert actual["stop_reason"] == expected["stop_reason"]


def test_positive_alpha_matches_hand_computed_formula_and_schur_score() -> None:
    X = np.array([[0.4, -0.2], [0.1, -0.3], [0.3, -0.1]])
    d = np.array([0.6, -0.4])
    sigma, alpha, scale = 0.7, 0.25, 1.8
    expected = (
        linear_kernel(X[0], d)
        + alpha
        * scale
        * anchored_kernel(X[0], d, kernel="rbf", sigma=sigma)
    ) / (1.0 + alpha)
    np.testing.assert_allclose(
        hybrid_kernel(
            X[0],
            d,
            sigma=sigma,
            alpha=alpha,
            kernel_scale_c=scale,
        ),
        expected,
        atol=1e-15,
        rtol=0.0,
    )
    before = hybrid_set_score(
        X,
        d,
        [0],
        sigma=sigma,
        alpha=alpha,
        kernel_scale_c=scale,
    )
    after = hybrid_set_score(
        X,
        d,
        [0, 1],
        sigma=sigma,
        alpha=alpha,
        kernel_scale_c=scale,
    )
    marginal = hybrid_marginal_gain(
        X,
        d,
        [0],
        1,
        sigma=sigma,
        alpha=alpha,
        kernel_scale_c=scale,
    )
    np.testing.assert_allclose(marginal.gain, after - before, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize(
    "updates",
    [
        {"alpha": -0.1},
        {"alpha": np.inf},
        {"kernel_scale_c": 0.0},
        {"kernel_scale_c": np.nan},
        {"sigma": 0.0},
        {"sigma": np.inf},
    ],
)
def test_invalid_hybrid_parameters_fail_closed(updates: dict[str, float]) -> None:
    params = {"sigma": 0.4, "alpha": 0.0, "kernel_scale_c": 1.0}
    params.update(updates)
    with pytest.raises(ValueError):
        hybrid_kernel(np.array([1.0]), np.array([1.0]), **params)


def test_positive_alpha_gate_remains_strictly_linear() -> None:
    X = np.array([[-0.3, 0.3], [0.3, -0.3], [0.0, 0.0]])
    d = np.array([0.5, -0.5])
    result = hybrid_linear_gate_signed_greedy(
        X,
        d,
        top_m=3,
        sigma=0.01,
        alpha=1.0,
        kernel_scale_c=1.0,
        stop_threshold=0.0,
    )
    assert result["alignments"] == (X @ d).tolist()
    assert result["admissible"] == [False, True, False]
    assert 0 not in result["selected_indices"]


def test_scale_fit_is_deterministic_label_free_and_uses_positive_medians() -> None:
    effects = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    first = fit_kernel_scale_c(effects, sigma=1.0, split="train_core")
    second = fit_kernel_scale_c(effects.copy(), sigma=1.0, split="train_core")
    assert first == second
    expected_rbf = np.array(
        [
            2.0 - 2.0 * np.exp(-0.5),
            2.0 - 2.0 * np.exp(-2.0),
        ],
        dtype=np.float64,
    )
    assert first.linear_diag_median_positive == 2.5
    np.testing.assert_allclose(
        first.rbf_diag_median_positive,
        np.median(expected_rbf),
        atol=1e-15,
        rtol=0.0,
    )
    assert first.kernel_scale_c == (
        first.linear_diag_median_positive / first.rbf_diag_median_positive
    )
    assert first.linear_diag_eligible_count == 2
    assert first.rbf_diag_eligible_count == 2
    assert first.linear_diag_near_zero_exclusion_count == 1
    assert first.rbf_diag_near_zero_exclusion_count == 1
    assert first.normalization_policy == KERNEL_SCALE_NORMALIZATION_POLICY
    assert set(inspect.signature(fit_kernel_scale_c).parameters) == {
        "effects",
        "sigma",
        "split",
    }


def test_scale_fit_rejects_non_train_core_and_degenerate_effects() -> None:
    with pytest.raises(ValueError, match="train_core"):
        fit_kernel_scale_c(np.array([[1.0]]), sigma=1.0, split="validation")
    with pytest.raises(ValueError, match="positive component diagonals"):
        fit_kernel_scale_c(np.zeros((3, 2)), sigma=1.0, split="train_core")


def test_ratio_summary_excludes_zero_near_zero_and_nonfinite_denominators() -> None:
    result = audit.summarize_ratio(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.0, 1e-16, 2.0, np.inf]),
    )
    assert result["ratio_eligible_count"] == 1
    assert result["excluded_zero_count"] == 1
    assert result["excluded_near_zero_count"] == 1
    assert result["excluded_nonfinite_count"] == 1
    assert result["count"] == 1
    assert result["mean"] == 1.5
    assert result["tau"] == max(
        1e-15,
        64.0 * np.finfo(np.float64).eps * 2.0,
    )


def test_audit_payload_records_scale_statistics_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "dataset_id": "fm2_official_closed_page_v1",
                "generator": {"generator_id": "qwen2.5-1.5b-instruct"},
                "cbwdm": {
                    "L_type": "euclidean_posterior_shift",
                    "eps_smooth": 0.001,
                    "target_smoothing": "paper_mixture",
                },
            }
        ),
        encoding="utf-8",
    )
    posterior = tmp_path / "posteriors.jsonl"
    write_jsonl(
        posterior,
        [
            {
                "id": "train:1",
                "query": "claim",
                "label": "SUPPORTS",
                "split": "train_core",
                "labels": ["SUPPORTS", "REFUTES"],
                "eta0": [0.5, 0.5],
                "candidates": [
                    {"doc_id": "d1", "eta": [0.7, 0.3]},
                    {"doc_id": "d2", "eta": [0.4, 0.6]},
                ],
            }
        ],
    )
    manifest = tmp_path / "posteriors.manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        audit,
        "validate_posterior_provenance",
        lambda *args, **kwargs: {
            "posterior_sha256": "posterior-sha",
            "manifest_sha256": "manifest-sha",
            "manifest_fingerprint": "manifest-fingerprint",
            "identity_fingerprint": "identity-fingerprint",
        },
    )
    monkeypatch.setattr(
        audit,
        "git_state",
        lambda project_root: {"commit": "test", "dirty": False},
    )
    payload = audit.build_audit_payload(
        posterior_path=posterior,
        posterior_manifest_path=manifest,
        config_path=config,
        sigma=0.4,
        generator_id="qwen2.5-1.5b-instruct",
        dataset_id="fm2_official_closed_page_v1",
    )
    assert payload["schema_version"] == audit.AUDIT_SCHEMA_VERSION
    assert payload["hybrid_kernel"] == {
        "description": "scale-preserving / scale-controlled hybrid kernel",
        "formula": "(u^T v + alpha * c * k0_rbf(u,v)) / (1 + alpha)",
        "directional_gate": "x_j^T d_i > alignment_eps",
        "effective_ridge_claim": "not_exactly_constant",
    }
    assert payload["split"] == "train_core"
    assert payload["row_count"] == payload["query_count"] == 1
    assert payload["candidate_count"] == 2
    assert payload["normalization_policy"] == KERNEL_SCALE_NORMALIZATION_POLICY
    assert payload["kernel_scale_c"] > 0
    assert payload["effect_construction_identity"]["helper"] == (
        "src.cbwdm_score.build_local_effects"
    )
    assert payload["effect_construction_identity"][
        "candidate_posterior_smoothing"
    ] is False
    assert payload["posterior_manifest"]["fingerprint"] == "manifest-fingerprint"
    assert payload["near_zero_policy"]["epsilon_replacement"] is False
    assert payload["near_zero_policy"]["winsorization"] is False
    assert set(payload["statistics"]) == {
        "linear_diag",
        "rbf_diag",
        "linear_target",
        "rbf_target",
        "rbf_diag_over_linear_diag",
        "abs_rbf_target_over_abs_linear_target",
    }

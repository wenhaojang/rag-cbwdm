from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from src.cbwdm_score import (
    EUCLIDEAN_L_TYPE,
    build_local_effects,
    canonical_l_type,
    greedy_teacher,
    marginal_gain,
    smooth_probability_vector,
    theta_for_indices,
)


class CBWDMScoreTests(unittest.TestCase):
 def test_smoothing_and_gold_direction_fever2_and_fever3(self) -> None:
    for labels, eta0 in [
        (["SUPPORTS", "REFUTES"], np.array([1.0, 0.0])),
        (["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"], np.array([0.8, 0.1, 0.1])),
    ]:
        eps = 0.1
        smoothed = smooth_probability_vector(eta0, eps)
        np.testing.assert_allclose(smoothed, (1 - eps) * eta0 + eps / len(labels))
        candidates = np.vstack([eta0, np.full(len(labels), 1 / len(labels))])
        X, d = build_local_effects(
            eta0,
            candidates,
            labels[0],
            labels,
            eps_smooth=eps,
            target_smoothing="paper_mixture",
        )
        expected_target = smooth_probability_vector(
            np.eye(len(labels))[0], eps
        )
        np.testing.assert_allclose(d, expected_target - smoothed)
        self.assertEqual(X.shape, (2, len(labels)))
        self.assertTrue(np.all(np.isfinite(X)))


 def test_eps_zero_exact_and_l_type_validation(self) -> None:
    p = np.array([0.2, 0.8])
    np.testing.assert_array_equal(smooth_probability_vector(p, 0.0), p)
    self.assertEqual(canonical_l_type("identity"), EUCLIDEAN_L_TYPE)
    with self.assertRaises(NotImplementedError):
        canonical_l_type("bw_local_hessian")
    with self.assertRaises(ValueError):
        canonical_l_type("unknown")


 def test_theta_closed_form_and_monotonic_gains(self) -> None:
  for ridge in [1e-10, 0.01, 1000.0]:
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    d = np.array([0.6, -0.2])
    self.assertEqual(theta_for_indices(X, d, [], ridge), 0.0)
    previous = 0.0
    selected: list[int] = []
    for idx in range(len(X)):
        gain, after = marginal_gain(X, d, selected, idx, ridge)
        self.assertGreaterEqual(after, previous - 1e-8)
        self.assertGreaterEqual(gain, -1e-8)
        selected.append(idx)
        previous = after


 def test_closed_form_matches_direct_quadratic_maximum(self) -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(3, 4))
    d = rng.normal(size=4)
    ridge = 0.3
    c = X @ d
    G = X @ X.T + ridge * np.eye(3)
    optimum = np.linalg.solve(G, c)
    # max_a 2 a^T c - a^T G a = c^T G^-1 c
    direct_value = 2 * optimum @ c - optimum @ G @ optimum
    # The associated generalized Rayleigh quotient has the same maximum.
    rayleigh_value = float((c @ optimum) ** 2 / (optimum @ G @ optimum))
    np.testing.assert_allclose(
        theta_for_indices(X, d, [0, 1, 2], ridge), direct_value, rtol=1e-10
    )
    np.testing.assert_allclose(direct_value, rayleigh_value, rtol=1e-10)


 def test_greedy_duplicate_orthogonal_zero_threshold_and_short_pool(self) -> None:
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    result = greedy_teacher(X, np.array([1.0, 0.5]), top_m=10, ridge_lambda=0.01)
    self.assertEqual(len(result["selected_indices"]), len(set(result["selected_indices"])))
    self.assertEqual(len(result["selected_indices"]), 4)
    stopped = greedy_teacher(
        X, np.array([1.0, 0.5]), top_m=4, ridge_lambda=0.01, stop_threshold=10.0
    )
    self.assertEqual(stopped["selected_indices"], [])
    self.assertEqual(stopped["stop_reason"], "gain_below_threshold")


 def test_teacher_schema_v2_fixture(self) -> None:
    script = Path(__file__).parents[1] / "scripts" / "04_build_cbwdm_teacher.py"
    spec = importlib.util.spec_from_file_location("build_teacher_script", script)
    self.assertTrue(spec and spec.loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = {
        "id": "x",
        "query": "claim",
        "label": "SUPPORTS",
        "split": "train",
        "labels": ["SUPPORTS", "REFUTES"],
        "eta0": [0.5, 0.5],
        "candidates": [
            {"doc_id": "d1", "rank": 1, "title": "t", "text": "e", "eta": [0.8, 0.2]}
        ],
    }
    params = {
        "top_m": 2,
        "ridge_lambda": 0.01,
        "stop_threshold": 0.0,
        "eps_smooth": 0.001,
        "l_type": EUCLIDEAN_L_TYPE,
        "target_smoothing": "paper_mixture",
        "gain_tolerance": 1e-10,
        "store_all_gains": True,
    }
    output = module.build_teacher_row(row, params)
    self.assertEqual(output["schema_version"], "rag_cbwdm_teacher.v2")
    self.assertEqual(output["teacher_type"], "cbwdm_greedy_gold_label")
    self.assertEqual(output["l_type"], EUCLIDEAN_L_TYPE)
    self.assertTrue(output["steps"][0]["stop_decision"]["selected"])

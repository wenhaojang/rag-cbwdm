from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.cbwdm_score import build_local_effects, theta_for_indices
from src.diagnostics.method_failure import (PosteriorCache, joint_posterior_greedy,
    require_diagnostic_output, signed_admissible, signed_gated_greedy)


class MethodFailureDiagnosticsTest(unittest.TestCase):
    def test_binary_theta_is_sign_blind_for_equal_magnitude_shift(self) -> None:
        eta0=np.asarray([0.5,0.5]);etas=np.asarray([[0.7,0.3],[0.3,0.7]])
        X,d=build_local_effects(eta0,etas,"SUPPORTS",["SUPPORTS","REFUTES"])
        self.assertGreater(float(X[0]@d),0);self.assertLess(float(X[1]@d),0)
        self.assertAlmostEqual(theta_for_indices(X,d,[0],.01),theta_for_indices(X,d,[1],.01),places=12)

    def test_signed_gate_admissibility(self) -> None:
        np.testing.assert_array_equal(signed_admissible(np.asarray([.1,-.1,0]),0),[True,False,False])

    def test_signed_greedy_rejects_large_harmful_shift(self) -> None:
        X=np.asarray([[.2,-.2],[-.8,.8],[.4,-.4]])
        d=np.asarray([.5,-.5])
        result=signed_gated_greedy(X,d,top_m=2,ridge_lambda=.01)
        self.assertNotIn(1,result["selected_indices"])
        self.assertEqual(result["selected_indices"][0],2)
        self.assertTrue(all(result["alignments"][i]>0 for i in result["selected_indices"]))

    def test_joint_greedy_positive_stop_and_top_m(self) -> None:
        values={(0,):[.6,.4],(1,):[.8,.2],(2,):[.4,.6],
                (1,0):[.85,.15],(1,2):[.7,.3],(1,0,2):[.84,.16]}
        result=joint_posterior_greedy(gold_index=0,eta_empty=[.5,.5],num_candidates=3,
            score_states=lambda states:[values[tuple(s)] for s in states],top_m=3)
        self.assertEqual(result["selected_indices"],[1,0])
        self.assertEqual(result["stop_reason"],"no_positive_gold_marginal")
        capped=joint_posterior_greedy(gold_index=0,eta_empty=[.5,.5],num_candidates=3,
            score_states=lambda states:[values[tuple(s)] for s in states],top_m=1)
        self.assertEqual(capped["selected_indices"],[1]);self.assertEqual(capped["stop_reason"],"top_m_reached")

    def test_artifact_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);good=run/"artifacts/diagnostics/method_failure_audit/signed_gate"
            self.assertEqual(require_diagnostic_output(run,good),good.resolve())
            with self.assertRaises(ValueError):require_diagnostic_output(run,run/"artifacts/formal/teacher.jsonl")
            with self.assertRaises(ValueError):require_diagnostic_output(run,run/"anything_else")

    def test_cache_resume_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"cache.jsonl";cache=PosteriorCache(path,"a");calls=0
            if cache.get("state") is None:
                calls+=1;cache.put("state",[.7,.3])
            cache.put("state",[.7,.3])
            resumed=PosteriorCache(path,"a",resume=True)
            self.assertEqual(resumed.get("state"),[.7,.3])
            if resumed.get("state") is None:
                calls+=1
            self.assertEqual(calls,1,"a completed cached state must not be scored again")
            with self.assertRaises(ValueError):PosteriorCache(path,"b",resume=True)
            with self.assertRaises(FileExistsError):PosteriorCache(path,"a",resume=False)


if __name__=="__main__":unittest.main()

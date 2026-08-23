from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.cbwdm_score import build_local_effects
from src.diagnostics.method_failure import require_diagnostic_output, signed_gated_greedy
from src.diagnostics.signed_selector_v1 import greedy_select_without_gold
from src.diagnostics.signed_teacher_v1 import (build_signed_teacher_row,
    build_signed_training_groups, supervision_for_candidate)
from src.run_manifest import stable_hash, validate_resume_manifest


def _params(**updates):
    values={"top_m":4,"stop_threshold":.001,"alignment_eps":0.0,"b_plus":.01,"b_minus":.001,
        "neutral_sample_policy":"negative","ridge_lambda":.01,"eps_smooth":0.0,
        "l_type":"euclidean_posterior_shift","target_smoothing":"paper_mixture","gain_tolerance":1e-10}
    values.update(updates);return values


def _row(etas):
    return {"id":"q1","query":"claim","label":"SUPPORTS","split":"train_core","labels":["SUPPORTS","REFUTES"],
        "eta0":[.5,.5],"candidates":[{"doc_id":f"d{i}","rank":i+1,"title":f"T{i}","text":f"text {i}","eta":eta}
        for i,eta in enumerate(etas)]}


class SignedTeacherV1Tests(unittest.TestCase):
    def test_harmful_high_unsigned_gain_is_explicit_negative(self):
        item=supervision_for_candidate(alignment=-.5,gain=.9,alignment_eps=0,b_plus=.01,b_minus=.001)
        self.assertEqual(item["signed_supervision_class"],"explicit_harmful_negative")
        self.assertEqual(item["effective_bce_target"],0);self.assertEqual(item["ranking_role"],"negative")
        self.assertEqual(item["effective_training_gain"],0.0)

    def test_admissible_uses_production_threshold_contract(self):
        self.assertEqual(supervision_for_candidate(alignment=.1,gain=.02,alignment_eps=0,b_plus=.01,b_minus=.001)["signed_supervision_class"],"positive")
        self.assertEqual(supervision_for_candidate(alignment=.1,gain=.005,alignment_eps=0,b_plus=.01,b_minus=.001)["signed_supervision_class"],"neutral")
        self.assertEqual(supervision_for_candidate(alignment=.1,gain=.0005,alignment_eps=0,b_plus=.01,b_minus=.001)["signed_supervision_class"],"negative")

    def test_action_trajectory_matches_signed_gate(self):
        row=_row([[.7,.3],[.3,.7],[.6,.4]]);params=_params(stop_threshold=0)
        teacher=build_signed_teacher_row(row,params)
        X,d=build_local_effects(np.asarray(row["eta0"]),np.asarray([c["eta"] for c in row["candidates"]]),row["label"],row["labels"])
        oracle=signed_gated_greedy(X,d,top_m=4,ridge_lambda=.01,stop_threshold=0)
        self.assertEqual(teacher["teacher_selected_indices"],oracle["selected_indices"])
        self.assertEqual(teacher["stop_reason"],oracle["stop_reason"])

    def test_zero_admissible_creates_terminal_step0_group(self):
        teacher=build_signed_teacher_row(_row([[.3,.7],[.4,.6]]),_params())
        self.assertEqual(len(teacher["states"]),1);state=teacher["states"][0]
        self.assertTrue(state["is_terminal_state"]);self.assertEqual(state["teacher_action"],"STOP")
        self.assertEqual(state["stop_reason"],"no_admissible_candidates")
        self.assertTrue(all(c["signed_supervision_class"]=="explicit_harmful_negative" for c in state["remaining_candidates"]))

    def test_gain_below_threshold_creates_terminal_group(self):
        teacher=build_signed_teacher_row(_row([[.5001,.4999],[.50005,.49995]]),_params(stop_threshold=.01))
        self.assertEqual(teacher["states"][0]["stop_reason"],"gain_below_threshold")
        self.assertTrue(teacher["states"][0]["is_terminal_state"])

    def test_top_m_has_no_extra_semantic_stop_group(self):
        teacher=build_signed_teacher_row(_row([[.7,.3],[.6,.4]]),_params(top_m=1,stop_threshold=0))
        self.assertEqual(teacher["stop_reason"],"top_m_reached");self.assertEqual(len(teacher["states"]),1)
        self.assertFalse(teacher["states"][0]["is_terminal_state"])

    def test_terminal_all_negative_group_is_retained_for_bce(self):
        row=_row([[.3,.7]])
        teacher=build_signed_teacher_row(row,_params())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);posterior=root/"posterior.jsonl";teacher_path=root/"teacher.jsonl"
            posterior.write_text(json.dumps(row)+"\n",encoding="utf-8");teacher_path.write_text(json.dumps(teacher)+"\n",encoding="utf-8")
            groups=build_signed_training_groups(teacher_path=teacher_path,posteriors_path=posterior)
        self.assertEqual(len(groups),1);self.assertTrue(groups[0].is_terminal_state)
        self.assertEqual(groups[0].effective_gains,[0.0])
        self.assertEqual(groups[0].supervision_classes,["explicit_harmful_negative"])

    def test_inference_api_and_action_have_no_gold_inputs(self):
        names=set(inspect.signature(greedy_select_without_gold).parameters)
        self.assertFalse(names & {"gold","label","d_i","alignment"})
        candidates=[{"doc_id":"d1","rank":1},{"doc_id":"d2","rank":2}]
        result=greedy_select_without_gold(query="claim",candidates=candidates,
            score_remaining=lambda query,selected,remaining:[.2,.1][:len(remaining)],top_m=1,min_docs=0,score_threshold=0)
        self.assertEqual([d["doc_id"] for d in result["selected_docs"]],["d1"])

    def test_min_docs_zero_negative_step0_returns_zero_docs(self):
        result=greedy_select_without_gold(query="claim",candidates=[{"doc_id":"d1","rank":1}],
            score_remaining=lambda query,selected,remaining:[-.2],top_m=4,min_docs=0,score_threshold=0)
        self.assertEqual(result["selected_docs"],[]);self.assertEqual(result["stop_reason"],"score_below_threshold")

    def test_artifact_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);good=run/"artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher"
            self.assertEqual(require_diagnostic_output(run,good),good.resolve())
            with self.assertRaises(ValueError):require_diagnostic_output(run,run/"artifacts/formal/teacher.jsonl")

    def test_resume_fingerprint_protection(self):
        first=stable_hash({"teacher":"a","max_groups":100});second=stable_hash({"teacher":"a","max_groups":None})
        self.assertNotEqual(first,second);validate_resume_manifest({"fingerprint":first},first,stage="signed")
        with self.assertRaises(ValueError):validate_resume_manifest({"fingerprint":first},second,stage="signed")


if __name__=="__main__":unittest.main()

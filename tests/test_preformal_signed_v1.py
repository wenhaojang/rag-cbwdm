from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from src.diagnostics.signed_selector_v1 import greedy_select_without_gold
from src.diagnostics.signed_teacher_v1 import build_signed_teacher_row
from src.formal_splits import publish_splits
from src.preformal.registry import (
    SIGNED_V1_CONTRACT,
    assert_frozen_signed_contract,
    assert_no_held_out_reference,
    assert_not_preformal_parameter_source,
)
from src.preformal.fairness import audit_preformal
from src.preformal.splits import build_strategy_a_rows, publish_strategy_a
from src.preformal.statistics import paired_comparison
from src.run_manifest import stable_hash
from src.run_manifest import sha256_file


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PreformalSplitTests(unittest.TestCase):
    def build_manifest(self, root: Path) -> Path:
        train = root / "train.jsonl"; dev = root / "dev.jsonl"
        rows = []
        for index in range(1, 13):
            rows.append({"id": index, "claim": f"unique claim {index}", "label": "SUPPORTS" if index % 2 else "REFUTES", "evidence": []})
        write_jsonl(train, rows)
        write_jsonl(dev, [{"id": 100 + index, "claim": f"held out {index}", "label": "SUPPORTS" if index % 2 else "REFUTES", "evidence": []} for index in range(1, 5)])
        output = root / "formal"
        publish_splits(train, dev, output, seed=13, validation_size=6, train_limit=4, validation_limit=2,
                       test_limit=None, project_root=Path(__file__).resolve().parents[1])
        return output / "fever2_formal_splits.manifest.json"

    def test_strategy_a_is_clean_and_uses_remaining_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.build_manifest(root)
            rows, audit = build_strategy_a_rows(manifest)
            self.assertEqual(audit["full_validation_rows"], 6)
            self.assertEqual(audit["pilot_validation_rows"], 2)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(value == 0 for value in audit["overlap_checks"].values()))
            published = publish_strategy_a(manifest, root / "preformal", project_root=Path(__file__).resolve().parents[1])
            self.assertEqual(published["row_count"], 4)
            self.assertEqual(published["source_strategy"], "A")
            self.assertFalse(published["held_out_test_consumed_for_modeling"])


class PreformalSafetyTests(unittest.TestCase):
    def test_preformal_cannot_calibrate_or_freeze_parameters(self) -> None:
        with self.assertRaises(ValueError): assert_not_preformal_parameter_source({"split": "preformal_eval"})
        with self.assertRaises(ValueError): assert_not_preformal_parameter_source("artifacts/preformal_signed_v1/results.json")

    def test_held_out_reference_fails_closed(self) -> None:
        with self.assertRaises(ValueError): assert_no_held_out_reference("/run/artifacts/formal/fever2_held_out_test_posteriors.jsonl")
        with self.assertRaises(ValueError): assert_no_held_out_reference({"stage": "posterior_test"})

    def test_signed_contract_rejects_parameter_changes(self) -> None:
        assert_frozen_signed_contract({"top_m": 4, "min_docs": 0, "score_threshold": 0.0})
        with self.assertRaises(ValueError): assert_frozen_signed_contract({"min_docs": 2})

    def test_seed_changes_training_fingerprint_and_same_seed_is_stable(self) -> None:
        base = {"method": "rag_cbwdm_signed_v1", "split_sha": "x", "model_sha": "y"}
        first = stable_hash({**base, "seed": 13}); resumed = stable_hash({**base, "seed": 13}); other = stable_hash({**base, "seed": 21})
        self.assertEqual(first, resumed); self.assertNotEqual(first, other)


class SignedRegressionTests(unittest.TestCase):
    def teacher_input(self) -> dict:
        return {"id":"q","query":"claim","label":"SUPPORTS","split":"train_core","labels":["SUPPORTS","REFUTES"],
                "eta0":[0.5,0.5],"candidates":[{"doc_id":"d1","rank":1,"eta":[0.8,0.2]},{"doc_id":"d2","rank":2,"eta":[0.2,0.8]}]}

    def test_formal_teacher_is_diagnostic_trajectory(self) -> None:
        params={"top_m":4,"stop_threshold":.001,"alignment_eps":0.0,"b_plus":.01,"b_minus":.001,
                "neutral_sample_policy":"negative","ridge_lambda":.01,"eps_smooth":.001,
                "l_type":"euclidean_posterior_shift","target_smoothing":"paper_mixture","gain_tolerance":1e-10}
        diagnostic=build_signed_teacher_row(self.teacher_input(),params)
        formal=build_signed_teacher_row(self.teacher_input(),params)
        self.assertEqual(formal["states"],diagnostic["states"])
        self.assertEqual(formal["teacher_selected_doc_ids"],diagnostic["teacher_selected_doc_ids"])

    def test_formal_inference_is_shared_and_has_no_gold_api(self) -> None:
        candidates=[{"doc_id":"d1","rank":1},{"doc_id":"d2","rank":2}]
        scorer=lambda query,selected,remaining:[1.0 if row["doc_id"]=="d1" else -1.0 for row in remaining]
        first=greedy_select_without_gold(query="claim",candidates=candidates,score_remaining=scorer,top_m=4,min_docs=0,score_threshold=0.0)
        second=greedy_select_without_gold(query="claim",candidates=candidates,score_remaining=scorer,top_m=4,min_docs=0,score_threshold=0.0)
        self.assertEqual(first,second);self.assertNotIn("gold",greedy_select_without_gold.__annotations__)


class PairedStatisticsTests(unittest.TestCase):
    def test_paired_comparison_requires_identical_ids(self) -> None:
        signed=[{"id":"a","gold":"SUPPORTS","pred":"SUPPORTS"}]
        baseline=[{"id":"b","gold":"SUPPORTS","pred":"SUPPORTS"}]
        with self.assertRaises(ValueError):paired_comparison(signed,baseline,samples=10)

    def test_paired_correctness_and_bootstrap_are_deterministic(self) -> None:
        signed=[{"id":"a","gold":"SUPPORTS","pred":"SUPPORTS"},{"id":"b","gold":"REFUTES","pred":"REFUTES"}]
        baseline=[{"id":"a","gold":"SUPPORTS","pred":"REFUTES"},{"id":"b","gold":"REFUTES","pred":"REFUTES"}]
        first=paired_comparison(signed,baseline,bootstrap_seed=7,samples=100)
        second=paired_comparison(signed,baseline,bootstrap_seed=7,samples=100)
        self.assertEqual(first,second);self.assertEqual(first["correctness_table"]["signed_only_correct"],1)


class FairnessContractTests(unittest.TestCase):
    def test_shared_retrieval_generator_prompt_and_verbalizer_shas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);split_rows=[{"id":"q","original_id":"1","query":"claim","label":"SUPPORTS","split":"preformal_eval"}]
            split_path=root/"split.jsonl";retrieval_path=root/"retrieval.jsonl";posterior_path=root/"posterior.jsonl";selection_path=root/"selection.jsonl";prediction_path=root/"predictions.jsonl";metrics_path=root/"metrics.json"
            write_jsonl(split_path,split_rows);write_jsonl(retrieval_path,[{"id":"q"}]);write_jsonl(posterior_path,[{"id":"q"}]);write_jsonl(selection_path,[{"id":"q","split":"preformal_eval","uses_gold_at_test":False}]);write_jsonl(prediction_path,[{"id":"q","gold":"SUPPORTS","pred":"SUPPORTS"}]);metrics_path.write_text("{}",encoding="utf-8")
            split_manifest=root/"split.manifest.json";retrieval_manifest=root/"retrieval.manifest.json";posterior_manifest=root/"posterior.manifest.json";selection_manifest=root/"selection.manifest.json";evaluation_manifest=root/"evaluation.manifest.json"
            split_manifest.write_text(json.dumps({"preformal_eval_sha256":sha256_file(split_path),"preformal_eval_path":str(split_path)}),encoding="utf-8")
            retrieval_manifest.write_text(json.dumps({"query_input_sha256":sha256_file(split_path),"output_sha256":sha256_file(retrieval_path)}),encoding="utf-8")
            posterior_manifest.write_text(json.dumps({"output_sha256":sha256_file(posterior_path),"provenance":{"input_sha256":sha256_file(retrieval_path),"prompt_template_hash":"p","verbalizers_hash":"v","generator_sha256":"g"}}),encoding="utf-8")
            selection_manifest.write_text(json.dumps({"stage":"selection","status":"completed","method":"naive_topm","output_path":str(selection_path),"output_sha256":sha256_file(selection_path),"contract":{"inputs":{"posteriors":{"sha256":sha256_file(posterior_path)}}}}),encoding="utf-8")
            evaluation_manifest.write_text(json.dumps({"stage":"evaluation","status":"completed","method":"naive_topm","predictions_path":str(prediction_path),"metrics_path":str(metrics_path),"contract":{"selection_sha256":sha256_file(selection_path),"split":"preformal_eval","prompt_hash":"p","verbalizer_hash":"v","generator_sha256":"g"}}),encoding="utf-8")
            result=audit_preformal(split_manifest_path=split_manifest,retrieval_manifest_path=retrieval_manifest,posterior_manifest_path=posterior_manifest,selection_manifests={"naive_topm:13":selection_manifest},evaluation_manifests={"naive_topm:13":evaluation_manifest},required_methods={"naive_topm"})
            self.assertEqual(result["status"],"comparable")
            payload=json.loads(evaluation_manifest.read_text(encoding="utf-8"));payload["contract"]["prompt_hash"]="changed";evaluation_manifest.write_text(json.dumps(payload),encoding="utf-8")
            blocked=audit_preformal(split_manifest_path=split_manifest,retrieval_manifest_path=retrieval_manifest,posterior_manifest_path=posterior_manifest,selection_manifests={"naive_topm:13":selection_manifest},evaluation_manifests={"naive_topm:13":evaluation_manifest},required_methods={"naive_topm"})
            self.assertEqual(blocked["status"],"blocked")


if __name__ == "__main__": unittest.main()

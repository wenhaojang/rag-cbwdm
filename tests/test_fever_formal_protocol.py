from __future__ import annotations

import json
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

import yaml

from src.calibration.fever import (
    calibrate,
    enumerate_parameter_grid,
    load_candidate_records,
    publish_calibration,
)
from src.cbwdm_diagnostics import build_diagnostics
from src.formal_config import (
    publish_frozen_config,
    reject_critical_cli_overrides,
    validate_frozen_manifest,
)
from src.formal_readiness import CANONICAL_METHODS, check_readiness
from src.formal_splits import publish_splits, validate_split_manifest
from src.run_manifest import sha256_file


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def raw_rows(start: int, per_label: int) -> list[dict]:
    rows = []
    for offset in range(per_label):
        rows.append(
            {
                "id": start + offset,
                "claim": f"support claim {start + offset}",
                "label": "SUPPORTS",
            }
        )
        rows.append(
            {
                "id": start + per_label + offset,
                "claim": f"refute claim {start + per_label + offset}",
                "label": "REFUTES",
            }
        )
    rows.append(
        {
            "id": start + 2 * per_label,
            "claim": f"nei claim {start}",
            "label": "NOT ENOUGH INFO",
        }
    )
    return rows


def calibration_config() -> dict:
    return {
        "calibration": {
            "infogain": {
                "teacher_quantiles": [[0.25, 0.75]],
                "filter_thresholds": [0.3, 0.5],
                "min_docs": [1],
                "top_m": [4],
                "beta": [0.75],
            },
            "cbwdm": {
                "stop_thresholds": [0.0],
                "score_thresholds": [None],
                "min_docs": [1],
                "top_m": [4],
                "beta": [0.5],
                "gamma": [1.0],
                "b_plus": [0.01],
                "b_minus": [0.001],
            },
        }
    }


def full_config() -> dict:
    config = calibration_config()
    config.update(
        {
            "dataset": "fever2",
            "seed": 13,
            "profile_limits": {"seeds": [13, 21, 42]},
            "task": {
                "labels": ["SUPPORTS", "REFUTES"],
                "verbalizers": {
                    "SUPPORTS": ["A", " A"],
                    "REFUTES": ["B", " B"],
                },
            },
            "retrieval": {
                "backend": "pyserini_lucene",
                "top_n": 20,
                "bm25": {"k1": 0.9, "b": 0.4},
            },
            "generator": {
                "max_context_tokens": 4096,
                "posterior_batch_size": 4,
                "dtype": "auto",
            },
            "cbwdm": {"stop_threshold": 0.0, "top_m": 4},
            "selector": {"max_length": 512, "top_m": 4},
            "baselines": {
                "bge": {},
                "infogain_fever": {"max_length": 512},
            },
        }
    )
    return config


class FormalSplitTests(unittest.TestCase):
    def test_deterministic_stratified_filter_then_limit_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            rows = raw_rows(1, 8)
            write_jsonl(train, rows)
            write_jsonl(dev, raw_rows(100, 3))
            first, _ = publish_splits(
                train,
                dev,
                root / "first",
                seed=13,
                validation_size=4,
                train_limit=6,
                validation_limit=2,
            )
            shuffled = list(rows)
            random.Random(7).shuffle(shuffled)
            write_jsonl(train, shuffled)
            second, _ = publish_splits(
                train,
                dev,
                root / "second",
                seed=13,
                validation_size=4,
                train_limit=6,
                validation_limit=2,
            )
            for role in ("train_core", "validation", "held_out_test"):
                self.assertEqual(
                    first["splits"][role]["id_sha256"],
                    second["splits"][role]["id_sha256"],
                )
            self.assertEqual(first["splits"]["train_core"]["num_rows"], 6)
            self.assertEqual(first["splits"]["validation"]["num_rows"], 2)
            self.assertEqual(first["splits"]["held_out_test"]["num_rows"], 6)
            self.assertTrue(all(value == 0 for value in first["overlap_checks"].values()))
            validate_split_manifest(root / "first" / "fever2_formal_splits.manifest.json")

    def test_duplicate_claim_and_source_change_refuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            write_jsonl(train, raw_rows(1, 4))
            write_jsonl(dev, raw_rows(100, 2))
            output = root / "splits"
            publish_splits(train, dev, output, seed=13, validation_size=2)
            with train.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"id": 999, "claim": "new", "label": "SUPPORTS"}) + "\n"
                )
            with self.assertRaisesRegex(ValueError, "source SHA|parameters changed"):
                publish_splits(
                    train,
                    dev,
                    output,
                    seed=13,
                    validation_size=2,
                    resume=True,
                )
            duplicate = raw_rows(1, 4)
            duplicate.append(
                {"id": 99, "claim": "  SUPPORT  claim 1 ", "label": "REFUTES"}
            )
            write_jsonl(train, duplicate)
            with self.assertRaisesRegex(ValueError, "Duplicate normalized claim"):
                publish_splits(
                    train,
                    dev,
                    root / "duplicates",
                    seed=13,
                    validation_size=2,
                )


class CalibrationAndFreezeTests(unittest.TestCase):
    def _split_fixture(self, root: Path) -> Path:
        train = root / "train.jsonl"
        dev = root / "dev.jsonl"
        write_jsonl(train, raw_rows(1, 6))
        write_jsonl(dev, raw_rows(100, 2))
        output = root / "splits"
        publish_splits(train, dev, output, seed=13, validation_size=4)
        return output / "fever2_formal_splits.manifest.json"

    def test_validation_only_missing_metric_tie_break_and_stable_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_path = self._split_fixture(root)
            split = validate_split_manifest(split_path)
            config = calibration_config()
            grid = enumerate_parameter_grid(config)
            self.assertEqual(grid, enumerate_parameter_grid(config))
            info_a, info_b = grid["infogain_fever"]
            cbwdm = grid["rag_cbwdm"][0]
            records = [
                {
                    "method": "infogain_fever",
                    "split": "validation",
                    "split_sha256": split["splits"]["validation"]["sha256"],
                    "parameters": info_a,
                    "metrics": {
                        "macro_f1": 0.8,
                        "accuracy": 0.8,
                        "avg_num_docs": 3,
                        "avg_evidence_chars": 30,
                    },
                },
                {
                    "method": "infogain_fever",
                    "split": "validation",
                    "split_sha256": split["splits"]["validation"]["sha256"],
                    "parameters": info_b,
                    "metrics": {
                        "macro_f1": 0.8,
                        "accuracy": 0.8,
                        "avg_num_docs": 2,
                        "avg_evidence_chars": 40,
                    },
                },
                {
                    "method": "rag_cbwdm",
                    "split": "validation",
                    "split_sha256": split["splits"]["validation"]["sha256"],
                    "parameters": cbwdm,
                    "metrics": {"accuracy": 0.7},
                },
            ]
            result = calibrate(config, split, records)
            self.assertEqual(result["selected"]["infogain_fever"]["parameters"], info_b)
            self.assertEqual(result["status"], "blocked")
            missing = result["candidates"][-1]
            self.assertIsNone(missing["macro_f1"])
            self.assertIn("missing macro_f1", missing["reason"])
            held_out = dict(records[0], split="held_out_test")
            with self.assertRaisesRegex(ValueError, "validation"):
                calibrate(config, split, [held_out])
            held_path = root / "held_out_test_metrics.json"
            held_path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "held_out_test"):
                load_candidate_records(held_path)

    def test_publish_freeze_hashes_and_critical_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_path = self._split_fixture(root)
            split = validate_split_manifest(split_path)
            config = full_config()
            grid = enumerate_parameter_grid(config)
            rows = []
            for method in ("infogain_fever", "rag_cbwdm"):
                rows.append(
                    {
                        "method": method,
                        "split": "validation",
                        "split_sha256": split["splits"]["validation"]["sha256"],
                        "parameters": grid[method][0],
                        "metrics": {
                            "macro_f1": 0.7,
                            "accuracy": 0.7,
                            "avg_num_docs": 2,
                            "avg_evidence_chars": 20,
                        },
                    }
                )
            config_path = root / "base.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            metrics_path = root / "metrics.json"
            metrics_path.write_text(json.dumps(rows), encoding="utf-8")
            calibration_dir = root / "calibration"
            manifest, _ = publish_calibration(
                config_path,
                split_path,
                metrics_path,
                calibration_dir,
                project_root=Path(__file__).resolve().parents[1],
            )
            self.assertEqual(manifest["status"], "completed")
            models = {}
            revisions = {}
            for name in ("generator", "tokenizer", "bge", "infogain", "rag_cbwdm"):
                path = root / f"{name}.bin"
                path.write_bytes(name.encode())
                models[name] = path
                revisions[name] = f"{name}-revision"
            corpus = root / "corpus.jsonl"
            corpus.write_text("{}\n", encoding="utf-8")
            index = root / "index.manifest.json"
            index.write_text(
                '{"status":"completed","fingerprint":"index-fingerprint"}\n',
                encoding="utf-8",
            )
            frozen_config, frozen_manifest, diff = publish_frozen_config(
                config_path,
                split_path,
                calibration_dir / "calibration.manifest.json",
                root / "generated",
                models=models,
                revisions=revisions,
                corpus=corpus,
                index=index,
                project_root=Path(__file__).resolve().parents[1],
            )
            payload = validate_frozen_manifest(frozen_manifest)
            self.assertEqual(payload["split_manifest_sha256"], sha256_file(split_path))
            self.assertEqual(
                payload["calibration_manifest_sha256"],
                sha256_file(calibration_dir / "calibration.manifest.json"),
            )
            self.assertTrue(frozen_config.is_file())
            self.assertTrue(diff.is_file())
            models["generator"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "generator"):
                validate_frozen_manifest(frozen_manifest)
            with self.assertRaisesRegex(ValueError, "critical CLI overrides"):
                reject_critical_cli_overrides(["--seed", "21"])


class DiagnosticTests(unittest.TestCase):
    def test_validation_retrieval_keeps_recall_keys_but_held_out_does_not(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "formal_retrieve_test", root / "scripts" / "02_retrieve_bm25.py"
        )
        self.assertTrue(spec and spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Retriever:
            def search(self, query: str, top_n: int) -> list[dict]:
                return [
                    {
                        "doc_id": "d1",
                        "title": "Page",
                        "text": "sentence",
                        "meta": {"page_id": "Page", "sentence_id": 2},
                    }
                ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            base = {
                "id": "q",
                "query": "claim",
                "label": "SUPPORTS",
                "meta": {"evidence": [[[0, 1, "Page", 2]]]},
            }
            write_jsonl(path, [{**base, "split": "validation"}])
            validation = list(module.iter_results(Retriever(), path, 20, None, []))[0]
            self.assertEqual(validation["gold_evidence_keys"], ["Page\t2"])
            write_jsonl(path, [{**base, "split": "held_out_test"}])
            held_out = list(module.iter_results(Retriever(), path, 20, None, []))[0]
            self.assertNotIn("gold_evidence_keys", held_out)

    def test_gain_stop_selection_flips_recall_and_label_mapping(self) -> None:
        ids = ["q1", "q2"]
        labels = {"q1": "SUPPORTS", "q2": "REFUTES"}

        def predictions(values: dict[str, str]) -> dict[str, dict]:
            return {
                identifier: {
                    "id": identifier,
                    "gold": labels[identifier],
                    "pred": values[identifier],
                    "probs": [0.8, 0.2],
                }
                for identifier in ids
            }

        selections = {
            identifier: {"id": identifier, "selected_doc_ids": [f"{identifier}-gold"]}
            for identifier in ids
        }
        teacher = {
            identifier: {
                "id": identifier,
                "schema_version": "rag_cbwdm_teacher.v2",
                "teacher_selected_doc_ids": [f"{identifier}-gold"],
                "stop_reason": "top_m_reached",
                "steps": [
                    {
                        "best_gain": 0.3,
                        "candidate_gains": [
                            {"gain": 0.3},
                            {"gain": 0.0},
                        ],
                    }
                ],
            }
            for identifier in ids
        }
        retrieval = {
            identifier: {
                "id": identifier,
                "gold_evidence_doc_ids": [f"{identifier}-gold"],
                "candidates": [{"doc_id": f"{identifier}-gold"}],
            }
            for identifier in ids
        }
        posteriors = {
            identifier: {
                "id": identifier,
                "eta0": [0.6, 0.4],
                "candidates": [
                    {"doc_id": f"{identifier}-gold", "eta": [0.9, 0.1]}
                ],
            }
            for identifier in ids
        }
        config = {
            "task": {
                "labels": ["SUPPORTS", "REFUTES"],
                "verbalizers": {"SUPPORTS": ["A"], "REFUTES": ["B"]},
            },
            "selector": {"top_m": 4},
            "cbwdm": {"gain_tolerance": 1e-10},
            "diagnostics": {
                "min_pilot_examples": 2,
                "oracle_naive_tolerance": 0.02,
            },
        }
        result = build_diagnostics(
            config=config,
            no_evidence=predictions({"q1": "SUPPORTS", "q2": "SUPPORTS"}),
            naive=predictions(labels),
            cbwdm=predictions(labels),
            oracle=predictions(labels),
            naive_selection=selections,
            cbwdm_selection=selections,
            oracle_selection=selections,
            teacher=teacher,
            retrieval=retrieval,
            posteriors=posteriors,
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["gains"]["sign_counts"], {"positive": 2, "negative": 0, "zero": 2})
        self.assertEqual(
            result["gold_evidence"]["selected_recall_at_4"]["value"], 1.0
        )
        self.assertFalse(result["gate"]["scientific_risk"])


class ReadinessTests(unittest.TestCase):
    def test_runner_exposes_all_formal_data_roles(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "formal_runner_test", root / "scripts" / "run_fever_cbwdm.py"
        )
        self.assertTrue(spec and spec.loader)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        self.assertTrue(
            {
                "prepare_formal_splits",
                "retrieve_train_core",
                "retrieve_validation",
                "retrieve_test",
                "posterior_train_core",
                "posterior_validation",
                "posterior_test",
                "calibrate_methods",
            }
            <= set(runner.ALL_STAGES)
        )
        self.assertNotIn("retrieve_test", runner.FORMAL_PILOT_STAGES)
        self.assertNotIn("posterior_test", runner.FORMAL_PILOT_STAGES)

    def test_nonzero_overlap_and_failed_diagnostics_block(self) -> None:
        # Focused aggregation behavior is covered without manufacturing every
        # frozen artifact here; each failed input must be converted to a blocker.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in (
                "split",
                "calibration",
                "frozen",
                "fairness",
                "summary",
                "diagnostics",
                "tests",
            ):
                path = root / f"{name}.json"
                path.write_text("{}\n", encoding="utf-8")
                paths[name] = path
            paths["diagnostics"].write_text(
                json.dumps(
                    {
                        "status": "blocked",
                        "gate": {
                            "status": "blocked",
                            "scientific_risk": True,
                            "blockers": ["oracle_significantly_below_naive"],
                        },
                        "git": {"commit": "x"},
                    }
                ),
                encoding="utf-8",
            )
            result = check_readiness(
                split_manifest_path=paths["split"],
                calibration_manifest_path=paths["calibration"],
                frozen_manifest_path=paths["frozen"],
                fairness_audit_path=paths["fairness"],
                baseline_summary_path=paths["summary"],
                diagnostics_path=paths["diagnostics"],
                tests_status_path=paths["tests"],
                verify_model_artifacts=False,
            )
            self.assertEqual(result["status"], "blocked")
            self.assertTrue(result["blockers"])
            self.assertIn(
                "cbwdm_pilot_diagnostic_gate",
                {item["check"] for item in result["blockers"]},
            )
            self.assertEqual(
                CANONICAL_METHODS,
                {
                    "no_evidence",
                    "naive_topm",
                    "bge",
                    "infogain_fever",
                    "rag_cbwdm",
                    "cbwdm_oracle",
                },
            )


if __name__ == "__main__":
    unittest.main()

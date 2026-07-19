from __future__ import annotations

import json
import importlib.util
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.calibration.fever import (
    calibrate,
    enumerate_parameter_grid,
    load_candidate_records,
    publish_calibration,
)
from src.calibration.grid import (
    CANDIDATE_SCHEMA_VERSION,
    PARAMETER_DEPENDENCIES,
    _common_record,
    _publish_candidates,
    _teacher_paths,
    build_grid_plan,
    dry_run_text,
    execute_grid,
    selected_plan_nodes,
)
from src.cbwdm_diagnostics import build_diagnostics
from src.formal_config import (
    publish_frozen_config,
    reject_critical_cli_overrides,
    validate_frozen_manifest,
)
from src.formal_readiness import CANONICAL_METHODS, check_readiness
from src.formal_splits import publish_splits, validate_split_manifest
from src.io_utils import read_jsonl
from src.run_manifest import sha256_file, stable_hash


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

    def test_conflicting_group_is_excluded_and_source_change_refuses_resume(self) -> None:
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
            conflict_output = root / "duplicates"
            manifest, _ = publish_splits(
                train,
                dev,
                conflict_output,
                seed=13,
                validation_size=2,
            )
            conflict_rows = list(
                read_jsonl(conflict_output / "conflicting_claim_groups.jsonl")
            )
            self.assertEqual(len(conflict_rows), 1)
            self.assertEqual(
                {record["id"] for record in conflict_rows[0]["records"]},
                {"1", "99"},
            )
            split_ids = {
                row["original_id"]
                for role in ("train_core", "validation", "held_out_test")
                for row in read_jsonl(conflict_output / f"{role}.jsonl")
            }
            self.assertFalse({"1", "99"} & split_ids)
            self.assertEqual(manifest["conflicting_label_group_count"], 1)
            self.assertEqual(manifest["conflicting_label_row_count"], 2)
            self.assertEqual(manifest["conflicting_train_group_count"], 1)
            self.assertEqual(manifest["conflicting_train_row_count"], 2)
            self.assertEqual(
                manifest["source_stats"]["official_train"][
                    "raw_fever2_row_count"
                ],
                9,
            )
            self.assertEqual(
                manifest["source_stats"]["official_train"][
                    "eligible_row_count"
                ],
                7,
            )
            self.assertEqual(
                manifest["conflicting_claims_sha256"],
                sha256_file(
                    conflict_output / "conflicting_claim_groups.jsonl"
                ),
            )
            validate_split_manifest(
                conflict_output / "fever2_formal_splits.manifest.json"
            )
            manifest_path = (
                conflict_output / "fever2_formal_splits.manifest.json"
            )
            original_manifest = manifest_path.read_text(encoding="utf-8")
            tampered = json.loads(original_manifest)
            tampered["contract"]["filter"]["conflict_policy_version"] = (
                "legacy.v0"
            )
            tampered["conflict_policy_version"] = "legacy.v0"
            tampered["fingerprint"] = stable_hash(tampered["contract"])
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "claim-group policy"):
                publish_splits(
                    train,
                    dev,
                    conflict_output,
                    seed=13,
                    validation_size=2,
                    resume=True,
                )
            manifest_path.write_text(original_manifest, encoding="utf-8")
            with (
                conflict_output / "conflicting_claim_groups.jsonl"
            ).open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "artifact SHA256 changed"):
                validate_split_manifest(
                    conflict_output / "fever2_formal_splits.manifest.json"
                )

    def test_duplicate_original_id_is_fatal_even_if_one_row_is_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            write_jsonl(
                train,
                [
                    {"id": 1, "claim": "filtered", "label": "NOT ENOUGH INFO"},
                    {"id": 1, "claim": "accepted", "label": "SUPPORTS"},
                ],
            )
            write_jsonl(dev, raw_rows(100, 2))
            with self.assertRaisesRegex(
                ValueError, "Duplicate original FEVER id '1'.*lines 1 and 2"
            ):
                publish_splits(
                    train,
                    dev,
                    root / "duplicate_ids",
                    seed=13,
                    validation_size=1,
                )

    def test_same_label_duplicate_group_is_kept_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            rows = raw_rows(1, 6)
            rows.append(
                {
                    "id": 999,
                    "claim": " SUPPORT   claim 1 ",
                    "label": "SUPPORTS",
                }
            )
            write_jsonl(train, rows)
            write_jsonl(dev, raw_rows(100, 2))
            manifest, _ = publish_splits(
                train,
                dev,
                root / "splits",
                seed=13,
                validation_size=4,
            )
            roles = {}
            for role in ("train_core", "validation"):
                for row in read_jsonl(root / "splits" / f"{role}.jsonl"):
                    roles[row["original_id"]] = role
            self.assertEqual(roles["1"], roles["999"])
            self.assertEqual(manifest["duplicate_claim_group_count"], 1)
            self.assertEqual(manifest["rows_in_duplicate_claim_groups"], 2)
            self.assertEqual(manifest["max_duplicate_group_size"], 2)
            self.assertEqual(manifest["conflicting_label_group_count"], 0)
            self.assertEqual(
                manifest["partition_unit"], "normalized_claim_group"
            )
            self.assertIn("keep_same_label", manifest["duplicate_policy"])

    def test_joint_label_allocation_reaches_exact_total_when_deviations_cancel(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            rows = []
            identifier = 1
            for claim, size, label in (
                ("support a", 2, "SUPPORTS"),
                ("support b", 2, "SUPPORTS"),
                ("support c", 2, "SUPPORTS"),
                ("refute a", 1, "REFUTES"),
                ("refute b", 3, "REFUTES"),
            ):
                for _ in range(size):
                    rows.append(
                        {"id": identifier, "claim": claim, "label": label}
                    )
                    identifier += 1
            write_jsonl(train, rows)
            write_jsonl(dev, raw_rows(100, 2))
            manifest, _ = publish_splits(
                train,
                dev,
                root / "splits",
                seed=13,
                validation_size=5,
            )
            self.assertEqual(manifest["requested_validation_size"], 5)
            self.assertEqual(manifest["actual_validation_size"], 5)
            self.assertEqual(manifest["validation_size_difference_rows"], 0)
            self.assertEqual(
                manifest["validation_label_actual_rows"],
                {"REFUTES": 3, "SUPPORTS": 2},
            )

    def test_group_row_target_order_independence_limits_and_policy_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            rows = [
                {"id": 1, "claim": "support a", "label": "SUPPORTS"},
                {"id": 2, "claim": " SUPPORT A ", "label": "SUPPORTS"},
                {"id": 3, "claim": "support b", "label": "SUPPORTS"},
                {"id": 4, "claim": "support  b", "label": "SUPPORTS"},
                {"id": 5, "claim": "refute a", "label": "REFUTES"},
                {"id": 6, "claim": " REFUTE A ", "label": "REFUTES"},
                {"id": 7, "claim": "refute b", "label": "REFUTES"},
                {"id": 8, "claim": "refute  b", "label": "REFUTES"},
            ]
            write_jsonl(train, rows)
            write_jsonl(dev, raw_rows(100, 2))
            first, _ = publish_splits(
                train,
                dev,
                root / "first",
                seed=13,
                validation_size=3,
                train_limit=1,
                validation_limit=1,
            )
            shuffled = list(rows)
            random.Random(99).shuffle(shuffled)
            write_jsonl(train, shuffled)
            second, _ = publish_splits(
                train,
                dev,
                root / "second",
                seed=13,
                validation_size=3,
                train_limit=1,
                validation_limit=1,
            )
            self.assertEqual(first["requested_validation_size"], 3)
            self.assertEqual(first["actual_validation_size"], 4)
            self.assertEqual(first["validation_size_difference_rows"], 1)
            self.assertEqual(
                first["actual_validation_size"], second["actual_validation_size"]
            )
            for role in ("train_core", "validation", "held_out_test"):
                self.assertEqual(
                    first["splits"][role]["id_sha256"],
                    second["splits"][role]["id_sha256"],
                )
            self.assertEqual(first["splits"]["train_core"]["num_rows"], 2)
            self.assertEqual(first["splits"]["validation"]["num_rows"], 2)
            self.assertTrue(all(value == 0 for value in first["overlap_checks"].values()))

            manifest_path = root / "first" / "fever2_formal_splits.manifest.json"
            tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
            tampered["contract"]["partition"]["duplicate_policy_version"] = "legacy.v1"
            tampered["duplicate_policy_version"] = "legacy.v1"
            tampered["fingerprint"] = stable_hash(tampered["contract"])
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            write_jsonl(train, rows)
            with self.assertRaisesRegex(
                ValueError, "source SHA|parameters changed|claim-group policy"
            ):
                publish_splits(
                    train,
                    dev,
                    root / "first",
                    seed=13,
                    validation_size=3,
                    train_limit=1,
                    validation_limit=1,
                    resume=True,
                )

    def test_cross_source_held_out_precedence_agree_and_disagree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            write_jsonl(train, raw_rows(1, 6))
            dev_rows = raw_rows(100, 2)
            dev_rows[0]["claim"] = " support CLAIM 1 "
            write_jsonl(dev, dev_rows)
            first, _ = publish_splits(
                train,
                dev,
                root / "agree",
                seed=13,
                validation_size=4,
            )
            train_ids = {
                row["original_id"]
                for role in ("train_core", "validation")
                for row in read_jsonl(root / "agree" / f"{role}.jsonl")
            }
            held_ids = {
                row["original_id"]
                for row in read_jsonl(root / "agree" / "held_out_test.jsonl")
            }
            self.assertNotIn("1", train_ids)
            self.assertIn("100", held_ids)
            self.assertEqual(first["cross_source_overlap_group_count"], 1)
            self.assertEqual(first["cross_source_train_row_count"], 1)
            self.assertEqual(first["cross_source_dev_row_count"], 1)
            self.assertEqual(
                first["train_rows_excluded_for_held_out_overlap"], 1
            )
            self.assertEqual(
                first["dev_rows_retained_after_overlap_resolution"], 1
            )
            self.assertEqual(
                first["cross_source_label_agreement_group_count"], 1
            )
            self.assertTrue(all(value == 0 for value in first["overlap_checks"].values()))
            self.assertEqual(
                first["cross_source_overlap_sha256"],
                sha256_file(
                    root
                    / "agree"
                    / "cross_source_overlap_groups.jsonl"
                ),
            )
            validate_split_manifest(
                root / "agree" / "fever2_formal_splits.manifest.json"
            )
            manifest_path = (
                root / "agree" / "fever2_formal_splits.manifest.json"
            )
            tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
            tampered["contract"]["filter"][
                "cross_source_overlap_policy_version"
            ] = "legacy.v0"
            tampered["cross_source_overlap_policy_version"] = "legacy.v0"
            tampered["fingerprint"] = stable_hash(tampered["contract"])
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "claim-group policy"):
                publish_splits(
                    train,
                    dev,
                    root / "agree",
                    seed=13,
                    validation_size=4,
                    resume=True,
                )

            dev_rows[0]["label"] = "REFUTES"
            write_jsonl(dev, dev_rows)
            second, _ = publish_splits(
                train,
                dev,
                root / "disagree",
                seed=13,
                validation_size=4,
            )
            self.assertEqual(
                second["cross_source_label_disagreement_group_count"], 1
            )
            retained = {
                row["original_id"]: row["label"]
                for row in read_jsonl(
                    root / "disagree" / "held_out_test.jsonl"
                )
            }
            self.assertEqual(retained["100"], "REFUTES")

    def test_dev_conflict_excludes_dev_group_and_train_counterpart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            train_rows = raw_rows(1, 6)
            train_rows[0]["claim"] = "shared conflict"
            train_rows.append(
                {
                    "id": 888,
                    "claim": " SHARED CONFLICT ",
                    "label": "SUPPORTS",
                }
            )
            dev_rows = raw_rows(100, 3)
            dev_rows[0].update(
                {"claim": "shared conflict", "label": "SUPPORTS"}
            )
            dev_rows[1].update(
                {"claim": " SHARED   CONFLICT ", "label": "REFUTES"}
            )
            write_jsonl(train, train_rows)
            write_jsonl(dev, dev_rows)
            manifest, _ = publish_splits(
                train,
                dev,
                root / "splits",
                seed=13,
                validation_size=4,
            )
            all_ids = {
                row["original_id"]
                for role in ("train_core", "validation", "held_out_test")
                for row in read_jsonl(root / "splits" / f"{role}.jsonl")
            }
            self.assertFalse({"1", "888", "100", "103"} & all_ids)
            self.assertEqual(manifest["conflicting_dev_group_count"], 1)
            self.assertEqual(manifest["conflicting_dev_row_count"], 2)
            cross = list(
                read_jsonl(
                    root / "splits" / "cross_source_overlap_groups.jsonl"
                )
            )
            self.assertEqual(cross[0]["dev_action"], "exclude_conflicting_group")
            self.assertEqual(
                {record["id"] for record in cross[0]["train_records"]},
                {"1", "888"},
            )

    def test_conflict_and_cross_source_membership_is_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            train_rows = raw_rows(1, 8)
            train_rows.append(
                {
                    "id": 999,
                    "claim": " support claim 2 ",
                    "label": "REFUTES",
                }
            )
            dev_rows = raw_rows(100, 3)
            dev_rows[0]["claim"] = "support claim 1"
            write_jsonl(train, train_rows)
            write_jsonl(dev, dev_rows)
            first, _ = publish_splits(
                train,
                dev,
                root / "first",
                seed=13,
                validation_size=6,
            )
            random.Random(17).shuffle(train_rows)
            random.Random(23).shuffle(dev_rows)
            write_jsonl(train, train_rows)
            write_jsonl(dev, dev_rows)
            second, _ = publish_splits(
                train,
                dev,
                root / "second",
                seed=13,
                validation_size=6,
            )
            for role in ("train_core", "validation", "held_out_test"):
                self.assertEqual(
                    first["splits"][role]["id_sha256"],
                    second["splits"][role]["id_sha256"],
                )

            def membership(path: Path) -> list[dict]:
                rows = list(read_jsonl(path))
                for item in rows:
                    for key in ("records", "train_records", "dev_records"):
                        for record in item.get(key, []):
                            record.pop("line", None)
                return rows

            for name in (
                "conflicting_claim_groups.jsonl",
                "cross_source_overlap_groups.jsonl",
            ):
                self.assertEqual(
                    membership(root / "first" / name),
                    membership(root / "second" / name),
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


class CalibrationGridTests(unittest.TestCase):
    def _plan(self, root: Path, output_name: str = "calibration_grid") -> dict:
        train = root / "train.jsonl"
        dev = root / "dev.jsonl"
        write_jsonl(train, raw_rows(1, 6))
        write_jsonl(dev, raw_rows(100, 2))
        split_dir = root / "splits"
        publish_splits(
            train,
            dev,
            split_dir,
            seed=13,
            validation_size=4,
            resume=True,
        )
        inputs = {}
        for name, split in (
            ("train_retrieval", "train_core"),
            ("validation_retrieval", "validation"),
            ("train_posteriors", "train_core"),
            ("validation_posteriors", "validation"),
        ):
            path = root / f"{name}.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "id": f"{split}-1",
                        "query": "claim",
                        "label": "SUPPORTS",
                        "split": split,
                        "candidates": [],
                    }
                ],
            )
            inputs[name] = path
        return build_grid_plan(
            config_path=Path(__file__).resolve().parents[1]
            / "configs"
            / "fever2_server_pilot_5000_500.yaml",
            split_manifest_path=split_dir / "fever2_formal_splits.manifest.json",
            train_retrieval_path=inputs["train_retrieval"],
            validation_retrieval_path=inputs["validation_retrieval"],
            train_posteriors_path=inputs["train_posteriors"],
            validation_posteriors_path=inputs["validation_posteriors"],
            output_dir=root / "artifacts" / "formal" / output_name,
            project_root=Path(__file__).resolve().parents[1],
        )

    def test_yaml_grid_dependency_layering_and_reuse_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._plan(root)
            self.assertEqual(
                plan["totals"],
                {
                    "teacher_candidates": 6,
                    "training_candidates": 24,
                    "selection_candidates": 108,
                },
            )
            self.assertEqual(
                PARAMETER_DEPENDENCIES["infogain_fever"]["filter_threshold"],
                ["selection"],
            )
            self.assertEqual(
                PARAMETER_DEPENDENCIES["rag_cbwdm"]["beta"], ["training"]
            )
            info_nodes = plan["methods"]["infogain_fever"][
                "training_candidates"
            ]
            self.assertEqual(len(info_nodes), 6)
            self.assertTrue(all(len(node["selections"]) == 6 for node in info_nodes))
            one = info_nodes[0]
            self.assertEqual(
                len(
                    {
                        item["training_fingerprint"]
                        for item in one["selections"]
                    }
                ),
                1,
            )
            self.assertGreater(
                len(
                    {
                        item["selection_fingerprint"]
                        for item in one["selections"]
                    }
                ),
                1,
            )
            canonical_record = _common_record(
                plan, one, one["selections"][0]
            )
            self.assertTrue(
                canonical_record["selection_manifest"].endswith(
                    ".grid.manifest.json"
                )
            )
            self.assertNotIn("cbwdm_oracle", plan["methods"])

    def test_plan_fingerprint_filters_and_dry_run_do_not_execute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._plan(root, "grid_a")
            second = self._plan(root, "grid_b")
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            nodes = selected_plan_nodes(
                first,
                methods={"infogain_fever"},
                candidate_limit=2,
                candidate_fingerprint=None,
                max_training_candidates=None,
            )
            self.assertEqual(len(nodes), 1)
            self.assertEqual(len(nodes[0]["selections"]), 2)
            rendered = dry_run_text(first, nodes)
            self.assertIn("no models will be loaded", rendered)
            self.assertIn("training_candidates=1 selection_candidates=2", rendered)
            self.assertIn("checkpoint_reuse=true", rendered)
            with patch(
                "src.calibration.grid._run_command",
                side_effect=AssertionError("dry-run executed a model command"),
            ):
                dry_result = execute_grid(
                    first,
                    config_path=first["inputs"]["config"]["path"],
                    project_root=Path(__file__).resolve().parents[1],
                    methods={"infogain_fever"},
                    candidate_limit=2,
                    dry_run=True,
                )
            self.assertEqual(dry_result["status"], "dry_run")
            with self.assertRaisesRegex(ValueError, "Unknown calibration candidate"):
                selected_plan_nodes(
                    first,
                    methods=None,
                    candidate_limit=None,
                    candidate_fingerprint="missing",
                    max_training_candidates=None,
                )

    def test_failed_candidate_schema_preserves_null_and_held_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._plan(root)
            node = plan["methods"]["infogain_fever"]["training_candidates"][0]
            selection = node["selections"][0]
            record = {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "method": "infogain_fever",
                "split": "validation",
                "candidate_fingerprint": selection["candidate_fingerprint"],
                "training_fingerprint": node["training_fingerprint"],
                "selection_fingerprint": selection["selection_fingerprint"],
                "parameters": selection["parameters"],
                "canonical_parameter_json": selection[
                    "canonical_parameter_json"
                ],
                "metrics": {
                    "accuracy": None,
                    "macro_f1": None,
                    "avg_num_docs": None,
                    "avg_evidence_chars": None,
                },
                "status": "failed",
                "reason": "simulated",
            }
            output_root = root / "aggregate"
            output_root.mkdir()
            manifest = _publish_candidates(
                plan,
                [record],
                output_root,
                execution_contract={
                    "training_candidate_count": 1,
                    "selection_candidate_count": 1,
                },
                project_root=Path(__file__).resolve().parents[1],
            )
            payload = json.loads(
                (output_root / "calibration_candidates.json").read_text(
                    encoding="utf-8"
                )
            )
            candidate = payload["candidates"][0]
            self.assertEqual(candidate["status"], "failed")
            self.assertIsNone(candidate["metrics"]["macro_f1"])
            self.assertEqual(manifest["status"], "completed_with_failures")

            held = root / "held_out_test_posteriors.jsonl"
            write_jsonl(held, [{"id": "x"}])
            with self.assertRaisesRegex(ValueError, "held_out_test"):
                build_grid_plan(
                    config_path=Path(__file__).resolve().parents[1]
                    / "configs"
                    / "fever2_server_pilot_5000_500.yaml",
                    split_manifest_path=plan["inputs"]["split_manifest"]["path"],
                    train_retrieval_path=plan["inputs"]["train_retrieval"]["path"],
                    validation_retrieval_path=plan["inputs"][
                        "validation_retrieval"
                    ]["path"],
                    train_posteriors_path=plan["inputs"]["train_posteriors"][
                        "path"
                    ],
                    validation_posteriors_path=held,
                    output_dir=root / "forbidden",
                    project_root=Path(__file__).resolve().parents[1],
                )

    def test_failed_teacher_manifest_retries_and_completed_outputs_reuse(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._plan(root)
            node = plan["methods"]["infogain_fever"]["training_candidates"][0]
            candidate = node["selections"][0]["candidate_fingerprint"]
            plan["output_dir"] = str(root / "g")
            node["directory"] = str(root / "n")
            node["selections"][0]["selection_path"] = str(
                root / "n" / "s.jsonl"
            )
            node["selections"][0]["evaluation_dir"] = str(root / "n" / "e")
            teacher_path, teacher_grid_manifest = _teacher_paths(
                Path(plan["output_dir"]),
                "infogain_fever",
                node["teacher_fingerprint"],
            )
            teacher_path.parent.mkdir(parents=True)
            teacher_grid_manifest.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "fingerprint": "previous-failed-fingerprint",
                        "reason": "train_core role was rejected",
                    }
                ),
                encoding="utf-8",
            )
            commands: list[list[str]] = []

            def fake_run(command, **kwargs):
                del kwargs
                commands.append(command)
                script = Path(command[1]).name
                if script == "12a_build_infogain_teacher.py":
                    output = Path(command[command.index("--output") + 1])
                    write_jsonl(
                        output,
                        [
                            {
                                "query_id": "q",
                                "doc_id": "positive",
                                "split": "train_core",
                                "teacher_purpose": "training",
                                "dig": 0.3,
                            },
                            {
                                "query_id": "q",
                                "doc_id": "negative",
                                "split": "train_core",
                                "teacher_purpose": "training",
                                "dig": -0.3,
                            },
                        ],
                    )
                    output.with_suffix(".manifest.json").write_text(
                        json.dumps(
                            {
                                "status": "completed",
                                "completed": True,
                                "teacher_role": "train_core",
                                "teacher_purpose": "training",
                                "training_eligible": True,
                                "thresholds": {
                                    "label_distribution": {
                                        "positive": 1,
                                        "negative": 1,
                                    }
                                },
                                "output_sha256": sha256_file(output),
                            }
                        ),
                        encoding="utf-8",
                    )
                elif script == "12b_train_infogain_reranker.py":
                    output = Path(command[command.index("--output-dir") + 1])
                    checkpoint = output / "checkpoint"
                    checkpoint.mkdir(parents=True, exist_ok=True)
                    (checkpoint / "heads.pt").write_bytes(b"heads")
                    (checkpoint / "infogain_config.json").write_text(
                        "{}", encoding="utf-8"
                    )
                elif script == "12c_select_infogain_reranker.py":
                    output = Path(command[command.index("--output") + 1])
                    write_jsonl(output, [{"id": "validation-1"}])
                elif script == "07_eval_rag_classification.py":
                    predictions = Path(command[command.index("--output") + 1])
                    metrics = Path(
                        command[command.index("--metrics-output") + 1]
                    )
                    write_jsonl(predictions, [{"id": "validation-1"}])
                    metrics.write_text(
                        json.dumps(
                            {
                                "accuracy": 1.0,
                                "macro_f1": 1.0,
                                "avg_num_docs": 1.0,
                                "avg_evidence_chars": 10.0,
                            }
                        ),
                        encoding="utf-8",
                    )
                else:
                    raise AssertionError(f"unexpected command: {command}")
                return 0.01

            with patch("src.calibration.grid._run_command", side_effect=fake_run):
                first = execute_grid(
                    plan,
                    config_path=plan["inputs"]["config"]["path"],
                    project_root=Path(__file__).resolve().parents[1],
                    methods={"infogain_fever"},
                    candidate_fingerprint=candidate,
                    skip_completed=True,
                )
            self.assertEqual(first["status"], "completed")
            self.assertEqual(len(commands), 4)
            teacher_command = next(
                command
                for command in commands
                if Path(command[1]).name == "12a_build_infogain_teacher.py"
            )
            self.assertIn("--resume", teacher_command)
            self.assertEqual(
                teacher_command[teacher_command.index("--purpose") + 1],
                "training",
            )
            completed_teacher = json.loads(
                teacher_grid_manifest.read_text(encoding="utf-8")
            )
            self.assertEqual(completed_teacher["status"], "completed")
            self.assertEqual(
                completed_teacher["output_sha256"][str(teacher_path.resolve())],
                sha256_file(teacher_path),
            )
            candidates = json.loads(
                (
                    Path(plan["output_dir"]).parent
                    / "calibration_candidates.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                candidates["candidates"][0]["candidate_fingerprint"],
                candidate,
            )
            self.assertEqual(
                candidates["candidates"][0]["teacher_role"], "train_core"
            )

            with patch(
                "src.calibration.grid._run_command",
                side_effect=AssertionError("completed stage was rerun"),
            ):
                reused = execute_grid(
                    plan,
                    config_path=plan["inputs"]["config"]["path"],
                    project_root=Path(__file__).resolve().parents[1],
                    methods={"infogain_fever"},
                    candidate_fingerprint=candidate,
                    skip_completed=True,
                )
            self.assertEqual(reused["status"], "completed")

    def test_calibrate_methods_consumes_canonical_grid_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._plan(root)
            records = []
            for method in ("infogain_fever", "rag_cbwdm"):
                node = plan["methods"][method]["training_candidates"][0]
                selection = node["selections"][0]
                records.append(
                    {
                        "schema_version": CANDIDATE_SCHEMA_VERSION,
                        "method": method,
                        "split": "validation",
                        "candidate_fingerprint": selection[
                            "candidate_fingerprint"
                        ],
                        "training_fingerprint": node[
                            "training_fingerprint"
                        ],
                        "selection_fingerprint": selection[
                            "selection_fingerprint"
                        ],
                        "parameters": selection["parameters"],
                        "canonical_parameter_json": selection[
                            "canonical_parameter_json"
                        ],
                        "metrics": {
                            "accuracy": 0.7,
                            "macro_f1": 0.7,
                            "avg_num_docs": 2.0,
                            "avg_evidence_chars": 20.0,
                        },
                        "status": "completed",
                        "reason": None,
                        "split_manifest_sha256": plan[
                            "split_manifest_sha256"
                        ],
                        "split_sha256": plan["validation_sha256"],
                        "validation_sha256": plan["validation_sha256"],
                    }
                )
            output = root / "formal"
            output.mkdir()
            _publish_candidates(
                plan,
                records,
                output,
                execution_contract={
                    "training_candidate_count": 2,
                    "selection_candidate_count": 2,
                },
                project_root=Path(__file__).resolve().parents[1],
            )
            split = validate_split_manifest(
                plan["inputs"]["split_manifest"]["path"]
            )
            config = yaml.safe_load(
                Path(plan["inputs"]["config"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            result = calibrate(
                config,
                split,
                load_candidate_records(output / "calibration_candidates.json"),
            )
            self.assertEqual(result["status"], "completed")
            for method in ("infogain_fever", "rag_cbwdm"):
                self.assertEqual(
                    result["selected"][method]["candidate_fingerprint"],
                    next(
                        record["candidate_fingerprint"]
                        for record in records
                        if record["method"] == method
                    ),
                )

    def test_cbwdm_teacher_stage_manifest_resume_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "grid_teacher_test",
            root / "scripts" / "04_build_cbwdm_teacher.py",
        )
        self.assertTrue(spec and spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            config = temp / "config.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "dataset": "fever2",
                        "paths": {"processed_dir": str(temp)},
                        "cbwdm": {
                            "top_m": 1,
                            "ridge_lambda": 0.01,
                            "stop_threshold": 0.0,
                            "eps_smooth": 0.001,
                            "L_type": "euclidean_posterior_shift",
                            "target_smoothing": "paper_mixture",
                            "gain_tolerance": 1e-10,
                            "store_all_gains": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            posterior = temp / "posterior.jsonl"
            write_jsonl(
                posterior,
                [
                    {
                        "id": "q1",
                        "query": "claim",
                        "label": "SUPPORTS",
                        "split": "train_core",
                        "labels": ["SUPPORTS", "REFUTES"],
                        "eta0": [0.5, 0.5],
                        "candidates": [
                            {
                                "doc_id": "d1",
                                "rank": 1,
                                "title": "t",
                                "text": "e",
                                "eta": [0.8, 0.2],
                            }
                        ],
                    }
                ],
            )
            output = temp / "teacher.jsonl"
            argv = [
                "04_build_cbwdm_teacher.py",
                "--config",
                str(config),
                "--split",
                "train_core",
                "--posteriors",
                str(posterior),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                module.main()
            manifest = output.with_suffix(".manifest.json")
            first = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["output_sha256"], sha256_file(output))
            with patch.object(sys, "argv", [*argv, "--resume"]):
                module.main()
            second = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(first["fingerprint"], second["fingerprint"])


class DiagnosticTests(unittest.TestCase):
    def test_diagnostics_resolves_selected_calibration_candidate_not_first(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "calibrated_diagnostics_test",
            root / "scripts" / "16a_diagnose_cbwdm_pilot.py",
        )
        self.assertTrue(spec and spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            calibration = temp / "calibration.json"
            candidates = temp / "candidates.json"
            calibration.write_text(
                json.dumps(
                    {
                        "selected": {
                            "rag_cbwdm": {
                                "status": "selected",
                                "candidate_fingerprint": "winner",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            candidates.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "method": "rag_cbwdm",
                                "candidate_fingerprint": "first",
                                "status": "completed",
                                "prediction_path": "wrong-predictions",
                                "selection_path": "wrong-selection",
                                "teacher_path": "wrong-teacher",
                            },
                            {
                                "method": "rag_cbwdm",
                                "candidate_fingerprint": "winner",
                                "training_fingerprint": "train",
                                "selection_fingerprint": "selection",
                                "status": "completed",
                                "prediction_path": "winner-predictions",
                                "selection_path": "winner-selection",
                                "teacher_path": "winner-teacher",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "16a_diagnose_cbwdm_pilot.py",
                "--config",
                "config.yaml",
                "--no-evidence",
                "no.jsonl",
                "--naive",
                "naive.jsonl",
                "--oracle",
                "oracle.jsonl",
                "--naive-selection",
                "naive-selection.jsonl",
                "--oracle-selection",
                "oracle-selection.jsonl",
                "--retrieval",
                "retrieval.jsonl",
                "--posteriors",
                "posteriors.jsonl",
                "--calibration-manifest",
                str(calibration),
                "--calibration-candidates",
                str(candidates),
                "--output-dir",
                str(temp / "output"),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                module,
                "publish_diagnostics",
                return_value={"status": "passed"},
            ) as publish:
                module.main()
            call = publish.call_args.kwargs
            self.assertEqual(call["paths"]["cbwdm"], "winner-predictions")
            self.assertEqual(
                call["paths"]["cbwdm_selection"], "winner-selection"
            )
            self.assertEqual(call["paths"]["teacher"], "winner-teacher")
            self.assertEqual(
                call["calibration_selection"]["candidate_fingerprint"],
                "winner",
            )

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
                "run_calibration_grid",
                "calibrate_methods",
                "cbwdm_diagnostics",
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

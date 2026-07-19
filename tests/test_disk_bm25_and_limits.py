from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from src.retrieval.pyserini_bm25 import (
    SCHEMA_VERSION,
    build_index,
    contract_diff,
    index_contract,
    index_fingerprint,
    index_inventory,
    validate_index,
)
from src.run_manifest import sha256_file, stable_hash

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_fever_test_module", "00_prepare_fever.py")
retrieve = load_script("retrieve_test_module", "02_retrieve_bm25.py")
runner = load_script("runner_test_module", "run_fever_cbwdm.py")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class DiskBM25AndLimitTests(unittest.TestCase):
    def test_v2_index_fingerprint_covers_every_material_contract_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.jsonl"
            write_jsonl(
                corpus,
                [
                    {
                        "doc_id": "d1",
                        "title": "Title",
                        "text": "Text",
                        "meta": {"page_id": "Title", "sentence_id": 0},
                    }
                ],
            )
            base = {
                "backend_version": "test",
                "k1": 0.9,
                "b": 0.4,
                "analyzer": "english",
            }
            contract = index_contract(corpus, **base)
            self.assertEqual(
                index_fingerprint(contract),
                index_fingerprint(index_contract(corpus, **base)),
            )
            variants = [
                {**base, "k1": 1.2},
                {**base, "b": 0.7},
                {**base, "analyzer": "standard"},
                {**base, "store_raw": False},
                {
                    **base,
                    "raw_metadata_contract_version": "metadata.v2",
                },
            ]
            for arguments in variants:
                self.assertNotEqual(
                    index_fingerprint(contract),
                    index_fingerprint(index_contract(corpus, **arguments)),
                )
            corpus.write_text(
                corpus.read_text(encoding="utf-8") + " \n",
                encoding="utf-8",
            )
            self.assertNotEqual(
                index_fingerprint(contract),
                index_fingerprint(index_contract(corpus, **base)),
            )

    def test_shared_index_planner_avoids_legacy_collision_and_is_read_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / "_shared"
            corpus = root / "corpus.jsonl"
            write_jsonl(
                corpus,
                [
                    {
                        "doc_id": "d1",
                        "title": "Title",
                        "text": "Text",
                        "meta": {"page_id": "Title", "sentence_id": 0},
                    }
                ],
            )
            corpus.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "rag_cbwdm_corpus.v1",
                        "completed": True,
                        "source_fingerprint": "corpus-source",
                        "output_sha256": sha256_file(corpus),
                    }
                ),
                encoding="utf-8",
            )
            retrieval = {
                "backend": "pyserini_lucene",
                "top_n": 20,
                "bm25": {"k1": 0.9, "b": 0.4},
                "index": {
                    "path": None,
                    "analyzer": "english",
                    "store_raw": True,
                },
            }
            corpus_key = "legacy-corpus-key"
            legacy_key = stable_hash(
                {"corpus_key": corpus_key, "retrieval": retrieval}
            )[:16]
            legacy = shared / "indexes" / legacy_key
            legacy.mkdir(parents=True)
            legacy_manifest = legacy / "index_manifest.json"
            legacy_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "rag_cbwdm_bm25_index.v1",
                        "completed": True,
                        "contract": {
                            "backend": "pyserini_lucene",
                            "bm25": {"k1": 0.9, "b": 0.4},
                        },
                    }
                ),
                encoding="utf-8",
            )
            before = sha256_file(legacy_manifest)
            plan = runner.plan_shared_index(
                corpus=corpus,
                shared=shared,
                corpus_key=corpus_key,
                retrieval_config=retrieval,
                backend_version="test",
            )
            self.assertNotEqual(plan["path"], legacy)
            self.assertEqual(plan["path"].name, plan["fingerprint"])
            self.assertTrue(plan["legacy_collision"])
            self.assertFalse(plan["existing_compatible"])
            self.assertEqual(plan["action"], "build")
            self.assertEqual(sha256_file(legacy_manifest), before)
            self.assertFalse(plan["path"].exists())

    def test_legacy_manifest_without_v2_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory)
            (index / "segments_1").write_bytes(b"segments")
            (index / "_0.tim").write_bytes(b"terms")
            (index / "index_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "rag_cbwdm_bm25_index.v1",
                        "completed": True,
                        "contract": {"backend": "pyserini_lucene"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not completed"):
                validate_index(index)

    def test_runner_dry_run_prints_v2_index_plan_and_retrieval_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "experiments"
            config_path = ROOT / "configs" / "fever2_server_pilot_5000_500.yaml"
            config = runner.load_yaml(config_path)
            corpus_key = stable_hash(
                {
                    "wiki": config["paths"].get("raw_wiki_pages_dir"),
                    "corpus": config.get("corpus", {}),
                    "limit": config.get("profile_limits", {}).get("corpus"),
                }
            )[:16]
            corpus = (
                output_root
                / "_shared"
                / "corpora"
                / corpus_key
                / "fever_corpus_sentence.jsonl"
            )
            corpus.parent.mkdir(parents=True)
            write_jsonl(
                corpus,
                [
                    {
                        "doc_id": "d1",
                        "title": "Title",
                        "text": "Text",
                        "meta": {"page_id": "Title", "sentence_id": 0},
                    }
                ],
            )
            corpus.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "rag_cbwdm_corpus.v1",
                        "completed": True,
                        "source_fingerprint": "source",
                        "output_sha256": sha256_file(corpus),
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "run_fever_cbwdm.py",
                "--config",
                str(config_path),
                "--run-name",
                "index-plan",
                "--stages",
                "corpus,index,retrieve_train_core,retrieve_validation",
                "--output-root",
                str(output_root),
                "--cache-root",
                str(root / "cache"),
                "--dry-run",
            ]
            stream = io.StringIO()
            with patch.object(sys, "argv", argv), patch.object(
                runner, "pyserini_version", return_value="test"
            ), redirect_stdout(stream):
                runner.main()
            output = stream.getvalue()
            self.assertIn("contract_version=rag_cbwdm_lucene_index_contract.v2", output)
            self.assertIn("corpus_reused=yes", output)
            self.assertIn("action=build", output)
            plan = runner.plan_shared_index(
                corpus=corpus,
                shared=output_root / "_shared",
                corpus_key=corpus_key,
                retrieval_config=config["retrieval"],
                backend_version="test",
            )
            self.assertIn(str(plan["path"]), output)
            self.assertIn(
                f"--index '{str(plan['path'])}'",
                output,
            )
            self.assertNotIn("e0dc68ba1711fd74", str(plan["path"]))

    def test_limit_counts_valid_rows_after_fever2_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            rows = []
            for index in range(15):
                label = "NOT ENOUGH INFO" if index < 10 and index % 2 == 0 else "SUPPORTS"
                rows.append({"id": index, "claim": f"claim {index}", "label": label})
            write_jsonl(raw, rows)
            converted, stats = prepare.iter_converted_rows(
                raw,
                "fever2",
                "train",
                {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES"},
                {"NOT ENOUGH INFO"},
                limit=10,
            )
            self.assertEqual(len(converted), 10)
            self.assertEqual(stats["read"], 15)
            self.assertEqual(stats["drop_reason_counts"], {"NOT ENOUGH INFO": 5})
            converted_raw_cap, capped = prepare.iter_converted_rows(
                raw,
                "fever2",
                "train",
                {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES"},
                {"NOT ENOUGH INFO"},
                limit=10,
                raw_limit=10,
            )
            self.assertEqual(len(converted_raw_cap), 5)
            self.assertEqual(capped["stop_reason"], "raw_limit_reached")
            fever3, fever3_stats = prepare.iter_converted_rows(
                raw,
                "fever3",
                "train",
                {
                    "SUPPORTS": "SUPPORTS",
                    "REFUTES": "REFUTES",
                    "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
                },
                set(),
                limit=10,
            )
            self.assertEqual(len(fever3), 10)
            self.assertIn("NOT_ENOUGH_INFO", {row["label"] for row in fever3})
            self.assertEqual(fever3_stats["dropped"], 0)

    def test_resolved_limits_obey_priority_and_change_fingerprint(self) -> None:
        config = {"profile_limits": {"train": 100, "dev": 50, "corpus": 200_000}}
        args = Namespace(
            train_limit=10,
            dev_limit=None,
            corpus_limit=None,
            raw_limit=99,
            limit=7,
        )
        limits = runner.resolve_limits(args, config)
        self.assertEqual(
            limits, {"train": 10, "dev": 50, "corpus": 200_000, "raw": 99}
        )
        from src.run_manifest import stable_hash

        self.assertNotEqual(stable_hash(limits), stable_hash({**limits, "dev": 51}))

    def test_index_manifest_reuse_mismatch_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            index = root / "index"
            write_jsonl(
                corpus,
                [
                    {
                        "doc_id": "d1",
                        "title": "Hawaii",
                        "text": "Obama was born here.",
                        "meta": {"page_id": "Hawaii", "sentence_id": 0},
                    },
                    {
                        "doc_id": "d2",
                        "title": "Paris",
                        "text": "The Eiffel Tower.",
                        "meta": {"page_id": "Paris", "sentence_id": 1},
                    },
                ],
            )

            run_calls = 0

            def fake_run(command, check):
                nonlocal run_calls
                run_calls += 1
                target = Path(command[command.index("--index") + 1])
                (target / "segments_1").write_bytes(b"lucene")
                (target / "_0_Lucene104_0.tim").write_bytes(b"terms")
                (target / "write.lock").write_bytes(b"")

            lucene_module = types.ModuleType("pyserini.search.lucene")

            class FakeSearcher:
                def __init__(self, path):
                    self.path = path

                def set_bm25(self, k1, b):
                    self.params = (k1, b)

                def search(self, query, k=1):
                    return [types.SimpleNamespace(docid="d1")]

                def doc(self, docid):
                    payload = {
                        "id": "d1",
                        "doc_id": "d1",
                        "title": "Hawaii",
                        "text": "Obama was born here.",
                        "meta": {"page_id": "Hawaii", "sentence_id": 0},
                    }
                    return types.SimpleNamespace(
                        raw=lambda: json.dumps(payload)
                    )

            lucene_module.LuceneSearcher = FakeSearcher
            modules = {
                "pyserini": types.ModuleType("pyserini"),
                "pyserini.search": types.ModuleType("pyserini.search"),
                "pyserini.search.lucene": lucene_module,
            }
            with patch("src.retrieval.pyserini_bm25.pyserini_version", return_value="test"), patch.dict(
                sys.modules, modules
            ):
                first = build_index(corpus, index, run_command=fake_run)
                self.assertTrue(first["completed"])
                self.assertEqual(first["num_documents"], 2)
                calls_after_build = run_calls
                reused = build_index(corpus, index, resume=True, run_command=fake_run)
                self.assertEqual(reused["fingerprint"], first["fingerprint"])
                self.assertEqual(
                    run_calls,
                    calls_after_build,
                    "resume must not call the indexing subprocess",
                )
                original_corpus = corpus.read_text(encoding="utf-8")
                with corpus.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"doc_id": "d3", "title": "Berlin", "text": "Germany"})
                        + "\n"
                    )
                with self.assertRaisesRegex(ValueError, "document count"):
                    build_index(corpus, index, resume=True, run_command=fake_run)
                corpus.write_text(original_corpus, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    build_index(corpus, index, resume=True, k1=1.2, run_command=fake_run)
                rebuilt = build_index(
                    corpus, index, overwrite=True, k1=1.2, run_command=fake_run
                )
                self.assertNotEqual(rebuilt["fingerprint"], first["fingerprint"])
                validate_index(index)
                manifest_path = index / "index_manifest.json"
                incomplete = json.loads(manifest_path.read_text(encoding="utf-8"))
                incomplete["completed"] = False
                manifest_path.write_text(json.dumps(incomplete), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "not completed"):
                    validate_index(index)

    def test_lucene_10_inventory_allows_zero_byte_write_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            index = root / "index"
            index.mkdir()
            write_jsonl(
                corpus,
                [
                    {
                        "doc_id": "d1",
                        "title": "Hawaii",
                        "text": "Obama was born here.",
                        "meta": {"page_id": "Hawaii", "sentence_id": 0},
                    }
                ],
            )
            files = {
                "_0_Lucene104_0.doc": b"doc",
                "_0_Lucene104_0.pos": b"pos",
                "_0_Lucene104_0.tim": b"tim",
                "_0_Lucene104_0.tip": b"tip",
                "_0_Lucene90_0.dvd": b"dvd",
                "segments_1": b"segments",
                "write.lock": b"",
            }
            for name, content in files.items():
                (index / name).write_bytes(content)
            contract = index_contract(
                corpus,
                backend_version="2.3.0",
                k1=0.9,
                b=0.4,
                analyzer="english",
            )
            manifest = {
                **contract,
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "completed": True,
                "index_fingerprint": index_fingerprint(contract),
                "fingerprint": index_fingerprint(contract),
                "contract": contract,
                "num_documents": 1,
                "index_file_inventory": index_inventory(index),
                "probe_passed": True,
                "probe_query": "Hawaii",
                "raw_metadata_recovery": {
                    "passed": True,
                    "recovered": {
                        "doc_id": "d1",
                        "title": "Hawaii",
                        "text": "Obama was born here.",
                        "page_id": "Hawaii",
                        "sentence_id": 0,
                    },
                },
            }
            (index / "index_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            class Searcher:
                def search(self, query, k=10):
                    return [types.SimpleNamespace(docid="d1")]

                def doc(self, docid):
                    stored = {
                        "doc_id": "d1",
                        "title": "Hawaii",
                        "text": "Obama was born here.",
                        "meta": {"page_id": "Hawaii", "sentence_id": 0},
                    }
                    return types.SimpleNamespace(raw=lambda: json.dumps(stored))

            searcher_factory = Mock(return_value=Searcher())
            validated = validate_index(
                index, contract, searcher_factory=searcher_factory
            )
            self.assertTrue(validated["completed"])
            self.assertEqual(validated["fingerprint"], stable_hash(contract))
            searcher_factory.assert_called_once_with(index)

    def test_retrieval_failure_keeps_only_partial_then_atomic_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text(
                "retrieval:\n  backend: memory_rank_bm25\n  top_n: 2\n", encoding="utf-8"
            )
            queries = root / "queries.jsonl"
            corpus = root / "corpus.jsonl"
            output = root / "output.jsonl"
            write_jsonl(
                queries,
                [{"id": "q1", "query": "x", "label": "SUPPORTS", "split": "train"}],
            )
            write_jsonl(corpus, [{"doc_id": "d", "title": "t", "text": "x"}])

            class Failing:
                def search(self, query, top_n):
                    raise RuntimeError("simulated")

            argv = [
                "02_retrieve_bm25.py",
                "--config",
                str(config),
                "--split",
                "train",
                "--queries",
                str(queries),
                "--corpus",
                str(corpus),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                retrieve.MemoryBM25Retriever, "from_jsonl", return_value=Failing()
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    retrieve.main()
            self.assertFalse(output.exists())
            self.assertTrue((root / "output.jsonl.partial").exists())

            class Working:
                def search(self, query, top_n):
                    return [
                        {"doc_id": "d", "rank": 1, "score": 1.0, "title": "t", "text": "x"}
                    ]

            with patch.object(sys, "argv", argv), patch.object(
                retrieve.MemoryBM25Retriever, "from_jsonl", return_value=Working()
            ):
                retrieve.main()
            self.assertTrue(output.exists())
            manifest = json.loads(
                output.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["completed"])
            self.assertEqual(manifest["num_output_rows"], 1)

    def test_posterior_status_manifest_validation_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config = {
                "dataset": "fever2",
                "task": {
                    "labels": ["SUPPORTS", "REFUTES"],
                    "verbalizers": {"SUPPORTS": ["A"], "REFUTES": ["B"]},
                },
                "generator": {
                    "model_name": "local-generator",
                    "dtype": "auto",
                    "device_map": "auto",
                    "trust_remote_code": False,
                    "max_context_tokens": 128,
                    "posterior_batch_size": 1,
                },
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            query = root / "query.jsonl"
            retrieval_path = root / "retrieval.jsonl"
            posterior_path = root / "posterior.jsonl"
            row = {"id": "q1", "query": "claim", "label": "SUPPORTS", "split": "train"}
            write_jsonl(query, [row])
            write_jsonl(retrieval_path, [{**row, "candidates": []}])
            write_jsonl(posterior_path, [{**row, "labels": ["SUPPORTS", "REFUTES"]}])
            provenance = runner.posterior_provenance(
                config=config,
                config_path=config_path,
                retrieval_path=retrieval_path,
                split="train",
                model_name="local-generator",
                batch_size=1,
            )
            manifest_path = posterior_path.with_suffix(".manifest.json")

            def write_manifest(status: str = "completed") -> None:
                manifest_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "rag_cbwdm_posterior_manifest.v1",
                            "stage": "posterior",
                            "status": status,
                            "fingerprint": stable_hash(provenance),
                            "provenance": provenance,
                            "expected_rows": 1,
                            "completed_rows": 1,
                            "output_sha256": sha256_file(posterior_path),
                        }
                    ),
                    encoding="utf-8",
                )

            write_manifest()
            reasons = runner.validate_posterior_artifact(
                split="train",
                output_path=posterior_path,
                manifest_path=manifest_path,
                retrieval_path=retrieval_path,
                query_path=query,
                expected_provenance=provenance,
            )
            self.assertEqual(reasons, [])

            write_manifest(status="running")
            reasons = runner.validate_posterior_artifact(
                split="train",
                output_path=posterior_path,
                manifest_path=manifest_path,
                retrieval_path=retrieval_path,
                query_path=query,
                expected_provenance=provenance,
            )
            self.assertTrue(any(reason.startswith("status:") for reason in reasons))

            write_manifest()
            fingerprint_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            fingerprint_manifest["fingerprint"] = "wrong"
            manifest_path.write_text(
                json.dumps(fingerprint_manifest), encoding="utf-8"
            )
            reasons = runner.validate_posterior_artifact(
                split="train",
                output_path=posterior_path,
                manifest_path=manifest_path,
                retrieval_path=retrieval_path,
                query_path=query,
                expected_provenance=provenance,
            )
            self.assertTrue(
                any(reason.startswith("fingerprint:") for reason in reasons)
            )

            write_manifest()
            write_jsonl(posterior_path, [{**row, "labels": ["changed"]}])
            reasons = runner.validate_posterior_artifact(
                split="train",
                output_path=posterior_path,
                manifest_path=manifest_path,
                retrieval_path=retrieval_path,
                query_path=query,
                expected_provenance=provenance,
            )
            self.assertTrue(any(reason.startswith("output_sha256:") for reason in reasons))

            missing, invalid = runner.validate_stage_outputs(
                "posterior",
                [posterior_path, manifest_path],
                {
                    "posterior": {"train": posterior_path, "dev": posterior_path},
                    "retrieval": {"train": retrieval_path, "dev": retrieval_path},
                    "query": {"train": query, "dev": query},
                    "posterior_provenance": lambda split: provenance,
                },
            )
            self.assertEqual(missing, [])
            self.assertTrue(invalid)
            posterior_path.unlink()
            missing, invalid = runner.validate_stage_outputs(
                "posterior",
                [posterior_path, manifest_path],
                {},
            )
            self.assertEqual(missing, [str(posterior_path)])
            self.assertEqual(invalid, [])

    def test_runner_resume_reuses_completed_posterior_without_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "runs"
            run_name = "posterior-resume"
            config_path = root / "config.yaml"
            config = {
                "dataset": "fever2",
                "profile": "test",
                "profile_limits": {"train": 1, "dev": 1, "corpus": None},
                "paths": {
                    "raw_fever_train": "unused-train.jsonl",
                    "raw_fever_dev": "unused-dev.jsonl",
                    "raw_fever_test": "unused-test.jsonl",
                    "raw_wiki_pages_dir": "unused-wiki",
                    "processed_dir": str(root / "processed"),
                },
                "task": {
                    "labels": ["SUPPORTS", "REFUTES"],
                    "verbalizers": {"SUPPORTS": ["A"], "REFUTES": ["B"]},
                    "label_map": {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES"},
                    "drop_labels": ["NOT ENOUGH INFO"],
                },
                "corpus": {"mode": "sentence", "min_text_chars": 1},
                "retrieval": {"backend": "memory_rank_bm25", "top_n": 2},
                "generator": {
                    "model_name": "local-generator",
                    "dtype": "auto",
                    "device_map": "auto",
                    "trust_remote_code": False,
                    "max_context_tokens": 128,
                    "posterior_batch_size": 1,
                },
                "cbwdm": {"top_m": 1},
                "selector": {"model_name": "local-selector", "candidate_batch_size": 1},
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            artifacts = output_root / run_name / "artifacts"
            artifacts.mkdir(parents=True)
            for split in ("train", "dev"):
                row = {
                    "id": f"{split}-1",
                    "query": "claim",
                    "label": "SUPPORTS",
                    "split": split,
                }
                query_path = artifacts / f"fever2_{split}.jsonl"
                retrieval_path = artifacts / f"fever2_{split}_bm25_top2.jsonl"
                posterior_path = artifacts / f"fever2_{split}_posteriors.jsonl"
                write_jsonl(query_path, [row])
                write_jsonl(retrieval_path, [{**row, "candidates": []}])
                write_jsonl(
                    posterior_path,
                    [{**row, "labels": ["SUPPORTS", "REFUTES"], "candidates": []}],
                )
                provenance = runner.posterior_provenance(
                    config=config,
                    config_path=config_path,
                    retrieval_path=retrieval_path,
                    split=split,
                    model_name="local-generator",
                    batch_size=1,
                )
                posterior_path.with_suffix(".manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "rag_cbwdm_posterior_manifest.v1",
                            "stage": "posterior",
                            "status": "completed",
                            "fingerprint": stable_hash(provenance),
                            "provenance": provenance,
                            "expected_rows": 1,
                            "completed_rows": 1,
                            "output_sha256": sha256_file(posterior_path),
                        }
                    ),
                    encoding="utf-8",
                )
            run_contract = {
                "config": config,
                "seed": 13,
                "generator_model_override": None,
                "selector_model_override": None,
                "resolved_limits": {
                    "train": 1,
                    "dev": 1,
                    "corpus": None,
                    "raw": None,
                },
            }
            run_manifest = {
                "fingerprint": stable_hash(run_contract),
                "stages": {
                    stage: {"status": "failed" if stage == "posterior" else "pending"}
                    for stage in runner.ALL_STAGES
                },
            }
            (output_root / run_name / "run_manifest.json").write_text(
                json.dumps(run_manifest), encoding="utf-8"
            )
            argv = [
                "run_fever_cbwdm.py",
                "--config",
                str(config_path),
                "--run-name",
                run_name,
                "--stages",
                "posterior",
                "--resume",
                "--output-root",
                str(output_root),
                "--cache-root",
                str(root / "cache"),
            ]
            run_subprocess = Mock(side_effect=AssertionError("posterior subprocess called"))
            with patch.object(sys, "argv", argv), patch.object(
                runner.subprocess, "run", run_subprocess
            ):
                runner.main()
            run_subprocess.assert_not_called()
            updated = json.loads(
                (output_root / run_name / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(updated["stages"]["posterior"]["status"], "skipped")
            self.assertEqual(
                updated["stages"]["posterior"]["reason"],
                "validated_completed_outputs",
            )


if __name__ == "__main__":
    unittest.main()

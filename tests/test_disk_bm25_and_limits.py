from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from src.retrieval.pyserini_bm25 import build_index, validate_index

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
                    {"doc_id": "d1", "title": "Hawaii", "text": "Obama was born here."},
                    {"doc_id": "d2", "title": "Paris", "text": "The Eiffel Tower."},
                ],
            )

            def fake_run(command, check):
                target = Path(command[command.index("--index") + 1])
                (target / "segments_1").write_bytes(b"lucene")

            lucene_module = types.ModuleType("pyserini.search.lucene")

            class FakeSearcher:
                def __init__(self, path):
                    self.path = path

                def set_bm25(self, k1, b):
                    self.params = (k1, b)

                def search(self, query, k=1):
                    return [object()]

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
                reused = build_index(corpus, index, resume=True, run_command=fake_run)
                self.assertEqual(reused["fingerprint"], first["fingerprint"])
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


if __name__ == "__main__":
    unittest.main()

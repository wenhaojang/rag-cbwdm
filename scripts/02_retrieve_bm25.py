from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml, read_jsonl, require_keys
from src.retrieval.memory_bm25 import MemoryBM25Retriever
from src.retrieval.pyserini_bm25 import PyseriniBM25Retriever, validate_index
from src.run_manifest import atomic_write_json, sha256_file, stable_hash, utc_now

SCHEMA_VERSION = "rag_cbwdm_retrieval.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 retrieval from a reusable index.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "dev", "train_core", "validation", "held_out_test"],
    )
    parser.add_argument("--queries", required=True)
    parser.add_argument("--index", help="Persistent Lucene index directory.")
    parser.add_argument("--corpus", help="Only for explicit memory_rank_bm25 debug runs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_results(
    retriever: Any, queries: Path, top_n: int, limit: int | None, counts: list[int]
) -> Iterator[dict[str, Any]]:
    seen_ids: set[str] = set()
    for row_index, row in enumerate(read_jsonl(queries, limit=limit), start=1):
        require_keys(row, ["id", "query", "label", "split"], f"query row {row_index}")
        row_id = str(row["id"])
        if row_id in seen_ids:
            raise ValueError(f"Duplicate query id: {row_id}")
        seen_ids.add(row_id)
        candidates = retriever.search(str(row["query"]), top_n)
        counts.append(len(candidates))
        output = {
            "id": row_id,
            "query": row["query"],
            "label": row["label"],
            "split": row["split"],
            "candidates": candidates,
        }
        if row.get("original_id") is not None:
            output["original_id"] = row["original_id"]
        # Validation diagnostics need gold sentence keys to measure retrieval
        # recall. They are deliberately omitted from held_out_test artifacts.
        if row["split"] in {"train", "train_core", "validation"}:
            meta = row.get("meta")
            evidence = meta.get("evidence") if isinstance(meta, dict) else None
            keys: set[str] = set()
            if isinstance(evidence, list):
                for group in evidence:
                    if not isinstance(group, list):
                        continue
                    for item in group:
                        if (
                            isinstance(item, list)
                            and len(item) >= 4
                            and item[2] is not None
                            and item[3] is not None
                        ):
                            keys.add(f"{item[2]}\t{item[3]}")
            if keys:
                output["gold_evidence_keys"] = sorted(keys)
        yield output


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    retrieval = config.get("retrieval", {})
    backend = retrieval.get("backend", "memory_rank_bm25")
    top_n = args.top_n or int(retrieval.get("top_n", 20))
    queries = Path(args.queries).resolve()
    output = Path(args.output).resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if (output.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Retrieval output exists; use --overwrite: {output}")
    bm25 = retrieval.get("bm25", {})
    k1, b = float(bm25.get("k1", 0.9)), float(bm25.get("b", 0.4))
    if backend == "pyserini_lucene":
        if not args.index:
            raise ValueError("--index is required for backend=pyserini_lucene")
        index_manifest = validate_index(args.index)
        if index_manifest.get("backend") != backend or index_manifest.get("bm25") != {
            "k1": k1,
            "b": b,
        }:
            raise ValueError(
                "Index backend/BM25 parameters differ from retrieval config; rebuild the index."
            )
        retriever = PyseriniBM25Retriever(args.index, k1=k1, b=b)
        index_fingerprint = index_manifest["fingerprint"]
    elif backend == "memory_rank_bm25":
        if not args.corpus:
            raise ValueError("--corpus is required for backend=memory_rank_bm25")
        max_docs = int(retrieval.get("memory_max_documents", 50_000))
        retriever = MemoryBM25Retriever.from_jsonl(
            args.corpus, max_documents=max_docs, k1=k1, b=b
        )
        index_fingerprint = stable_hash(
            {"backend": backend, "corpus_sha256": sha256_file(args.corpus), "k1": k1, "b": b}
        )
    else:
        raise ValueError(f"Unsupported retrieval backend: {backend}")

    contract = {
        "split": args.split,
        "query_input_sha256": sha256_file(queries),
        "index_fingerprint": index_fingerprint,
        "backend": backend,
        "top_n": top_n,
        "limit": args.limit,
        "gold_evidence_key_policy": (
            "validation_diagnostics_only"
            if args.split in {"train", "train_core", "validation"}
            else "omitted"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    counts: list[int] = []
    started = utc_now()
    rows = 0
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as handle:
            for row in iter_results(retriever, queries, top_n, args.limit, counts):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, output)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "completed": True,
            "fingerprint": stable_hash(contract),
            **contract,
            "query_input_path": str(queries),
            "num_queries": rows,
            "num_output_rows": rows,
            "candidate_count_statistics": {
                "min": min(counts, default=0),
                "max": max(counts, default=0),
                "mean": statistics.fmean(counts) if counts else 0.0,
            },
            "output_sha256": sha256_file(output),
            "start_time": started,
            "end_time": utc_now(),
        }
        atomic_write_json(manifest_path, manifest)
    except Exception:
        # A partial is deliberately retained for diagnosis, but never promoted.
        raise
    print(f"[retrieve] split={args.split} rows={rows} output={output}")


if __name__ == "__main__":
    main()

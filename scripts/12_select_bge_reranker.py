from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.bge_reranker import (
    INPUT_TEMPLATE_VERSION,
    HuggingFaceReranker,
    format_pair,
    make_bge_selection,
)
from src.baselines.common import build_selection_contract, publish_selection
from src.io_utils import read_jsonl
from src.run_manifest import atomic_write_json, sha256_file, stable_hash, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score/select FEVER BM25 candidates with a BGE reranker.")
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-cache", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--min-docs", type=int, default=4)
    parser.add_argument("--normalize-score", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def cache_contract(args: argparse.Namespace, retrieval: Path) -> dict[str, Any]:
    return {
        "retrieval_sha256": sha256_file(retrieval),
        "model": args.model_name_or_path,
        "revision": args.revision,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "normalize_score": args.normalize_score,
        "input_template": INPUT_TEMPLATE_VERSION,
        "limit": args.limit,
    }


def load_valid_cache(path: Path, contract: dict[str, Any]) -> dict[str, dict[str, float]] | None:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("fingerprint") != stable_hash(contract)
        or manifest.get("output_sha256") != sha256_file(path)
    ):
        return None
    result: dict[str, dict[str, float]] = {}
    for row in read_jsonl(path):
        result[str(row["id"])] = {
            str(item["doc_id"]): float(item["score"]) for item in row["scores"]
        }
    return result


def write_score_cache(
    path: Path,
    rows: list[dict[str, Any]],
    scorer: HuggingFaceReranker,
    contract: dict[str, Any],
    batch_size: int,
) -> dict[str, dict[str, float]]:
    partial = path.with_name(path.name + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, float]] = {}
    count = 0
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            candidates = list(row.get("candidates", []))
            pairs = [format_pair(str(row.get("query") or ""), item) for item in candidates]
            scores = scorer.score_pairs(pairs, batch_size=batch_size)
            items = [
                {"doc_id": str(candidate.get("doc_id")), "score": score}
                for candidate, score in zip(candidates, scores)
            ]
            payload = {"id": row.get("id"), "scores": items}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            result[str(row.get("id"))] = {
                item["doc_id"]: float(item["score"]) for item in items
            }
            count += len(items)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    atomic_write_json(
        path.with_suffix(".manifest.json"),
        {
            "schema_version": "rag_cbwdm_bge_score_cache.v1",
            "stage": "score_bge",
            "status": "completed",
            "completed": True,
            "fingerprint": stable_hash(contract),
            "contract": contract,
            "num_candidate_scores": count,
            "output_sha256": sha256_file(path),
            "end_time": utc_now(),
        },
    )
    return result


def main() -> None:
    args = parse_args()
    retrieval = absolute(args.retrieval)
    output = absolute(args.output)
    cache = absolute(args.score_cache)
    rows = list(read_jsonl(retrieval, limit=args.limit))
    scoring_contract = cache_contract(args, retrieval)
    scores = load_valid_cache(cache, scoring_contract) if args.resume and not args.overwrite else None
    cache_reused = scores is not None
    if (
        scores is None
        and args.resume
        and not args.overwrite
        and (cache.exists() or cache.with_suffix(".manifest.json").exists())
    ):
        raise ValueError(
            "Cannot resume BGE scoring: score cache fingerprint/checksum mismatch. "
            "Use a new cache/output path or explicit --overwrite."
        )
    if scores is None:
        scorer = HuggingFaceReranker(
            args.model_name_or_path,
            revision=args.revision,
            device=args.device,
            dtype=args.dtype,
            max_length=args.max_length,
            normalize_score=args.normalize_score,
            local_files_only=args.local_files_only,
        )
        scores = write_score_cache(cache, rows, scorer, scoring_contract, args.batch_size)
    if args.score_only:
        print(f"[bge_score] rows={len(rows)} cache_reused={cache_reused} cache={cache}")
        return
    method = "bge"
    model_metadata = {
        "model": args.model_name_or_path,
        "revision": args.revision,
        "backend": "transformers.AutoModelForSequenceClassification",
        "normalized_score": args.normalize_score,
    }
    contract = build_selection_contract(
        method=method,
        input_paths={"retrieval": retrieval, "score_cache": cache},
        parameters={
            "top_m": args.top_m,
            "min_docs": args.min_docs,
            "score_threshold": args.score_threshold,
            "limit": args.limit,
        },
        model=model_metadata,
    )
    selected, reused = publish_selection(
        output,
        (
            make_bge_selection(
                row,
                scores[str(row["id"])],
                method=method,
                top_m=args.top_m,
                score_threshold=args.score_threshold,
                min_docs=args.min_docs,
                model_metadata=model_metadata,
            )
            for row in rows
        ),
        contract=contract,
        project_root=PROJECT_ROOT,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(
        f"[bge_select] rows={selected} cache_reused={cache_reused} "
        f"selection_reused={reused} output={output}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import build_selection_contract, publish_selection
from src.io_utils import load_yaml, read_jsonl, require_keys
from src.selection_schema import make_selection_row, normalize_selected_doc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for naive top-M selection."""
    parser = argparse.ArgumentParser(description="Select the first M BM25 candidates as a baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--retrieval", required=True, help="Retrieval JSONL path.")
    parser.add_argument("--output", required=True, help="Selection JSONL output path.")
    parser.add_argument("--top-m", type=int, required=True, help="Number of top candidates to select.")
    parser.add_argument("--min-docs", type=int, default=None, help="Required minimum candidates per row.")
    parser.add_argument("--method-name", default=None, help="Method name written to output.")
    parser.add_argument("--limit", type=int, default=None, help="Max retrieval rows to process.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def candidate_score(candidate: Dict[str, Any]) -> Any:
    """Return retrieval score under either supported field name."""
    return candidate.get("retrieval_score", candidate.get("score"))


def selected_doc(candidate: Dict[str, Any], step: int) -> Dict[str, Any]:
    """Convert one retrieval candidate to selection output schema."""
    require_keys(candidate, ["doc_id", "title", "text"], "candidate")
    return normalize_selected_doc(candidate, selector_score=None, selection_step=step)


def iter_selection_rows(
    retrieval_path: Path,
    top_m: int,
    method_name: str,
    limit: int | None = None,
    min_docs: int = 0,
) -> Iterator[Dict[str, Any]]:
    """Stream naive top-M selection rows."""
    if top_m < 0:
        raise ValueError(f"top_m must be non-negative, got {top_m}")
    if min_docs < 0 or min_docs > top_m:
        raise ValueError(f"min_docs must satisfy 0 <= min_docs <= top_m, got {min_docs}")
    for row_index, row in enumerate(read_jsonl(retrieval_path, limit=limit), start=1):
        require_keys(row, ["id", "query", "label", "split", "candidates"], f"retrieval row {row_index}")
        candidates = row["candidates"]
        if not isinstance(candidates, list):
            raise ValueError(f"retrieval row {row_index} has non-list candidates")
        if len(candidates) < min_docs:
            raise ValueError(
                f"retrieval row {row_index} has {len(candidates)} candidates, "
                f"fewer than min_docs={min_docs}"
            )
        docs = [selected_doc(candidate, step) for step, candidate in enumerate(candidates[:top_m])]
        yield make_selection_row(
            row,
            method=method_name,
            selected_docs=docs,
            selection_steps=[
                {
                    "step": step,
                    "selected_doc_id": str(doc["doc_id"]),
                    "predicted_score": None,
                    "remaining_count": max(len(candidates) - step, 0),
                    "stop": False,
                }
                for step, doc in enumerate(docs)
            ],
            stop_reason="top_m_reached" if len(docs) >= top_m else "no_candidates",
            max_docs=top_m,
            selection_metadata={
                "state_aware": False,
                "uses_gold_at_test": False,
                "ranking": "bm25_source_rank",
                "min_docs": min_docs,
            },
        )


def main() -> None:
    args = parse_args()
    load_yaml(args.config)
    method_name = args.method_name or f"naive_top{args.top_m}"
    min_docs = args.top_m if args.min_docs is None else args.min_docs
    output_path = resolve_project_path(args.output)
    retrieval_path = resolve_project_path(args.retrieval)
    contract = build_selection_contract(
        method=method_name,
        input_paths={"retrieval": retrieval_path},
        parameters={"top_m": args.top_m, "min_docs": min_docs, "limit": args.limit},
    )
    written, reused = publish_selection(
        output_path,
        iter_selection_rows(
            retrieval_path=retrieval_path,
            top_m=args.top_m,
            method_name=method_name,
            limit=args.limit,
            min_docs=min_docs,
        ),
        contract=contract,
        project_root=PROJECT_ROOT,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(
        f"[naive_topm] rows={written} top_m={args.top_m} reused={reused} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()

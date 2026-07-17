from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml, read_jsonl, require_keys, write_jsonl
from src.selection_schema import make_selection_row, normalize_selected_doc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CBWDM oracle teacher selection."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert CBWDM oracle teacher selections to selection JSONL. "
            "This uses gold-label teacher output and is diagnostic only."
        )
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--teacher", required=True, help="CBWDM teacher JSONL path.")
    parser.add_argument("--retrieval", default=None, help="Retrieval JSONL path for document text recovery.")
    parser.add_argument("--posteriors", default=None, help="Posterior JSONL path for document text recovery.")
    parser.add_argument("--output", required=True, help="Selection JSONL output path.")
    parser.add_argument("--top-m", type=int, default=None, help="Maximum teacher docs to select.")
    parser.add_argument("--method-name", default="cbwdm_oracle", help="Method name written to output.")
    parser.add_argument("--limit", type=int, default=None, help="Max teacher rows to process.")
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_candidate_source(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load retrieval or posterior rows keyed by sample id."""
    if path is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        require_keys(row, ["id", "candidates"], "candidate source row")
        rows[row["id"]] = row
    return rows


def candidate_score(candidate: Dict[str, Any]) -> Any:
    """Return retrieval score under either supported field name."""
    return candidate.get("retrieval_score", candidate.get("score"))


def candidate_to_selected_doc(candidate: Dict[str, Any], step: int) -> Dict[str, Any]:
    """Convert a candidate object into selected_docs schema."""
    require_keys(candidate, ["doc_id", "title", "text"], "candidate")
    return normalize_selected_doc(candidate, selector_score=None, selection_step=step)


def recover_candidate_map(teacher_row: Dict[str, Any], source_row: Dict[str, Any] | None) -> dict[str, Dict[str, Any]]:
    """Return candidate metadata from source row, falling back to teacher candidates."""
    candidates = source_row.get("candidates", []) if source_row else teacher_row.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError(f"Candidate source for row {teacher_row.get('id')} has non-list candidates")
    return {candidate.get("doc_id"): candidate for candidate in candidates}


def iter_oracle_rows(
    teacher_path: Path,
    candidate_source: dict[str, dict[str, Any]],
    top_m: int | None,
    method_name: str,
    limit: int | None = None,
) -> Iterator[Dict[str, Any]]:
    """Stream CBWDM oracle teacher rows converted to selection JSONL."""
    if top_m is not None and top_m < 0:
        raise ValueError(f"top_m must be non-negative, got {top_m}")

    for row_index, teacher_row in enumerate(read_jsonl(teacher_path, limit=limit), start=1):
        require_keys(
            teacher_row,
            ["id", "query", "label", "split", "teacher_selected_doc_ids"],
            f"teacher row {row_index}",
        )
        selected_ids = list(teacher_row["teacher_selected_doc_ids"])
        if top_m is not None:
            selected_ids = selected_ids[:top_m]

        source_row = candidate_source.get(teacher_row["id"])
        candidate_map = recover_candidate_map(teacher_row, source_row)
        selected_docs = []
        for step, doc_id in enumerate(selected_ids):
            if doc_id not in candidate_map:
                raise KeyError(f"Teacher selected doc_id {doc_id!r} not found for row {teacher_row['id']}")
            selected_docs.append(candidate_to_selected_doc(candidate_map[doc_id], step))

        output = make_selection_row(
            teacher_row,
            method=method_name,
            selected_docs=selected_docs,
            selection_steps=[
                {
                    "step": step,
                    "selected_doc_id": str(doc["doc_id"]),
                    "predicted_score": None,
                    "remaining_count": max(len(candidate_map) - step, 0),
                    "stop": False,
                }
                for step, doc in enumerate(selected_docs)
            ],
            stop_reason=teacher_row.get("stop_reason", "teacher_trajectory_complete"),
            diagnostic_only=True,
        )
        output["oracle_note"] = "cbwdm_oracle uses gold labels and is diagnostic only"
        yield output


def main() -> None:
    args = parse_args()
    load_yaml(args.config)
    if args.retrieval is None and args.posteriors is None:
        raise ValueError("Provide either --retrieval or --posteriors to recover selected document text.")
    source_path = args.posteriors or args.retrieval
    source = load_candidate_source(resolve_project_path(source_path))
    output_path = resolve_project_path(args.output)
    written = write_jsonl(
        output_path,
        iter_oracle_rows(
            teacher_path=resolve_project_path(args.teacher),
            candidate_source=source,
            top_m=args.top_m,
            method_name=args.method_name,
            limit=args.limit,
        ),
    )
    print(f"[cbwdm_oracle] rows={written} method={args.method_name} output={output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml, read_jsonl, write_jsonl
from src.selector_dataset import build_candidate_feature
from src.selector_model import load_feature_selector_checkpoint
from src.selection_schema import make_selection_row, normalize_selected_doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedily select evidence with a trained feature MLP selector.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--posteriors", required=True, help="Posterior JSONL path.")
    parser.add_argument("--checkpoint", required=True, help="Feature MLP selector checkpoint.")
    parser.add_argument("--output", required=True, help="Selection JSONL output path.")
    parser.add_argument("--top-m", type=int, default=None, help="Maximum number of selected documents.")
    parser.add_argument("--score-threshold", type=float, default=None, help="Stop if best score is below threshold.")
    parser.add_argument("--min-docs", type=int, default=None, help="Minimum number of documents to keep.")
    parser.add_argument("--limit", type=int, default=None, help="Max posterior rows to process.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Only consider first K candidates per row.")
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def selector_params(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    selector = config.get("selector", {})
    return {
        "top_m": args.top_m if args.top_m is not None else int(selector.get("top_m", 4)),
        "min_docs": args.min_docs if args.min_docs is not None else int(selector.get("min_docs", 1)),
        "score_threshold": (
            args.score_threshold if args.score_threshold is not None else selector.get("score_threshold")
        ),
    }


def _candidate_score(candidate: Dict[str, Any]) -> Any:
    return candidate.get("retrieval_score", candidate.get("score"))


def select_row(row: Dict[str, Any], model: torch.nn.Module, params: Dict[str, Any], max_candidates: int | None) -> Dict[str, Any]:
    """Run greedy feature-MLP selection for one posterior row."""
    candidates = list(row.get("candidates", []))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    selected_doc_ids: list[str] = []
    selected_docs: list[Dict[str, Any]] = []
    threshold = params["score_threshold"]
    top_m = max(int(params["top_m"]), 0)
    min_docs = max(int(params["min_docs"]), 0)
    selection_steps: list[dict[str, Any]] = []
    stop_reason = "top_m_reached" if top_m == 0 else "no_candidates"

    for step_idx in range(min(top_m, len(candidates))):
        remaining = [candidate for candidate in candidates if candidate["doc_id"] not in selected_doc_ids]
        if not remaining:
            break

        scored = []
        for candidate in remaining:
            features = build_candidate_feature(row, selected_doc_ids, candidate["doc_id"], step_idx)
            with torch.inference_mode():
                score = float(model(torch.from_numpy(features).float().unsqueeze(0)).item())
            scored.append((score, candidate))

        best_score, best_candidate = max(scored, key=lambda item: (item[0], -int(item[1].get("rank", 0) or 0)))
        if threshold is not None and len(selected_doc_ids) >= min_docs and best_score < float(threshold):
            stop_reason = "score_below_threshold"
            selection_steps.append(
                {
                    "step": step_idx,
                    "selected_doc_id": None,
                    "predicted_score": best_score,
                    "remaining_count": len(remaining),
                    "stop": True,
                    "stop_reason": stop_reason,
                }
            )
            break

        selected_doc_ids.append(best_candidate["doc_id"])
        selected_docs.append(
            normalize_selected_doc(
                best_candidate, selector_score=best_score, selection_step=step_idx
            )
        )
        selection_steps.append(
            {
                "step": step_idx,
                "selected_doc_id": str(best_candidate["doc_id"]),
                "predicted_score": best_score,
                "remaining_count": len(remaining),
                "stop": False,
            }
        )
        stop_reason = "top_m_reached" if len(selected_docs) >= top_m else "no_candidates"
    return make_selection_row(
        row,
        method="feature_mlp_selector",
        selected_docs=selected_docs,
        selection_steps=selection_steps,
        stop_reason=stop_reason,
    )


def iter_selection_rows(
    posteriors_path: Path,
    model: torch.nn.Module,
    params: Dict[str, Any],
    limit: int | None,
    max_candidates: int | None,
) -> Iterator[Dict[str, Any]]:
    """Stream selection rows from posterior JSONL input."""
    for row in read_jsonl(posteriors_path, limit=limit):
        yield select_row(row, model, params, max_candidates=max_candidates)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    params = selector_params(config, args)
    checkpoint = load_feature_selector_checkpoint(resolve_project_path(args.checkpoint))
    model = checkpoint["model"]
    output_path = resolve_project_path(args.output)
    written = write_jsonl(
        output_path,
        iter_selection_rows(
            resolve_project_path(args.posteriors),
            model,
            params,
            limit=args.limit,
            max_candidates=args.max_candidates,
        ),
    )
    print(
        f"[selector_select] rows={written} top_m={params['top_m']} "
        f"checkpoint={resolve_project_path(args.checkpoint)} output={output_path}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import build_selection_contract, publish_selection
from src.io_utils import load_yaml, read_jsonl
from src.selector_cross_encoder import CrossEncoderSelector, build_selector_input
from src.selection_schema import make_selection_row, normalize_selected_doc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for cross-encoder selector inference."""
    parser = argparse.ArgumentParser(description="Greedily select evidence with a cross-encoder selector.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--posteriors", required=True, help="Posterior JSONL path.")
    parser.add_argument("--checkpoint-dir", required=True, help="Cross-encoder checkpoint directory.")
    parser.add_argument("--output", required=True, help="Selection JSONL output path.")
    parser.add_argument("--top-m", type=int, default=None, help="Maximum number of selected documents.")
    parser.add_argument("--max-length", type=int, default=None, help="Tokenizer max sequence length override.")
    parser.add_argument("--batch-size", type=int, default=8, help="Candidate scoring batch size.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--score-threshold", type=float, default=None, help="Stop if best score is below threshold.")
    parser.add_argument("--min-docs", type=int, default=None, help="Minimum documents before threshold stopping.")
    parser.add_argument("--limit", type=int, default=None, help="Max posterior rows to process.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Only consider first K candidates per row.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def selector_params(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Resolve selector parameters from CLI overrides and config defaults."""
    selector_config = config.get("selector", {})
    return {
        "top_m": args.top_m if args.top_m is not None else int(selector_config.get("top_m", 4)),
        "min_docs": args.min_docs if args.min_docs is not None else int(selector_config.get("min_docs", 1)),
        "score_threshold": (
            args.score_threshold if args.score_threshold is not None else selector_config.get("score_threshold")
        ),
    }


def _candidate_score(candidate: dict[str, Any]) -> Any:
    return candidate.get("retrieval_score", candidate.get("score"))


def select_row(
    row: dict[str, Any],
    selector: CrossEncoderSelector,
    params: dict[str, Any],
    batch_size: int,
    max_candidates: int | None,
) -> dict[str, Any]:
    """Run state-aware greedy cross-encoder selection for one posterior row."""
    candidates = list(row.get("candidates", []))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    top_m = max(int(params["top_m"]), 0)
    min_docs = max(int(params["min_docs"]), 0)
    threshold = params["score_threshold"]
    selected_doc_ids: list[str] = []
    selected_docs: list[dict[str, Any]] = []
    selection_steps: list[dict[str, Any]] = []
    stop_reason = "top_m_reached" if top_m == 0 else "no_candidates"

    for step_idx in range(min(top_m, len(candidates))):
        remaining = [candidate for candidate in candidates if candidate.get("doc_id") not in selected_doc_ids]
        if not remaining:
            break

        texts = [
            build_selector_input(
                query=str(row.get("query") or ""),
                selected_docs=selected_docs,
                candidate_doc=candidate,
            )
            for candidate in remaining
        ]
        scores = selector.score_texts(texts, batch_size=batch_size, requires_grad=False)
        score_values = [float(value) for value in scores.detach().cpu().tolist()]
        scored = list(zip(score_values, remaining))
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
                    "top_candidates": [
                        {"doc_id": item[1].get("doc_id"), "score": item[0]}
                        for item in sorted(scored, reverse=True, key=lambda value: value[0])[:5]
                    ],
                }
            )
            break

        selected_doc_ids.append(str(best_candidate["doc_id"]))
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
                "top_candidates": [
                    {"doc_id": item[1].get("doc_id"), "score": item[0]}
                    for item in sorted(scored, reverse=True, key=lambda value: value[0])[:5]
                ],
            }
        )
        stop_reason = (
            "top_m_reached"
            if len(selected_docs) >= top_m
            else "no_candidates"
        )
    return make_selection_row(
        row,
        method="rag_cbwdm",
        selected_docs=selected_docs,
        selection_steps=selection_steps,
        stop_reason=stop_reason,
        max_docs=top_m,
        selection_metadata={
            "method": "rag_cbwdm",
            "state_aware": True,
            "uses_gold_at_test": False,
            "teacher_uses_gold_train": True,
            "score_variant": "euclidean_posterior_shift",
            "min_docs": min_docs,
            "score_threshold": threshold,
        },
    )


def iter_selection_rows(
    posteriors_path: Path,
    selector: CrossEncoderSelector,
    params: dict[str, Any],
    batch_size: int,
    limit: int | None,
    max_candidates: int | None,
) -> Iterator[dict[str, Any]]:
    """Stream cross-encoder selection rows from posterior JSONL input."""
    for row in read_jsonl(posteriors_path, limit=limit):
        yield select_row(
            row=row,
            selector=selector,
            params=params,
            batch_size=batch_size,
            max_candidates=max_candidates,
        )


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_project_path(args.config))
    params = selector_params(config, args)
    output_path = resolve_project_path(args.output)
    posterior_path = resolve_project_path(args.posteriors)
    checkpoint_path = resolve_project_path(args.checkpoint_dir)
    checkpoint_config = checkpoint_path / "config.json"
    checkpoint_weights = next(
        (
            path
            for name in ("model.safetensors", "pytorch_model.bin")
            if (path := checkpoint_path / name).is_file()
        ),
        None,
    )
    if checkpoint_weights is None:
        raise FileNotFoundError(f"No selector weights found under {checkpoint_path}")
    contract = build_selection_contract(
        method="rag_cbwdm",
        input_paths={
            "posteriors": posterior_path,
            "checkpoint_config": checkpoint_config,
            "checkpoint_weights": checkpoint_weights,
        },
        parameters={
            **params,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "max_candidates": args.max_candidates,
            "limit": args.limit,
        },
        model={"checkpoint": str(checkpoint_path.resolve())},
    )
    if args.resume and output_path.exists() and not args.overwrite:
        written, reused = publish_selection(
            output_path,
            [],
            contract=contract,
            project_root=PROJECT_ROOT,
            resume=True,
        )
        print(
            f"[cross_encoder_select] rows={written} top_m={params['top_m']} "
            f"checkpoint={checkpoint_path} reused={reused} output={output_path}"
        )
        return
    selector = CrossEncoderSelector.load_checkpoint(
        checkpoint_dir=checkpoint_path,
        max_length=args.max_length,
        device=args.device,
    )
    selector.model.eval()
    written, reused = publish_selection(
        output_path,
        iter_selection_rows(
            posteriors_path=posterior_path,
            selector=selector,
            params=params,
            batch_size=args.batch_size,
            limit=args.limit,
            max_candidates=args.max_candidates,
        ),
        contract=contract,
        project_root=PROJECT_ROOT,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(
        f"[cross_encoder_select] rows={written} top_m={params['top_m']} "
        f"checkpoint={checkpoint_path} reused={reused} output={output_path}"
    )


if __name__ == "__main__":
    main()

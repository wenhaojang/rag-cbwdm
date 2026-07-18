from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import build_selection_contract, publish_selection
from src.baselines.infogain_fever import TEACHER_DEFINITION, pointwise_input
from src.baselines.infogain_selector import InfoGainPointwiseReranker
from src.io_utils import read_jsonl
from src.run_manifest import sha256_file
from src.selection_schema import make_selection_row, normalize_selected_doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select FEVER evidence with a trained InfoGain-FEVER reranker.")
    parser.add_argument("--retrieval", required=True, help="Gold-free BM25 retrieval JSONL.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--filter-threshold", type=float)
    parser.add_argument("--min-docs", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_row(
    row: dict[str, Any],
    model: InfoGainPointwiseReranker,
    *,
    batch_size: int,
    top_m: int,
    filter_threshold: float | None,
    min_docs: int,
    checkpoint_metadata: dict[str, Any],
) -> dict[str, Any]:
    # Deliberately read only id/query/candidates; gold is never inspected.
    candidates = list(row.get("candidates", []))
    texts = [
        pointwise_input(str(row.get("query") or ""), item.get("title"), item.get("text"))
        for item in candidates
    ]
    rank_scores, filter_probabilities = model.score(texts, batch_size)
    ranked = sorted(
        zip(candidates, rank_scores, filter_probabilities),
        key=lambda item: (-float(item[1]), int(item[0].get("rank") or 10**9)),
    )
    chosen = []
    for candidate, rank_score, filter_probability in ranked:
        if len(chosen) >= top_m:
            break
        passes = filter_threshold is None or filter_probability >= filter_threshold
        fallback = not passes and len(chosen) < min_docs
        if passes or fallback:
            chosen.append((candidate, rank_score, fallback))
    rank_by_id = {
        str(candidate.get("doc_id")): rank_scores[index]
        for index, candidate in enumerate(candidates)
    }
    filter_by_id = {
        str(candidate.get("doc_id")): filter_probabilities[index]
        for index, candidate in enumerate(candidates)
    }
    docs, steps = [], []
    for step, (candidate, _, fallback) in enumerate(chosen):
        doc_id = str(candidate.get("doc_id"))
        doc = normalize_selected_doc(
            candidate, selector_score=rank_by_id[doc_id], selection_step=step
        )
        doc.update(
            {
                "rank_score": rank_by_id[doc_id],
                "filter_probability": filter_by_id[doc_id],
                "min_docs_fallback": fallback,
            }
        )
        docs.append(doc)
        steps.append(
            {
                "step": step,
                "selected_doc_id": doc_id,
                "predicted_score": rank_by_id[doc_id],
                "filter_probability": filter_by_id[doc_id],
                "min_docs_fallback": fallback,
                "stop": False,
            }
        )
    return make_selection_row(
        row,
        method="infogain_fever",
        selected_docs=docs,
        selection_steps=steps,
        stop_reason="top_m_reached" if len(docs) == top_m else "threshold_or_candidates_exhausted",
        max_docs=top_m,
        selection_metadata={
            "official_fidelity": "adapted",
            "teacher_definition": TEACHER_DEFINITION,
            "state_aware": False,
            "uses_gold_at_test": False,
            "deployable": True,
            "filter_threshold": filter_threshold,
            "min_docs": min_docs,
            "checkpoint_fingerprint": checkpoint_metadata.get("fingerprint"),
        },
    )


def main() -> None:
    args = parse_args()
    retrieval = absolute(args.retrieval)
    checkpoint = absolute(args.checkpoint_dir)
    output = absolute(args.output)
    checkpoint_metadata = json.loads(
        (checkpoint / "infogain_config.json").read_text(encoding="utf-8")
    )
    encoder_weights = next(
        (
            path
            for name in ("model.safetensors", "pytorch_model.bin")
            if (path := checkpoint / "encoder" / name).is_file()
        ),
        None,
    )
    if encoder_weights is None:
        raise FileNotFoundError(f"No encoder weights found under {checkpoint / 'encoder'}")
    contract = build_selection_contract(
        method="infogain_fever",
        input_paths={
            "retrieval": retrieval,
            "checkpoint_config": checkpoint / "infogain_config.json",
            "checkpoint_heads": checkpoint / "heads.pt",
            "checkpoint_encoder": encoder_weights,
        },
        parameters={
            "top_m": args.top_m,
            "min_docs": args.min_docs,
            "filter_threshold": args.filter_threshold,
            "limit": args.limit,
        },
        model={
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_fingerprint": checkpoint_metadata.get("fingerprint"),
        },
    )
    if args.resume and output.exists() and not args.overwrite:
        written, reused = publish_selection(
            output,
            [],
            contract=contract,
            project_root=PROJECT_ROOT,
            resume=True,
        )
        print(f"[infogain_select] rows={written} reused={reused} output={output}")
        return
    model, checkpoint_metadata = InfoGainPointwiseReranker.load(
        checkpoint, device=args.device
    )
    rows = read_jsonl(retrieval, limit=args.limit)
    written, reused = publish_selection(
        output,
        (
            select_row(
                row,
                model,
                batch_size=args.batch_size,
                top_m=args.top_m,
                filter_threshold=args.filter_threshold,
                min_docs=args.min_docs,
                checkpoint_metadata=checkpoint_metadata,
            )
            for row in rows
        ),
        contract=contract,
        project_root=PROJECT_ROOT,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(f"[infogain_select] rows={written} reused={reused} output={output}")


if __name__ == "__main__":
    main()

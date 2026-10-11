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
from src.baselines.infogain import TEACHER_DEFINITION, pointwise_input
from src.baselines.infogain_selector import InfoGainPointwiseReranker
from src.baselines.infogain_protocol import (
    budget_protocol_id,
    rank_and_select,
    stop_reason_for_selection,
)
from src.artifact_binding import validate_formal_training_binding
from src.io_utils import read_jsonl
from src.run_manifest import sha256_file
from src.selection_schema import make_selection_row, normalize_selected_doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select evidence with a trained classification InfoGain reranker.")
    parser.add_argument("--retrieval", required=True, help="Shared candidate-pool JSONL.")
    parser.add_argument("--retrieval-manifest")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--training-manifest")
    parser.add_argument("--dataset-id")
    parser.add_argument("--generator-id")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--formal-v2-identity", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--filter-threshold", type=float)
    parser.add_argument("--min-docs", type=int, default=2)
    parser.add_argument("--method-name", default="infogain_fever")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolved_protocol_id(
    *, top_m: int, filter_threshold: float | None, min_docs: int
) -> str | None:
    if filter_threshold is None:
        return None
    try:
        return budget_protocol_id(
            top_m=top_m,
            filter_threshold=filter_threshold,
            min_docs=min_docs,
        )
    except ValueError:
        return None


def select_row(
    row: dict[str, Any],
    model: InfoGainPointwiseReranker,
    *,
    batch_size: int,
    top_m: int,
    filter_threshold: float | None,
    min_docs: int,
    checkpoint_metadata: dict[str, Any],
    method_name: str = "infogain_fever",
) -> dict[str, Any]:
    # Deliberately read only id/query/candidates; gold is never inspected.
    candidates = list(row.get("candidates", []))
    texts = [
        pointwise_input(str(row.get("query") or ""), item.get("title"), item.get("text"))
        for item in candidates
    ]
    rank_scores, filter_probabilities = model.score(texts, batch_size)
    ranked, chosen = rank_and_select(
        candidates,
        rank_scores,
        filter_probabilities,
        top_m=top_m,
        filter_threshold=filter_threshold,
        min_docs=min_docs,
    )
    docs, steps = [], []
    for step, item in enumerate(chosen):
        candidate = item["candidate"]
        doc_id = item["doc_id"]
        doc = normalize_selected_doc(
            candidate, selector_score=item["rank_score"], selection_step=step
        )
        doc.update(
            {
                "rank_score": item["rank_score"],
                "filter_probability": item["filter_probability"],
                "min_docs_fallback": item["min_docs_fallback"],
            }
        )
        docs.append(doc)
        steps.append(
            {
                "step": step,
                "selected_doc_id": doc_id,
                "predicted_score": item["rank_score"],
                "filter_probability": item["filter_probability"],
                "min_docs_fallback": item["min_docs_fallback"],
                "stop": False,
            }
        )
    return make_selection_row(
        row,
        method=method_name,
        selected_docs=docs,
        selection_steps=steps,
        stop_reason=stop_reason_for_selection(
            candidate_count=len(candidates),
            selected_count=len(docs),
            top_m=top_m,
            min_docs=min_docs,
            filter_threshold=filter_threshold,
            used_fallback=any(item["min_docs_fallback"] for item in chosen),
        ),
        max_docs=top_m,
        selection_metadata={
            "official_fidelity": "adapted",
            "teacher_definition": TEACHER_DEFINITION,
            "state_aware": False,
            "uses_gold_at_test": False,
            "deployable": True,
            "filter_threshold": filter_threshold,
            "top_m": top_m,
            "min_docs": min_docs,
            "selection_protocol_id": (
                resolved_protocol_id(
                    top_m=top_m,
                    filter_threshold=filter_threshold,
                    min_docs=min_docs,
                )
            ),
            "ranking_score_semantics": "unconstrained_rank_head_score",
            "filter_probability_semantics": "softmax_filter_head_positive_class",
            "candidate_score_audit": [
                {
                    key: item[key]
                    for key in (
                        "doc_id",
                        "source_position",
                        "rerank_position",
                        "rank_score",
                        "filter_probability",
                        "within_top_m",
                        "passes_filter",
                    )
                }
                for item in ranked
            ],
            "checkpoint_fingerprint": checkpoint_metadata.get("fingerprint"),
        },
    )


def main() -> None:
    args = parse_args()
    retrieval = absolute(args.retrieval)
    retrieval_manifest = (
        absolute(args.retrieval_manifest) if args.retrieval_manifest else None
    )
    checkpoint = absolute(args.checkpoint_dir)
    output = absolute(args.output)
    training_manifest = (
        absolute(args.training_manifest)
        if args.training_manifest
        else checkpoint.parent / "training_manifest.json"
    )
    artifact_binding = None
    if args.formal_v2_identity:
        artifact_binding = validate_formal_training_binding(
            training_manifest,
            checkpoint,
            method=args.method_name,
            expected_dataset_id=args.dataset_id,
            expected_conditioning_generator_id=args.generator_id,
            expected_seed=args.expected_seed,
        )
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
        method=args.method_name,
        input_paths={
            "retrieval": retrieval,
            **(
                {"retrieval_manifest": retrieval_manifest}
                if retrieval_manifest is not None
                else {}
            ),
            "checkpoint_config": checkpoint / "infogain_config.json",
            "checkpoint_heads": checkpoint / "heads.pt",
            "checkpoint_encoder": encoder_weights,
            **(
                {"training_manifest": training_manifest}
                if args.formal_v2_identity
                else {}
            ),
        },
        parameters={
            "top_m": args.top_m,
            "min_docs": args.min_docs,
            "filter_threshold": args.filter_threshold,
            "limit": args.limit,
            **(
                {"selection_protocol_id": protocol_id}
                if (
                    protocol_id := resolved_protocol_id(
                        top_m=args.top_m,
                        filter_threshold=args.filter_threshold,
                        min_docs=args.min_docs,
                    )
                )
                else {}
            ),
        },
        model={
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_fingerprint": checkpoint_metadata.get("fingerprint"),
        },
        artifact_binding=artifact_binding,
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
                method_name=args.method_name,
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

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import build_selection_contract, publish_selection
from src.diagnostics.method_failure import require_diagnostic_output
from src.diagnostics.signed_selector_v21 import (
    SIGNED_V21_ARCHITECTURE,
    SIGNED_V21_METHOD,
    SIGNED_V21_VARIANT,
    select_row_without_gold,
)
from src.formal_provenance import sha256_path
from src.io_utils import load_yaml, read_jsonl
from src.selector_cross_encoder import CrossEncoderSelector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deployable experimental rag_cbwdm_signed_v21 inference"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--min-docs", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    fixed = {"top_m": 4, "min_docs": 0, "score_threshold": 0.0}
    changed = {
        name: (getattr(args, name), expected)
        for name, expected in fixed.items()
        if getattr(args, name) != expected
    }
    if changed:
        raise ValueError(f"signed-v2.1 first-round inference is frozen: {changed}")
    if args.max_candidates is not None and args.max_candidates < 1:
        raise ValueError("max-candidates must be >= 1")
    if args.validation_limit is not None and args.validation_limit < 1:
        raise ValueError("validation-limit must be >= 1")

    config_path = Path(args.config).resolve()
    load_yaml(config_path)
    run = Path(args.run_dir).resolve()
    output = require_diagnostic_output(run, Path(args.output).resolve())
    root = (
        run / "artifacts/diagnostics/method_failure_audit/signed_v21/selections"
    ).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"selection output must be below {root}") from exc

    posterior = Path(args.posteriors).resolve()
    checkpoint = Path(args.checkpoint_dir).resolve()
    weights = next(
        (
            checkpoint / name
            for name in ("model.safetensors", "pytorch_model.bin")
            if (checkpoint / name).is_file()
        ),
        None,
    )
    if weights is None:
        raise FileNotFoundError(f"No selector weights under {checkpoint}")
    checkpoint_config = checkpoint / "selector_config.json"
    training_manifest = checkpoint.parent / "training_manifest.json"
    for path in (posterior, checkpoint_config, training_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    training_payload = json.loads(training_manifest.read_text(encoding="utf-8"))
    try:
        checkpoint.relative_to(
            (
                run
                / "artifacts/diagnostics/method_failure_audit/signed_v21/training"
            ).resolve()
        )
    except ValueError as exc:
        raise ValueError("Checkpoint must be under isolated signed_v21/training") from exc
    if (
        training_payload.get("status") != "completed"
        or training_payload.get("method") != SIGNED_V21_METHOD
        or training_payload.get("variant") != SIGNED_V21_VARIANT
        or training_payload.get("architecture") != SIGNED_V21_ARCHITECTURE
        or training_payload.get("contract", {}).get("variant")
        != "signed_selector_v21_training"
        or not training_payload.get("experimental")
    ):
        raise ValueError("Checkpoint is not a completed experimental signed-v2.1 artifact")
    actual_checkpoint_sha = sha256_path(checkpoint)
    if training_payload.get("checkpoint_sha256") != actual_checkpoint_sha:
        raise ValueError("signed-v2.1 checkpoint checksum differs from training manifest")

    parameters = {
        "top_m": args.top_m,
        "min_docs": args.min_docs,
        "score_threshold": args.score_threshold,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_candidates": args.max_candidates,
        "validation_limit": args.validation_limit,
        "experimental": True,
        "uses_gold_at_inference": False,
        "score_quantity": "raw_sequence_classification_logit",
    }
    contract = build_selection_contract(
        method=SIGNED_V21_METHOD,
        input_paths={
            "posteriors": posterior,
            "checkpoint_config": checkpoint_config,
            "checkpoint_weights": weights,
            "training_manifest": training_manifest,
        },
        parameters=parameters,
        model={
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_checkpoint_sha,
            "architecture": SIGNED_V21_ARCHITECTURE,
        },
    )
    if args.resume and output.is_file():
        written, reused = publish_selection(
            output,
            [],
            contract=contract,
            project_root=PROJECT_ROOT,
            resume=True,
        )
        print(
            f"[signed_selector_v21] rows={written} reused={reused} output={output}"
        )
        return
    selector = CrossEncoderSelector.load_checkpoint(
        checkpoint, max_length=args.max_length, device=args.device
    )
    selector.model.eval()
    rows = (
        select_row_without_gold(
            row,
            selector,
            top_m=args.top_m,
            min_docs=args.min_docs,
            score_threshold=args.score_threshold,
            batch_size=args.batch_size,
            max_candidates=args.max_candidates,
        )
        for row in read_jsonl(posterior, limit=args.validation_limit)
    )
    written, reused = publish_selection(
        output,
        rows,
        contract=contract,
        project_root=PROJECT_ROOT,
        resume=args.resume,
    )
    print(
        f"[signed_selector_v21] rows={written} reused={reused} "
        f"uses_gold_at_inference=false output={output}"
    )


if __name__ == "__main__":
    main()

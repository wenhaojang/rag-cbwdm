from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import build_selection_contract, publish_selection
from src.artifact_binding import validate_formal_training_binding
from src.diagnostics.signed_selector_v1 import select_row_without_gold
from src.formal_provenance import sha256_path
from src.io_utils import read_jsonl
from src.preformal.registry import (
    KCBWDM_LINEAR_GATE_V2_CONTRACT,
    KCBWDM_LINEAR_GATE_V2_METHOD,
    KCBWDM_SIGNED_V1_CONTRACT,
    KCBWDM_SIGNED_V1_METHOD,
    SIGNED_V1_CONTRACT,
    SIGNED_V1_METHOD,
    assert_frozen_kcbwdm_linear_gate_v2_contract,
    assert_frozen_kcbwdm_contract,
    assert_frozen_signed_contract,
    validate_training_method_contract,
)
from src.run_manifest import sha256_file
from src.selector_cross_encoder import CrossEncoderSelector


SUPPORTED_SIGNED_METHODS = (
    SIGNED_V1_METHOD,
    KCBWDM_SIGNED_V1_METHOD,
    KCBWDM_LINEAR_GATE_V2_METHOD,
)


def method_contract(method_name: str) -> dict:
    if method_name == SIGNED_V1_METHOD:
        return SIGNED_V1_CONTRACT
    if method_name == KCBWDM_SIGNED_V1_METHOD:
        return KCBWDM_SIGNED_V1_CONTRACT
    if method_name == KCBWDM_LINEAR_GATE_V2_METHOD:
        return KCBWDM_LINEAR_GATE_V2_CONTRACT
    raise ValueError(f"Unsupported signed selector method: {method_name!r}")


def validate_checkpoint_method_identity(
    payload: dict, *, method_name: str, seed: int
) -> str:
    version = validate_training_method_contract(payload, method_name=method_name)
    if payload.get("seed") != seed or payload.get("status") != "completed":
        raise ValueError("Checkpoint is not the requested completed method/seed")
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal deployable rag_cbwdm_signed_v1 selection")
    parser.add_argument("--posteriors", required=True); parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--training-manifest"); parser.add_argument("--dataset-id"); parser.add_argument("--generator-id")
    parser.add_argument("--formal-v2-identity", action="store_true")
    parser.add_argument(
        "--method-name", choices=SUPPORTED_SIGNED_METHODS, default=SIGNED_V1_METHOD
    )
    parser.add_argument("--output", required=True); parser.add_argument("--seed", type=int, required=True, choices=[13,21,42])
    parser.add_argument("--device", default="auto"); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=["dev", "test", "validation", "preformal_eval", "held_out_test"], default="preformal_eval")
    parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    active_contract = method_contract(args.method_name)
    frozen = active_contract["selector"]
    frozen_parameters = {"top_m": frozen["top_m"], "min_docs": frozen["min_docs"], "score_threshold": frozen["score_threshold"]}
    if args.method_name == SIGNED_V1_METHOD:
        assert_frozen_signed_contract(frozen_parameters)
    elif args.method_name == KCBWDM_SIGNED_V1_METHOD:
        assert_frozen_kcbwdm_contract(
            {**frozen_parameters, "method": args.method_name, "seed": args.seed}
        )
        if args.split == "held_out_test":
            raise ValueError("kcbwdm_signed_v1 is development-only and cannot select held_out_test")
    else:
        assert_frozen_kcbwdm_linear_gate_v2_contract(
            {**frozen_parameters, "method": args.method_name, "seed": args.seed}
        )
        if args.split == "held_out_test":
            raise ValueError(
                "kcbwdm_linear_gate_v2 is development-only and cannot select held_out_test"
            )
    posterior = Path(args.posteriors).resolve(); checkpoint = Path(args.checkpoint_dir).resolve(); output = Path(args.output).resolve()
    training_manifest = (Path(args.training_manifest).resolve() if args.training_manifest
        else checkpoint.parent / "training_manifest.json")
    payload = json.loads(training_manifest.read_text(encoding="utf-8"))
    validated_contract_version = validate_checkpoint_method_identity(
        payload, method_name=args.method_name, seed=args.seed
    )
    if payload.get("checkpoint_sha256") != sha256_path(checkpoint): raise ValueError("Checkpoint SHA mismatch")
    if payload.get("train_core_posterior_sha256") == sha256_file(posterior):
        raise ValueError("Refusing selection on the training posterior artifact")
    artifact_binding = None
    if args.formal_v2_identity:
        artifact_binding = validate_formal_training_binding(
            training_manifest, checkpoint, method=args.method_name,
            expected_dataset_id=args.dataset_id,
            expected_conditioning_generator_id=args.generator_id,
            expected_seed=args.seed,
        )
    parameters = {"seed": args.seed, "top_m": frozen["top_m"], "min_docs": frozen["min_docs"],
        "score_threshold": frozen["score_threshold"], "uses_gold_at_inference": False, "split": args.split}
    if args.limit is not None:
        parameters["limit"] = args.limit
    contract = build_selection_contract(method=args.method_name,
        method_contract_version=validated_contract_version,
        input_paths={"posteriors": posterior,
        "training_manifest": training_manifest}, parameters=parameters,
        model={"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_path(checkpoint)},
        artifact_binding=artifact_binding)
    if args.resume and output.is_file():
        written, reused = publish_selection(output, [], contract=contract, project_root=PROJECT_ROOT, resume=True)
        print(f"[preformal_signed_select] rows={written} reused={reused} output={output}"); return
    selector = CrossEncoderSelector.load_checkpoint(checkpoint, device=args.device); selector.model.eval()
    all_rows = list(read_jsonl(posterior))
    if not all_rows or {row.get("split") for row in all_rows} != {args.split}:
        raise ValueError(f"Selection input must contain only {args.split} rows")
    rows = all_rows[: args.limit] if args.limit is not None else all_rows
    selected = (select_row_without_gold(row, selector, method=args.method_name, top_m=frozen["top_m"], min_docs=frozen["min_docs"],
        score_threshold=frozen["score_threshold"], batch_size=args.batch_size, max_candidates=None) for row in rows)
    written, reused = publish_selection(output, selected, contract=contract, project_root=PROJECT_ROOT, resume=args.resume)
    print(f"[preformal_signed_select] rows={written} reused={reused} uses_gold_at_inference=false output={output}")


if __name__ == "__main__": main()

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.method_failure import require_diagnostic_output
from src.diagnostics.signed_selector_v2 import (
    build_signed_v2_training_groups,
    signed_v2_loss,
)
from src.diagnostics.signed_v2_model import (
    SIGNED_V2_ARCHITECTURE,
    SIGNED_V2_METHOD,
    SignedV2DualHeadSelector,
)
from src.formal_provenance import sha256_path
from src.io_utils import load_yaml
from src.run_manifest import (
    atomic_write_json,
    git_state,
    sha256_file,
    stable_hash,
    utc_now,
)
from src.selector_cross_encoder import build_selector_input


TRAINING_MANIFEST_SCHEMA = "rag_cbwdm_signed_v2_training_manifest.v1"


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def build_completed_manifest(
    *,
    contract: dict[str, Any],
    fingerprint: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    outputs: list[Path],
    num_groups: int,
) -> dict[str, Any]:
    """Build a manifest that cannot be mistaken for signed-v1."""
    return {
        "schema_version": TRAINING_MANIFEST_SCHEMA,
        "method": SIGNED_V2_METHOD,
        "architecture": SIGNED_V2_ARCHITECTURE,
        "status": "completed",
        "completed": True,
        "fingerprint": fingerprint,
        "contract": contract,
        "diagnostic_only": True,
        "experimental": True,
        "num_groups": num_groups,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "outputs": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in outputs
        },
        "git": git_state(PROJECT_ROOT),
        "completed_at": utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train experimental rag_cbwdm_signed_v2 dual-head selector"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-name", default="/root/models/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-groups", type=int, default=None)
    parser.add_argument("--alignment-eps", type=float, default=0.0)
    parser.add_argument("--b-plus", type=float, default=0.01)
    parser.add_argument("--b-minus", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda-gate", type=float, default=1.0)
    parser.add_argument("--lambda-utility", type=float, default=1.0)
    parser.add_argument(
        "--neutral-sample-policy", choices=["negative", "ignore"], default="negative"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be >= 1")
    if args.max_train_groups is not None and args.max_train_groups < 1:
        raise ValueError("max-train-groups must be >= 1")
    fixed = {
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 8,
        "alignment_eps": 0.0,
        "b_plus": 0.01,
        "b_minus": 0.001,
        "beta": 0.25,
        "gamma": 1.0,
        "lambda_gate": 1.0,
        "lambda_utility": 1.0,
        "neutral_sample_policy": "negative",
        "seed": 13,
    }
    changed = {
        name: (getattr(args, name), expected)
        for name, expected in fixed.items()
        if getattr(args, name) != expected
    }
    if changed:
        raise ValueError(f"signed-v2 first-round hyperparameters are frozen: {changed}")

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    run = Path(args.run_dir).resolve()
    output = require_diagnostic_output(run, Path(args.output_dir).resolve())
    signed_root = (
        run / "artifacts/diagnostics/method_failure_audit/signed_teacher_v2"
    ).resolve()
    try:
        output.relative_to(signed_root / "training")
    except ValueError as exc:
        raise ValueError(
            f"training output must be below {signed_root / 'training'}"
        ) from exc

    teacher = Path(args.teacher).resolve()
    posteriors = Path(args.posteriors).resolve()
    retrieval = Path(args.retrieval).resolve()
    model = Path(args.model_name).resolve()
    teacher_manifest = teacher.parent / "manifest.json"
    for path in (config_path, teacher, posteriors, retrieval, teacher_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not model.exists():
        raise FileNotFoundError(model)

    teacher_payload = json.loads(teacher_manifest.read_text(encoding="utf-8"))
    teacher_contract = teacher_payload.get("contract", {})
    if teacher_contract.get("variant") != "signed_teacher_v1":
        raise ValueError("v2 must reuse an authoritative signed_teacher_v1 artifact")
    teacher_params = teacher_contract.get("parameters", {})
    expected_teacher = {
        "alignment_eps": args.alignment_eps,
        "b_plus": args.b_plus,
        "b_minus": args.b_minus,
        "neutral_sample_policy": args.neutral_sample_policy,
        "l_type": config["cbwdm"].get("L_type", "euclidean_posterior_shift"),
        "eps_smooth": float(config["cbwdm"].get("eps_smooth", 0)),
        "target_smoothing": config["cbwdm"].get(
            "target_smoothing", "paper_mixture"
        ),
    }
    mismatched = {
        key: (teacher_params.get(key), expected)
        for key, expected in expected_teacher.items()
        if teacher_params.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"Teacher contract differs from signed-v2 contract: {mismatched}")

    training_contract = {
        "gate_target_definition": "1 iff authoritative alignment > alignment_eps; else 0",
        "utility_target_definition": (
            "signed-v1 effective gains/classes on authoritative-admissible candidates"
        ),
        "utility_mask": "authoritative alignment > alignment_eps",
        "alignment_definition": "X_j^T d from src.cbwdm_score.build_local_effects",
        "alignment_eps": args.alignment_eps,
        "b_plus": args.b_plus,
        "b_minus": args.b_minus,
        "neutral_sample_policy": args.neutral_sample_policy,
        "lambda_gate": args.lambda_gate,
        "lambda_utility": args.lambda_utility,
        "beta": args.beta,
        "gamma": args.gamma,
        "raw_gate_implemented": False,
        "step_balancing_implemented": False,
    }
    contract = {
        "variant": "signed_selector_v2_training",
        "method": SIGNED_V2_METHOD,
        "architecture": SIGNED_V2_ARCHITECTURE,
        "experimental": True,
        "config_sha256": sha256_file(config_path),
        "inputs": {
            "teacher": {"path": str(teacher), "sha256": sha256_file(teacher)},
            "teacher_manifest": {
                "path": str(teacher_manifest),
                "sha256": sha256_file(teacher_manifest),
            },
            "posteriors": {
                "path": str(posteriors),
                "sha256": sha256_file(posteriors),
            },
            "retrieval": {
                "path": str(retrieval),
                "sha256": sha256_file(retrieval),
            },
            "base_model": {"path": str(model), "sha256": sha256_path(model)},
        },
        "training_contract": training_contract,
        "parameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "max_train_groups": args.max_train_groups,
            "seed": args.seed,
            "device": args.device,
        },
    }
    fingerprint = stable_hash(contract)
    manifest_path = output / "training_manifest.json"
    checkpoint = output / "checkpoint"
    outputs = [output / "training_config.json", output / "training_history.json"]
    if args.resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "completed"
            and manifest.get("method") == SIGNED_V2_METHOD
            and manifest.get("fingerprint") == fingerprint
            and checkpoint.is_dir()
            and all(path.is_file() for path in outputs)
            and manifest.get("checkpoint_sha256") == sha256_path(checkpoint)
            and all(
                manifest["outputs"][path.name]["sha256"] == sha256_file(path)
                for path in outputs
            )
        ):
            print(f"[signed_selector_v2_train] reused=true output={output}")
            return
        raise ValueError("Cannot resume signed-v2 training: fingerprint/checksum mismatch")
    if manifest_path.exists() or checkpoint.exists() or any(path.exists() for path in outputs):
        raise FileExistsError(
            "signed-v2 training artifact exists; use matching --resume or a new directory"
        )

    groups = build_signed_v2_training_groups(
        teacher_path=teacher,
        posteriors_path=posteriors,
        retrieval_path=retrieval,
        max_train_groups=args.max_train_groups,
    )
    _seed(args.seed)
    selector = SignedV2DualHeadSelector(
        model_name=str(model),
        max_length=args.max_length,
        device=args.device,
        head_seed=args.seed,
    )
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        selector.train()
        ordered = list(groups)
        random.Random(args.seed + epoch).shuffle(ordered)
        optimizer.zero_grad()
        totals = {
            "total": 0.0,
            "gate": 0.0,
            "utility": 0.0,
            "utility_ce": 0.0,
            "utility_rank": 0.0,
            "valid_rank": 0,
            "skipped_rank": 0,
            "groups": 0,
        }
        for index, group in enumerate(ordered, start=1):
            texts = [
                build_selector_input(group.query, group.selected_docs, candidate)
                for candidate in group.candidate_docs
            ]
            logits = selector.score_texts(
                texts, batch_size=len(texts), requires_grad=True
            )
            loss, details = signed_v2_loss(
                logits.gate_logit,
                logits.utility_logit,
                alignments=group.alignments,
                effective_gains=group.effective_gains,
                alignment_eps=group.alignment_eps,
                b_plus=args.b_plus,
                b_minus=args.b_minus,
                beta=args.beta,
                gamma=args.gamma,
                neutral_sample_policy=args.neutral_sample_policy,
                lambda_gate=args.lambda_gate,
                lambda_utility=args.lambda_utility,
            )
            (loss / args.batch_size).backward()
            totals["total"] += float(loss.detach().cpu())
            totals["gate"] += float(details["gate_loss"].detach().cpu())
            totals["utility"] += float(details["utility_loss"].detach().cpu())
            totals["utility_ce"] += float(details["utility_ce_loss"].detach().cpu())
            totals["utility_rank"] += float(details["utility_rank_loss"].detach().cpu())
            totals["valid_rank"] += int(details["valid_ranking_group"])
            totals["skipped_rank"] += int(details["skipped_ranking_group"])
            totals["groups"] += 1
            if index % args.batch_size == 0 or index == len(ordered):
                optimizer.step()
                optimizer.zero_grad()
            if index % 100 == 0:
                print(
                    f"[signed_selector_v2_train] epoch={epoch} "
                    f"group={index}/{len(ordered)} loss={float(loss.detach().cpu()):.6f}"
                )
        denominator = totals["groups"]
        history.append(
            {
                "epoch": epoch,
                "avg_total_loss": totals["total"] / denominator,
                "avg_gate_loss": totals["gate"] / denominator,
                "avg_utility_loss": totals["utility"] / denominator,
                "avg_utility_ce_loss": totals["utility_ce"] / denominator,
                "avg_utility_rank_loss": totals["utility_rank"] / denominator,
                "valid_utility_ranking_groups": totals["valid_rank"],
                "skipped_utility_ranking_groups": totals["skipped_rank"],
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    selector.save_checkpoint(checkpoint, training_contract=training_contract)
    training_config = {
        **contract["parameters"],
        **training_contract,
        "method": SIGNED_V2_METHOD,
        "architecture": SIGNED_V2_ARCHITECTURE,
        "base_model": str(model),
        "num_groups": len(groups),
        "terminal_groups": sum(group.is_terminal_state for group in groups),
        "gate_positive_candidates": sum(
            int(target) for group in groups for target in group.gate_targets
        ),
        "gate_negative_candidates": sum(
            int(not target) for group in groups for target in group.gate_targets
        ),
    }
    atomic_write_json(outputs[0], training_config)
    atomic_write_json(outputs[1], {"epochs": history})
    checkpoint_sha = sha256_path(checkpoint)
    atomic_write_json(
        manifest_path,
        build_completed_manifest(
            contract=contract,
            fingerprint=fingerprint,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            outputs=outputs,
            num_groups=len(groups),
        ),
    )
    print(f"[signed_selector_v2_train] groups={len(groups)} checkpoint={checkpoint}")


if __name__ == "__main__":
    main()

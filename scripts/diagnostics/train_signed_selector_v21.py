from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.method_failure import require_diagnostic_output
from src.diagnostics.signed_selector_v21 import (
    SIGNED_V21_ARCHITECTURE,
    SIGNED_V21_LOSS_TYPE,
    SIGNED_V21_METHOD,
    SIGNED_V21_VARIANT,
    signed_v21_multitask_loss,
)
from src.diagnostics.signed_teacher_v1 import build_signed_training_groups
from src.formal_provenance import sha256_path
from src.io_utils import load_yaml
from src.run_manifest import (
    atomic_write_json,
    git_state,
    sha256_file,
    stable_hash,
    utc_now,
)
from src.selector_cross_encoder import CrossEncoderSelector, build_selector_input


TRAINING_MANIFEST_SCHEMA = "rag_cbwdm_signed_v21_training_manifest.v1"


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def create_selector(
    *, model_name: str, max_length: int, device: str, seed: int
) -> CrossEncoderSelector:
    """Create the unchanged pretrained single-head v1 selector."""
    _seed(seed)
    return CrossEncoderSelector(
        model_name=model_name,
        max_length=max_length,
        device=device,
    )


def build_completed_manifest(
    *,
    contract: dict[str, Any],
    fingerprint: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    outputs: list[Path],
    num_groups: int,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_MANIFEST_SCHEMA,
        "method": SIGNED_V21_METHOD,
        "variant": SIGNED_V21_VARIANT,
        "architecture": SIGNED_V21_ARCHITECTURE,
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


def _group_coverage(groups: list[Any], b_plus: float, b_minus: float) -> dict[str, Any]:
    base_valid = 0
    signed_valid = 0
    signed_pairs = 0
    classes: Counter[str] = Counter()
    for group in groups:
        positive = sum(float(gain) > b_plus for gain in group.effective_gains)
        negative = sum(float(gain) < b_minus for gain in group.effective_gains)
        base_valid += int(positive > 0 and negative > 0)
        group_classes = Counter(group.supervision_classes)
        ordinary_low = group_classes["negative"] + group_classes["neutral"]
        harmful = group_classes["explicit_harmful_negative"]
        signed_valid += int(ordinary_low > 0 and harmful > 0)
        signed_pairs += ordinary_low * harmful
        classes.update(group.supervision_classes)
    return {
        "total_groups": len(groups),
        "base_valid_ranking_groups": base_valid,
        "base_skipped_ranking_groups": len(groups) - base_valid,
        "signed_valid_ranking_groups": signed_valid,
        "signed_skipped_ranking_groups": len(groups) - signed_valid,
        "signed_pair_count_total": signed_pairs,
        "candidate_class_counts": {
            name: classes[name]
            for name in (
                "positive",
                "negative",
                "neutral",
                "explicit_harmful_negative",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train experimental rag_cbwdm_signed_v21 ordinal selector"
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
    parser.add_argument("--b-plus", type=float, default=0.01)
    parser.add_argument("--b-minus", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda-signed", type=float, default=0.25)
    parser.add_argument("--gamma-signed", type=float, default=1.0)
    parser.add_argument(
        "--neutral-sample-policy", choices=["negative", "ignore"], default="negative"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_train_groups is not None and args.max_train_groups < 1:
        raise ValueError("max-train-groups must be >= 1")
    fixed = {
        "epochs": 3,
        "lr": 2e-5,
        "batch_size": 8,
        "max_length": 512,
        "b_plus": 0.01,
        "b_minus": 0.001,
        "beta": 0.25,
        "gamma": 1.0,
        "lambda_signed": 0.25,
        "gamma_signed": 1.0,
        "neutral_sample_policy": "negative",
        "seed": 13,
    }
    changed = {
        name: (getattr(args, name), expected)
        for name, expected in fixed.items()
        if getattr(args, name) != expected
    }
    if changed:
        raise ValueError(f"signed-v2.1 first-round hyperparameters are frozen: {changed}")

    config_path = Path(args.config).resolve()
    load_yaml(config_path)
    run = Path(args.run_dir).resolve()
    output = require_diagnostic_output(run, Path(args.output_dir).resolve())
    signed_root = (
        run / "artifacts/diagnostics/method_failure_audit/signed_v21"
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
        raise ValueError("signed-v2.1 must reuse a signed_teacher_v1 artifact")
    teacher_params = teacher_contract.get("parameters", {})
    expected_teacher = {
        "b_plus": args.b_plus,
        "b_minus": args.b_minus,
        "neutral_sample_policy": args.neutral_sample_policy,
    }
    mismatched = {
        key: (teacher_params.get(key), expected)
        for key, expected in expected_teacher.items()
        if teacher_params.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"Training thresholds differ from teacher: {mismatched}")

    contract = {
        "variant": "signed_selector_v21_training",
        "method": SIGNED_V21_METHOD,
        "architecture": SIGNED_V21_ARCHITECTURE,
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
            "model": {"path": str(model), "sha256": sha256_path(model)},
        },
        "parameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "max_train_groups": args.max_train_groups,
            "loss_type": SIGNED_V21_LOSS_TYPE,
            "b_plus": args.b_plus,
            "b_minus": args.b_minus,
            "beta": args.beta,
            "gamma": args.gamma,
            "lambda_signed": args.lambda_signed,
            "gamma_signed": args.gamma_signed,
            "neutral_sample_policy": args.neutral_sample_policy,
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
            and manifest.get("method") == SIGNED_V21_METHOD
            and manifest.get("fingerprint") == fingerprint
            and checkpoint.is_dir()
            and all(path.is_file() for path in outputs)
            and manifest.get("checkpoint_sha256") == sha256_path(checkpoint)
            and all(
                manifest["outputs"][path.name]["sha256"] == sha256_file(path)
                for path in outputs
            )
        ):
            print(f"[signed_selector_v21_train] reused=true output={output}")
            return
        raise ValueError("Cannot resume signed-v2.1 training: fingerprint/checksum mismatch")
    if manifest_path.exists() or checkpoint.exists() or any(path.exists() for path in outputs):
        raise FileExistsError(
            "signed-v2.1 training artifact exists; use matching --resume or a new directory"
        )

    groups = build_signed_training_groups(
        teacher_path=teacher,
        posteriors_path=posteriors,
        retrieval_path=retrieval,
        max_train_groups=args.max_train_groups,
    )
    coverage = _group_coverage(groups, args.b_plus, args.b_minus)
    selector = create_selector(
        model_name=str(model),
        max_length=args.max_length,
        device=args.device,
        seed=args.seed,
    )
    optimizer = torch.optim.AdamW(selector.model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        selector.model.train()
        ordered = list(groups)
        random.Random(args.seed + epoch).shuffle(ordered)
        optimizer.zero_grad()
        totals = Counter()
        for index, group in enumerate(ordered, start=1):
            texts = [
                build_selector_input(group.query, group.selected_docs, candidate)
                for candidate in group.candidate_docs
            ]
            scores = selector.score_texts(
                texts, batch_size=len(texts), requires_grad=True
            )
            loss, details = signed_v21_multitask_loss(
                scores,
                group.effective_gains,
                group.supervision_classes,
                b_plus=args.b_plus,
                b_minus=args.b_minus,
                gamma=args.gamma,
                beta=args.beta,
                neutral_sample_policy=args.neutral_sample_policy,
                lambda_signed=args.lambda_signed,
                gamma_signed=args.gamma_signed,
            )
            (loss / args.batch_size).backward()
            totals["total_loss"] += float(loss.detach().cpu())
            totals["base_loss"] += float(details["base_loss"].detach().cpu())
            totals["ce_loss"] += float(details["ce_loss"].detach().cpu())
            totals["base_rank_loss"] += float(
                details["base_rank_loss"].detach().cpu()
            )
            totals["signed_rank_loss"] += float(
                details["signed_rank_loss"].detach().cpu()
            )
            totals["base_valid_ranking_groups"] += int(
                details["base_valid_ranking_group"]
            )
            totals["base_skipped_ranking_groups"] += int(
                details["base_skipped_ranking_group"]
            )
            totals["signed_valid_ranking_groups"] += int(
                details["signed_valid_ranking_group"]
            )
            totals["signed_skipped_ranking_groups"] += int(
                details["signed_skipped_ranking_group"]
            )
            totals["signed_pair_count_total"] += details["signed_pair_count"]
            for name in (
                "positive_candidate_count",
                "ordinary_negative_count",
                "neutral_count",
                "harmful_count",
            ):
                totals[name] += details[name]
            totals["groups"] += 1
            if index % args.batch_size == 0 or index == len(ordered):
                optimizer.step()
                optimizer.zero_grad()
            if index % 100 == 0:
                print(
                    f"[signed_selector_v21_train] epoch={epoch} "
                    f"group={index}/{len(ordered)} loss={float(loss.detach().cpu()):.6f}"
                )
        denominator = totals["groups"]
        history.append(
            {
                "epoch": epoch,
                "avg_total_loss": totals["total_loss"] / denominator,
                "avg_base_loss": totals["base_loss"] / denominator,
                "avg_ce_loss": totals["ce_loss"] / denominator,
                "avg_base_rank_loss": totals["base_rank_loss"] / denominator,
                "avg_signed_rank_loss": totals["signed_rank_loss"] / denominator,
                "base_valid_ranking_groups": totals["base_valid_ranking_groups"],
                "base_skipped_ranking_groups": totals["base_skipped_ranking_groups"],
                "signed_valid_ranking_groups": totals["signed_valid_ranking_groups"],
                "signed_skipped_ranking_groups": totals[
                    "signed_skipped_ranking_groups"
                ],
                "signed_pair_count_total": totals["signed_pair_count_total"],
                "positive_candidate_count": totals["positive_candidate_count"],
                "ordinary_negative_count": totals["ordinary_negative_count"],
                "neutral_count": totals["neutral_count"],
                "harmful_count": totals["harmful_count"],
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    selector.save_checkpoint(
        checkpoint,
        extra_config={
            "variant": SIGNED_V21_VARIANT,
            "method": SIGNED_V21_METHOD,
            "architecture": SIGNED_V21_ARCHITECTURE,
            "experimental": True,
            "loss_type": SIGNED_V21_LOSS_TYPE,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "b_plus": args.b_plus,
            "b_minus": args.b_minus,
            "beta": args.beta,
            "gamma": args.gamma,
            "lambda_signed": args.lambda_signed,
            "gamma_signed": args.gamma_signed,
            "neutral_sample_policy": args.neutral_sample_policy,
            "seed": args.seed,
            "num_groups": len(groups),
        },
    )
    training_config = {
        **contract["parameters"],
        "method": SIGNED_V21_METHOD,
        "variant": SIGNED_V21_VARIANT,
        "architecture": SIGNED_V21_ARCHITECTURE,
        "model_name": str(model),
        "num_groups": len(groups),
        "terminal_groups": sum(group.is_terminal_state for group in groups),
        "ranking_coverage": coverage,
    }
    atomic_write_json(outputs[0], training_config)
    atomic_write_json(outputs[1], {"epochs": history, "ranking_coverage": coverage})
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
    print(
        f"[signed_selector_v21_train] groups={len(groups)} "
        f"base_valid={coverage['base_valid_ranking_groups']} "
        f"signed_valid={coverage['signed_valid_ranking_groups']} checkpoint={checkpoint}"
    )


if __name__ == "__main__":
    main()

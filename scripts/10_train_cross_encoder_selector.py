from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_dir, load_yaml
from src.selector_cross_encoder import (
    CrossEncoderSelector,
    build_cross_encoder_groups,
    build_selector_input,
    cbwdm_multitask_loss,
    listwise_distillation_loss,
    shuffle_groups,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for cross-encoder selector training."""
    parser = argparse.ArgumentParser(description="Train a state-aware cross-encoder selector.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--posteriors", required=True, help="Training posterior JSONL path.")
    parser.add_argument("--teacher", required=True, help="CBWDM teacher JSONL path.")
    parser.add_argument("--retrieval", default=None, help="Optional retrieval JSONL for text recovery.")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoint and config.")
    parser.add_argument("--model-name", default=None, help="HF encoder model name.")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Groups per optimizer step.")
    parser.add_argument("--max-length", type=int, default=None, help="Tokenizer max sequence length.")
    parser.add_argument("--max-train-groups", type=int, default=None, help="Limit number of training groups.")
    parser.add_argument("--max-candidates-per-group", type=int, default=None, help="Limit candidates per group.")
    parser.add_argument("--teacher-temperature", type=float, default=None)
    parser.add_argument("--loss-type", choices=["listwise_distill", "cbwdm_multitask"], default=None)
    parser.add_argument("--b-plus", type=float, default=None)
    parser.add_argument("--b-minus", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--neutral-sample-policy", choices=["ignore", "negative"], default=None)
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    return parser.parse_args()


def resolve_project_path(path_value: str | Path | None) -> Path | None:
    """Resolve relative paths against the project root."""
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and torch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_args(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Fill CLI omissions from config while preserving legacy listwise behavior."""
    selector = config.get("selector", {})
    args.model_name = args.model_name or selector.get(
        "model_name", "hf-internal-testing/tiny-random-bert"
    )
    args.model_revision = args.model_revision or selector.get("revision")
    args.tokenizer_revision = (
        args.tokenizer_revision or selector.get("tokenizer_revision") or args.model_revision
    )
    args.epochs = args.epochs if args.epochs is not None else int(selector.get("epochs", 1))
    args.lr = args.lr if args.lr is not None else float(selector.get("lr", 2e-5))
    args.batch_size = (
        args.batch_size if args.batch_size is not None else int(selector.get("batch_size", 1))
    )
    args.max_length = (
        args.max_length if args.max_length is not None else int(selector.get("max_length", 512))
    )
    args.teacher_temperature = (
        args.teacher_temperature
        if args.teacher_temperature is not None
        else float(selector.get("teacher_temperature", 0.1))
    )
    # Missing loss_type means old configs retain the previous listwise objective.
    args.loss_type = args.loss_type or selector.get("loss_type", "listwise_distill")
    args.b_plus = args.b_plus if args.b_plus is not None else float(selector.get("b_plus", 0.01))
    args.b_minus = args.b_minus if args.b_minus is not None else float(selector.get("b_minus", 0.001))
    args.gamma = args.gamma if args.gamma is not None else float(selector.get("gamma", 1.0))
    args.beta = args.beta if args.beta is not None else float(selector.get("beta", 0.5))
    args.neutral_sample_policy = (
        args.neutral_sample_policy
        or selector.get("neutral_sample_policy", "negative")
    )


def group_texts(group: Any) -> list[str]:
    """Build cross-encoder texts for every candidate in a group."""
    return [
        build_selector_input(
            query=group.query,
            selected_docs=group.selected_docs,
            candidate_doc=candidate_doc,
        )
        for candidate_doc in group.candidate_docs
    ]


def save_training_config(output_dir: Path, args: argparse.Namespace, config: dict[str, Any], num_groups: int) -> None:
    """Save a small JSON training manifest next to the checkpoint."""
    payload = {
        "config_path": str(resolve_project_path(args.config)),
        "posteriors": str(resolve_project_path(args.posteriors)),
        "teacher": str(resolve_project_path(args.teacher)),
        "retrieval": str(resolve_project_path(args.retrieval)) if args.retrieval else None,
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_train_groups": args.max_train_groups,
        "max_candidates_per_group": args.max_candidates_per_group,
        "teacher_temperature": args.teacher_temperature,
        "loss_type": args.loss_type,
        "b_plus": args.b_plus,
        "b_minus": args.b_minus,
        "gamma": args.gamma,
        "beta": args.beta,
        "neutral_sample_policy": args.neutral_sample_policy,
        "device": args.device,
        "seed": args.seed,
        "num_groups": num_groups,
        "dataset": config.get("dataset"),
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_project_path(args.config) or args.config)
    resolve_training_args(args, config)
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    set_seed(args.seed)

    groups = build_cross_encoder_groups(
        posterior_path=resolve_project_path(args.posteriors) or args.posteriors,
        teacher_path=resolve_project_path(args.teacher) or args.teacher,
        retrieval_path=resolve_project_path(args.retrieval),
        max_train_groups=args.max_train_groups,
        max_candidates_per_group=args.max_candidates_per_group,
    )
    output_dir = ensure_dir(resolve_project_path(args.output_dir) or args.output_dir)
    checkpoint_dir = output_dir / "checkpoint"

    selector = CrossEncoderSelector(
        model_name=args.model_name,
        max_length=args.max_length,
        device=args.device,
        revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
    )
    optimizer = torch.optim.AdamW(selector.model.parameters(), lr=args.lr)

    print(
        f"[cross_encoder_train] groups={len(groups)} model={args.model_name} "
        f"device={selector.device} max_length={args.max_length}"
    )
    for epoch in range(1, args.epochs + 1):
        selector.model.train()
        epoch_groups = shuffle_groups(groups, seed=args.seed + epoch)
        total_loss = 0.0
        total_groups = 0
        total_ce = 0.0
        total_rank = 0.0
        num_positive = num_negative = num_neutral = valid_rank = skipped_rank = 0
        optimizer.zero_grad()

        for idx, group in enumerate(epoch_groups, start=1):
            texts = group_texts(group)
            scores = selector.score_texts(texts, batch_size=len(texts), requires_grad=True)
            if args.loss_type == "listwise_distill":
                loss = listwise_distillation_loss(
                    scores=scores,
                    gains=group.gains,
                    teacher_temperature=args.teacher_temperature,
                )
                details = {
                    "ce_loss": scores.sum() * 0.0,
                    "rank_loss": scores.sum() * 0.0,
                    "num_positive": 0,
                    "num_negative": 0,
                    "num_neutral": len(group.gains),
                    "valid_ranking_group": False,
                    "skipped_ranking_group": False,
                }
            else:
                loss, details = cbwdm_multitask_loss(
                    scores,
                    group.gains,
                    b_plus=args.b_plus,
                    b_minus=args.b_minus,
                    gamma=args.gamma,
                    beta=args.beta,
                    neutral_sample_policy=args.neutral_sample_policy,
                )
            scaled_loss = loss / args.batch_size
            scaled_loss.backward()
            total_loss += float(loss.detach().cpu().item())
            total_groups += 1
            total_ce += float(details["ce_loss"].detach().cpu().item())
            total_rank += float(details["rank_loss"].detach().cpu().item())
            num_positive += details["num_positive"]
            num_negative += details["num_negative"]
            num_neutral += details["num_neutral"]
            valid_rank += int(details["valid_ranking_group"])
            skipped_rank += int(details["skipped_ranking_group"])

            if idx % args.batch_size == 0 or idx == len(epoch_groups):
                optimizer.step()
                optimizer.zero_grad()

            print(
                f"[cross_encoder_train] epoch={epoch} step={idx} "
                f"total_loss={float(loss.detach().cpu().item()):.6f} "
                f"ce_loss={float(details['ce_loss'].detach().cpu().item()):.6f} "
                f"rank_loss={float(details['rank_loss'].detach().cpu().item()):.6f} "
                f"positive={details['num_positive']} negative={details['num_negative']} "
                f"neutral={details['num_neutral']} lr={optimizer.param_groups[0]['lr']:.3g}"
            )

        avg_loss = total_loss / max(total_groups, 1)
        print(
            f"[cross_encoder_train] epoch={epoch} avg_total_loss={avg_loss:.6f} "
            f"avg_ce_loss={total_ce/max(total_groups,1):.6f} "
            f"avg_rank_loss={total_rank/max(total_groups,1):.6f} "
            f"positive={num_positive} negative={num_negative} neutral={num_neutral} "
            f"valid_ranking_groups={valid_rank} skipped_ranking_groups={skipped_rank} "
            f"lr={optimizer.param_groups[0]['lr']:.3g}"
        )

    selector.save_checkpoint(
        checkpoint_dir,
        extra_config={
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "teacher_temperature": args.teacher_temperature,
            "loss_type": args.loss_type,
            "b_plus": args.b_plus,
            "b_minus": args.b_minus,
            "gamma": args.gamma,
            "beta": args.beta,
            "neutral_sample_policy": args.neutral_sample_policy,
            "num_groups": len(groups),
        },
    )
    save_training_config(output_dir, args, config, num_groups=len(groups))
    print(f"[cross_encoder_train] saved={checkpoint_dir}")


if __name__ == "__main__":
    main()

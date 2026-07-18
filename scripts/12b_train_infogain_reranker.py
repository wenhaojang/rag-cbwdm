from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.infogain_fever import (
    group_teacher_rows,
    infogain_multitask_loss,
    pointwise_input,
)
from src.baselines.infogain_selector import InfoGainPointwiseReranker
from src.io_utils import read_jsonl
from src.run_manifest import atomic_write_json, git_state, sha256_file, stable_hash, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pointwise InfoGain-FEVER RankNet/filter reranker.")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument("--b-pos", type=float)
    parser.add_argument("--b-neg", type=float)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    import torch

    teacher = absolute(args.teacher)
    teacher_manifest = json.loads(
        teacher.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    thresholds = teacher_manifest["thresholds"]
    b_pos = float(args.b_pos if args.b_pos is not None else thresholds["b_pos"])
    b_neg = float(args.b_neg if args.b_neg is not None else thresholds["b_neg"])
    output = absolute(args.output_dir)
    checkpoint = output / "checkpoint"
    manifest_path = output / "training_manifest.json"
    contract = {
        "teacher_sha256": sha256_file(teacher),
        "teacher_fingerprint": teacher_manifest["fingerprint"],
        "model": args.model_name_or_path,
        "revision": args.revision,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "b_pos": b_pos,
        "b_neg": b_neg,
        "seed": args.seed,
        "max_groups": args.max_groups,
        "loss": "pairwise_logistic_plus_filter_ce",
    }
    fingerprint = stable_hash(contract)
    if args.resume and manifest_path.exists() and not args.overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "completed"
            and manifest.get("fingerprint") == fingerprint
            and (checkpoint / "infogain_config.json").is_file()
            and (checkpoint / "heads.pt").is_file()
        ):
            print(f"[infogain_train] reused=true checkpoint={checkpoint}")
            return
        raise ValueError("Cannot resume InfoGain training: checkpoint contract mismatch")
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError("InfoGain checkpoint exists; use --resume or --overwrite")
    groups = group_teacher_rows(read_jsonl(teacher))
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise ValueError("No InfoGain teacher groups")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = InfoGainPointwiseReranker(
        args.model_name_or_path,
        revision=args.revision,
        device=args.device,
        max_length=args.max_length,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(groups)
        model.train()
        totals = {"total": 0.0, "rank": 0.0, "filter": 0.0}
        for group in groups:
            texts = [
                pointwise_input(row["query"], row.get("title"), row.get("text"))
                for row in group
            ]
            rank, filter_logits = model.forward(texts)
            loss, details = infogain_multitask_loss(
                rank,
                filter_logits,
                [float(row["dig"]) for row in group],
                b_pos=b_pos,
                b_neg=b_neg,
                beta=args.beta,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["total"] += float(loss.detach().cpu())
            totals["rank"] += float(details["rank_loss"].detach().cpu())
            totals["filter"] += float(details["filter_loss"].detach().cpu())
        record = {
            "epoch": epoch,
            "total_loss": totals["total"] / len(groups),
            "rank_loss": totals["rank"] / len(groups),
            "filter_loss": totals["filter"] / len(groups),
        }
        history.append(record)
        print(f"[infogain_train] {record}")
    checkpoint_config = {**contract, "fingerprint": fingerprint, "history": history}
    model.save(checkpoint, checkpoint_config)
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "rag_cbwdm_infogain_training_manifest.v1",
            "stage": "train_infogain",
            "status": "completed",
            "completed": True,
            "fingerprint": fingerprint,
            "contract": contract,
            "history": history,
            "checkpoint": str(checkpoint.resolve()),
            "git": git_state(PROJECT_ROOT),
            "end_time": utc_now(),
        },
    )
    print(f"[infogain_train] reused=false checkpoint={checkpoint}")


if __name__ == "__main__":
    main()

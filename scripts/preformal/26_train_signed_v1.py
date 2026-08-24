from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.signed_teacher_v1 import build_signed_training_groups
from src.formal_provenance import sha256_path
from src.preformal.registry import SIGNED_V1_CONTRACT, assert_frozen_signed_contract, assert_no_held_out_reference
from src.run_manifest import atomic_write_json, git_state, sha256_file, stable_hash, utc_now
from src.selector_cross_encoder import CrossEncoderSelector, build_selector_input, cbwdm_multitask_loss


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def training_contract(*, seed: int, config: Path, teacher: Path, posteriors: Path,
                      retrieval: Path, model: Path, training_split: str = "train_core") -> dict:
    frozen = SIGNED_V1_CONTRACT["selector"]
    return {"method": "rag_cbwdm_signed_v1", "stage": "training", "seed": seed,
        "config_sha256": sha256_file(config), "teacher_sha256": sha256_file(teacher),
        "train_core_posterior_sha256": sha256_file(posteriors), "train_core_retrieval_sha256": sha256_file(retrieval),
        "model_path": str(model), "model_sha256": sha256_path(model), "epochs": frozen["epochs"],
        "lr": frozen["lr"], "batch_size": frozen["batch_size"], "beta": frozen["beta"],
        "gamma": frozen["gamma"], "loss_type": frozen["loss_type"], "b_plus": 0.01,
        "b_minus": 0.001, "neutral_sample_policy": "negative", "max_length": 512,
        "teacher_temperature": 0.1, "training_split": training_split,
        "preformal_eval_used_for_training": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train formal experimental rag_cbwdm_signed_v1")
    parser.add_argument("--config", required=True); parser.add_argument("--teacher", required=True)
    parser.add_argument("--posteriors", required=True); parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--model-name", default="/root/models/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--training-split", choices=["train", "train_core"], default="train_core")
    parser.add_argument("--seed", type=int, required=True, choices=[13,21,42]); parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    config = Path(args.config).resolve(); teacher = Path(args.teacher).resolve(); posteriors = Path(args.posteriors).resolve()
    retrieval = Path(args.retrieval).resolve(); model = Path(args.model_name).resolve(); output = Path(args.output_dir).resolve()
    assert_no_held_out_reference({"teacher": str(teacher), "posteriors": str(posteriors), "retrieval": str(retrieval)})
    frozen = SIGNED_V1_CONTRACT["selector"]
    assert_frozen_signed_contract({"model_name": str(model).replace("\\", "/"), "epochs": frozen["epochs"], "lr": frozen["lr"],
        "batch_size": frozen["batch_size"], "beta": frozen["beta"], "gamma": frozen["gamma"], "loss_type": frozen["loss_type"]})
    teacher_manifest = teacher.parent / "manifest.json"; teacher_payload = json.loads(teacher_manifest.read_text(encoding="utf-8"))
    if teacher_payload.get("method") != "rag_cbwdm_signed_v1" or teacher_payload.get("teacher_sha256") != sha256_file(teacher):
        raise ValueError("Training teacher is not checksum-compatible formal signed-v1 supervision")
    if teacher_payload.get("contract", {}).get("split") != args.training_split:
        raise ValueError("Teacher manifest split does not match --training-split")
    contract = training_contract(seed=args.seed, config=config, teacher=teacher, posteriors=posteriors,
                                 retrieval=retrieval, model=model, training_split=args.training_split)
    fingerprint = stable_hash(contract); manifest_path = output / "training_manifest.json"; checkpoint = output / "checkpoint"
    history_path = output / "training_history.json"; config_path = output / "training_config.json"
    if args.resume and all(path.exists() for path in (manifest_path, checkpoint, history_path, config_path)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint and manifest.get("checkpoint_sha256") == sha256_path(checkpoint):
            print(f"[preformal_signed_train] seed={args.seed} reused=true checkpoint={checkpoint}"); return
        raise ValueError("Cannot resume signed-v1 training: fingerprint/checkpoint mismatch")
    if any(path.exists() for path in (manifest_path, checkpoint, history_path, config_path)):
        raise FileExistsError("Training artifacts exist; use matching --resume")
    groups = build_signed_training_groups(teacher_path=teacher, posteriors_path=posteriors, retrieval_path=retrieval)
    seed_everything(args.seed); selector = CrossEncoderSelector(model_name=str(model), max_length=512, device=args.device)
    optimizer = torch.optim.AdamW(selector.model.parameters(), lr=float(frozen["lr"])); history = []
    for epoch in range(1, int(frozen["epochs"]) + 1):
        selector.model.train(); ordered = list(groups); random.Random(args.seed + epoch).shuffle(ordered); optimizer.zero_grad()
        totals = {"loss": 0.0, "ce": 0.0, "rank": 0.0, "valid_rank": 0, "skipped_rank": 0, "groups": 0}
        for index, group in enumerate(ordered, start=1):
            texts = [build_selector_input(group.query, group.selected_docs, doc) for doc in group.candidate_docs]
            scores = selector.score_texts(texts, batch_size=len(texts), requires_grad=True)
            loss, details = cbwdm_multitask_loss(scores, group.effective_gains, b_plus=0.01, b_minus=0.001,
                gamma=1.0, beta=0.25, neutral_sample_policy="negative")
            (loss / 8).backward(); totals["loss"] += float(loss.detach().cpu()); totals["ce"] += float(details["ce_loss"].detach().cpu())
            totals["rank"] += float(details["rank_loss"].detach().cpu()); totals["valid_rank"] += int(details["valid_ranking_group"])
            totals["skipped_rank"] += int(details["skipped_ranking_group"]); totals["groups"] += 1
            if index % 8 == 0 or index == len(ordered): optimizer.step(); optimizer.zero_grad()
            if index % 100 == 0: print(f"[preformal_signed_train] seed={args.seed} epoch={epoch} group={index}/{len(ordered)}")
        history.append({"epoch": epoch, "avg_total_loss": totals["loss"] / totals["groups"],
            "avg_ce_loss": totals["ce"] / totals["groups"], "avg_rank_loss": totals["rank"] / totals["groups"],
            "valid_ranking_groups": totals["valid_rank"], "skipped_ranking_groups": totals["skipped_rank"]})
    output.mkdir(parents=True, exist_ok=True)
    selector.save_checkpoint(checkpoint, extra_config={**contract, "variant": "signed_selector_v1", "experimental": True})
    atomic_write_json(history_path, {"epochs": history}); atomic_write_json(config_path, {**contract, "num_groups": len(groups)})
    checkpoint_sha = sha256_path(checkpoint)
    atomic_write_json(manifest_path, {"schema_version": "rag_cbwdm_preformal_signed_training.v1", "status": "completed",
        "completed": True, "method": "rag_cbwdm_signed_v1", "seed": args.seed, "fingerprint": fingerprint,
        "contract": contract, "train_core_posterior_sha256": contract["train_core_posterior_sha256"],
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_fingerprint": stable_hash({"contract": fingerprint, "checkpoint_sha256": checkpoint_sha}),
        "git": git_state(PROJECT_ROOT), "completed_at": utc_now()})
    print(f"[preformal_signed_train] seed={args.seed} groups={len(groups)} checkpoint={checkpoint}")


if __name__ == "__main__": main()

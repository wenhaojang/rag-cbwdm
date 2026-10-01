from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.signed_teacher_v1 import build_signed_training_groups
from src.artifact_binding import (
    complete_training_binding,
    validate_formal_teacher_binding,
)
from src.experiment_identity import resolve_dataset_identity
from src.formal_provenance import sha256_path
from src.io_utils import load_yaml
from src.preformal.registry import SIGNED_V1_CONTRACT, assert_frozen_signed_contract, assert_no_held_out_reference
from src.run_manifest import atomic_write_json, git_state, sha256_file, stable_hash, utc_now
from src.selector_cross_encoder import CrossEncoderSelector, build_selector_input, cbwdm_multitask_loss


SIGNED_TRAINING_RUNTIME_VERSION = "signed_optimizer_block_v1"
OPTIMIZER_GROUP_BATCH_SIZE = 8


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def training_contract(*, seed: int, config: Path, teacher: Path, posteriors: Path,
                      retrieval: Path, model: Path, training_split: str = "train_core",
                      runtime_implementation: str = "block_v1",
                      forward_batch_size: int = 32,
                      max_groups: int | None = None) -> dict:
    frozen = SIGNED_V1_CONTRACT["selector"]
    return {"method": "rag_cbwdm_signed_v1", "stage": "training", "seed": seed,
        "config_sha256": sha256_file(config), "teacher_sha256": sha256_file(teacher),
        "train_core_posterior_sha256": sha256_file(posteriors), "train_core_retrieval_sha256": sha256_file(retrieval),
        "model_path": str(model), "model_sha256": sha256_path(model), "epochs": frozen["epochs"],
        "lr": frozen["lr"], "batch_size": frozen["batch_size"], "beta": frozen["beta"],
        "gamma": frozen["gamma"], "loss_type": frozen["loss_type"], "b_plus": 0.01,
        "b_minus": 0.001, "neutral_sample_policy": "negative", "max_length": 512,
        "teacher_temperature": 0.1, "training_split": training_split,
        "preformal_eval_used_for_training": False, "max_groups": max_groups,
        "training_runtime": {
            "implementation_version": SIGNED_TRAINING_RUNTIME_VERSION,
            "runtime_implementation": runtime_implementation,
            "optimizer_group_batch_size": OPTIMIZER_GROUP_BATCH_SIZE,
            "forward_batch_size": (
                forward_batch_size if runtime_implementation == "block_v1" else None
            ),
            "vectorized_infogain_rank_loss": False,
        }}


def optimizer_blocks(
    groups: Sequence[Any], size: int = OPTIMIZER_GROUP_BATCH_SIZE
) -> list[list[Any]]:
    if size < 1:
        raise ValueError("optimizer block size must be positive")
    return [list(groups[start : start + size]) for start in range(0, len(groups), size)]


def group_texts(group: Any) -> list[str]:
    return [
        build_selector_input(group.query, group.selected_docs, doc)
        for doc in group.candidate_docs
    ]


def signed_group_loss(scores: torch.Tensor, group: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    return cbwdm_multitask_loss(
        scores,
        group.effective_gains,
        b_plus=0.01,
        b_minus=0.001,
        gamma=1.0,
        beta=0.25,
        neutral_sample_policy="negative",
    )


def block_forward_and_loss(
    selector: Any,
    block: Sequence[Any],
    *,
    forward_batch_size: int,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, dict[str, Any]]], int]:
    """Tokenize one optimizer block, microbatch its forward, and preserve group losses."""
    text_groups = [group_texts(group) for group in block]
    lengths = [len(texts) for texts in text_groups]
    flat_texts = [text for texts in text_groups for text in texts]
    encoded = selector.encode_texts(flat_texts)
    flat_scores = selector.forward_encoded(
        encoded, batch_size=forward_batch_size, requires_grad=True
    )
    losses: list[tuple[torch.Tensor, dict[str, Any]]] = []
    offset = 0
    for group, length in zip(block, lengths):
        losses.append(signed_group_loss(flat_scores[offset : offset + length], group))
        offset += length
    if offset != int(flat_scores.numel()):
        raise RuntimeError("Signed block score/group offset mismatch")
    # Deliberately divide incomplete final blocks by the frozen constant eight.
    block_loss = torch.stack([loss for loss, _ in losses]).sum() / OPTIMIZER_GROUP_BATCH_SIZE
    return block_loss, losses, len(flat_texts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train formal experimental rag_cbwdm_signed_v1")
    parser.add_argument("--config", required=True); parser.add_argument("--teacher", required=True)
    parser.add_argument("--posteriors", required=True); parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--model-name", default="/root/models/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--training-split", choices=["train", "train_core"], default="train_core")
    parser.add_argument("--teacher-manifest")
    parser.add_argument("--dataset-id")
    parser.add_argument("--generator-id")
    parser.add_argument("--formal-v2-identity", action="store_true")
    parser.add_argument("--seed", type=int, required=True, choices=[13,21,42]); parser.add_argument("--device", default="auto")
    parser.add_argument("--runtime-implementation", choices=["legacy", "block_v1"], default="block_v1")
    parser.add_argument("--forward-batch-size", type=int, default=32)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    if args.forward_batch_size < 1:
        raise ValueError("--forward-batch-size must be positive")
    if args.max_groups is not None and args.max_groups < 1:
        raise ValueError("--max-groups must be positive")
    config = Path(args.config).resolve(); teacher = Path(args.teacher).resolve(); posteriors = Path(args.posteriors).resolve()
    retrieval = Path(args.retrieval).resolve(); model = Path(args.model_name).resolve(); output = Path(args.output_dir).resolve()
    assert_no_held_out_reference({"teacher": str(teacher), "posteriors": str(posteriors), "retrieval": str(retrieval)})
    frozen = SIGNED_V1_CONTRACT["selector"]
    assert_frozen_signed_contract({"model_name": str(model).replace("\\", "/"), "epochs": frozen["epochs"], "lr": frozen["lr"],
        "batch_size": frozen["batch_size"], "beta": frozen["beta"], "gamma": frozen["gamma"], "loss_type": frozen["loss_type"]})
    teacher_manifest = (Path(args.teacher_manifest).resolve() if args.teacher_manifest
        else teacher.parent / "manifest.json"); teacher_payload = json.loads(teacher_manifest.read_text(encoding="utf-8"))
    if teacher_payload.get("method") != "rag_cbwdm_signed_v1" or teacher_payload.get("teacher_sha256") != sha256_file(teacher):
        raise ValueError("Training teacher is not checksum-compatible formal signed-v1 supervision")
    if teacher_payload.get("contract", {}).get("split") != args.training_split:
        raise ValueError("Teacher manifest split does not match --training-split")
    artifact_binding = None
    if args.formal_v2_identity:
        config_payload = load_yaml(config)
        dataset = resolve_dataset_identity(
            config_payload["dataset"], explicit_dataset_id=args.dataset_id
        )
        teacher_binding = validate_formal_teacher_binding(
            teacher,
            teacher_manifest,
            method="rag_cbwdm_signed_v1",
            expected_dataset_id=dataset.dataset_id,
            expected_conditioning_generator_id=args.generator_id,
        )
        if teacher_binding["posterior"]["posterior_sha256"] != sha256_file(
            posteriors
        ):
            raise ValueError("Training posterior SHA differs from teacher binding")
        if teacher_binding.get("retrieval_source_sha256") != sha256_file(retrieval):
            raise ValueError("Training retrieval SHA differs from teacher binding")
        artifact_binding = complete_training_binding(
            teacher_binding, seed=args.seed, config_path=config
        )
    contract = training_contract(seed=args.seed, config=config, teacher=teacher, posteriors=posteriors,
                                 retrieval=retrieval, model=model, training_split=args.training_split,
                                 runtime_implementation=args.runtime_implementation,
                                 forward_batch_size=args.forward_batch_size,
                                 max_groups=args.max_groups)
    if artifact_binding is not None:
        contract["artifact_binding"] = artifact_binding
    fingerprint = stable_hash(contract); manifest_path = output / "training_manifest.json"; checkpoint = output / "checkpoint"
    history_path = output / "training_history.json"; config_path = output / "training_config.json"
    if args.resume and all(path.exists() for path in (manifest_path, checkpoint, history_path, config_path)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint and manifest.get("checkpoint_sha256") == sha256_path(checkpoint):
            print(f"[preformal_signed_train] seed={args.seed} reused=true checkpoint={checkpoint}"); return
        raise ValueError("Cannot resume signed-v1 training: fingerprint/checkpoint mismatch")
    if any(path.exists() for path in (manifest_path, checkpoint, history_path, config_path)):
        raise FileExistsError("Training artifacts exist; use matching --resume")
    groups = build_signed_training_groups(teacher_path=teacher, posteriors_path=posteriors,
                                          retrieval_path=retrieval, max_train_groups=args.max_groups)
    seed_everything(args.seed); selector = CrossEncoderSelector(model_name=str(model), max_length=512, device=args.device)
    optimizer = torch.optim.AdamW(selector.model.parameters(), lr=float(frozen["lr"])); history = []
    if torch.cuda.is_available() and selector.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selector.device)
    train_started = time.perf_counter()
    for epoch in range(1, int(frozen["epochs"]) + 1):
        selector.model.train(); ordered = list(groups); random.Random(args.seed + epoch).shuffle(ordered)
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "ce": 0.0, "rank": 0.0, "valid_rank": 0, "skipped_rank": 0, "groups": 0}
        forward_calls = tokenizer_calls = optimizer_steps = text_count = 0
        if args.runtime_implementation == "legacy":
            for index, group in enumerate(ordered, start=1):
                texts = group_texts(group)
                scores = selector.score_texts(texts, batch_size=len(texts), requires_grad=True)
                loss, details = signed_group_loss(scores, group)
                (loss / OPTIMIZER_GROUP_BATCH_SIZE).backward()
                forward_calls += 1; tokenizer_calls += 1; text_count += len(texts)
                totals["loss"] += float(loss.detach().cpu()); totals["ce"] += float(details["ce_loss"].detach().cpu())
                totals["rank"] += float(details["rank_loss"].detach().cpu()); totals["valid_rank"] += int(details["valid_ranking_group"])
                totals["skipped_rank"] += int(details["skipped_ranking_group"]); totals["groups"] += 1
                if index % OPTIMIZER_GROUP_BATCH_SIZE == 0 or index == len(ordered):
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
                if index % 100 == 0: print(f"[preformal_signed_train] seed={args.seed} epoch={epoch} group={index}/{len(ordered)}")
        else:
            completed = 0
            for block in optimizer_blocks(ordered):
                block_loss, block_results, block_texts = block_forward_and_loss(
                    selector, block, forward_batch_size=args.forward_batch_size
                )
                block_loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
                tokenizer_calls += 1; optimizer_steps += 1; text_count += block_texts
                forward_calls += (block_texts + args.forward_batch_size - 1) // args.forward_batch_size
                for loss, details in block_results:
                    totals["loss"] += float(loss.detach().cpu()); totals["ce"] += float(details["ce_loss"].detach().cpu())
                    totals["rank"] += float(details["rank_loss"].detach().cpu()); totals["valid_rank"] += int(details["valid_ranking_group"])
                    totals["skipped_rank"] += int(details["skipped_ranking_group"]); totals["groups"] += 1
                completed += len(block)
                if completed % 100 < len(block): print(f"[preformal_signed_train] seed={args.seed} epoch={epoch} group={completed}/{len(ordered)}")
        history.append({"epoch": epoch, "avg_total_loss": totals["loss"] / totals["groups"],
            "avg_ce_loss": totals["ce"] / totals["groups"], "avg_rank_loss": totals["rank"] / totals["groups"],
            "valid_ranking_groups": totals["valid_rank"], "skipped_ranking_groups": totals["skipped_rank"],
            "optimizer_steps": optimizer_steps, "forward_calls": forward_calls,
            "tokenizer_calls": tokenizer_calls, "texts": text_count})
    train_seconds = time.perf_counter() - train_started
    runtime_metrics = {
        "groups": len(groups) * len(history),
        "texts": sum(int(record["texts"]) for record in history),
        "forward_calls": sum(int(record["forward_calls"]) for record in history),
        "tokenizer_calls": sum(int(record["tokenizer_calls"]) for record in history),
        "optimizer_steps": sum(int(record["optimizer_steps"]) for record in history),
        "wall_clock_seconds": train_seconds,
        "groups_per_second": (len(groups) * len(history)) / train_seconds,
        "texts_per_second": sum(int(record["texts"]) for record in history) / train_seconds,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(selector.device))
            if torch.cuda.is_available() and selector.device.type == "cuda"
            else None
        ),
        "gpu_utilization": None,
        "gpu_utilization_note": "not sampled by worker",
        "final_loss": history[-1]["avg_total_loss"],
        "loss_trajectory": [record["avg_total_loss"] for record in history],
    }
    output.mkdir(parents=True, exist_ok=True)
    selector.save_checkpoint(checkpoint, extra_config={**contract, "variant": "signed_selector_v1", "experimental": True})
    atomic_write_json(history_path, {"epochs": history}); atomic_write_json(config_path, {**contract, "num_groups": len(groups)})
    checkpoint_sha = sha256_path(checkpoint)
    training_manifest = {"schema_version": (
            "rag_cbwdm_preformal_signed_training.v2"
            if artifact_binding is not None
            else "rag_cbwdm_preformal_signed_training.v1"
        ), "status": "completed",
        "completed": True, "method": "rag_cbwdm_signed_v1", "seed": args.seed, "fingerprint": fingerprint,
        "contract": contract, "train_core_posterior_sha256": contract["train_core_posterior_sha256"],
        "runtime_metrics": runtime_metrics,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_fingerprint": stable_hash({"contract": fingerprint, "checkpoint_sha256": checkpoint_sha}),
        "git": git_state(PROJECT_ROOT), "completed_at": utc_now()}
    if artifact_binding is not None:
        training_manifest.update({
            "identity_mode": "formal_v2",
            "artifact_binding": artifact_binding,
            "dataset_id": artifact_binding["dataset_id"],
            "conditioning_generator_id": artifact_binding[
                "conditioning_generator_id"
            ],
            "generator_identity_fingerprint": artifact_binding[
                "generator_identity_fingerprint"
            ],
            "retrieval_protocol_id": artifact_binding["retrieval_protocol_id"],
        })
    atomic_write_json(manifest_path, training_manifest)
    print(f"[preformal_signed_train] seed={args.seed} groups={len(groups)} checkpoint={checkpoint}")


if __name__ == "__main__": main()

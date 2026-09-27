from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.signed_teacher_v1 import build_signed_teacher_row, teacher_statistics
from src.experiment_identity import (
    load_optional_posterior_binding,
    posterior_binding_contract,
    resolve_dataset_identity,
)
from src.io_utils import load_yaml, read_jsonl
from src.preformal.registry import SIGNED_V1_CONTRACT, assert_frozen_signed_contract, assert_no_held_out_reference
from src.run_manifest import atomic_write_json, git_state, sha256_file, stable_hash, utc_now


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize formal rag_cbwdm_signed_v1 training teacher")
    parser.add_argument("--config", required=True)
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-split", choices=["train", "train_core"], default="train_core")
    parser.add_argument("--posterior-manifest")
    parser.add_argument("--dataset-id")
    parser.add_argument("--generator-id")
    parser.add_argument("--retrieval-protocol-id")
    parser.add_argument("--formal-v2-identity", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve(); config = load_yaml(config_path)
    posterior = Path(args.posteriors).resolve(); retrieval = Path(args.retrieval).resolve(); output = Path(args.output_dir).resolve()
    assert_no_held_out_reference({"posteriors": str(posterior), "retrieval": str(retrieval)})
    configured_dataset = resolve_dataset_identity(
        config["dataset"], explicit_dataset_id=args.dataset_id
    )
    posterior_binding = load_optional_posterior_binding(
        posterior,
        Path(args.posterior_manifest).resolve() if args.posterior_manifest else None,
        formal_v2=args.formal_v2_identity,
        expected_dataset_id=(
            configured_dataset.dataset_id
            if args.formal_v2_identity or args.dataset_id
            else None
        ),
        expected_split=args.training_split,
        expected_generator_id=args.generator_id,
        expected_retrieval_protocol_id=args.retrieval_protocol_id,
    )
    identity_opt_in = bool(
        args.formal_v2_identity
        or args.posterior_manifest
        or args.dataset_id
        or args.generator_id
        or args.retrieval_protocol_id
    )
    has_stable_identity = bool(
        posterior_binding
        and posterior_binding["dataset_identity"].get("dataset_id")
        and posterior_binding["generator_identity"].get("generator_id")
        and posterior_binding["retrieval_protocol_identity"].get(
            "retrieval_protocol_id"
        )
    )
    binding_contract = (
        posterior_binding_contract(posterior_binding)
        if posterior_binding is not None and (identity_opt_in or has_stable_identity)
        else None
    )
    if binding_contract is not None:
        source_retrieval_sha = binding_contract[
            "retrieval_protocol_identity"
        ].get("source_artifact_sha256")
        if source_retrieval_sha and source_retrieval_sha != sha256_file(retrieval):
            raise ValueError(
                "Signed-v1 retrieval SHA differs from posterior retrieval identity"
            )
    frozen = SIGNED_V1_CONTRACT
    params = {
        "top_m": frozen["teacher"]["top_m"], "stop_threshold": frozen["teacher"]["stop_threshold"],
        "alignment_eps": frozen["teacher"]["alignment_eps"], "b_plus": frozen["teacher"]["b_plus"],
        "b_minus": frozen["teacher"]["b_minus"], "neutral_sample_policy": frozen["teacher"]["neutral_sample_policy"],
        "ridge_lambda": float(config["cbwdm"]["ridge_lambda"]), "eps_smooth": float(config["cbwdm"].get("eps_smooth", 0)),
        "l_type": config["cbwdm"].get("L_type", "euclidean_posterior_shift"),
        "target_smoothing": config["cbwdm"].get("target_smoothing", "paper_mixture"),
        "gain_tolerance": float(config["cbwdm"].get("gain_tolerance", 1e-10)),
    }
    assert_frozen_signed_contract({"top_m": params["top_m"], "teacher_stop_threshold": params["stop_threshold"],
        "alignment_eps": params["alignment_eps"], "b_plus": params["b_plus"], "b_minus": params["b_minus"],
        "neutral_sample_policy": params["neutral_sample_policy"]})
    contract = {"method": "rag_cbwdm_signed_v1", "stage": "teacher_training_only", "split": args.training_split,
        "config_sha256": sha256_file(config_path), "posterior_sha256": sha256_file(posterior),
        "retrieval_sha256": sha256_file(retrieval), "parameters": params, "uses_gold_for_teacher": True,
        "evaluation_eligible": False, "calibration_eligible": False}
    if binding_contract is not None:
        contract["posterior_binding"] = binding_contract
    fingerprint = stable_hash(contract); teacher_path = output / "teacher.jsonl"; stats_path = output / "statistics.json"; manifest_path = output / "manifest.json"
    if args.resume and all(path.is_file() for path in (teacher_path, stats_path, manifest_path)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint and manifest.get("teacher_sha256") == sha256_file(teacher_path):
            print(f"[preformal_signed_teacher] reused=true output={output}"); return
        raise ValueError("Cannot resume signed teacher: fingerprint/checksum mismatch")
    if any(path.exists() for path in (teacher_path, stats_path, manifest_path)):
        raise FileExistsError("Signed teacher artifacts exist; use matching --resume")
    source_rows = list(read_jsonl(posterior))
    if not source_rows or {row.get("split") for row in source_rows} != {args.training_split}:
        raise ValueError(f"Signed teacher input must contain only {args.training_split} rows")
    rows = [build_signed_teacher_row(row, params) for row in source_rows]
    _write_jsonl(teacher_path, rows); atomic_write_json(stats_path, teacher_statistics(rows))
    teacher_manifest = {"schema_version": (
            "rag_cbwdm_preformal_signed_teacher.v2"
            if binding_contract is not None
            else "rag_cbwdm_preformal_signed_teacher.v1"
        ), "status": "completed",
        "completed": True, "fingerprint": fingerprint, "contract": contract, "method": "rag_cbwdm_signed_v1",
        "num_rows": len(rows), "teacher_sha256": sha256_file(teacher_path), "statistics_sha256": sha256_file(stats_path),
        "diagnostic_trajectory_implementation": "src.diagnostics.signed_teacher_v1.build_signed_teacher_row",
        "git": git_state(PROJECT_ROOT), "completed_at": utc_now()}
    if binding_contract is not None:
        teacher_manifest.update({
            "identity_mode": (
                "formal_v2" if args.formal_v2_identity else "legacy_compatible"
            ),
            "posterior_binding": binding_contract,
            "dataset_identity": binding_contract["dataset_identity"],
            "generator_identity": binding_contract["generator_identity"],
            "retrieval_protocol_identity": binding_contract[
                "retrieval_protocol_identity"
            ],
            "generator_id": binding_contract["generator_identity"].get(
                "generator_id"
            ),
        })
    atomic_write_json(manifest_path, teacher_manifest)
    print(f"[preformal_signed_teacher] rows={len(rows)} output={output}")


if __name__ == "__main__": main()

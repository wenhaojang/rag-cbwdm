from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.infogain import (
    TEACHER_DEFINITION,
    posterior_to_teacher_rows,
    resolve_thresholds,
    validate_teacher_roles,
)
from src.experiment_identity import (
    load_optional_posterior_binding,
    posterior_binding_contract,
)
from src.io_utils import read_jsonl
from src.run_manifest import atomic_write_json, git_state, sha256_file, stable_hash, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build classification probability-difference DIG teacher rows.")
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--purpose",
        choices=["training", "validation_diagnostic"],
        default="training",
    )
    parser.add_argument("--threshold-mode", choices=["explicit", "train_quantile", "validation_calibrated"], default="train_quantile")
    parser.add_argument("--b-pos", type=float)
    parser.add_argument("--b-neg", type=float)
    parser.add_argument("--positive-quantile", type=float, default=0.75)
    parser.add_argument("--negative-quantile", type=float, default=0.25)
    parser.add_argument("--generator-model")
    parser.add_argument("--generator-revision")
    parser.add_argument("--prompt-hash")
    parser.add_argument("--verbalizer-hash")
    parser.add_argument("--posterior-manifest")
    parser.add_argument("--dataset-id")
    parser.add_argument("--generator-id")
    parser.add_argument("--retrieval-protocol-id")
    parser.add_argument("--formal-v2-identity", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    source = absolute(args.posteriors)
    output = absolute(args.output)
    manifest_path = output.with_suffix(".manifest.json")
    posterior_rows = list(read_jsonl(source, limit=args.limit))
    roles = {str(row.get("split") or "") for row in posterior_rows}
    teacher_role = validate_teacher_roles(roles, purpose=args.purpose)
    posterior_binding = load_optional_posterior_binding(
        source,
        absolute(args.posterior_manifest) if args.posterior_manifest else None,
        formal_v2=args.formal_v2_identity,
        expected_dataset_id=args.dataset_id,
        expected_split=teacher_role,
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
    bound_generator = (
        binding_contract["generator_identity"] if binding_contract else {}
    )
    bound_prompt_hash = bound_generator.get("prompt_template_hash")
    bound_verbalizer_hash = bound_generator.get("verbalizer_hash")
    if args.formal_v2_identity:
        comparisons = (
            (
                "generator model",
                args.generator_model,
                bound_generator.get("model_name_or_path"),
            ),
            (
                "generator revision",
                args.generator_revision,
                bound_generator.get("model_revision"),
            ),
            ("prompt hash", args.prompt_hash, bound_prompt_hash),
            ("verbalizer hash", args.verbalizer_hash, bound_verbalizer_hash),
        )
        for field, requested, authoritative in comparisons:
            if requested is not None and requested != authoritative:
                raise ValueError(
                    f"InfoGain {field} conflicts with posterior manifest: "
                    f"requested={requested!r} authoritative={authoritative!r}"
                )
    if args.threshold_mode == "train_quantile" and args.purpose != "training":
        raise ValueError("train_quantile thresholds require purpose=training")
    if (
        args.threshold_mode == "validation_calibrated"
        and args.purpose != "validation_diagnostic"
    ):
        raise ValueError(
            "validation_calibrated thresholds require "
            "purpose=validation_diagnostic"
        )
    provenance = {
        "posterior_path": str(source.resolve()),
        "posterior_sha256": sha256_file(source),
        "teacher_role": teacher_role,
        "teacher_purpose": args.purpose,
        "generator_model": args.generator_model
        or bound_generator.get("model_name_or_path"),
        "generator_revision": args.generator_revision
        or bound_generator.get("model_revision"),
        "prompt_hash": args.prompt_hash or bound_prompt_hash,
        "verbalizer_hash": args.verbalizer_hash or bound_verbalizer_hash,
        "teacher_definition": TEACHER_DEFINITION,
        "threshold_mode": args.threshold_mode,
        "b_pos": args.b_pos,
        "b_neg": args.b_neg,
        "positive_quantile": args.positive_quantile,
        "negative_quantile": args.negative_quantile,
        "limit": args.limit,
    }
    if binding_contract is not None:
        provenance["posterior_binding"] = binding_contract
    fingerprint = stable_hash(provenance)
    if args.resume and output.exists() and manifest_path.exists() and not args.overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "completed"
            and manifest.get("fingerprint") == fingerprint
            and manifest.get("output_sha256") == sha256_file(output)
        ):
            print(f"[infogain_teacher] reused=true rows={manifest['num_rows']} output={output}")
            return
        raise ValueError("Cannot resume InfoGain teacher: manifest/checksum/fingerprint mismatch")
    if (output.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError("InfoGain teacher exists; use --resume or --overwrite")
    rows = [
        teacher
        for posterior in posterior_rows
        for teacher in posterior_to_teacher_rows(
            posterior, purpose=args.purpose
        )
    ]
    thresholds = resolve_thresholds(
        (row["dig"] for row in rows),
        mode=args.threshold_mode,
        b_pos=args.b_pos,
        b_neg=args.b_neg,
        positive_quantile=args.positive_quantile,
        negative_quantile=args.negative_quantile,
    )
    partial = output.with_name(output.name + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, output)
    teacher_manifest = {
        "schema_version": (
            "rag_cbwdm_infogain_teacher_manifest.v2"
            if binding_contract is not None
            else "rag_cbwdm_infogain_teacher_manifest.v1"
        ),
        "stage": "build_infogain_teacher",
        "status": "completed",
        "completed": True,
        "fingerprint": fingerprint,
        "provenance": provenance,
        "teacher_role": teacher_role,
        "teacher_purpose": args.purpose,
        "training_eligible": args.purpose == "training",
        "diagnostic_only": args.purpose != "training",
        "thresholds": thresholds,
        "num_rows": len(rows),
        "output_sha256": sha256_file(output),
        "git": git_state(PROJECT_ROOT),
        "end_time": utc_now(),
    }
    if binding_contract is not None:
        teacher_manifest.update(
            {
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
            }
        )
    atomic_write_json(
        manifest_path,
        teacher_manifest,
    )
    print(f"[infogain_teacher] reused=false rows={len(rows)} output={output}")


if __name__ == "__main__":
    main()

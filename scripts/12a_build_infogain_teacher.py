from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.infogain_fever import (
    TEACHER_DEFINITION,
    posterior_to_teacher_rows,
    resolve_thresholds,
    validate_teacher_roles,
)
from src.io_utils import read_jsonl
from src.run_manifest import atomic_write_json, git_state, sha256_file, stable_hash, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FEVER probability-difference DIG teacher rows.")
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
        "generator_model": args.generator_model,
        "generator_revision": args.generator_revision,
        "prompt_hash": args.prompt_hash,
        "verbalizer_hash": args.verbalizer_hash,
        "teacher_definition": TEACHER_DEFINITION,
        "threshold_mode": args.threshold_mode,
        "b_pos": args.b_pos,
        "b_neg": args.b_neg,
        "positive_quantile": args.positive_quantile,
        "negative_quantile": args.negative_quantile,
        "limit": args.limit,
    }
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
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "rag_cbwdm_infogain_teacher_manifest.v1",
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
        },
    )
    print(f"[infogain_teacher] reused=false rows={len(rows)} output={output}")


if __name__ == "__main__":
    main()

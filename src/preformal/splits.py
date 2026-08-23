"""Recover the unused portion of the frozen formal validation split."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.formal_splits import build_splits, normalize_claim, validate_split_manifest
from src.io_utils import read_jsonl
from src.run_manifest import atomic_write_json, git_state, sha256_file, stable_hash, utc_now

SCHEMA_VERSION = "rag_cbwdm_preformal_split_manifest.v1"


def _set_sha(values: Iterable[str]) -> str:
    return stable_hash(sorted(set(map(str, values))))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row["original_id"]) for row in rows}


def _claims(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {normalize_claim(str(row["query"])) for row in rows}


def build_strategy_a_rows(split_manifest_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return full frozen validation minus the exact pilot-validation groups."""
    manifest_path = Path(split_manifest_path).resolve()
    manifest = validate_split_manifest(manifest_path)
    contract = manifest["contract"]
    partition = contract["partition"]
    limits = contract["limits"]
    if limits.get("validation") is None:
        raise ValueError("Strategy A requires a pilot validation limit")
    source = contract["source"]
    full, details = build_splits(
        source["official_train_path"],
        source["official_dev_path"],
        seed=int(partition["seed"]),
        validation_size=partition.get("validation_size"),
        validation_fraction=partition.get("validation_fraction"),
        train_limit=None,
        validation_limit=None,
        test_limit=None,
    )
    pilot_validation = list(read_jsonl(manifest["splits"]["validation"]["path"]))
    current_train = list(read_jsonl(manifest["splits"]["train_core"]["path"]))
    pilot_claims = _claims(pilot_validation)
    preformal = [
        {**row, "split": "preformal_eval"}
        for row in full["validation"]
        if normalize_claim(str(row["query"])) not in pilot_claims
    ]
    preformal.sort(key=lambda row: (str(row["original_id"]), str(row["id"])))
    sets = {
        "preformal_ids": _ids(preformal),
        "preformal_claims": _claims(preformal),
        "pilot_ids": _ids(pilot_validation),
        "pilot_claims": pilot_claims,
        "train_ids": _ids(current_train),
        "train_claims": _claims(current_train),
        "held_out_ids": _ids(full["held_out_test"]),
        "held_out_claims": _claims(full["held_out_test"]),
    }
    checks = {
        "id_overlap_preformal_pilot_validation": len(sets["preformal_ids"] & sets["pilot_ids"]),
        "normalized_claim_overlap_preformal_pilot_validation": len(sets["preformal_claims"] & sets["pilot_claims"]),
        "id_overlap_preformal_current_train_core": len(sets["preformal_ids"] & sets["train_ids"]),
        "normalized_claim_overlap_preformal_current_train_core": len(sets["preformal_claims"] & sets["train_claims"]),
        "id_overlap_preformal_held_out_test": len(sets["preformal_ids"] & sets["held_out_ids"]),
        "normalized_claim_overlap_preformal_held_out_test": len(sets["preformal_claims"] & sets["held_out_claims"]),
    }
    if any(checks.values()):
        raise ValueError(f"Preformal split overlap detected: {checks}")
    if len(preformal) + len(pilot_validation) != len(full["validation"]):
        raise ValueError("Pilot validation is not an exact whole-group subset of full validation")
    audit = {
        "strategy": "A",
        "full_split_rows": details["full_split_rows"],
        "full_validation_rows": len(full["validation"]),
        "pilot_validation_rows": len(pilot_validation),
        "preformal_eval_rows": len(preformal),
        "overlap_checks": checks,
        "id_set_sha256": _set_sha(sets["preformal_ids"]),
        "normalized_claim_set_sha256": _set_sha(sets["preformal_claims"]),
        "held_out_id_set_sha256_audit_only": _set_sha(sets["held_out_ids"]),
        "source_details": details,
    }
    return preformal, audit


def publish_strategy_a(
    split_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    split_path = output / "preformal_eval.jsonl"
    manifest_path = output / "preformal_eval.manifest.json"
    audit_path = output / "preformal_split_audit.md"
    rows, audit = build_strategy_a_rows(split_manifest_path)
    source_manifest_path = Path(split_manifest_path).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    contract = {
        "strategy": "A",
        "source_split_manifest_sha256": sha256_file(source_manifest_path),
        "excluded_pilot_validation_sha256": source_manifest["splits"]["validation"]["sha256"],
        "train_core_sha256": source_manifest["splits"]["train_core"]["sha256"],
        "held_out_test_sha256_audit_only": source_manifest["splits"]["held_out_test"]["sha256"],
        "seed": source_manifest["contract"]["partition"]["seed"],
        "partition": source_manifest["contract"]["partition"],
    }
    fingerprint = stable_hash(contract)
    if resume and all(path.is_file() for path in (split_path, manifest_path, audit_path)):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("fingerprint") == fingerprint and existing.get("preformal_eval_sha256") == sha256_file(split_path):
            return existing
        raise ValueError("Cannot resume preformal split: fingerprint/checksum mismatch")
    if any(path.exists() for path in (split_path, manifest_path, audit_path)):
        raise FileExistsError("Preformal split artifacts exist; use matching --resume or a new directory")
    _write_jsonl(split_path, rows)
    label_counts = dict(sorted(Counter(str(row["label"]) for row in rows).items()))
    markdown = [
        "# Preformal split audit", "", "Strategy: **A** — full frozen formal validation minus exact current pilot-validation normalized-claim groups.", "",
        f"- Full frozen validation rows: {audit['full_validation_rows']}",
        f"- Excluded pilot validation rows: {audit['pilot_validation_rows']}",
        f"- Clean preformal_eval rows: {audit['preformal_eval_rows']}",
        f"- Label counts: `{json.dumps(label_counts, sort_keys=True)}`", "", "## Overlap checks", "",
    ]
    markdown.extend(f"- {key}: {value}" for key, value in audit["overlap_checks"].items())
    markdown.extend(("", "Official-dev held_out_test was used only for deterministic leakage auditing; it is not emitted, trained on, calibrated on, or evaluated here.", ""))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(markdown), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "completed": True,
        "fingerprint": fingerprint,
        "source_strategy": "A",
        "contract": contract,
        "source_split_sha256": contract["source_split_manifest_sha256"],
        "excluded_pilot_validation_sha256": contract["excluded_pilot_validation_sha256"],
        "train_core_sha256": contract["train_core_sha256"],
        "preformal_eval_path": str(split_path),
        "preformal_eval_sha256": sha256_file(split_path),
        "row_count": len(rows),
        "label_counts": label_counts,
        "id_set_sha256": audit["id_set_sha256"],
        "normalized_claim_set_sha256": audit["normalized_claim_set_sha256"],
        "overlap_checks": audit["overlap_checks"],
        "held_out_test_consumed_for_modeling": False,
        "held_out_test_audit_only": True,
        "git": git_state(project_root),
        "created_at": utc_now(),
        "audit_path": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest

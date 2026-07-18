"""Deterministic, leakage-checked FEVER-2 formal data splits."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.io_utils import read_jsonl
from src.run_manifest import git_state, sha256_file, stable_hash, utc_now

SCHEMA_VERSION = "rag_cbwdm_fever_split_manifest.v1"
FILTER_VERSION = "fever2_supports_refutes.v1"
LABELS = ("SUPPORTS", "REFUTES")
SPLIT_NAMES = ("train_core", "validation", "held_out_test")


def normalize_claim(text: str) -> str:
    """Normalize claim text for conservative cross-split leakage detection."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _partition_key(seed: int, original_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{original_id}".encode("utf-8")).hexdigest()


def _stable_id(original_id: str) -> str:
    return f"fever_{original_id}"


def _load_source(path: Path, source_name: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_ids: dict[str, int] = {}
    seen_claims: dict[str, tuple[str, int]] = {}
    for line_no, row in enumerate(read_jsonl(path), start=1):
        label = str(row.get("label", "")).strip().upper().replace(" ", "_")
        if label == "NOT_ENOUGH_INFO":
            counts["filtered_not_enough_info"] += 1
            continue
        if label not in LABELS:
            raise ValueError(
                f"{source_name} line {line_no} has unsupported FEVER label {row.get('label')!r}"
            )
        if "id" not in row or "claim" not in row:
            raise KeyError(f"{source_name} line {line_no} must contain id and claim")
        original_id = str(row["id"])
        claim = str(row["claim"])
        normalized = normalize_claim(claim)
        if not normalized:
            raise ValueError(f"{source_name} line {line_no} has an empty normalized claim")
        if original_id in seen_ids:
            raise ValueError(
                f"Duplicate original FEVER id {original_id!r} in {source_name} "
                f"(lines {seen_ids[original_id]} and {line_no})"
            )
        if normalized in seen_claims:
            previous_id, previous_line = seen_claims[normalized]
            raise ValueError(
                f"Duplicate normalized claim in {source_name}: ids {previous_id!r} "
                f"(line {previous_line}) and {original_id!r} (line {line_no})"
            )
        seen_ids[original_id] = line_no
        seen_claims[normalized] = (original_id, line_no)
        accepted.append(
            {
                "id": _stable_id(original_id),
                "original_id": original_id,
                "query": claim,
                "label": label,
                "_normalized_claim": normalized,
                "_source_name": source_name,
                "_raw": row,
            }
        )
        counts[f"accepted_{label}"] += 1
    counts["accepted"] = len(accepted)
    return accepted, dict(sorted(counts.items()))


def _validation_quotas(
    rows: list[dict[str, Any]],
    validation_size: int | None,
    validation_fraction: float | None,
) -> dict[str, int]:
    counts = Counter(row["label"] for row in rows)
    if validation_size is not None and validation_fraction is not None:
        raise ValueError("Specify only one of validation_size and validation_fraction")
    if validation_size is None and validation_fraction is None:
        raise ValueError("One of validation_size and validation_fraction is required")
    if validation_fraction is not None:
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be strictly between 0 and 1")
        quotas = {
            label: int(round(counts[label] * validation_fraction)) for label in LABELS
        }
    else:
        assert validation_size is not None
        if validation_size <= 0 or validation_size >= len(rows):
            raise ValueError(
                f"validation_size must be in [1, {len(rows) - 1}], got {validation_size}"
            )
        exact = {
            label: validation_size * counts[label] / len(rows) for label in LABELS
        }
        quotas = {label: int(exact[label]) for label in LABELS}
        remainder = validation_size - sum(quotas.values())
        order = sorted(
            LABELS,
            key=lambda label: (-(exact[label] - quotas[label]), label),
        )
        for label in order[:remainder]:
            quotas[label] += 1
    for label, quota in quotas.items():
        if quota <= 0 or quota >= counts[label]:
            raise ValueError(
                f"Validation quota for {label} must leave rows in both train_core and "
                f"validation (quota={quota}, available={counts[label]})"
            )
    return quotas


def _apply_limit(rows: list[dict[str, Any]], limit: int | None, seed: int, role: str) -> list[dict[str, Any]]:
    if limit is None:
        selected = rows
    else:
        if limit < 0:
            raise ValueError(f"{role} limit must be non-negative")
        selected = sorted(
            rows,
            key=lambda row: (_partition_key(seed, f"{role}\0{row['original_id']}"), row["original_id"]),
        )[:limit]
    return sorted(selected, key=lambda row: (row["original_id"], row["id"]))


def build_splits(
    official_train: str | Path,
    official_dev: str | Path,
    *,
    seed: int,
    validation_size: int | None,
    validation_fraction: float | None,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build formal roles from official train/dev without writing artifacts."""
    train_path = Path(official_train).resolve()
    dev_path = Path(official_dev).resolve()
    train_rows, train_stats = _load_source(train_path, "official_train")
    dev_rows, dev_stats = _load_source(dev_path, "official_dev")

    train_ids = {row["original_id"] for row in train_rows}
    dev_ids = {row["original_id"] for row in dev_rows}
    duplicate_ids = sorted(train_ids & dev_ids)
    if duplicate_ids:
        raise ValueError(
            f"Original FEVER IDs occur in both official train and dev: {duplicate_ids[:5]}"
        )
    train_claims = {row["_normalized_claim"] for row in train_rows}
    dev_claims = {row["_normalized_claim"] for row in dev_rows}
    duplicate_claims = train_claims & dev_claims
    if duplicate_claims:
        raise ValueError(
            f"Normalized claims occur in both official train and dev ({len(duplicate_claims)} overlaps)"
        )

    quotas = _validation_quotas(train_rows, validation_size, validation_fraction)
    validation_ids: set[str] = set()
    for label in LABELS:
        label_rows = sorted(
            (row for row in train_rows if row["label"] == label),
            key=lambda row: (_partition_key(seed, row["original_id"]), row["original_id"]),
        )
        validation_ids.update(row["original_id"] for row in label_rows[: quotas[label]])

    full = {
        "train_core": [row for row in train_rows if row["original_id"] not in validation_ids],
        "validation": [row for row in train_rows if row["original_id"] in validation_ids],
        "held_out_test": list(dev_rows),
    }
    limited = {
        "train_core": _apply_limit(full["train_core"], train_limit, seed, "train_core"),
        "validation": _apply_limit(full["validation"], validation_limit, seed, "validation"),
        "held_out_test": _apply_limit(full["held_out_test"], test_limit, seed, "held_out_test"),
    }
    public: dict[str, list[dict[str, Any]]] = {}
    for role, rows in limited.items():
        public[role] = []
        for row in rows:
            raw = row["_raw"]
            meta: dict[str, Any] = {
                "original_id": row["original_id"],
                "official_source": row["_source_name"],
            }
            if raw.get("evidence") is not None:
                meta["evidence"] = raw["evidence"]
            public[role].append(
                {
                    "id": row["id"],
                    "original_id": row["original_id"],
                    "query": row["query"],
                    "label": row["label"],
                    "split": role,
                    "meta": meta,
                }
            )

    checks = overlap_checks(public)
    if any(checks.values()):
        raise ValueError(f"Formal split overlap detected: {checks}")
    details = {
        "source_stats": {"official_train": train_stats, "official_dev": dev_stats},
        "full_split_rows": {role: len(rows) for role, rows in full.items()},
        "validation_label_quotas": quotas,
        "overlap_checks": checks,
    }
    return public, details


def overlap_checks(splits: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    ids = {role: {str(row["original_id"]) for row in rows} for role, rows in splits.items()}
    claims = {
        role: {normalize_claim(str(row["query"])) for row in rows}
        for role, rows in splits.items()
    }
    return {
        "id_overlap_train_validation": len(ids["train_core"] & ids["validation"]),
        "id_overlap_train_test": len(ids["train_core"] & ids["held_out_test"]),
        "id_overlap_validation_test": len(ids["validation"] & ids["held_out_test"]),
        "normalized_claim_overlap_train_validation": len(
            claims["train_core"] & claims["validation"]
        ),
        "normalized_claim_overlap_train_test": len(
            claims["train_core"] & claims["held_out_test"]
        ),
        "normalized_claim_overlap_validation_test": len(
            claims["validation"] & claims["held_out_test"]
        ),
    }


def _write_jsonl_temp(target: Path, rows: Iterable[dict[str, Any]]) -> Path:
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _split_info(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "num_rows": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "id_sha256": stable_hash(sorted(str(row["original_id"]) for row in rows)),
    }


def split_contract(
    official_train: Path,
    official_dev: Path,
    *,
    seed: int,
    validation_size: int | None,
    validation_fraction: float | None,
    train_limit: int | None,
    validation_limit: int | None,
    test_limit: int | None,
) -> dict[str, Any]:
    return {
        "source": {
            "official_train_path": str(official_train.resolve()),
            "official_train_sha256": sha256_file(official_train),
            "official_dev_path": str(official_dev.resolve()),
            "official_dev_sha256": sha256_file(official_dev),
        },
        "filter": {
            "task": "FEVER-2",
            "included_labels": list(LABELS),
            "excluded_labels": ["NOT ENOUGH INFO"],
            "filter_version": FILTER_VERSION,
        },
        "partition": {
            "method": "stable_hash_stratified",
            "seed": seed,
            "validation_size": validation_size,
            "validation_fraction": validation_fraction,
            "id_field": "official FEVER id",
            "source_order_independent": True,
            "limits_applied_after_partition": True,
        },
        "limits": {
            "train_core": train_limit,
            "validation": validation_limit,
            "held_out_test": test_limit,
        },
    }


def validate_split_manifest(
    manifest_path: str | Path,
    expected_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "completed":
        raise ValueError(f"Split manifest is not completed: {path}")
    contract = manifest.get("contract")
    if not isinstance(contract, dict) or manifest.get("fingerprint") != stable_hash(contract):
        raise ValueError("Split manifest contract fingerprint mismatch")
    if expected_contract is not None and contract != expected_contract:
        raise ValueError("Cannot resume formal splits: source SHA or split parameters changed")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for role in SPLIT_NAMES:
        info = manifest.get("splits", {}).get(role, {})
        output = Path(str(info.get("path", "")))
        if not output.is_file():
            raise ValueError(f"Missing {role} split artifact: {output}")
        if sha256_file(output) != info.get("sha256"):
            raise ValueError(f"{role} split SHA256 changed")
        rows = list(read_jsonl(output))
        if len(rows) != info.get("num_rows"):
            raise ValueError(f"{role} split row count changed")
        if stable_hash(sorted(str(row["original_id"]) for row in rows)) != info.get("id_sha256"):
            raise ValueError(f"{role} split ID set changed")
        if any(row.get("split") != role for row in rows):
            raise ValueError(f"{role} artifact contains rows with another split role")
        loaded[role] = rows
    checks = overlap_checks(loaded)
    if checks != manifest.get("overlap_checks") or any(checks.values()):
        raise ValueError("Split manifest overlap checks do not match current artifacts")
    return manifest


def publish_splits(
    official_train: str | Path,
    official_dev: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 13,
    validation_size: int | None = 5000,
    validation_fraction: float | None = None,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
    project_root: str | Path | None = None,
) -> tuple[dict[str, Any], bool]:
    """Build and atomically publish all formal split outputs and their manifest."""
    train_path = Path(official_train).resolve()
    dev_path = Path(official_dev).resolve()
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "fever2_formal_splits.manifest.json"
    contract = split_contract(
        train_path,
        dev_path,
        seed=seed,
        validation_size=validation_size,
        validation_fraction=validation_fraction,
        train_limit=train_limit,
        validation_limit=validation_limit,
        test_limit=test_limit,
    )
    if resume and manifest_path.is_file() and not overwrite:
        return validate_split_manifest(manifest_path, contract), True

    targets = {role: directory / f"{role}.jsonl" for role in SPLIT_NAMES}
    existing = [path for path in [*targets.values(), manifest_path] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Formal split output already exists ({existing[0]}). Use --resume or --overwrite."
        )

    splits, details = build_splits(
        train_path,
        dev_path,
        seed=seed,
        validation_size=validation_size,
        validation_fraction=validation_fraction,
        train_limit=train_limit,
        validation_limit=validation_limit,
        test_limit=test_limit,
    )
    temporaries: list[Path] = []
    try:
        for role, target in targets.items():
            temporaries.append(_write_jsonl_temp(target, splits[role]))
        for role, target in targets.items():
            os.replace(target.with_name(target.name + ".tmp"), target)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "fingerprint": stable_hash(contract),
            "contract": contract,
            **contract,
            "splits": {
                role: _split_info(targets[role], splits[role]) for role in SPLIT_NAMES
            },
            "overlap_checks": details["overlap_checks"],
            "source_stats": details["source_stats"],
            "full_split_rows_before_limits": details["full_split_rows"],
            "validation_label_quotas": details["validation_label_quotas"],
            "git": git_state(project_root or Path(__file__).resolve().parents[1]),
            "created_at": utc_now(),
        }
        temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        with temporary_manifest.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_manifest, manifest_path)
        return manifest, False
    finally:
        for temporary in temporaries + [manifest_path.with_name(manifest_path.name + ".tmp")]:
            if temporary.exists():
                temporary.unlink()

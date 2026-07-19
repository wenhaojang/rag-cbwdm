"""Deterministic, leakage-checked FEVER-2 formal data splits."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.io_utils import read_jsonl
from src.run_manifest import git_state, sha256_file, stable_hash, utc_now

SCHEMA_VERSION = "rag_cbwdm_fever_split_manifest.v1"
FILTER_VERSION = "fever2_supports_refutes.v1"
DUPLICATE_POLICY = "keep_same_label_normalized_claim_groups_together"
DUPLICATE_POLICY_VERSION = "normalized_claim_group.v2"
LABELS = ("SUPPORTS", "REFUTES")
SPLIT_NAMES = ("train_core", "validation", "held_out_test")


def normalize_claim(text: str) -> str:
    """Normalize claim text for conservative cross-split leakage detection."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _partition_key(seed: int, group_key: str) -> str:
    return hashlib.sha256(f"{seed}\0{group_key}".encode("utf-8")).hexdigest()


def _stable_id(original_id: str) -> str:
    return f"fever_{original_id}"


def _load_source(path: Path, source_name: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_ids: dict[str, int] = {}
    for line_no, row in enumerate(read_jsonl(path), start=1):
        if "id" not in row or "claim" not in row:
            raise KeyError(f"{source_name} line {line_no} must contain id and claim")
        original_id = str(row["id"])
        if original_id in seen_ids:
            raise ValueError(
                f"Duplicate original FEVER id {original_id!r} in {source_name} "
                f"(lines {seen_ids[original_id]} and {line_no})"
            )
        seen_ids[original_id] = line_no
        label = str(row.get("label", "")).strip().upper().replace(" ", "_")
        if label == "NOT_ENOUGH_INFO":
            counts["filtered_not_enough_info"] += 1
            continue
        if label not in LABELS:
            raise ValueError(
                f"{source_name} line {line_no} has unsupported FEVER label {row.get('label')!r}"
            )
        claim = str(row["claim"])
        normalized = normalize_claim(claim)
        if not normalized:
            raise ValueError(f"{source_name} line {line_no} has an empty normalized claim")
        accepted.append(
            {
                "id": _stable_id(original_id),
                "original_id": original_id,
                "query": claim,
                "label": label,
                "_normalized_claim": normalized,
                "_source_name": source_name,
                "_line_no": line_no,
                "_raw": row,
            }
        )
        counts[f"accepted_{label}"] += 1
    grouped = _claim_groups(accepted)
    conflicting = []
    for normalized, rows in grouped.items():
        labels = {row["label"] for row in rows}
        if len(labels) > 1:
            conflicting.append(
                {
                    "normalized_claim": normalized,
                    "records": [
                        {
                            "id": row["original_id"],
                            "label": row["label"],
                            "line": row["_line_no"],
                        }
                        for row in rows
                    ],
                }
            )
    counts["conflicting_label_group_count"] = len(conflicting)
    if conflicting:
        raise ValueError(
            f"Conflicting labels in normalized claim group(s) in {source_name}: "
            + json.dumps(conflicting, ensure_ascii=False, sort_keys=True)
        )
    duplicate_sizes = [len(rows) for rows in grouped.values() if len(rows) > 1]
    counts["accepted"] = len(accepted)
    counts["claim_group_count"] = len(grouped)
    counts["duplicate_claim_group_count"] = len(duplicate_sizes)
    counts["rows_in_duplicate_claim_groups"] = sum(duplicate_sizes)
    counts["max_duplicate_group_size"] = max(duplicate_sizes, default=0)
    return accepted, dict(sorted(counts.items()))


def _claim_groups(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("_normalized_claim") or normalize_claim(str(row["query"])))
        groups.setdefault(key, []).append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: (str(row["original_id"]), str(row["id"])))
    return groups


def _allocate_groups_by_row_target(
    groups: list[tuple[str, list[dict[str, Any]]]],
    target_rows: int,
    *,
    seed: int,
    namespace: str,
    require_nonempty_remainder: bool,
    require_nonempty_selection: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose a closest whole-group subset using deterministic row-count subset sum."""
    if target_rows < 0:
        raise ValueError(f"{namespace} target must be non-negative")
    total_rows = sum(len(rows) for _, rows in groups)
    if target_rows >= total_rows and not require_nonempty_remainder:
        chosen_groups = list(groups)
    else:
        ordered = sorted(
            groups,
            key=lambda item: (
                _partition_key(seed, f"{namespace}\0{item[0]}"),
                item[0],
            ),
        )
        max_groups = len(ordered) - int(require_nonempty_remainder)
        if max_groups <= 0:
            raise ValueError(
                f"{namespace} has only one normalized-claim group and cannot be split "
                "without claim leakage"
            )
        max_group_size = max(len(rows) for _, rows in ordered)
        max_proper_subset = total_rows - min(len(rows) for _, rows in ordered)
        cap = min(target_rows + max_group_size, max_proper_subset)
        if cap <= 0:
            raise ValueError(f"{namespace} has no legal non-leaking allocation")
        mask = (1 << (cap + 1)) - 1
        reachable = 1
        predecessor: dict[int, tuple[int, int]] = {}
        processed = 0
        for index, (_, group_rows) in enumerate(ordered):
            size = len(group_rows)
            shifted = (reachable << size) & mask
            new_sums = shifted & ~reachable
            while new_sums:
                bit = new_sums & -new_sums
                row_sum = bit.bit_length() - 1
                predecessor[row_sum] = (row_sum - size, index)
                new_sums ^= bit
            reachable |= shifted
            processed = index + 1
            if (reachable >> target_rows) & 1:
                break
        candidates = [
            row_sum
            for row_sum in range(cap + 1)
            if (reachable >> row_sum) & 1
            and (row_sum > 0 or not require_nonempty_selection)
        ]
        if not candidates:
            raise ValueError(f"{namespace} has no legal whole-group allocation")
        actual_target = min(
            candidates,
            key=lambda row_sum: (
                abs(row_sum - target_rows),
                row_sum > target_rows,
                row_sum,
            ),
        )
        selected_indices: set[int] = set()
        cursor = actual_target
        while cursor:
            previous, index = predecessor[cursor]
            selected_indices.add(index)
            cursor = previous
        chosen_groups = [
            item
            for index, item in enumerate(ordered[:processed])
            if index in selected_indices
        ]
    chosen_keys = {key for key, _ in chosen_groups}
    selected = [
        row for key, rows in groups if key in chosen_keys for row in rows
    ]
    selected.sort(key=lambda row: (str(row["original_id"]), str(row["id"])))
    return selected, {
        "requested_rows": target_rows,
        "actual_rows": len(selected),
        "difference_rows": len(selected) - target_rows,
        "selected_group_count": len(chosen_keys),
        "available_group_count": len(groups),
        "allocation": "deterministic_row_count_aware_whole_group_subset_sum",
    }


def _reachable_proper_group_row_counts(
    groups: list[tuple[str, list[dict[str, Any]]]],
    *,
    cap: int,
    namespace: str,
) -> list[int]:
    """Return positive reachable row counts that leave at least one group behind."""
    if len(groups) < 2:
        raise ValueError(
            f"{namespace} has only one normalized-claim group and cannot be split "
            "without claim leakage"
        )
    total_rows = sum(len(rows) for _, rows in groups)
    max_proper_subset = total_rows - min(len(rows) for _, rows in groups)
    effective_cap = min(cap, max_proper_subset)
    reachable = 1
    mask = (1 << (effective_cap + 1)) - 1
    for _, rows in groups:
        reachable |= (reachable << len(rows)) & mask
    return [
        row_count
        for row_count in range(1, effective_cap + 1)
        if (reachable >> row_count) & 1
    ]


def _joint_validation_row_targets(
    groups_by_label: dict[str, list[tuple[str, list[dict[str, Any]]]]],
    quotas: dict[str, int],
) -> dict[str, int]:
    """Choose stratified reachable counts with closest possible total row count."""
    requested = sum(quotas.values())
    max_group_size = max(
        len(rows)
        for groups in groups_by_label.values()
        for _, rows in groups
    )
    options = {
        label: _reachable_proper_group_row_counts(
            groups_by_label[label],
            cap=requested + max_group_size,
            namespace=f"validation\0{label}",
        )
        for label in LABELS
    }
    if any(not values for values in options.values()):
        raise ValueError("Validation has no legal stratified whole-group allocation")
    left_label, right_label = LABELS
    right_values = options[right_label]
    candidates: list[tuple[tuple[Any, ...], int, int]] = []
    for left_rows in options[left_label]:
        desired_right = requested - left_rows
        index = bisect_left(right_values, desired_right)
        for neighbor in (index - 1, index):
            if 0 <= neighbor < len(right_values):
                right_rows = right_values[neighbor]
                total = left_rows + right_rows
                key = (
                    abs(total - requested),
                    abs(left_rows - quotas[left_label])
                    + abs(right_rows - quotas[right_label]),
                    total > requested,
                    abs(left_rows - quotas[left_label]),
                    left_rows,
                    right_rows,
                )
                candidates.append((key, left_rows, right_rows))
    _, left_rows, right_rows = min(candidates)
    return {left_label: left_rows, right_label: right_rows}


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


def _apply_limit(
    rows: list[dict[str, Any]], limit: int | None, seed: int, role: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if limit is None:
        selected = sorted(rows, key=lambda row: (row["original_id"], row["id"]))
        return selected, {
            "requested_rows": None,
            "actual_rows": len(selected),
            "difference_rows": None,
            "group_preserving": True,
        }
    groups = list(_claim_groups(rows).items())
    selected, allocation = _allocate_groups_by_row_target(
        groups,
        limit,
        seed=seed,
        namespace=f"limit\0{role}",
        require_nonempty_remainder=False,
        require_nonempty_selection=limit > 0,
    )
    allocation["group_preserving"] = True
    return selected, allocation


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
    groups_by_label = {
        label: [
            (key, rows)
            for key, rows in _claim_groups(train_rows).items()
            if rows[0]["label"] == label
        ]
        for label in LABELS
    }
    allocation_targets = _joint_validation_row_targets(
        groups_by_label, quotas
    )
    validation_claims: set[str] = set()
    label_allocations: dict[str, Any] = {}
    for label in LABELS:
        selected, allocation = _allocate_groups_by_row_target(
            groups_by_label[label],
            allocation_targets[label],
            seed=seed,
            namespace=f"validation\0{label}",
            require_nonempty_remainder=True,
            require_nonempty_selection=True,
        )
        allocation["stratified_requested_rows"] = quotas[label]
        validation_claims.update(row["_normalized_claim"] for row in selected)
        label_allocations[label] = allocation

    full = {
        "train_core": [
            row for row in train_rows if row["_normalized_claim"] not in validation_claims
        ],
        "validation": [
            row for row in train_rows if row["_normalized_claim"] in validation_claims
        ],
        "held_out_test": list(dev_rows),
    }
    train_limited, train_limit_info = _apply_limit(
        full["train_core"], train_limit, seed, "train_core"
    )
    validation_limited, validation_limit_info = _apply_limit(
        full["validation"], validation_limit, seed, "validation"
    )
    test_limited, test_limit_info = _apply_limit(
        full["held_out_test"], test_limit, seed, "held_out_test"
    )
    limited = {
        "train_core": train_limited,
        "validation": validation_limited,
        "held_out_test": test_limited,
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
    duplicate_stats = {
        key: sum(
            int(stats.get(key, 0)) for stats in (train_stats, dev_stats)
        )
        for key in (
            "duplicate_claim_group_count",
            "rows_in_duplicate_claim_groups",
            "conflicting_label_group_count",
        )
    }
    duplicate_stats["max_duplicate_group_size"] = max(
        int(train_stats.get("max_duplicate_group_size", 0)),
        int(dev_stats.get("max_duplicate_group_size", 0)),
    )
    actual_validation_size = len(full["validation"])
    requested_validation_size = sum(quotas.values())
    details = {
        "source_stats": {"official_train": train_stats, "official_dev": dev_stats},
        "full_split_rows": {role: len(rows) for role, rows in full.items()},
        "validation_label_quotas": quotas,
        "validation_label_actual_rows": {
            label: sum(row["label"] == label for row in full["validation"])
            for label in LABELS
        },
        "validation_requested_rows": requested_validation_size,
        "validation_actual_rows": actual_validation_size,
        "validation_size_difference_rows": (
            actual_validation_size - requested_validation_size
        ),
        "validation_group_allocations": label_allocations,
        "limit_allocations": {
            "train_core": train_limit_info,
            "validation": validation_limit_info,
            "held_out_test": test_limit_info,
        },
        "duplicate_claim_statistics": duplicate_stats,
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
            "method": "stable_hash_stratified_row_count_aware_groups",
            "seed": seed,
            "validation_size": validation_size,
            "validation_fraction": validation_fraction,
            "partition_unit": "normalized_claim_group",
            "group_key": "normalized claim text",
            "hash_input": "seed + normalized claim/group key",
            "duplicate_policy": DUPLICATE_POLICY,
            "duplicate_policy_version": DUPLICATE_POLICY_VERSION,
            "source_order_independent": True,
            "limits_applied_after_partition": True,
            "limits_preserve_claim_groups": True,
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
    partition = contract.get("partition", {})
    if (
        partition.get("duplicate_policy") != DUPLICATE_POLICY
        or partition.get("duplicate_policy_version") != DUPLICATE_POLICY_VERSION
        or partition.get("partition_unit") != "normalized_claim_group"
    ):
        raise ValueError("Split manifest duplicate-claim policy is incompatible")
    if (
        manifest.get("duplicate_policy") != DUPLICATE_POLICY
        or manifest.get("duplicate_policy_version") != DUPLICATE_POLICY_VERSION
        or manifest.get("partition_unit") != "normalized_claim_group"
    ):
        raise ValueError("Split manifest duplicate-claim statistics/policy are missing")
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
        for claim, grouped_rows in _claim_groups(rows).items():
            labels = {row.get("label") for row in grouped_rows}
            if len(labels) != 1:
                raise ValueError(
                    f"{role} contains conflicting labels for normalized claim {claim!r}"
                )
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
            "validation_label_actual_rows": details[
                "validation_label_actual_rows"
            ],
            "requested_validation_size": details["validation_requested_rows"],
            "actual_validation_size": details["validation_actual_rows"],
            "validation_size_difference_rows": details[
                "validation_size_difference_rows"
            ],
            "validation_group_allocations": details[
                "validation_group_allocations"
            ],
            "limit_allocations": details["limit_allocations"],
            **details["duplicate_claim_statistics"],
            "duplicate_policy": DUPLICATE_POLICY,
            "duplicate_policy_version": DUPLICATE_POLICY_VERSION,
            "partition_unit": "normalized_claim_group",
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

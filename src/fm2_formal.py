"""Fail-closed formal-v2 validation for FM2 official candidate pools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.datasets.fm2 import (
    FM2_DATASET_FAMILY,
    FM2_DATASET_ID,
    FM2_EXPECTED_SHA256,
    FM2_OFFICIAL_SPLIT_BY_FORMAL_ROLE,
    FM2_PREPARE_MANIFEST_SCHEMA_VERSION,
    FM2_RETRIEVAL_PROTOCOL_ID,
    FM2_SOURCE_COMMIT,
)
from src.io_utils import read_jsonl
from src.run_manifest import sha256_file


_FORBIDDEN_GOLD_KEYS = {
    "gold_evidence",
    "gold_evidence_keys",
    "any_gold_in_official_pool",
    "all_gold_in_official_pool",
}


def official_split_for_formal_role(formal_role: str) -> str:
    try:
        return FM2_OFFICIAL_SPLIT_BY_FORMAL_ROLE[formal_role]
    except KeyError as exc:
        raise ValueError(f"Unsupported FM2 formal role: {formal_role!r}") from exc


def _recorded_path(value: Any, manifest_path: Path, field: str) -> Path:
    if not value:
        raise ValueError(f"FM2 manifest is missing {field}")
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def validate_fm2_pool_row(
    row: Mapping[str, Any],
    *,
    official_split: str,
    formal_role: str,
    row_number: int,
) -> None:
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier.startswith(
        f"fm2:{official_split}:"
    ) or not identifier.removeprefix(f"fm2:{official_split}:"):
        raise ValueError(f"FM2 pool row {row_number} has an unstable id")
    if not isinstance(row.get("query"), str) or not str(row["query"]).strip():
        raise ValueError(f"FM2 pool row {row_number} has an empty query")
    if row.get("split") != formal_role:
        raise ValueError(
            f"FM2 pool row {row_number} formal role mismatch: "
            f"expected={formal_role!r} actual={row.get('split')!r}"
        )
    if row.get("official_source_split") != official_split:
        raise ValueError(
            f"FM2 pool row {row_number} official split mismatch: "
            f"expected={official_split!r} actual={row.get('official_source_split')!r}"
        )
    if _FORBIDDEN_GOLD_KEYS & set(row):
        raise ValueError(f"FM2 selector pool row {row_number} contains gold diagnostics")
    contract = row.get("candidate_pool")
    if not isinstance(contract, Mapping):
        raise ValueError(f"FM2 pool row {row_number} lacks candidate_pool metadata")
    if contract.get("protocol") != FM2_RETRIEVAL_PROTOCOL_ID:
        raise ValueError(f"FM2 pool row {row_number} retrieval protocol mismatch")
    if contract.get("official_source_split") != official_split:
        raise ValueError(f"FM2 pool row {row_number} candidate official split mismatch")
    if contract.get("formal_role") != formal_role:
        raise ValueError(f"FM2 pool row {row_number} candidate formal role mismatch")
    if contract.get("source_order_preserved") is not True:
        raise ValueError(f"FM2 pool row {row_number} does not preserve source order")
    if contract.get("construction_gold_free") is not True:
        raise ValueError(f"FM2 pool row {row_number} is not construction-gold-free")
    if _FORBIDDEN_GOLD_KEYS & set(contract):
        raise ValueError(f"FM2 pool row {row_number} metadata contains gold diagnostics")

    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"FM2 pool row {row_number} candidates must be non-empty")
    doc_ids: set[str] = set()
    for source_rank, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"FM2 pool row {row_number} candidate is not an object")
        if _FORBIDDEN_GOLD_KEYS & set(candidate):
            raise ValueError(f"FM2 pool row {row_number} candidate contains gold diagnostics")
        doc_id = candidate.get("doc_id")
        if not isinstance(doc_id, str) or doc_id != f"{identifier}:official:{source_rank:02d}":
            raise ValueError(f"FM2 pool row {row_number} candidate has an unstable doc_id")
        if doc_id in doc_ids:
            raise ValueError(f"FM2 pool row {row_number} has duplicate doc_id {doc_id!r}")
        doc_ids.add(doc_id)
        rank = candidate.get("rank")
        source = candidate.get("source_rank")
        if isinstance(rank, bool) or isinstance(source, bool):
            raise ValueError(f"FM2 pool row {row_number} candidate rank must be an integer")
        if rank != source_rank or source != source_rank:
            raise ValueError(
                f"FM2 pool row {row_number} candidate order/rank mismatch at {source_rank}"
            )
        if not isinstance(candidate.get("text"), str) or not str(candidate["text"]).strip():
            raise ValueError(f"FM2 pool row {row_number} candidate text is empty")


def validate_fm2_retrieval_manifest(
    manifest_path: str | Path,
    pool_path: str | Path,
    *,
    expected_formal_role: str,
    expected_dataset_id: str = FM2_DATASET_ID,
    expected_retrieval_protocol_id: str = FM2_RETRIEVAL_PROTOCOL_ID,
) -> dict[str, Any]:
    """Validate the authoritative FM2 prepare manifest and one formal-role pool."""
    manifest_file = Path(manifest_path).resolve()
    pool_file = Path(pool_path).resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"FM2 retrieval manifest does not exist: {manifest_file}")
    if not pool_file.is_file():
        raise FileNotFoundError(f"FM2 candidate pool does not exist: {pool_file}")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid FM2 retrieval manifest {manifest_file}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("FM2 retrieval manifest must be an object")
    expected_top_level = {
        "schema_version": FM2_PREPARE_MANIFEST_SCHEMA_VERSION,
        "status": "completed",
        "dataset_id": expected_dataset_id,
        "dataset_family": FM2_DATASET_FAMILY,
        "retrieval_protocol_id": expected_retrieval_protocol_id,
    }
    for field, expected in expected_top_level.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"FM2 retrieval manifest {field} mismatch: "
                f"expected={expected!r} actual={manifest.get(field)!r}"
            )
    source = manifest.get("source")
    if not isinstance(source, Mapping) or source.get("commit") != FM2_SOURCE_COMMIT:
        raise ValueError("FM2 retrieval manifest official source commit mismatch")
    mapping = manifest.get("formal_role_mapping")
    official_split = official_split_for_formal_role(expected_formal_role)
    if not isinstance(mapping, Mapping) or mapping.get(official_split) != expected_formal_role:
        raise ValueError("FM2 retrieval manifest formal role mapping mismatch")
    splits = manifest.get("splits")
    split_info = splits.get(official_split) if isinstance(splits, Mapping) else None
    if not isinstance(split_info, Mapping):
        raise ValueError(f"FM2 manifest lacks official split {official_split!r}")
    if split_info.get("official_source_split") != official_split:
        raise ValueError("FM2 manifest official source split mismatch")
    if split_info.get("formal_role") != expected_formal_role:
        raise ValueError("FM2 manifest formal role mismatch")
    if split_info.get("raw_sha256") != FM2_EXPECTED_SHA256[official_split]:
        raise ValueError("FM2 manifest raw source SHA-256 mismatch")
    recorded_pool = _recorded_path(
        split_info.get("candidate_pool_path"), manifest_file, "candidate_pool_path"
    )
    if recorded_pool != pool_file:
        raise ValueError(
            f"FM2 manifest candidate pool path mismatch: recorded={recorded_pool} actual={pool_file}"
        )
    pool_sha = sha256_file(pool_file)
    if split_info.get("candidate_pool_sha256") != pool_sha:
        raise ValueError("FM2 candidate pool SHA-256 mismatch")
    pool_contract = manifest.get("candidate_pool_contract")
    if not isinstance(pool_contract, Mapping):
        raise ValueError("FM2 manifest lacks candidate_pool_contract")
    if pool_contract.get("protocol") != expected_retrieval_protocol_id:
        raise ValueError("FM2 manifest candidate-pool protocol mismatch")
    if pool_contract.get("source_order_preserved") is not True:
        raise ValueError("FM2 manifest does not guarantee source order")
    if pool_contract.get("construction_gold_free") is not True:
        raise ValueError("FM2 manifest does not guarantee gold-free construction")

    count = 0
    for count, row in enumerate(read_jsonl(pool_file), start=1):
        validate_fm2_pool_row(
            row,
            official_split=official_split,
            formal_role=expected_formal_role,
            row_number=count,
        )
    if count != split_info.get("num_rows"):
        raise ValueError(
            f"FM2 candidate pool row count mismatch: expected={split_info.get('num_rows')} actual={count}"
        )
    return {
        "dataset_id": expected_dataset_id,
        "dataset_family": FM2_DATASET_FAMILY,
        "retrieval_protocol_id": expected_retrieval_protocol_id,
        "official_source_split": official_split,
        "formal_role": expected_formal_role,
        "manifest_path": str(manifest_file),
        "manifest_sha256": sha256_file(manifest_file),
        "pool_path": str(pool_file),
        "pool_sha256": pool_sha,
        "num_rows": count,
    }

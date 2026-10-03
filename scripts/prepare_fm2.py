from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.fm2 import (
    FM2_DATASET_FAMILY,
    FM2_DATASET_ID,
    FM2_EXPECTED_ROWS,
    FM2_EXPECTED_SHA256,
    FM2_FORMAL_ROLE_BY_OFFICIAL_SPLIT,
    FM2_INTERNAL_LABELS,
    FM2_LABEL_MAPPING,
    FM2_PREPARE_MANIFEST_SCHEMA_VERSION,
    FM2_REQUIRED_FIELDS,
    FM2_RETRIEVAL_PROTOCOL_ID,
    FM2_SOURCE_COMMIT,
    FM2_SOURCE_REPOSITORY,
    adapt_raw_row,
    summarize_pool,
)
from src.io_utils import read_jsonl
from src.run_manifest import atomic_write_json, git_state, sha256_file, utc_now


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and adapt official FM2 into shared RAG schemas")
    parser.add_argument("--raw-dir", default="data/raw/fm2")
    parser.add_argument("--output-dir", default="data/processed/fm2")
    parser.add_argument(
        "--splits", nargs="+", choices=sorted(FM2_EXPECTED_ROWS),
        default=["train", "dev"],
        help="Defaults to train/dev; test must be named explicitly after parameter freeze.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-noncanonical-source", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    raw_dir, output_dir = resolve(args.raw_dir), resolve(args.output_dir)
    outputs = [output_dir / f"fm2_{split}{suffix}" for split in args.splits for suffix in (".jsonl", "_official_pool.jsonl", "_gold_diagnostics.jsonl")]
    normalized_splits = tuple(sorted(args.splits))
    manifest_name = (
        "fm2_prepare.manifest.json"
        if set(args.splits) == {"train", "dev"}
        else f"fm2_prepare_{'_'.join(normalized_splits)}.manifest.json"
    )
    manifest_path = output_dir / manifest_name
    if any(path.exists() for path in [*outputs, manifest_path]) and not args.overwrite:
        raise FileExistsError("FM2 prepared artifacts exist; use --overwrite explicitly")
    pages: dict[str, set[str]] = {}
    split_manifests = {}
    for split in args.splits:
        raw_path = raw_dir / f"{split}.jsonl"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        raw_sha = sha256_file(raw_path)
        if not args.allow_noncanonical_source and raw_sha != FM2_EXPECTED_SHA256[split]:
            raise ValueError(f"FM2 {split} checksum mismatch: {raw_sha}")
        raw_rows = list(read_jsonl(raw_path, limit=args.limit))
        if args.limit is None and len(raw_rows) != FM2_EXPECTED_ROWS[split]:
            raise ValueError(f"FM2 {split} row count mismatch: {len(raw_rows)}")
        canonical, pools, diagnostics = [], [], []
        original_ids = set()
        for index, raw in enumerate(raw_rows, start=1):
            if str(raw.get("id")) in original_ids:
                raise ValueError(f"Duplicate FM2 id in {split}: {raw.get('id')}")
            original_ids.add(str(raw.get("id")))
            query, pool, diagnostic = adapt_raw_row(raw, split=split, row_number=index)
            canonical.append(query); pools.append(pool); diagnostics.append(diagnostic)
        pages[split] = {row["metadata"]["wikipedia_page"] for row in canonical}
        query_path = output_dir / f"fm2_{split}.jsonl"
        pool_path = output_dir / f"fm2_{split}_official_pool.jsonl"
        diagnostic_path = output_dir / f"fm2_{split}_gold_diagnostics.jsonl"
        atomic_jsonl(query_path, canonical); atomic_jsonl(pool_path, pools); atomic_jsonl(diagnostic_path, diagnostics)
        split_manifests[split] = {
            "official_source_split": split,
            "formal_role": FM2_FORMAL_ROLE_BY_OFFICIAL_SPLIT[split],
            "raw_path": str(raw_path.resolve()), "raw_sha256": raw_sha,
            "num_rows": len(canonical), "num_pages": len(pages[split]),
            "label_distribution": dict(sorted(Counter(row["label"] for row in canonical).items())),
            "query_path": str(query_path.resolve()), "query_sha256": sha256_file(query_path),
            "candidate_pool_path": str(pool_path.resolve()), "candidate_pool_sha256": sha256_file(pool_path),
            "gold_diagnostic_path": str(diagnostic_path.resolve()), "gold_diagnostic_sha256": sha256_file(diagnostic_path),
            "candidate_statistics": summarize_pool(pools),
            "any_gold_coverage_rows": sum(row["any_gold_in_official_pool"] for row in diagnostics),
            "all_gold_coverage_rows": sum(row["all_gold_in_official_pool"] for row in diagnostics),
        }
    if "test" in args.splits:
        for audit_split in ("train", "dev"):
            if audit_split in pages:
                continue
            audit_path = raw_dir / f"{audit_split}.jsonl"
            if not audit_path.is_file():
                raise FileNotFoundError(
                    f"Preparing test requires {audit_path} for the page-disjoint audit"
                )
            if sha256_file(audit_path) != FM2_EXPECTED_SHA256[audit_split]:
                raise ValueError(f"FM2 {audit_split} checksum mismatch during held-out audit")
            pages[audit_split] = {
                str(row.get("wikipedia_page")) for row in read_jsonl(audit_path)
            }
    page_overlaps = {}
    audited_splits = sorted(pages)
    for left_index, left in enumerate(audited_splits):
        for right in audited_splits[left_index + 1:]:
            overlap = sorted(pages[left] & pages[right])
            page_overlaps[f"{left}/{right}"] = {"count": len(overlap), "sample": overlap[:10]}
            if overlap:
                raise ValueError(f"FM2 page-disjoint violation in {left}/{right}: {overlap[:3]}")
    atomic_write_json(manifest_path, {
        "schema_version": FM2_PREPARE_MANIFEST_SCHEMA_VERSION,
        "status": "completed",
        "dataset_id": FM2_DATASET_ID,
        "dataset_family": FM2_DATASET_FAMILY,
        "retrieval_protocol_id": FM2_RETRIEVAL_PROTOCOL_ID,
        "source": {"repository": FM2_SOURCE_REPOSITORY, "commit": FM2_SOURCE_COMMIT},
        "raw_schema_fields": sorted(FM2_REQUIRED_FIELDS),
        "label_mapping": FM2_LABEL_MAPPING, "internal_labels": list(FM2_INTERNAL_LABELS),
        "split_contract": "official partitions mapped to formal roles without resampling; wikipedia_page disjointness enforced",
        "formal_role_mapping": dict(FM2_FORMAL_ROLE_BY_OFFICIAL_SPLIT),
        "candidate_pool_contract": {
            "protocol": FM2_RETRIEVAL_PROTOCOL_ID, "source_field": "retrieved_evidence",
            "shared_by_all_methods": True, "source_order_preserved": True,
            "gold_evidence_stored_separately": True, "supports_top20": False,
            "construction_gold_free": True, "known_source_page_required": True,
            "open_domain_deployable": False,
        },
        "page_overlaps": page_overlaps, "splits": split_manifests,
        "git": git_state(PROJECT_ROOT), "completed_at": utc_now(),
    })
    print(f"[prepare_fm2] splits={','.join(args.splits)} output={output_dir} manifest={manifest_path}")


if __name__ == "__main__":
    main()

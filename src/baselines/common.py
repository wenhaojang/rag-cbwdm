"""Shared atomic selection artifacts and method-specific resume validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from src.run_manifest import (
    atomic_write_json,
    git_state,
    sha256_file,
    stable_hash,
    utc_now,
    validate_resume_manifest,
)
from src.selection_schema import validate_selection_row

SELECTION_MANIFEST_SCHEMA = "rag_cbwdm_selection_manifest.v1"


def selection_manifest_path(output_path: str | Path) -> Path:
    return Path(output_path).with_suffix(".manifest.json")


def build_selection_contract(
    *,
    method: str,
    input_paths: dict[str, str | Path],
    parameters: dict[str, Any],
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inputs = {
        name: {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(input_paths.items())
    }
    return {
        "method": method,
        "inputs": inputs,
        "parameters": parameters,
        "model": model or {},
    }


def validate_selection_artifact(
    output_path: str | Path,
    contract: dict[str, Any],
) -> list[str]:
    output = Path(output_path)
    manifest_path = selection_manifest_path(output)
    reasons: list[str] = []
    if not output.is_file():
        reasons.append(f"missing selection output: {output}")
    if not manifest_path.is_file():
        reasons.append(f"missing selection manifest: {manifest_path}")
    if reasons:
        return reasons
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid selection manifest: {exc}"]
    expected = {
        "schema_version": SELECTION_MANIFEST_SCHEMA,
        "stage": "selection",
        "status": "completed",
        "completed": True,
        "method": contract["method"],
        "fingerprint": stable_hash(contract),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            reasons.append(
                f"{field}: expected={value!r} actual={manifest.get(field)!r}"
            )
    actual_sha = sha256_file(output)
    if manifest.get("output_sha256") != actual_sha:
        reasons.append(
            f"output_sha256: expected={manifest.get('output_sha256')!r} actual={actual_sha!r}"
        )
    try:
        rows = 0
        with output.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                validate_selection_row(row)
                if row.get("method") != contract["method"]:
                    reasons.append(
                        f"row {line_no} method: expected={contract['method']!r} "
                        f"actual={row.get('method')!r}"
                    )
                rows += 1
        if manifest.get("num_rows") != rows:
            reasons.append(
                f"num_rows: expected={manifest.get('num_rows')!r} actual={rows}"
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        reasons.append(f"invalid selection JSONL: {exc}")
    return reasons


def publish_selection(
    output_path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    contract: dict[str, Any],
    project_root: str | Path,
    resume: bool = False,
    overwrite: bool = False,
) -> tuple[int, bool]:
    """Atomically publish a selection and completed manifest.

    Returns ``(num_rows, reused)``. A completed manifest is only published after
    every row validates and the JSONL has been fsynced and atomically replaced.
    """
    output = Path(output_path)
    manifest_path = selection_manifest_path(output)
    fingerprint = stable_hash(contract)
    if resume and output.exists() and manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_resume_manifest(existing, fingerprint, stage=contract["method"])
        reasons = validate_selection_artifact(output, contract)
        if reasons:
            raise ValueError("Cannot resume invalid selection artifact:\n- " + "\n- ".join(reasons))
        return int(existing["num_rows"]), True
    if (output.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Selection artifact exists: {output}. Use --resume or --overwrite."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    started = utc_now()
    count = 0
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                validate_selection_row(row)
                if row.get("method") != contract["method"]:
                    raise ValueError(
                        f"Selection row method {row.get('method')!r} does not match "
                        f"contract method {contract['method']!r}"
                    )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()
    manifest = {
        "schema_version": SELECTION_MANIFEST_SCHEMA,
        "stage": "selection",
        "method": contract["method"],
        "status": "completed",
        "completed": True,
        "fingerprint": fingerprint,
        "contract": contract,
        "num_rows": count,
        "output_path": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "start_time": started,
        "end_time": utc_now(),
        "git": git_state(project_root),
    }
    atomic_write_json(manifest_path, manifest)
    reasons = validate_selection_artifact(output, contract)
    if reasons:
        raise ValueError("Published selection failed validation:\n- " + "\n- ".join(reasons))
    return count, False

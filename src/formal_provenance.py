"""Hashing and atomic text helpers for frozen formal experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.run_manifest import sha256_file


def sha256_path(path: str | Path) -> str:
    """Hash one file or a directory tree, including relative file names."""
    source = Path(path).resolve()
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise FileNotFoundError(f"Artifact path does not exist: {source}")
    digest = hashlib.sha256()
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Cannot fingerprint empty artifact directory: {source}")
    for item in files:
        relative = item.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_digest = bytes.fromhex(sha256_file(item))
        digest.update(file_digest)
    return digest.hexdigest()


def artifact_identity(path: str | Path, *, revision: str | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    identity = {
        "path": str(resolved),
        "revision": revision,
        "sha256": sha256_path(resolved),
        "kind": "directory" if resolved.is_dir() else "file",
    }
    if resolved.is_file() and resolved.suffix.casefold() == ".json":
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            identity["manifest_schema_version"] = payload.get("schema_version")
            identity["manifest_status"] = payload.get(
                "status", "completed" if payload.get("completed") is True else None
            )
            identity["manifest_fingerprint"] = payload.get(
                "fingerprint", payload.get("source_fingerprint")
            )
    return identity


def atomic_write_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)

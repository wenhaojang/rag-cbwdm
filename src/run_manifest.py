"""Atomic manifests and provenance helpers used by long-running server stages."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def git_state(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_status": status,
    }


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    try:
        import torch

        info.update(
            {
                "torch": str(torch.__version__),
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
            }
        )
    except ImportError:
        info["torch"] = None
    try:
        import transformers

        info["transformers"] = str(transformers.__version__)
    except ImportError:
        info["transformers"] = None
    return info


def validate_resume_manifest(
    existing: dict[str, Any],
    expected_fingerprint: str,
    *,
    stage: str,
) -> None:
    """Refuse resume when a cache-defining field has changed."""
    actual = existing.get("fingerprint")
    if actual != expected_fingerprint:
        raise ValueError(
            f"Cannot resume {stage}: manifest fingerprint mismatch "
            f"(existing={actual}, requested={expected_fingerprint}). "
            "Use a new output path or explicit --overwrite."
        )

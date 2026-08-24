from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.fm2 import FM2_EXPECTED_SHA256, FM2_SOURCE_COMMIT, FM2_SOURCE_URLS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download checksum-pinned official FM2 JSONL files")
    parser.add_argument("--output-dir", default="data/raw/fm2")
    parser.add_argument(
        "--splits", nargs="+", choices=sorted(FM2_SOURCE_URLS),
        default=["train", "dev"],
        help="Defaults to train/dev; request test only after the held-out gate is open.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        target = output_dir / f"{split}.jsonl"
        expected = FM2_EXPECTED_SHA256[split]
        if target.is_file() and not args.overwrite:
            if sha256_file(target) != expected:
                raise ValueError(f"Existing {target} has an unexpected checksum")
            print(f"[download_fm2] split={split} reused=true sha256={expected}")
            continue
        partial = target.with_name(target.name + ".partial")
        if partial.exists():
            partial.unlink()
        try:
            with urllib.request.urlopen(FM2_SOURCE_URLS[split]) as response, partial.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            actual = sha256_file(partial)
            if actual != expected:
                raise ValueError(f"FM2 {split} checksum mismatch: {actual} != {expected}")
            os.replace(partial, target)
        except BaseException:
            if partial.exists():
                partial.unlink()
            raise
        print(f"[download_fm2] split={split} reused=false commit={FM2_SOURCE_COMMIT} sha256={expected}")


if __name__ == "__main__":
    main()

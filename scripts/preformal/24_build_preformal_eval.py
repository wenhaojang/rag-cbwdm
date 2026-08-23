from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preformal.splits import publish_strategy_a


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean preformal_eval using frozen strategy A")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = publish_strategy_a(args.split_manifest, args.output_dir, project_root=PROJECT_ROOT, resume=args.resume)
    print(f"[preformal_split] strategy=A rows={manifest['row_count']} sha256={manifest['preformal_eval_sha256']} output={args.output_dir}")


if __name__ == "__main__":
    main()

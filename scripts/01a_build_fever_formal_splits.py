from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.formal_splits import publish_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen train_core/validation/held_out_test FEVER-2 splits."
    )
    parser.add_argument("--official-train", required=True)
    parser.add_argument("--official-dev", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=13)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validation-size", type=int, default=5000)
    group.add_argument("--validation-fraction", type=float)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_size = None if args.validation_fraction is not None else args.validation_size
    manifest, reused = publish_splits(
        args.official_train,
        args.official_dev,
        args.output_dir,
        seed=args.seed,
        validation_size=validation_size,
        validation_fraction=args.validation_fraction,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        resume=args.resume,
        overwrite=args.overwrite,
        project_root=PROJECT_ROOT,
    )
    print(
        f"[formal_splits] status={manifest['status']} reused={reused} "
        f"fingerprint={manifest['fingerprint']} output={Path(args.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()

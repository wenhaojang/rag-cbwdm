from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.fever import publish_calibration


def _assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected non-empty NAME=PATH")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select InfoGain and RAG-CBWDM parameters using validation metrics only."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--validation-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument(
        "--artifact",
        action="append",
        type=_assignment,
        default=[],
        metavar="NAME=PATH",
        help="Fingerprint validation retrieval/posterior/checkpoint artifacts.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, reused = publish_calibration(
        args.config,
        args.split_manifest,
        args.validation_metrics,
        args.output_dir,
        objective=args.objective,
        artifacts=dict(args.artifact),
        resume=args.resume,
        overwrite=args.overwrite,
        project_root=PROJECT_ROOT,
    )
    print(
        f"[calibration] status={manifest['status']} reused={reused} "
        f"fingerprint={manifest['fingerprint']} output={Path(args.output_dir).resolve()}"
    )
    if manifest["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

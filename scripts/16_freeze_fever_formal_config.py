from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.formal_config import REQUIRED_MODELS, publish_frozen_config


def _assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=VALUE")
    name, item = value.split("=", 1)
    if not name or not item:
        raise argparse.ArgumentTypeError("Expected non-empty NAME=VALUE")
    return name, item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze an immutable FEVER formal config.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "configs" / "generated"))
    parser.add_argument("--model", action="append", type=_assignment, default=[])
    parser.add_argument("--revision", action="append", type=_assignment, default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = dict(args.model)
    revisions = dict(args.revision)
    missing_models = [name for name in REQUIRED_MODELS if name not in models]
    missing_revisions = [name for name in REQUIRED_MODELS if not revisions.get(name)]
    if missing_models or missing_revisions:
        raise ValueError(
            f"All model paths and immutable revisions are required; "
            f"missing models={missing_models}, missing revisions={missing_revisions}"
        )
    config_path, manifest_path, diff_path = publish_frozen_config(
        args.base_config,
        args.split_manifest,
        args.calibration_manifest,
        args.output_dir,
        models=models,
        revisions=revisions,
        corpus=args.corpus,
        index=args.index,
        overwrite=args.overwrite,
        project_root=PROJECT_ROOT,
    )
    print(f"[freeze_formal] config={config_path}")
    print(f"[freeze_formal] manifest={manifest_path}")
    print(f"[freeze_formal] diff={diff_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cbwdm_diagnostics import publish_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build validation pilot CBWDM diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-evidence", required=True)
    parser.add_argument("--naive", required=True)
    parser.add_argument("--cbwdm", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--naive-selection", required=True)
    parser.add_argument("--cbwdm-selection", required=True)
    parser.add_argument("--oracle-selection", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = (
        "no_evidence",
        "naive",
        "cbwdm",
        "oracle",
        "naive_selection",
        "cbwdm_selection",
        "oracle_selection",
        "teacher",
        "retrieval",
        "posteriors",
    )
    payload = publish_diagnostics(
        config_path=args.config,
        paths={key: getattr(args, key) for key in keys},
        output_dir=args.output_dir,
        project_root=PROJECT_ROOT,
    )
    print(
        f"[cbwdm_diagnostics] status={payload['status']} "
        f"output={Path(args.output_dir).resolve()}"
    )
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

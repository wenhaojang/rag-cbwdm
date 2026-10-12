"""Audit and merge the two completed FEVER budget-sweep summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fever_budget_sweep import merge_budget_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate the original and extra FEVER 30-point budget "
            "summaries, verify all k4 anchors, and optionally emit 60 rows."
        )
    )
    parser.add_argument("--original-csv", required=True)
    parser.add_argument("--extra-csv", required=True)
    parser.add_argument("--main-results", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--output-manifest")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        if args.output_csv or args.output_manifest:
            raise ValueError("--validate-only cannot be combined with output paths")
        output_csv = None
        output_manifest = None
    else:
        if not args.output_csv:
            raise ValueError("--output-csv is required unless --validate-only is used")
        output_csv = args.output_csv
        output_manifest = args.output_manifest
    report = merge_budget_summaries(
        args.original_csv,
        args.extra_csv,
        main_results_csv=args.main_results,
        output_csv=output_csv,
        output_manifest=output_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

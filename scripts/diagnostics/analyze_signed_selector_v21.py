from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.method_failure import require_diagnostic_output
from src.diagnostics.signed_selector_v1 import oracle_imitation, posthoc_alignment
from src.diagnostics.signed_selector_v21 import (
    SIGNED_V21_METHOD,
    harmful_selection_audit,
    score_separability_audit,
    stopping_and_budget_audit,
)
from src.formal_provenance import atomic_write_text
from src.io_utils import load_yaml, read_jsonl
from src.run_manifest import atomic_write_json, git_state, sha256_file, utc_now


def _map(path: Path) -> dict[str, dict]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only post-hoc diagnostics for rag_cbwdm_signed_v21"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--oracle-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alignment-eps", type=float, default=0.0)
    args = parser.parse_args()
    if args.alignment_eps != 0.0:
        raise ValueError("signed-v2.1 first-round alignment_eps is frozen at 0.0")

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    run = Path(args.run_dir).resolve()
    output = require_diagnostic_output(run, Path(args.output_dir).resolve())
    root = (
        run / "artifacts/diagnostics/method_failure_audit/signed_v21/reports"
    ).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"post-hoc output must be below {root}") from exc

    paths = {
        "config": config_path,
        "posteriors": Path(args.posteriors).resolve(),
        "selection": Path(args.selection).resolve(),
        "predictions": Path(args.predictions).resolve(),
        "metrics": Path(args.metrics).resolve(),
        "oracle_selection": Path(args.oracle_selection).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing signed-v2.1 inputs:\n- " + "\n- ".join(missing))

    posteriors = _map(paths["posteriors"])
    selections = _map(paths["selection"])
    predictions = _map(paths["predictions"])
    oracle = _map(paths["oracle_selection"])
    for name, rows in (
        ("posteriors", posteriors),
        ("predictions", predictions),
        ("oracle selection", oracle),
    ):
        missing_ids = sorted(set(selections) - set(rows))
        if missing_ids:
            raise ValueError(f"Selection IDs missing from {name}: {missing_ids[:5]}")
    for identifier, row in selections.items():
        metadata = row.get("selection_metadata", {})
        if row.get("method") != SIGNED_V21_METHOD:
            raise ValueError(f"Selection row is not signed-v2.1: id={identifier}")
        if row.get("uses_gold_at_test") or metadata.get("uses_gold_at_inference") is not False:
            raise ValueError(f"Selection does not prove gold-free inference: id={identifier}")

    payload = {
        "schema_version": "rag_cbwdm_signed_v21_posthoc.v1",
        "method": SIGNED_V21_METHOD,
        "created_at": utc_now(),
        "posthoc_only": True,
        "uses_gold_for_selection": False,
        "alignment_eps": args.alignment_eps,
        "production_evaluation_metrics": json.loads(
            paths["metrics"].read_text(encoding="utf-8")
        ),
        "budget": stopping_and_budget_audit(selections),
        "selected_authoritative_alignment": posthoc_alignment(
            selection_rows=selections,
            posterior_rows=posteriors,
            cbwdm=config["cbwdm"],
        ),
        "harmful_selection_and_supports_error_association": harmful_selection_audit(
            selection_rows=selections,
            posterior_rows=posteriors,
            prediction_rows=predictions,
            cbwdm=config["cbwdm"],
        ),
        "score_separability": score_separability_audit(
            selection_rows=selections,
            posterior_rows=posteriors,
            cbwdm=config["cbwdm"],
            alignment_eps=args.alignment_eps,
        ),
        "signed_oracle_imitation": oracle_imitation(
            selection_rows=selections,
            oracle_rows=oracle,
            posterior_rows=posteriors,
        ),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "git": git_state(PROJECT_ROOT),
    }
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "SIGNED_SELECTOR_V21_VALIDATION_DIAGNOSTICS.json"
    markdown_path = output / "SIGNED_SELECTOR_V21_VALIDATION_DIAGNOSTICS.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("signed-v2.1 diagnostics exist; use a new output directory")
    atomic_write_json(json_path, payload)
    atomic_write_text(
        markdown_path,
        "# signed-selector v2.1 Validation Diagnostics\n\n"
        "- Selection-time gold leakage: **NO GOLD LEAKAGE**\n"
        "- Raw and authoritative alignments below are post-hoc only.\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}\n```\n",
    )
    print(f"[signed_selector_v21_posthoc] rows={len(selections)} output={output}")


if __name__ == "__main__":
    main()

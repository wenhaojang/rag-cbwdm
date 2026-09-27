from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.method_failure import require_diagnostic_output
from src.diagnostics.signed_selector_v1 import oracle_imitation, posthoc_alignment
from src.diagnostics.signed_selector_v21 import (
    harmful_selection_audit,
    score_separability_audit,
    stopping_and_budget_audit,
)
from src.diagnostics.signed_selector_v22 import (
    SIGNED_V22_ARCHITECTURE,
    SIGNED_V22_METHOD,
    SIGNED_V22_VARIANT,
)
from src.formal_provenance import atomic_write_text
from src.io_utils import load_yaml, read_jsonl
from src.run_manifest import atomic_write_json, git_state, sha256_file, utc_now


def _map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _add_later_only(groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    enriched: dict[str, dict[str, Any]] = {}
    for key, source in groups.items():
        bucket = dict(source)
        bucket["later_only_raw_negative"] = (
            int(bucket.get("queries_with_any_raw_negative", 0))
            - int(bucket.get("first_selected_raw_negative", 0))
        )
        enriched[key] = bucket
    return enriched


def _supports_error_probabilities(
    groups: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    supports = groups.get("gold=SUPPORTS", {})
    wrong = groups.get("gold=SUPPORTS/wrong", {})
    total_queries = int(supports.get("queries", 0))
    wrong_queries = int(wrong.get("queries", 0))
    any_negative = int(supports.get("queries_with_any_raw_negative", 0))
    wrong_any_negative = int(wrong.get("queries_with_any_raw_negative", 0))
    first_negative = int(supports.get("first_selected_raw_negative", 0))
    wrong_first_negative = int(wrong.get("first_selected_raw_negative", 0))
    selected_queries = int(supports.get("queries_with_selected_docs", 0))
    wrong_selected_queries = int(wrong.get("queries_with_selected_docs", 0))
    no_negative = total_queries - any_negative
    wrong_no_negative = wrong_queries - wrong_any_negative
    first_nonnegative = selected_queries - first_negative
    wrong_first_nonnegative = wrong_selected_queries - wrong_first_negative
    return {
        "supports_queries": total_queries,
        "supports_wrong_queries": wrong_queries,
        "supports_queries_with_selected_docs": selected_queries,
        "queries_with_any_raw_negative": any_negative,
        "first_selected_raw_negative": first_negative,
        "later_only_raw_negative": any_negative - first_negative,
        "p_wrong_given_any_raw_negative": _ratio(
            wrong_any_negative, any_negative
        ),
        "p_wrong_given_no_raw_negative": _ratio(wrong_no_negative, no_negative),
        "p_wrong_given_first_raw_negative": _ratio(
            wrong_first_negative, first_negative
        ),
        "p_wrong_given_first_raw_nonnegative": _ratio(
            wrong_first_nonnegative, first_nonnegative
        ),
    }


def _priority_metrics(
    *,
    harmful_groups: dict[str, dict[str, Any]],
    separability: dict[str, Any],
    evaluation_metrics: dict[str, Any],
) -> dict[str, Any]:
    supports = harmful_groups.get("gold=SUPPORTS", {})
    all_group = harmful_groups.get("ALL", {})
    supports_step0 = separability.get("gold=SUPPORTS/step=0", {})
    raw_step0 = supports_step0.get("raw_pos_vs_neg", {})
    return {
        "priority_order": [
            "supports_first_selected_raw_negative",
            "supports_queries_with_any_raw_negative",
            "selected_raw_negative_ratio",
            "supports_step0_raw_sign_auroc",
            "accuracy_and_macro_f1",
        ],
        "supports_first_selected_raw_negative": {
            "signed_v1_baseline": 104,
            "signed_v21_baseline": 103,
            "signed_v22_observed": supports.get("first_selected_raw_negative"),
        },
        "supports_queries_with_any_raw_negative": {
            "signed_v1_baseline": 141,
            "signed_v21_baseline": 128,
            "signed_v22_observed": supports.get("queries_with_any_raw_negative"),
        },
        "selected_raw_negative_ratio": {
            "signed_v1_baseline": 0.3246,
            "signed_v21_baseline": 0.2991,
            "signed_v22_observed": all_group.get("selected_raw_negative_ratio"),
        },
        "supports_step0_raw_sign_auroc": {
            "signed_v1_baseline": 0.8177,
            "signed_v21_baseline": 0.8094,
            "signed_v22_observed": raw_step0.get("auroc"),
        },
        "accuracy": evaluation_metrics.get("accuracy"),
        "macro_f1": evaluation_metrics.get("macro_f1"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only post-hoc diagnostics for rag_cbwdm_signed_v22"
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
        raise ValueError("signed-v2.2 first-round alignment_eps is frozen at 0.0")

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    run = Path(args.run_dir).resolve()
    output = require_diagnostic_output(run, Path(args.output_dir).resolve())
    root = (
        run / "artifacts/diagnostics/method_failure_audit/signed_v22/reports"
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
        raise FileNotFoundError("Missing signed-v2.2 inputs:\n- " + "\n- ".join(missing))

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
        if row.get("method") != SIGNED_V22_METHOD:
            raise ValueError(f"Selection row is not signed-v2.2: id={identifier}")
        if metadata.get("variant") != SIGNED_V22_VARIANT:
            raise ValueError(f"Selection variant is not signed-v2.2: id={identifier}")
        if metadata.get("architecture") != SIGNED_V22_ARCHITECTURE:
            raise ValueError(f"Selection architecture mismatch: id={identifier}")
        if row.get("uses_gold_at_test") or metadata.get("uses_gold_at_inference") is not False:
            raise ValueError(f"Selection does not prove gold-free inference: id={identifier}")

    evaluation_metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    harmful = harmful_selection_audit(
        selection_rows=selections,
        posterior_rows=posteriors,
        prediction_rows=predictions,
        cbwdm=config["cbwdm"],
    )
    harmful_groups = _add_later_only(harmful)
    separability = score_separability_audit(
        selection_rows=selections,
        posterior_rows=posteriors,
        cbwdm=config["cbwdm"],
        alignment_eps=args.alignment_eps,
    )
    payload = {
        "schema_version": "rag_cbwdm_signed_v22_posthoc.v1",
        "method": SIGNED_V22_METHOD,
        "variant": SIGNED_V22_VARIANT,
        "created_at": utc_now(),
        "posthoc_only": True,
        "uses_gold_for_selection": False,
        "alignment_eps": args.alignment_eps,
        "production_evaluation_metrics": evaluation_metrics,
        "priority_success_metrics": _priority_metrics(
            harmful_groups=harmful_groups,
            separability=separability,
            evaluation_metrics=evaluation_metrics,
        ),
        "budget": stopping_and_budget_audit(selections),
        "selected_authoritative_alignment": posthoc_alignment(
            selection_rows=selections,
            posterior_rows=posteriors,
            cbwdm=config["cbwdm"],
        ),
        "selected_harmful_and_query_association": harmful_groups,
        "supports_error_probabilities": _supports_error_probabilities(
            harmful_groups
        ),
        "score_separability": separability,
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
    json_path = output / "SIGNED_SELECTOR_V22_VALIDATION_DIAGNOSTICS.json"
    markdown_path = output / "SIGNED_SELECTOR_V22_VALIDATION_DIAGNOSTICS.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("signed-v2.2 diagnostics exist; use a new output directory")
    atomic_write_json(json_path, payload)
    atomic_write_text(
        markdown_path,
        "# signed-selector v2.2 Validation Diagnostics\n\n"
        "- Selection-time gold leakage: **NO GOLD LEAKAGE**\n"
        "- Raw and authoritative alignments below are post-hoc only.\n"
        "- Priority: SUPPORTS first-selected raw-negative before accuracy.\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}\n```\n",
    )
    print(f"[signed_selector_v22_posthoc] rows={len(selections)} output={output}")


if __name__ == "__main__":
    main()

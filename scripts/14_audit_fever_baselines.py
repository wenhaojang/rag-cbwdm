from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import read_jsonl
from src.run_manifest import atomic_write_json, sha256_file, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FEVER baseline comparability.")
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--selection", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--evaluation-manifest", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--expected-top-m", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def assignments(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected METHOD=PATH, got {value!r}")
        method, path = value.split("=", 1)
        result[method] = absolute(path)
    return result


def retrieval_contract(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    ids, candidates = [], {}
    for row in read_jsonl(path):
        sample_id = str(row["id"])
        ids.append(sample_id)
        candidates[sample_id] = [str(item["doc_id"]) for item in row.get("candidates", [])]
    return ids, candidates


def main() -> None:
    args = parse_args()
    retrieval = absolute(args.retrieval)
    expected_ids, candidate_ids = retrieval_contract(retrieval)
    selections = assignments(args.selection)
    evaluations = assignments(args.evaluation_manifest)
    methods: dict[str, Any] = {}
    global_reasons: list[str] = []
    generator_contracts = set()
    for method, path in selections.items():
        if not path.is_file():
            methods[method] = {
                "status": "missing",
                "reasons": [f"selection artifact missing: {path}"],
                "num_queries": None,
                "retrieval_sha256": sha256_file(retrieval),
                "generator_contract": None,
                "state_aware": method == "rag_cbwdm",
                "teacher": None,
            }
            continue
        reasons = []
        rows = list(read_jsonl(path))
        ids = [str(row["id"]) for row in rows]
        if ids != expected_ids:
            reasons.append("prepared/evaluation query IDs differ from retrieval IDs")
        for row in rows:
            sample_id = str(row["id"])
            unknown = set(row.get("selected_doc_ids", [])) - set(candidate_ids.get(sample_id, []))
            if unknown:
                reasons.append(f"{sample_id}: selected docs absent from shared candidate pool: {sorted(unknown)}")
            if int(row.get("max_docs", -1)) != args.expected_top_m:
                reasons.append(
                    f"{sample_id}: max_docs={row.get('max_docs')} expected={args.expected_top_m}"
                )
        manifest_path = evaluations.get(method)
        generator = None
        if manifest_path and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            contract = manifest.get("contract", {})
            generator = (
                contract.get("generator_model"),
                contract.get("generator_revision"),
                contract.get("prompt_hash"),
                contract.get("verbalizer_hash"),
                contract.get("max_context_tokens"),
            )
            generator_contracts.add(generator)
            if contract.get("selection_sha256") != sha256_file(path):
                reasons.append("evaluation selection checksum does not match selection")
        methods[method] = {
            "status": "comparable" if not reasons else "not_comparable",
            "reasons": reasons,
            "num_queries": len(rows),
            "retrieval_sha256": sha256_file(retrieval),
            "generator_contract": generator,
            "state_aware": method == "rag_cbwdm",
            "teacher": (
                "pointwise probability-difference DIG"
                if method == "infogain_fever"
                else "set-dependent marginal gain"
                if method == "rag_cbwdm"
                else None
            ),
        }
        global_reasons.extend(f"{method}: {reason}" for reason in reasons)
    for method, manifest_path in evaluations.items():
        if method in methods:
            continue
        if not manifest_path.is_file():
            methods[method] = {
                "status": "missing",
                "reasons": [f"evaluation manifest missing: {manifest_path}"],
                "num_queries": None,
                "retrieval_sha256": sha256_file(retrieval),
                "generator_contract": None,
                "state_aware": False,
                "teacher": None,
            }
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = manifest.get("contract", {})
        generator = (
            contract.get("generator_model"),
            contract.get("generator_revision"),
            contract.get("prompt_hash"),
            contract.get("verbalizer_hash"),
            contract.get("max_context_tokens"),
        )
        generator_contracts.add(generator)
        methods[method] = {
            "status": "comparable",
            "reasons": [],
            "num_queries": manifest.get("num_rows"),
            "retrieval_sha256": sha256_file(retrieval),
            "generator_contract": generator,
            "state_aware": False,
            "teacher": None,
        }
    if len(generator_contracts) > 1:
        global_reasons.append("generator/prompt/verbalizer/max-context provenance differs")
    payload = {
        "schema_version": "rag_cbwdm_baseline_fairness_audit.v1",
        "status": "comparable" if not global_reasons else "not_comparable",
        "retrieval": str(retrieval.resolve()),
        "retrieval_sha256": sha256_file(retrieval),
        "expected_top_m": args.expected_top_m,
        "methods": methods,
        "reasons": global_reasons,
        "created_at": utc_now(),
    }
    atomic_write_json(absolute(args.output), payload)
    print(f"[fairness_audit] status={payload['status']} methods={len(methods)} output={absolute(args.output)}")
    if global_reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

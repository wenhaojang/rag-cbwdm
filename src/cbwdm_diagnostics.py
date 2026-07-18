"""Pilot diagnostics and P0 gates for the FEVER RAG-CBWDM selector."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from src.formal_provenance import atomic_write_text
from src.io_utils import load_yaml, read_jsonl
from src.metrics import ClassificationMetrics
from src.run_manifest import atomic_write_json, git_state, sha256_file, utc_now

DIAGNOSTIC_SCHEMA_VERSION = "rag_cbwdm_fever_diagnostics.v1"


def _map(path: str | Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        identifier = str(row.get("id", ""))
        if not identifier or identifier in result:
            raise ValueError(f"Duplicate or missing id in {path}: {identifier!r}")
        result[identifier] = row
    return result


def _prediction(row: dict[str, Any]) -> str | None:
    value = row.get("pred", row.get("prediction", row.get("predicted_label")))
    return None if value is None else str(value)


def _gold(row: dict[str, Any]) -> str | None:
    value = row.get("gold", row.get("label"))
    return None if value is None else str(value)


def _selected_ids(row: dict[str, Any]) -> list[str]:
    ids = row.get("selected_doc_ids")
    if isinstance(ids, list):
        return [str(item) for item in ids]
    docs = row.get("selected_docs")
    if isinstance(docs, list):
        return [str(item["doc_id"]) for item in docs if isinstance(item, dict) and "doc_id" in item]
    return []


def _summary(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "reason": "no finite values",
        }
    ordered = sorted(finite)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p10": ordered[int((len(ordered) - 1) * 0.10)],
        "p90": ordered[int((len(ordered) - 1) * 0.90)],
        "reason": None,
    }


def _classification(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    metrics = ClassificationMetrics(labels=["SUPPORTS", "REFUTES"])
    for identifier in ids:
        row = rows[identifier]
        metrics.update(
            _gold(row),
            _prediction(row),
            num_docs=len(_selected_ids(row)),
            probs=row.get("probs"),
        )
    return metrics.compute()


def _gold_ids(row: dict[str, Any]) -> set[str] | None:
    for container in (row, row.get("meta") if isinstance(row.get("meta"), dict) else {}):
        value = container.get("gold_evidence_doc_ids")
        if isinstance(value, list):
            return {str(item) for item in value}
    return None


def _candidate_key(candidate: dict[str, Any]) -> str | None:
    meta = candidate.get("meta")
    if not isinstance(meta, dict):
        return None
    page = meta.get("page_id")
    sentence = meta.get("sentence_id")
    if page is None or sentence is None:
        return None
    return f"{page}\t{sentence}"


def _recall(
    retrieval: dict[str, dict[str, Any]],
    selection: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    values: list[float] = []
    missing = 0
    for identifier in ids:
        gold = _gold_ids(retrieval[identifier])
        if gold is not None and gold:
            selected = set(_selected_ids(selection[identifier]))
            values.append(len(gold & selected) / len(gold))
            continue
        gold_keys = retrieval[identifier].get("gold_evidence_keys")
        if not isinstance(gold_keys, list) or not gold_keys:
            missing += 1
            continue
        selected_ids = set(_selected_ids(selection[identifier]))
        selected_keys = {
            key
            for candidate in retrieval[identifier].get("candidates", [])
            if str(candidate.get("doc_id")) in selected_ids
            if (key := _candidate_key(candidate)) is not None
        }
        values.append(len(set(map(str, gold_keys)) & selected_keys) / len(set(gold_keys)))
    if not values:
        return {
            "value": None,
            "num_examples": 0,
            "missing_examples": missing,
            "reason": "gold_evidence_doc_ids are unavailable in retrieval artifacts",
        }
    return {
        "value": statistics.fmean(values),
        "num_examples": len(values),
        "missing_examples": missing,
        "reason": None if missing == 0 else f"{missing} examples lack gold evidence ids",
    }


def _candidate_selection(retrieval: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        identifier: {
            "selected_doc_ids": [
                str(candidate["doc_id"]) for candidate in row.get("candidates", [])
            ]
        }
        for identifier, row in retrieval.items()
    }


def build_diagnostics(
    *,
    config: dict[str, Any],
    no_evidence: dict[str, dict[str, Any]],
    naive: dict[str, dict[str, Any]],
    cbwdm: dict[str, dict[str, Any]],
    oracle: dict[str, dict[str, Any]],
    naive_selection: dict[str, dict[str, Any]],
    cbwdm_selection: dict[str, dict[str, Any]],
    oracle_selection: dict[str, dict[str, Any]],
    teacher: dict[str, dict[str, Any]],
    retrieval: dict[str, dict[str, Any]],
    posteriors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    collections = {
        "no_evidence": no_evidence,
        "naive": naive,
        "cbwdm": cbwdm,
        "oracle": oracle,
        "naive_selection": naive_selection,
        "cbwdm_selection": cbwdm_selection,
        "oracle_selection": oracle_selection,
        "teacher": teacher,
        "retrieval": retrieval,
        "posteriors": posteriors,
    }
    common = set.intersection(*(set(rows) for rows in collections.values()))
    union = set.union(*(set(rows) for rows in collections.values()))
    ids = sorted(common)
    if not ids:
        raise ValueError("CBWDM diagnostics have no aligned pilot examples")

    gains: list[float] = []
    best_gains: list[float] = []
    stop_reasons: Counter[str] = Counter()
    teacher_doc_counts: list[int] = []
    schema_errors: list[str] = []
    oracle_mismatches = 0
    for identifier in ids:
        teacher_row = teacher[identifier]
        if teacher_row.get("schema_version") != "rag_cbwdm_teacher.v2":
            schema_errors.append(f"{identifier}: teacher schema")
        teacher_ids = [str(item) for item in teacher_row.get("teacher_selected_doc_ids", [])]
        teacher_doc_counts.append(len(teacher_ids))
        if _selected_ids(oracle_selection[identifier]) != teacher_ids:
            oracle_mismatches += 1
        stop_reasons[str(teacher_row.get("stop_reason", "missing"))] += 1
        for step in teacher_row.get("steps", []):
            if step.get("best_gain") is not None:
                best_gains.append(float(step["best_gain"]))
            for item in step.get("candidate_gains", []):
                if item.get("gain") is not None:
                    gains.append(float(item["gain"]))

    cbwdm_doc_counts = [len(_selected_ids(cbwdm_selection[item])) for item in ids]
    oracle_overlaps = []
    for identifier in ids:
        left = set(_selected_ids(cbwdm_selection[identifier]))
        right = set(_selected_ids(oracle_selection[identifier]))
        oracle_overlaps.append(len(left & right) / len(left | right) if left | right else 1.0)

    flips = Counter()
    for identifier in ids:
        gold = _gold(cbwdm[identifier])
        no_ok = _prediction(no_evidence[identifier]) == gold
        naive_ok = _prediction(naive[identifier]) == gold
        cbwdm_ok = _prediction(cbwdm[identifier]) == gold
        flips["no_evidence_correct_to_cbwdm_wrong"] += int(no_ok and not cbwdm_ok)
        flips["naive_correct_to_cbwdm_wrong"] += int(naive_ok and not cbwdm_ok)
        flips["cbwdm_correct_to_naive_wrong"] += int(cbwdm_ok and not naive_ok)

    query_confidence = []
    selected_confidence = []
    for identifier in ids:
        posterior = posteriors[identifier]
        eta0 = posterior.get("eta0")
        if isinstance(eta0, list) and eta0:
            query_confidence.append(max(map(float, eta0)))
        chosen = set(_selected_ids(cbwdm_selection[identifier]))
        eta_values = [
            max(map(float, candidate["eta"]))
            for candidate in posterior.get("candidates", [])
            if candidate.get("doc_id") in chosen and isinstance(candidate.get("eta"), list)
        ]
        if eta_values:
            selected_confidence.append(max(eta_values))

    metrics = {
        "no_evidence": _classification(no_evidence, ids),
        "naive_topm": _classification(naive, ids),
        "rag_cbwdm": _classification(cbwdm, ids),
        "cbwdm_oracle": _classification(oracle, ids),
    }
    candidate_recall = _recall(retrieval, _candidate_selection(retrieval), ids)
    selected_recall = _recall(retrieval, cbwdm_selection, ids)
    labels_ok = config.get("task", {}).get("labels") == ["SUPPORTS", "REFUTES"]
    verbalizers = config.get("task", {}).get("verbalizers", {})
    verbalizers_ok = (
        isinstance(verbalizers, dict)
        and "A" in verbalizers.get("SUPPORTS", [])
        and "B" in verbalizers.get("REFUTES", [])
    )
    top_m = int(config.get("selector", {}).get("top_m", 4))
    gain_tolerance = float(config.get("cbwdm", {}).get("gain_tolerance", 1e-10))
    gain_signs = {
        "positive": sum(value > gain_tolerance for value in gains),
        "negative": sum(value < -gain_tolerance for value in gains),
        "zero": sum(abs(value) <= gain_tolerance for value in gains),
    }
    checks = {
        "all_artifacts_aligned": {
            "passed": len(common) == len(union),
            "reason": None if len(common) == len(union) else f"aligned={len(common)} union={len(union)}",
        },
        "teacher_and_selector_schema": {
            "passed": not schema_errors,
            "reason": None if not schema_errors else "; ".join(schema_errors[:5]),
        },
        "oracle_matches_teacher": {
            "passed": oracle_mismatches == 0,
            "reason": None if oracle_mismatches == 0 else f"{oracle_mismatches} mismatches",
        },
        "gain_direction": {
            "passed": bool(gains) and gain_signs["positive"] > 0 and gain_signs["negative"] == 0,
            "reason": (
                None
                if bool(gains) and gain_signs["positive"] > 0 and gain_signs["negative"] == 0
                else f"gain signs={gain_signs}"
            ),
        },
        "document_budget": {
            "passed": bool(cbwdm_doc_counts) and max(cbwdm_doc_counts) <= top_m,
            "reason": None if bool(cbwdm_doc_counts) and max(cbwdm_doc_counts) <= top_m else f"top_m={top_m}",
        },
        "gold_evidence_recall": {
            "passed": (
                candidate_recall["value"] is not None
                and selected_recall["value"] is not None
                and candidate_recall["value"] >= selected_recall["value"]
                and candidate_recall["value"] > 0
            ),
            "reason": (
                None
                if candidate_recall["value"] is not None
                and selected_recall["value"] is not None
                and candidate_recall["value"] >= selected_recall["value"]
                and candidate_recall["value"] > 0
                else candidate_recall["reason"] or selected_recall["reason"] or "abnormal recall"
            ),
        },
        "label_order": {"passed": labels_ok, "reason": None if labels_ok else "label order mismatch"},
        "verbalizers": {
            "passed": verbalizers_ok,
            "reason": None if verbalizers_ok else "SUPPORTS=A / REFUTES=B mapping mismatch",
        },
        "pilot_scale": {
            "passed": len(ids)
            >= int(config.get("diagnostics", {}).get("min_pilot_examples", 500)),
            "reason": (
                None
                if len(ids) >= int(config.get("diagnostics", {}).get("min_pilot_examples", 500))
                else f"only {len(ids)} aligned examples"
            ),
        },
    }
    oracle_delta = (
        metrics["cbwdm_oracle"]["accuracy"] - metrics["naive_topm"]["accuracy"]
    )
    risk_tolerance = float(config.get("diagnostics", {}).get("oracle_naive_tolerance", 0.02))
    scientific_risk = oracle_delta < -risk_tolerance
    blockers = [name for name, check in checks.items() if not check["passed"]]
    if scientific_risk:
        blockers.append("oracle_significantly_below_naive")
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "data_role": "validation",
        "num_aligned_examples": len(ids),
        "alignment": {"intersection": len(common), "union": len(union)},
        "prediction_flips": dict(flips),
        "classification": metrics,
        "selection": {
            "cbwdm_vs_oracle_jaccard": _summary(oracle_overlaps),
            "teacher_num_docs": _summary([float(value) for value in teacher_doc_counts]),
            "selector_num_docs": _summary([float(value) for value in cbwdm_doc_counts]),
            "stop_reason_counts": dict(sorted(stop_reasons.items())),
            "stopping_trigger_rate": sum(
                count
                for reason, count in stop_reasons.items()
                if reason not in {"top_m_reached", "no_candidates"}
            )
            / len(ids),
        },
        "gains": {
            "all_candidate_gains": _summary(gains),
            "best_gains": _summary(best_gains),
            "sign_counts": gain_signs,
        },
        "gold_evidence": {
            "candidate_recall_at_20": candidate_recall,
            "selected_recall_at_4": selected_recall,
        },
        "posterior_confidence": {
            "query_only": _summary(query_confidence),
            "selected_evidence": _summary(selected_confidence),
        },
        "oracle_below_naive_analysis": {
            "accuracy_delta": oracle_delta,
            "significant_tolerance": risk_tolerance,
            "scientific_risk": scientific_risk,
            "classification": (
                "selection_or_teacher_scientific_risk"
                if scientific_risk
                else "not_significantly_below_naive"
            ),
        },
        "gate": {
            "status": "passed" if not blockers else "blocked",
            "checks": checks,
            "blockers": blockers,
            "scientific_risk": scientific_risk,
        },
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CBWDM Pilot Diagnostics",
        "",
        f"- Status: `{payload['status']}`",
        f"- Data role: `{payload['data_role']}`",
        f"- Aligned examples: {payload['num_aligned_examples']}",
        f"- Scientific risk: `{str(payload['gate']['scientific_risk']).lower()}`",
        "",
        "## P0 checks",
        "",
        "| Check | Passed | Reason |",
        "|---|---:|---|",
    ]
    for name, check in payload["gate"]["checks"].items():
        lines.append(f"| {name} | {check['passed']} | {check['reason'] or ''} |")
    lines.extend(
        [
            "",
            "## Prediction flips",
            "",
            "```json",
            json.dumps(payload["prediction_flips"], indent=2, sort_keys=True),
            "```",
            "",
            "## Gold evidence recall",
            "",
            "```json",
            json.dumps(payload["gold_evidence"], indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def publish_diagnostics(
    *,
    config_path: str | Path,
    paths: dict[str, str | Path],
    output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    required = {
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
    }
    missing = sorted(required - set(paths))
    if missing:
        raise ValueError(f"Missing diagnostic inputs: {missing}")
    payload = build_diagnostics(
        config=load_yaml(config_path),
        **{name: _map(path) for name, path in paths.items()},
    )
    payload["inputs"] = {
        name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    }
    payload["config"] = {
        "path": str(Path(config_path).resolve()),
        "sha256": sha256_file(config_path),
    }
    payload["git"] = git_state(project_root)
    payload["created_at"] = utc_now()
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / "cbwdm_pilot_diagnostics.json", payload)
    atomic_write_text(directory / "cbwdm_pilot_diagnostics.md", _markdown(payload))
    return payload

"""Aggregate all P0 evidence for a FEVER formal experiment readiness decision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from src.formal_config import validate_frozen_manifest
from src.formal_provenance import atomic_write_text
from src.formal_splits import validate_split_manifest
from src.run_manifest import atomic_write_json, git_state, sha256_file, utc_now

READINESS_SCHEMA_VERSION = "rag_cbwdm_fever_formal_readiness.v1"
CANONICAL_METHODS = {
    "no_evidence",
    "naive_topm",
    "bge",
    "infogain_fever",
    "rag_cbwdm",
    "cbwdm_oracle",
}


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _record(
    checks: dict[str, dict[str, Any]],
    name: str,
    action: Callable[[], tuple[bool, str | None, dict[str, Any] | None]],
) -> None:
    try:
        passed, reason, evidence = action()
    except Exception as exc:  # readiness must report all blockers rather than crash on one
        passed, reason, evidence = False, f"{type(exc).__name__}: {exc}", None
    checks[name] = {"passed": bool(passed), "reason": reason, "evidence": evidence}


def check_readiness(
    *,
    split_manifest_path: str | Path,
    calibration_manifest_path: str | Path,
    frozen_manifest_path: str | Path,
    fairness_audit_path: str | Path,
    baseline_summary_path: str | Path,
    diagnostics_path: str | Path,
    tests_status_path: str | Path,
    verify_model_artifacts: bool = True,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    state: dict[str, Any] = {}

    def split_check() -> tuple[bool, str | None, dict[str, Any]]:
        split = validate_split_manifest(split_manifest_path)
        state["split"] = split
        overlap = split.get("overlap_checks", {})
        passed = all(value == 0 for value in overlap.values())
        return passed, None if passed else f"non-zero overlap: {overlap}", {
            "fingerprint": split["fingerprint"],
            "splits": split["splits"],
            "overlap_checks": overlap,
        }

    _record(checks, "split_manifest_completed_and_no_leakage", split_check)

    def calibration_check() -> tuple[bool, str | None, dict[str, Any]]:
        calibration = _load(calibration_manifest_path)
        state["calibration"] = calibration
        split = state.get("split")
        expected_split_sha = (
            sha256_file(split_manifest_path) if split is not None else None
        )
        contract_text = json.dumps(calibration.get("contract", {})).casefold()
        passed = (
            calibration.get("status") == "completed"
            and calibration.get("held_out_test_used") is False
            and calibration.get("split_manifest_sha256") == expected_split_sha
            and calibration.get("validation_sha256")
            == (split or {}).get("splits", {}).get("validation", {}).get("sha256")
            and "held_out_test" not in contract_text
            and all(
                calibration.get("selected", {}).get(method, {}).get("status")
                == "selected"
                for method in ("infogain_fever", "rag_cbwdm")
            )
        )
        return passed, None if passed else "calibration incomplete or held-out leakage detected", {
            "status": calibration.get("status"),
            "held_out_test_used": calibration.get("held_out_test_used"),
            "validation_sha256": calibration.get("validation_sha256"),
        }

    _record(checks, "validation_only_calibration", calibration_check)

    def frozen_check() -> tuple[bool, str | None, dict[str, Any]]:
        frozen = validate_frozen_manifest(
            frozen_manifest_path, verify_artifacts=verify_model_artifacts
        )
        state["frozen"] = frozen
        split_ok = frozen.get("split_manifest_sha256") == sha256_file(split_manifest_path)
        calibration_ok = frozen.get("calibration_manifest_sha256") == sha256_file(
            calibration_manifest_path
        )
        models = frozen.get("models", {})
        revisions_and_hashes = all(
            item.get("revision") and item.get("sha256") for item in models.values()
        )
        hashes = bool(frozen.get("prompt_hash") and frozen.get("verbalizer_hash"))
        passed = split_ok and calibration_ok and revisions_and_hashes and hashes
        return passed, None if passed else "frozen provenance mismatch or missing hash", {
            "fingerprint": frozen.get("fingerprint"),
            "models": sorted(models),
            "prompt_hash": frozen.get("prompt_hash"),
            "verbalizer_hash": frozen.get("verbalizer_hash"),
        }

    _record(checks, "frozen_formal_config_and_artifact_hashes", frozen_check)

    def fairness_check() -> tuple[bool, str | None, dict[str, Any]]:
        audit = _load(fairness_audit_path)
        method_payload = audit.get("methods", {})
        if isinstance(method_payload, list):
            statuses = {
                str(item.get("method")): item.get("status")
                for item in method_payload
                if isinstance(item, dict)
            }
        else:
            statuses = {
                str(name): item.get("status") if isinstance(item, dict) else item
                for name, item in method_payload.items()
            }
        overall = audit.get("overall", audit.get("status"))
        passed = overall == "comparable" and all(
            statuses.get(method) == "comparable" for method in CANONICAL_METHODS
        )
        return passed, None if passed else "fairness audit is not comparable for all methods", {
            "overall": overall,
            "methods": statuses,
        }

    _record(checks, "fairness_audit_comparable", fairness_check)

    def baseline_check() -> tuple[bool, str | None, dict[str, Any]]:
        summary = _load(baseline_summary_path)
        rows = summary.get("methods", [])
        methods = {
            str(row.get("method"))
            for row in rows
            if isinstance(row, dict) and row.get("status", "completed") == "completed"
        }
        oracle_rows = [
            row for row in rows if isinstance(row, dict) and row.get("method") == "cbwdm_oracle"
        ]
        oracle_ok = bool(oracle_rows) and oracle_rows[0].get("deployable") is False and oracle_rows[
            0
        ].get("diagnostic_only") is True
        passed = (
            summary.get("status") == "completed"
            and summary.get("comparable") is True
            and methods == CANONICAL_METHODS
            and oracle_ok
        )
        return passed, None if passed else "baseline summary is incomplete or non-canonical", {
            "methods": sorted(methods),
            "oracle_diagnostic": oracle_ok,
        }

    _record(checks, "canonical_baseline_suite_completed", baseline_check)

    def diagnostic_check() -> tuple[bool, str | None, dict[str, Any]]:
        diagnostics = _load(diagnostics_path)
        gate = diagnostics.get("gate", {})
        passed = (
            diagnostics.get("status") == "passed"
            and gate.get("status") == "passed"
            and gate.get("scientific_risk") is False
        )
        return passed, None if passed else "CBWDM pilot diagnostic gate failed", {
            "status": diagnostics.get("status"),
            "blockers": gate.get("blockers"),
            "scientific_risk": gate.get("scientific_risk"),
        }

    _record(checks, "cbwdm_pilot_diagnostic_gate", diagnostic_check)

    def tests_check() -> tuple[bool, str | None, dict[str, Any]]:
        tests = _load(tests_status_path)
        passed = tests.get("status") == "passed" and bool(tests.get("commands"))
        return passed, None if passed else "required test commands are not recorded as passed", tests

    _record(checks, "required_tests_passed", tests_check)

    def git_check() -> tuple[bool, str | None, dict[str, Any]]:
        payloads = [
            state.get("split", {}),
            state.get("calibration", {}),
            state.get("frozen", {}),
            _load(diagnostics_path),
        ]
        recorded = all(isinstance(payload.get("git"), dict) and payload["git"].get("commit") for payload in payloads)
        return recorded, None if recorded else "one or more formal artifacts lack Git commit provenance", {
            "commits": [payload.get("git", {}).get("commit") for payload in payloads]
        }

    _record(checks, "git_state_recorded", git_check)
    blockers = [
        {"check": name, "reason": item["reason"]}
        for name, item in checks.items()
        if not item["passed"]
    ]
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "status": "ready" if not blockers else "blocked",
        "checks": checks,
        "blockers": blockers,
        "p0_passed": not blockers,
        "engineering_ready": not blockers,
        "experiment_protocol_ready": not blockers,
        "scientific_conclusion_ready": False,
        "scientific_conclusion_reason": (
            "Scientific conclusions require completed held-out formal runs and analysis; "
            "readiness alone is not a result."
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# FEVER Formal Readiness",
        "",
        f"- Status: `{payload['status']}`",
        f"- Engineering ready: `{str(payload['engineering_ready']).lower()}`",
        f"- Experiment protocol ready: `{str(payload['experiment_protocol_ready']).lower()}`",
        f"- Scientific conclusion ready: `{str(payload['scientific_conclusion_ready']).lower()}`",
        "",
        "| P0 check | Passed | Reason |",
        "|---|---:|---|",
    ]
    for name, check in payload["checks"].items():
        lines.append(f"| {name} | {check['passed']} | {check['reason'] or ''} |")
    if payload["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(
            f"- `{item['check']}`: {item['reason']}" for item in payload["blockers"]
        )
    return "\n".join(lines) + "\n"


def publish_readiness(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    **inputs: Any,
) -> dict[str, Any]:
    payload = check_readiness(**inputs)
    payload["inputs"] = {
        key: {
            "path": str(Path(value).resolve()),
            "sha256": sha256_file(value),
        }
        for key, value in inputs.items()
        if key.endswith("_path")
    }
    payload["git"] = git_state(project_root)
    payload["created_at"] = utc_now()
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / "formal_readiness.json", payload)
    atomic_write_text(directory / "formal_readiness.md", _markdown(payload))
    return payload

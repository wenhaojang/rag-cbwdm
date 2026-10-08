from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.diagnostics.run_fm2_budget_frontier import (
    EXPECTED_GENERATORS,
    EXPECTED_METHODS,
    FrontierPlanError,
    build_execution_plan,
    load_frontier_config,
    validate_generator_manifest,
    validate_metrics_reference,
    validate_selection_artifact,
    write_commands,
)
from scripts.diagnostics.summarize_fm2_budget_frontier import build_summary, write_csv
from src.io_utils import load_yaml
from src.run_manifest import sha256_file, stable_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN_CONFIG = (
    PROJECT_ROOT / "configs/formal/fm2_budget_frontier_development.seed13.yaml"
)
EXPECTED_ROWS = 1169


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_metrics(path: Path, *, correct: int = 700, avg_docs: float = 1.5) -> None:
    _write_json(
        path,
        {
            "split": "validation",
            "num_examples": EXPECTED_ROWS,
            "num_correct": correct,
            "accuracy": correct / EXPECTED_ROWS,
            "macro_f1": 0.6,
            "avg_num_docs": avg_docs,
            "avg_evidence_chars": 240.0,
            "avg_original_retrieval_rank": 1.7,
        },
    )


def _selection_rows(native_max_docs: int) -> list[dict]:
    rows = []
    for index in range(EXPECTED_ROWS):
        num_docs = 1 + (index % native_max_docs)
        docs = [
            {"doc_id": f"doc-{index}-{doc_index}", "text": f"evidence {doc_index}"}
            for doc_index in range(num_docs)
        ]
        rows.append(
            {
                "id": f"validation-{index}",
                "split": "validation",
                "selected_doc_ids": [doc["doc_id"] for doc in docs],
                "selected_docs": docs,
                "num_docs": num_docs,
            }
        )
    return rows


def _write_selection(path: Path, manifest_path: Path, method: str, native_max_docs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in _selection_rows(native_max_docs):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    binding = {
        "dataset_id": "fm2_official_closed_page_v1",
        "method": method,
    }
    contract = {"method": method, "artifact_binding": binding}
    _write_json(
        manifest_path,
        {
            "schema_version": "rag_cbwdm_selection_manifest.v2",
            "status": "completed",
            "completed": True,
            "method": method,
            "num_rows": EXPECTED_ROWS,
            "output_path": str(path.resolve()),
            "output_sha256": sha256_file(path),
            "contract": contract,
            "fingerprint": stable_hash(contract),
            "artifact_binding": binding,
        },
    )


def _refresh_selection_manifest(selection: Path, manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["output_sha256"] = sha256_file(selection)
    _write_json(manifest, payload)


def _synthetic_environment(tmp_path: Path) -> dict:
    config = deepcopy(load_yaml(CHECKED_IN_CONFIG))
    config["artifact_root"] = str(tmp_path / "frontier")
    config["server_python"] = "server-python"
    config["generator_manifest_template"] = str(
        tmp_path / "generators" / "{generator_id}" / "manifest.json"
    )
    config["no_evidence_metrics_template"] = str(
        tmp_path / "native" / "{generator_id}" / "no_evidence" / "metrics.json"
    )
    native_maxima = {
        "qwen2.5-0.5b-instruct": 3,
        "qwen2.5-1.5b-instruct": 4,
        "qwen2.5-7b-instruct": 2,
        "mistral-7b-instruct-v0.3": 2,
    }
    for generator in config["generators"]:
        generator_id = generator["generator_id"]
        generator_identity = {"generator_id": generator_id}
        _write_json(
            Path(config["generator_manifest_template"].format(generator_id=generator_id)),
            {
                "status": "completed",
                "dataset_identity": {"dataset_id": "fm2_official_closed_page_v1"},
                "generator_identity": generator_identity,
                "generator_identity_fingerprint": stable_hash(generator_identity),
            },
        )
        _write_metrics(
            Path(config["no_evidence_metrics_template"].format(generator_id=generator_id)),
            avg_docs=0.0,
        )

    for method in config["methods"]:
        frontier_id = method["frontier_method_id"]
        method["selection_template"] = str(
            tmp_path / "selections" / frontier_id / "{generator_id}" / "selection.jsonl"
        )
        method["selection_manifest_template"] = str(
            tmp_path
            / "selections"
            / frontier_id
            / "{generator_id}"
            / "selection.manifest.json"
        )
        method["native_metrics_template"] = str(
            tmp_path / "native" / "{generator_id}" / frontier_id / "metrics.json"
        )
        for generator in config["generators"]:
            generator_id = generator["generator_id"]
            native_max = native_maxima[generator_id] if frontier_id == "linear_cbwdm" else 4
            selection = Path(method["selection_template"].format(generator_id=generator_id))
            manifest = Path(
                method["selection_manifest_template"].format(generator_id=generator_id)
            )
            _write_selection(selection, manifest, method["underlying_method"], native_max)
            _write_metrics(
                Path(method["native_metrics_template"].format(generator_id=generator_id)),
                correct=700 + EXPECTED_GENERATORS.index(generator_id),
                avg_docs=float(native_max),
            )
    return config


def test_checked_in_frontier_config_is_exact_and_development_only() -> None:
    config = load_frontier_config(CHECKED_IN_CONFIG)
    assert tuple(item["frontier_method_id"] for item in config["methods"]) == EXPECTED_METHODS
    assert tuple(item["generator_id"] for item in config["generators"]) == EXPECTED_GENERATORS
    assert config["requested_caps"] == [1, 2, 3]
    assert config["split"] == "validation"
    assert config["held_out"] is False
    assert config["expected_rows"] == EXPECTED_ROWS


def test_plan_has_100_points_dynamic_reuse_labels_and_safe_commands(tmp_path: Path) -> None:
    config = _synthetic_environment(tmp_path)
    first = build_execution_plan(config, project_root=PROJECT_ROOT)
    second = build_execution_plan(config, project_root=PROJECT_ROOT)
    assert first == second
    assert first["plan_fingerprint"] == second["plan_fingerprint"]
    assert first["total_logical_points"] == 100
    assert first["evaluation_job_count"] == 67
    assert first["native_reuse_count"] == 5
    assert first["no_evidence_anchor_count"] == 4

    points = first["points"]
    no_evidence = [point for point in points if point["frontier_role"] == "no_evidence_anchor"]
    native = [point for point in points if point["frontier_role"] == "native_method_point"]
    assert len(no_evidence) == 4 and all(point["nominal_cap"] == 0 for point in no_evidence)
    assert len(native) == 24 and all(point["source"] == "native_reuse" for point in native)

    def point(method: str, generator: str, cap: int) -> dict:
        return next(
            item
            for item in points
            if item["frontier_method_id"] == method
            and item["generator_id"] == generator
            and item["nominal_cap"] == cap
        )

    assert point("linear_cbwdm", EXPECTED_GENERATORS[0], 3)["source"] == "native_reuse"
    assert point("linear_cbwdm", EXPECTED_GENERATORS[1], 3)["source"] == "evaluation"
    assert point("linear_cbwdm", EXPECTED_GENERATORS[2], 2)["native_max_docs"] == 2
    assert point("infogain", EXPECTED_GENERATORS[0], 1)["frontier_status"] == "posthoc_diagnostic"
    assert point("infogain", EXPECTED_GENERATORS[0], 1)["contract_preserving"] is False
    for cap in (2, 3):
        assert point("infogain", EXPECTED_GENERATORS[0], cap)["frontier_status"] == "contract_preserving_prefix"
        assert point("infogain", EXPECTED_GENERATORS[0], cap)["contract_preserving"] is True
    for method in ("retrieval", "bge"):
        assert point(method, EXPECTED_GENERATORS[0], 1)["frontier_status"] == "posthoc_rank_prefix"
        assert point(method, EXPECTED_GENERATORS[0], 1)["contract_preserving"] is False
    for method in ("linear_cbwdm", "raw_kcbwdm_v2a", "normalized_kcbwdm_rho1"):
        assert point(method, EXPECTED_GENERATORS[0], 1)["frontier_status"] == "capped_native_prefix"

    commands_path = tmp_path / "commands.sh"
    write_commands(commands_path, first)
    commands = commands_path.read_text(encoding="utf-8").splitlines()
    assert len(commands) == 67
    assert all("--experiment-type budget_frontier" in command for command in commands)
    assert all("--formal-v2-identity" in command for command in commands)
    assert all(sum(f"--max-docs {cap}" in command for cap in (1, 2, 3)) == 1 for command in commands)
    sanitized = "\n".join(commands).casefold().replace("infogain_fever", "")
    for forbidden in (
        "posterior",
        "teacher",
        "train_signed",
        "select_signed",
        "held_out",
        "fever",
        "pyserini",
        "lucene",
        "--limit",
    ):
        assert forbidden not in sanitized


@pytest.mark.parametrize(
    "failure",
    ("missing_file", "missing_manifest", "wrong_rows", "duplicate_ids", "selected_ids", "num_docs"),
)
def test_selection_validation_fails_closed(tmp_path: Path, failure: str) -> None:
    selection = tmp_path / "selection.jsonl"
    manifest = tmp_path / "selection.manifest.json"
    _write_selection(selection, manifest, "retrieval_topk", 4)
    if failure == "missing_file":
        selection.unlink()
    elif failure == "missing_manifest":
        manifest.unlink()
    else:
        rows = selection.read_text(encoding="utf-8").splitlines()
        if failure == "wrong_rows":
            rows.pop()
        else:
            row = json.loads(rows[1])
            if failure == "duplicate_ids":
                row["id"] = json.loads(rows[0])["id"]
            elif failure == "selected_ids":
                row["selected_doc_ids"] = ["wrong"]
            elif failure == "num_docs":
                row["num_docs"] += 1
            rows[1] = json.dumps(row, sort_keys=True)
        selection.write_text("\n".join(rows) + "\n", encoding="utf-8")
        _refresh_selection_manifest(selection, manifest)
    with pytest.raises(FrontierPlanError):
        validate_selection_artifact(
            selection,
            manifest,
            expected_method="retrieval_topk",
            expected_rows=EXPECTED_ROWS,
            expected_split="validation",
        )


def test_metrics_and_generator_references_fail_closed(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    _write_metrics(metrics)
    validate_metrics_reference(metrics, expected_rows=EXPECTED_ROWS, split="validation")
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    payload["num_examples"] = EXPECTED_ROWS - 1
    _write_json(metrics, payload)
    with pytest.raises(FrontierPlanError, match="num_examples"):
        validate_metrics_reference(metrics, expected_rows=EXPECTED_ROWS, split="validation")

    manifest = tmp_path / "generator.json"
    generator_identity = {"generator_id": "wrong"}
    _write_json(
        manifest,
        {
            "status": "completed",
            "dataset_identity": {"dataset_id": "fm2_official_closed_page_v1"},
            "generator_identity": generator_identity,
            "generator_identity_fingerprint": stable_hash(generator_identity),
        },
    )
    with pytest.raises(FrontierPlanError, match="Generator ID"):
        validate_generator_manifest(manifest, EXPECTED_GENERATORS[0])


def test_plan_rejects_cross_artifact_example_id_drift(tmp_path: Path) -> None:
    config = _synthetic_environment(tmp_path)
    method = config["methods"][1]
    generator_id = EXPECTED_GENERATORS[0]
    selection = Path(method["selection_template"].format(generator_id=generator_id))
    manifest = Path(
        method["selection_manifest_template"].format(generator_id=generator_id)
    )
    rows = selection.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["id"] = "validation-drifted"
    rows[0] = json.dumps(row, sort_keys=True)
    selection.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _refresh_selection_manifest(selection, manifest)
    with pytest.raises(FrontierPlanError, match="IDs/order mismatch"):
        build_execution_plan(config, project_root=PROJECT_ROOT)


def test_summary_reads_evaluated_reused_and_anchor_metrics_and_aggregates(tmp_path: Path) -> None:
    config = _synthetic_environment(tmp_path)
    plan = build_execution_plan(config, project_root=PROJECT_ROOT)
    for index, point in enumerate(plan["points"]):
        if point["source"] == "evaluation":
            _write_metrics(
                Path(point["output_metrics_path"]),
                correct=600 + (index % 4),
                avg_docs=float(point["nominal_cap"]),
            )
    summary = build_summary(plan)
    assert len(summary["rows"]) == 100
    assert len(summary["aggregates"]) == 25
    assert {row["frontier_role"] for row in summary["rows"]} == {
        "no_evidence_anchor",
        "native_method_point",
        "requested_cap",
    }
    aggregate = next(
        row
        for row in summary["aggregates"]
        if row["frontier_method_id"] == "retrieval" and row["nominal_cap"] == 1
    )
    members = [
        row
        for row in summary["rows"]
        if row["frontier_method_id"] == "retrieval" and row["nominal_cap"] == 1
    ]
    assert aggregate["generator_count"] == 4
    assert aggregate["pooled_correct"] == sum(row["num_correct"] for row in members)
    assert aggregate["pooled_n"] == 4 * EXPECTED_ROWS
    assert aggregate["pooled_accuracy"] == pytest.approx(
        aggregate["pooled_correct"] / aggregate["pooled_n"]
    )
    assert aggregate["mean_macro_f1"] == pytest.approx(0.6)
    csv_path = tmp_path / "summary.csv"
    write_csv(csv_path, summary["rows"])
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 101

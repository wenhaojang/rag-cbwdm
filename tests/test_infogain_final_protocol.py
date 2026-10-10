from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.baselines.infogain_protocol import FINAL_PROTOCOL_ID, rank_and_select
from src.metrics import ClassificationMetrics
from src.run_manifest import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def load_selector_script():
    path = ROOT / "scripts" / "12c_select_infogain_reranker.py"
    spec = importlib.util.spec_from_file_location("select_infogain_final", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate(doc_id: str, rank: int) -> dict:
    return {
        "doc_id": doc_id,
        "rank": rank,
        "retrieval_score": float(100 - rank),
        "title": f"title {doc_id}",
        "text": f"text {doc_id}",
    }


def source(count: int = 5, *, label: str = "SUPPORTS") -> dict:
    return {
        "id": "example",
        "query": "claim",
        "label": label,
        "split": "validation",
        "candidates": [candidate(f"d{index + 1}", index + 1) for index in range(count)],
    }


def select(
    scores: list[float],
    probabilities: list[float],
    *,
    count: int | None = None,
    threshold: float | None = 0.2,
    min_docs: int = 2,
) -> dict:
    module = load_selector_script()

    class FakeModel:
        def score(self, texts, batch_size):
            assert len(texts) == len(scores)
            return scores, probabilities

    return module.select_row(
        source(len(scores) if count is None else count),
        FakeModel(),
        batch_size=8,
        top_m=4,
        filter_threshold=threshold,
        min_docs=min_docs,
        checkpoint_metadata={"fingerprint": "checkpoint"},
    )


def test_counterexample_filters_only_inside_fixed_top4() -> None:
    selected = select(
        [4.0, 3.0, 2.0, 1.0, 100.0],
        [0.2393, 0.1730, 0.2233, 0.01, 1.0],
    )
    # d5 ranks first in this fixture, so use the pure protocol for the literal
    # four-document counterexample and separately assert no below-window fill.
    _, literal = rank_and_select(
        source(4)["candidates"], [4, 3, 2, 1], [0.2393, 0.1730, 0.2233, 0.01],
        top_m=4, filter_threshold=0.2, min_docs=2,
    )
    assert [item["doc_id"] for item in literal] == ["d1", "d3"]
    assert selected["selected_doc_ids"] == ["d5", "d1", "d3"]
    assert "d2" not in selected["selected_doc_ids"]


def test_candidate_below_top4_never_backfills() -> None:
    selected = select(
        [5, 4, 3, 2, 1],
        [0.9, 0.8, 0.1, 0.1, 1.0],
    )
    assert selected["selected_doc_ids"] == ["d1", "d2"]
    assert "d5" not in selected["selected_doc_ids"]


@pytest.mark.parametrize(
    ("probabilities", "expected_ids", "fallback"),
    [
        ([0.9, 0.8, 0.7, 0.6], ["d1", "d2", "d3", "d4"], [False] * 4),
        ([0.1, 0.1, 0.1, 0.1], ["d1", "d2"], [True, True]),
        ([0.1, 0.1, 0.9, 0.1], ["d1", "d3"], [True, False]),
        ([0.1, 0.1, 0.9, 0.8], ["d3", "d4"], [False, False]),
        ([0.2, 0.199999, 0.1, 0.1], ["d1", "d2"], [False, True]),
    ],
)
def test_threshold_and_fallback_cases(probabilities, expected_ids, fallback) -> None:
    selected = select([4, 3, 2, 1], probabilities)
    assert selected["selected_doc_ids"] == expected_ids
    assert [doc["min_docs_fallback"] for doc in selected["selected_docs"]] == fallback


@pytest.mark.parametrize(
    ("count", "probabilities", "expected_ids"),
    [
        (1, [0.1], ["d1"]),
        (2, [0.1, 0.1], ["d1", "d2"]),
        (3, [0.1, 0.9, 0.8], ["d2", "d3"]),
    ],
)
def test_short_candidate_pools(count, probabilities, expected_ids) -> None:
    selected = select(list(range(count, 0, -1)), probabilities, count=count)
    assert selected["selected_doc_ids"] == expected_ids
    if count < 2:
        assert selected["stop_reason"] == "candidates_exhausted"


def test_equal_scores_use_source_list_order_not_rank_field() -> None:
    candidates = source(4)["candidates"]
    candidates[0]["rank"] = 99
    candidates[1]["rank"] = 1
    ranked, selected = rank_and_select(
        candidates, [1, 1, 1, 1], [0.9, 0.8, 0.7, 0.6],
        top_m=4, filter_threshold=0.2, min_docs=2,
    )
    assert [item["doc_id"] for item in ranked] == ["d1", "d2", "d3", "d4"]
    assert [item["doc_id"] for item in selected] == ["d1", "d2", "d3", "d4"]


@pytest.mark.parametrize(
    ("scores", "probabilities", "message"),
    [
        ([1, 2], [0.5], "filter_probabilities length"),
        ([1, 2], [0.5, None], "finite number"),
        ([1, 2], [0.5, float("nan")], "finite number"),
        ([1, 2], [0.5, 1.1], r"\[0, 1\]"),
    ],
)
def test_missing_or_illegal_filter_probabilities_fail(scores, probabilities, message) -> None:
    with pytest.raises(ValueError, match=message):
        rank_and_select(
            source(2)["candidates"], scores, probabilities,
            top_m=4, filter_threshold=0.2, min_docs=2,
        )


def test_duplicate_candidate_ids_fail() -> None:
    candidates = source(2)["candidates"]
    candidates[1]["doc_id"] = candidates[0]["doc_id"]
    with pytest.raises(ValueError, match="duplicate candidate doc_id"):
        rank_and_select(
            candidates, [2, 1], [0.9, 0.8],
            top_m=4, filter_threshold=0.2, min_docs=2,
        )


def test_decision_does_not_depend_on_label_or_gold() -> None:
    module = load_selector_script()

    class FakeModel:
        def score(self, texts, batch_size):
            assert all("SUPPORTS" not in text and "REFUTES" not in text for text in texts)
            return [3, 2, 1], [0.1, 0.9, 0.8]

    rows = []
    for label in ("SUPPORTS", "REFUTES"):
        row = source(3, label=label)
        row["gold"] = "poison-" + label
        rows.append(
            module.select_row(
                row, FakeModel(), batch_size=4, top_m=4,
                filter_threshold=0.2, min_docs=2,
                checkpoint_metadata={"fingerprint": "checkpoint"},
            )
        )
    assert rows[0]["selected_doc_ids"] == rows[1]["selected_doc_ids"] == ["d2", "d3"]
    assert all(row["uses_gold_at_test"] is False for row in rows)


def test_selection_schema_fields_and_trace_are_consistent() -> None:
    selected = select([4, 3, 2, 1], [0.1, 0.9, 0.8, 0.1])
    assert selected["num_docs"] == 2
    assert selected["selected_doc_ids"] == ["d2", "d3"]
    assert [step["selected_doc_id"] for step in selected["selection_steps"]] == ["d2", "d3"]
    assert selected["stop_reason"] == "filter_threshold_applied"
    metadata = selected["selection_metadata"]
    assert metadata["selection_protocol_id"] == FINAL_PROTOCOL_ID
    assert metadata["top_m"] == 4
    assert metadata["ranking_score_semantics"] != metadata["filter_probability_semantics"]
    assert [item["doc_id"] for item in metadata["candidate_score_audit"]] == [
        "d1", "d2", "d3", "d4"
    ]


def test_legacy_none_mode_still_selects_rank_top_m_without_fallback() -> None:
    selected = select([1, 4, 3, 2, 100], [0, 0, 0, 0, 0], threshold=None)
    assert selected["selected_doc_ids"] == ["d5", "d2", "d3", "d4"]
    assert all(not doc["min_docs_fallback"] for doc in selected["selected_docs"])
    assert selected["selection_metadata"]["selection_protocol_id"] is None
    assert selected["stop_reason"] == "top_m_reached"

    short = select([2, 1], [0, 0], threshold=None)
    assert short["selected_doc_ids"] == ["d1", "d2"]
    assert short["stop_reason"] == "threshold_or_candidates_exhausted"


def test_metrics_emit_complete_histogram_and_measurement_bases() -> None:
    metrics = ClassificationMetrics(labels=["SUPPORTS", "REFUTES"])
    metrics.update(
        "SUPPORTS", "SUPPORTS", num_docs=2, evidence_chars=12, evidence_tokens=3
    )
    result = metrics.compute()
    assert result["num_docs"]["histogram"] == {
        "0": 0, "1": 0, "2": 1, "3": 0, "4": 0
    }
    assert result["missing_predictions"] == 0
    assert result["nan_inf"] == 0
    assert result["avg_evidence_tokens"] == 3
    assert "prompt" in result["evidence_chars_basis"]


def test_read_only_auditor_accepts_selector_output_and_rejects_tampering() -> None:
    audit_path = ROOT / "scripts" / "infogain_final_protocol.py"
    spec = importlib.util.spec_from_file_location("infogain_final_audit", audit_path)
    assert spec and spec.loader
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    source_row = source(5)
    module = load_selector_script()

    class FakeModel:
        def score(self, texts, batch_size):
            return [5, 4, 3, 2, 1], [0.1, 0.1, 0.9, 0.8, 1.0]

    selection = module.select_row(
        source_row, FakeModel(), batch_size=8, top_m=4,
        filter_threshold=0.2, min_docs=2,
        checkpoint_metadata={"fingerprint": "checkpoint"},
    )
    report = audit.audit_rows([source_row], [selection], expected_rows=1)
    assert report["status"] == "PASS"
    assert report["num_docs_histogram"]["2"] == 1

    selection["selected_doc_ids"] = ["d5"]
    with pytest.raises(ValueError, match="violate final protocol"):
        audit.audit_rows([source_row], [selection], expected_rows=1)


def test_final_plan_has_exact_eight_development_groups() -> None:
    module_path = ROOT / "scripts" / "infogain_final_protocol.py"
    spec = importlib.util.spec_from_file_location("infogain_final_protocol", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan, groups = module.load_plan(
        ROOT / "configs" / "formal" / "infogain_top4_filter02_final.development.server.yaml"
    )
    assert plan["split"] == "validation"
    assert len(groups) == 8
    assert {group["expected_rows"] for group in groups} == {5000, 1169}
    fever = [group for group in groups if group["dataset_id"] == "fever_binary_v2"]
    fm2 = [
        group
        for group in groups
        if group["dataset_id"] == "fm2_official_closed_page_v1"
    ]
    assert len(fever) == len(fm2) == 4
    assert {group["retrieval"] for group in fever} == {
        "/root/experiments/rag_cbwdm/formal_v2_external_inputs/"
        "fever_binary_v2/fever_bm25_v1/full_seed13/validation.jsonl"
    }
    assert {group["retrieval_manifest"] for group in fever} == {
        "/root/experiments/rag_cbwdm/formal_v2_external_inputs/"
        "fever_binary_v2/fever_bm25_v1/full_seed13/validation.manifest.json"
    }
    assert {group["expected_retrieval_sha256"] for group in fever} == {
        "a71fffda269f9112c6c243cee3384df072455a604eb801c838b69e32993786eb"
    }
    assert {group["expected_retrieval_sha256_prefix"] for group in fm2} == {
        "f371600059769a24"
    }


def test_fever_retrieval_sidecar_is_checked(tmp_path: Path) -> None:
    module_path = ROOT / "scripts" / "infogain_final_protocol.py"
    spec = importlib.util.spec_from_file_location("infogain_retrieval_audit", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    retrieval = tmp_path / "validation.jsonl"
    retrieval.write_text(json.dumps(source(1)) + "\n", encoding="utf-8")
    manifest = retrieval.with_suffix(".manifest.json")
    payload = {
        "schema_version": "rag_cbwdm_retrieval.v1",
        "completed": True,
        "split": "validation",
        "output_sha256": sha256_file(retrieval),
        "num_output_rows": 1,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    group = {
        "group_id": "fixture",
        "dataset_id": "fever_binary_v2",
        "retrieval_protocol_id": "fever_bm25_v1",
        "expected_rows": 1,
        "retrieval": str(retrieval),
        "retrieval_manifest": str(manifest),
        "expected_retrieval_sha256": sha256_file(retrieval),
    }
    assert module.validate_retrieval_source(group)["num_rows"] == 1
    payload["output_sha256"] = "tampered"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="output SHA"):
        module.validate_retrieval_source(group)


def test_fever_retrieval_missing_manifest_is_explicit(tmp_path: Path) -> None:
    module_path = ROOT / "scripts" / "infogain_final_protocol.py"
    spec = importlib.util.spec_from_file_location("infogain_missing_manifest", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    retrieval = tmp_path / "validation.jsonl"
    retrieval.write_text(json.dumps(source(1)) + "\n", encoding="utf-8")
    group = {
        "group_id": "missing-sidecar",
        "dataset_id": "fever_binary_v2",
        "retrieval_protocol_id": "fever_bm25_v1",
        "expected_rows": 1,
        "retrieval": str(retrieval),
        "retrieval_manifest": str(tmp_path / "missing.manifest.json"),
    }
    with pytest.raises(FileNotFoundError, match="retrieval manifest does not exist"):
        module.validate_retrieval_source(group)


def test_fever_retrieval_frozen_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    module_path = ROOT / "scripts" / "infogain_final_protocol.py"
    spec = importlib.util.spec_from_file_location("infogain_frozen_sha", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    retrieval = tmp_path / "validation.jsonl"
    retrieval.write_text(json.dumps(source(1)) + "\n", encoding="utf-8")
    manifest = retrieval.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "rag_cbwdm_retrieval.v1",
                "completed": True,
                "split": "validation",
                "output_sha256": sha256_file(retrieval),
                "num_output_rows": 1,
            }
        ),
        encoding="utf-8",
    )
    group = {
        "group_id": "wrong-frozen-sha",
        "dataset_id": "fever_binary_v2",
        "retrieval_protocol_id": "fever_bm25_v1",
        "expected_rows": 1,
        "retrieval": str(retrieval),
        "retrieval_manifest": str(manifest),
        "expected_retrieval_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="differs from frozen plan"):
        module.validate_retrieval_source(group)


def test_source_preflight_stops_at_first_failed_group(tmp_path: Path) -> None:
    module_path = ROOT / "scripts" / "infogain_final_protocol.py"
    spec = importlib.util.spec_from_file_location("infogain_source_preflight", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = [{"group_id": f"group-{index}"} for index in range(8)]
    visited: list[str] = []

    def fake_validator(group, *, artifact_root):
        assert artifact_root == tmp_path
        visited.append(group["group_id"])
        if group["group_id"] == "group-2":
            raise ValueError("injected source failure")
        return {"status": "PASS"}

    with pytest.raises(ValueError, match="injected source failure"):
        module.source_preflight_groups(
            groups, artifact_root=tmp_path, validator=fake_validator
        )
    assert visited == ["group-0", "group-1", "group-2"]


def test_launcher_failure_preserves_traceback_log_and_exitcode(tmp_path: Path) -> None:
    module_path = ROOT / "scripts" / "infogain_final_protocol.py"
    spec = importlib.util.spec_from_file_location("infogain_launcher_logging", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return_code = module.run_logged_stage_command(
        [sys.executable, "-c", "raise RuntimeError('launcher-boom')"],
        tmp_path,
        "02_selection",
    )
    assert return_code != 0
    assert (tmp_path / "02_selection.exitcode").read_text(encoding="utf-8").strip() != "0"
    for log_name in ("02_selection.log", "launcher.log"):
        log = (tmp_path / log_name).read_text(encoding="utf-8")
        assert "Traceback" in log
        assert "launcher-boom" in log
        assert "EXIT stage=02_selection code=" in log
    status = json.loads((tmp_path / "02_selection.status.json").read_text(encoding="utf-8"))
    assert status["status"] == "FAIL"
    assert status["exit_code"] == return_code


def test_pipeline_failure_stops_and_marks_later_stages_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = ROOT / "scripts" / "infogain_final_protocol.py"
    spec = importlib.util.spec_from_file_location("infogain_pipeline_failure", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    visited: list[str] = []

    def fail_first(command, attempt_dir, stage_name):
        visited.append(stage_name)
        return 17

    monkeypatch.setattr(module, "run_logged_stage_command", fail_first)
    return_code = module.run_pipeline("plan.yaml", "cpu", tmp_path)
    assert return_code == 17
    assert visited == ["01_source_preflight"]
    assert (tmp_path / "overall.exitcode").read_text(encoding="utf-8") == "17\n"
    assert (tmp_path / "02_selection.skipped").is_file()
    assert (tmp_path / "03_protocol_audit.skipped").is_file()
    status = json.loads((tmp_path / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "FAIL"
    assert status["failed_stage"] == "01_source_preflight"


def test_successful_pipeline_records_zero_exitcode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = ROOT / "scripts" / "infogain_final_protocol.py"
    spec = importlib.util.spec_from_file_location("infogain_pipeline_success", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    visited: list[str] = []

    def pass_stage(command, attempt_dir, stage_name):
        visited.append(stage_name)
        return 0

    monkeypatch.setattr(module, "run_logged_stage_command", pass_stage)
    return_code = module.run_pipeline("plan.yaml", "cpu", tmp_path)
    assert return_code == 0
    assert visited == ["01_source_preflight", "02_selection", "03_protocol_audit"]
    assert (tmp_path / "overall.exitcode").read_text(encoding="utf-8") == "0\n"
    status = json.loads((tmp_path / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "PASS"
    assert status["exit_code"] == 0


def test_tmux_launchers_gate_creation_and_use_unique_attempt_logs() -> None:
    expectations = {
        "run_infogain_final_cpu_tmux.sh": ("source-preflight", "cpu-pipeline"),
        "run_infogain_final_gpu_tmux.sh": ("preflight", "gpu-pipeline"),
    }
    for name, (preflight, pipeline) in expectations.items():
        script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert script.index(f"--stage {preflight}") < script.index("tmux new-session")
        assert f"--stage {pipeline}" in script
        assert 'ATTEMPT_DIR="$RUN_ROOT/attempts/$ATTEMPT_ID"' in script
        assert "launcher.console.log" in script
        assert "overall.exitcode" in script
        assert "cpu_selection_and_audit.exitcode" not in script
        assert "gpu_evaluation_and_summary.exitcode" not in script

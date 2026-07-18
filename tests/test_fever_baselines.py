from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.baselines.bge_reranker import (
    choose_scored_candidates,
    make_bge_selection,
    score_rows,
)
from src.baselines.common import (
    build_selection_contract,
    publish_selection,
    selection_manifest_path,
    validate_selection_artifact,
)
from src.baselines.infogain_fever import (
    group_teacher_rows,
    infogain_multitask_loss,
    pointwise_input,
    posterior_to_teacher_rows,
    resolve_thresholds,
)
from src.metrics import ClassificationMetrics
from src.run_manifest import atomic_write_json, sha256_file, stable_hash
from src.selection_schema import make_selection_row, normalize_selected_doc

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate(doc_id: str, rank: int, score: float) -> dict:
    return {
        "doc_id": doc_id,
        "rank": rank,
        "retrieval_score": score,
        "title": doc_id,
        "text": f"text {doc_id}",
    }


def source_row() -> dict:
    return {
        "id": "q1",
        "query": "claim",
        "label": "SUPPORTS",
        "split": "dev",
        "candidates": [
            candidate("d1", 1, 10.0),
            candidate("d2", 2, 9.0),
            candidate("d3", 3, 8.0),
        ],
    }


def write_evaluation_artifact(
    directory: Path,
    *,
    stem: str,
    method: str,
    metrics: dict | None = None,
    stage: str = "evaluation",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / f"{stem}.json"
    payload = {
        "method": method,
        "accuracy": 0.75,
        "macro_f1": 0.7,
        "avg_num_docs": 2.0,
        "avg_evidence_chars": 20.0,
        "num_examples": 10,
        "model_name": "test-generator",
        **(metrics or {}),
    }
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    metrics_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "rag_cbwdm_evaluation_manifest.v1",
                "stage": stage,
                "status": "completed",
                "completed": True,
                "method": method,
                "metrics_path": str(metrics_path.resolve()),
                "fingerprint": f"fingerprint-{method}",
            }
        ),
        encoding="utf-8",
    )
    return metrics_path


def test_selection_v2_and_atomic_manifest_resume(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.jsonl"
    retrieval.write_text(json.dumps(source_row()) + "\n", encoding="utf-8")
    output = tmp_path / "naive_top2.jsonl"
    contract = build_selection_contract(
        method="naive_topm",
        input_paths={"retrieval": retrieval},
        parameters={"top_m": 2, "min_docs": 2},
    )
    docs = [
        normalize_selected_doc(item, selector_score=None, selection_step=index)
        for index, item in enumerate(source_row()["candidates"][:2])
    ]
    row = make_selection_row(
        source_row(),
        method="naive_topm",
        selected_docs=docs,
        selection_steps=[],
        stop_reason="top_m_reached",
        max_docs=2,
        selection_metadata={"state_aware": False},
    )
    count, reused = publish_selection(
        output, [row], contract=contract, project_root=ROOT
    )
    assert (count, reused) == (1, False)
    assert not output.with_name(output.name + ".partial").exists()
    assert not validate_selection_artifact(output, contract)
    manifest = json.loads(selection_manifest_path(output).read_text(encoding="utf-8"))
    assert manifest["output_sha256"] == sha256_file(output)
    assert manifest["status"] == "completed"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "rag_cbwdm_selection.v2"
    assert payload["selected_doc_ids"] == ["d1", "d2"]
    assert payload["selected_docs"][0]["source_rank"] == 1
    assert payload["selected_docs"][0]["source_score"] == 10.0

    def explode():
        raise AssertionError("resume must not iterate or rebuild")
        yield {}

    assert publish_selection(
        output,
        explode(),
        contract=contract,
        project_root=ROOT,
        resume=True,
    ) == (1, True)


def test_bge_mock_batch_order_threshold_and_fallback() -> None:
    row = source_row()
    observed = []

    def scorer(pairs):
        observed.extend(pairs)
        return [0.1, 0.9, 0.2]

    scored = score_rows([row], scorer)
    assert [pair[1].split(":")[0] for pair in observed] == ["d1", "d2", "d3"]
    scores = {item["doc_id"]: item["score"] for item in scored[0]["scores"]}
    selected = make_bge_selection(
        row,
        scores,
        method="bge",
        top_m=2,
        score_threshold=0.5,
        min_docs=2,
        model_metadata={"model": "mock"},
    )
    assert selected["selected_doc_ids"] == ["d2", "d3"]
    assert selected["selected_docs"][1]["min_docs_fallback"] is True
    assert selected["selection_metadata"]["score_direction"] == "higher_is_more_relevant"
    assert choose_scored_candidates(
        row["candidates"], [0.1, 0.9, 0.2], top_m=1, score_threshold=None, min_docs=1
    )[0][0]["doc_id"] == "d2"


def test_naive_rank_and_bge_score_cache_reuse(tmp_path: Path) -> None:
    naive = load_script("08_select_naive_topm.py")
    retrieval = tmp_path / "retrieval.jsonl"
    retrieval.write_text(json.dumps(source_row()) + "\n", encoding="utf-8")
    rows = list(
        naive.iter_selection_rows(
            retrieval, top_m=2, method_name="naive_topm", min_docs=2
        )
    )
    assert rows[0]["selected_doc_ids"] == ["d1", "d2"]
    assert rows[0]["max_docs"] == 2
    with pytest.raises(ValueError, match="fewer than min_docs"):
        list(
            naive.iter_selection_rows(
                retrieval, top_m=4, method_name="naive_topm", min_docs=4
            )
        )

    bge_script = load_script("12_select_bge_reranker.py")
    cache = tmp_path / "scores.jsonl"
    cache.write_text(
        json.dumps(
            {
                "id": "q1",
                "scores": [
                    {"doc_id": "d1", "score": 0.1},
                    {"doc_id": "d2", "score": 0.9},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    contract = {"model": "mock", "input_template": "v1"}
    atomic_write_json(
        cache.with_suffix(".manifest.json"),
        {
            "status": "completed",
            "fingerprint": stable_hash(contract),
            "output_sha256": sha256_file(cache),
        },
    )
    loaded = bge_script.load_valid_cache(cache, contract)
    assert loaded == {"q1": {"d1": 0.1, "d2": 0.9}}


def test_infogain_teacher_label_order_thresholds_and_loss() -> None:
    posterior = {
        "id": "q",
        "query": "claim",
        "label": "REFUTES",
        "split": "train",
        "labels": ["SUPPORTS", "REFUTES"],
        "eta0": [0.6, 0.4],
        "candidates": [
            {**candidate("positive", 2, 5.0), "eta": [0.3, 0.7]},
            {**candidate("negative", 1, 6.0), "eta": [0.8, 0.2]},
            {**candidate("neutral", 3, 4.0), "eta": [0.55, 0.45]},
        ],
    }
    rows = posterior_to_teacher_rows(posterior)
    assert rows[0]["gold_index"] == 1
    assert rows[0]["dig"] == pytest.approx(0.3)
    assert rows[1]["dig"] == pytest.approx(-0.2)
    assert group_teacher_rows(rows) == [rows]
    thresholds = resolve_thresholds(
        (row["dig"] for row in rows),
        mode="explicit",
        b_pos=0.2,
        b_neg=-0.1,
    )
    assert thresholds["label_distribution"] == {
        "negative": 1,
        "neutral": 1,
        "positive": 1,
    }
    rank = torch.tensor([1.0, -1.0, 0.0], requires_grad=True)
    logits = torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 0.0]], requires_grad=True)
    loss, details = infogain_multitask_loss(
        rank,
        logits,
        [0.3, -0.2, 0.05],
        b_pos=0.2,
        b_neg=-0.1,
        beta=0.75,
    )
    assert torch.isfinite(loss)
    assert details["num_filter"] == 2
    assert details["num_neutral"] == 1
    assert pointwise_input("claim", "title", "text").find("Already selected") == -1
    with pytest.raises(ValueError, match="probabilities"):
        posterior_to_teacher_rows({**posterior, "eta0": [float("nan"), 0.4]})


def test_infogain_selection_is_gold_free_and_rank_ordered() -> None:
    module = load_script("12c_select_infogain_reranker.py")

    class FakeModel:
        def score(self, texts, batch_size):
            assert all("SUPPORTS" not in text for text in texts)
            return [0.2, 0.9, 0.5], [0.9, 0.1, 0.8]

    row = source_row()
    row.pop("label")
    selected = module.select_row(
        row,
        FakeModel(),
        batch_size=2,
        top_m=2,
        filter_threshold=0.5,
        min_docs=1,
        checkpoint_metadata={"fingerprint": "mock"},
    )
    assert selected["selected_doc_ids"] == ["d2", "d3"]
    assert selected["selected_docs"][0]["min_docs_fallback"] is True
    assert selected["selection_metadata"]["state_aware"] is False
    assert selected["uses_gold_at_test"] is False


def test_metrics_macro_f1_and_no_evidence() -> None:
    metrics = ClassificationMetrics(labels=["SUPPORTS", "REFUTES"])
    metrics.update("SUPPORTS", "SUPPORTS", num_docs=0, evidence_chars=0)
    metrics.update("REFUTES", "SUPPORTS", num_docs=0, evidence_chars=0)
    result = metrics.compute()
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["per_class"]["SUPPORTS"]["precision"] == 0.5
    assert result["avg_num_docs"] == 0.0
    assert result["avg_evidence_chars"] == 0.0


def test_summary_discovers_exactly_six_canonical_evaluations(tmp_path: Path) -> None:
    module = load_script("13_summarize_fever_baselines.py")
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "artifacts" / "eval"
    stems = {
        "no_evidence": "no_evidence_metrics",
        "naive_topm": "naive_top4_metrics",
        "bge": "bge_top4_metrics",
        "infogain_fever": "infogain_top4_metrics",
        "rag_cbwdm": "rag_cbwdm_metrics",
        "cbwdm_oracle": "cbwdm_oracle_top4_metrics",
    }
    canonical = [
        write_evaluation_artifact(eval_dir, stem=stem, method=method)
        for method, stem in stems.items()
    ]

    # This is the real failure shape: the old root no-evidence evaluation has
    # the same canonical method but lacks macro_f1. It must never enter the
    # manifest-selected baseline summary.
    legacy_root = run_dir / "artifacts" / "fever2_dev_no_evidence_metrics.json"
    legacy_root.parent.mkdir(parents=True, exist_ok=True)
    legacy_root.write_text(
        json.dumps({"method": "no_evidence", "accuracy": 0.5}), encoding="utf-8"
    )
    legacy_main = run_dir / "artifacts" / "fever2_dev_metrics.json"
    legacy_main.write_text(
        json.dumps(
            {"method": "cbwdm_cross_encoder_selector", "accuracy": 0.6}
        ),
        encoding="utf-8",
    )
    resource = write_evaluation_artifact(
        eval_dir,
        stem="resource_metrics",
        method="rag_cbwdm",
        stage="resource_monitoring",
    )
    training = write_evaluation_artifact(
        eval_dir,
        stem="training_metrics",
        method="infogain_fever",
        stage="training",
    )

    paths, excluded = module.discover_metric_artifacts(run_dir, [])
    assert paths == sorted(canonical, key=lambda path: module.ORDER[
        json.loads(path.read_text(encoding="utf-8"))["method"]
    ])
    assert legacy_root not in paths
    assert legacy_main not in paths
    assert resource not in paths
    assert training not in paths
    assert len(excluded) == 2

    summary = module.summarize(paths, module.DEFAULT_METHODS)
    assert [row["method"] for row in summary["methods"]] == module.DEFAULT_METHODS
    assert all(row["status"] == "completed" for row in summary["methods"])


def test_summary_missing_macro_f1_is_null_with_reason(tmp_path: Path) -> None:
    module = load_script("13_summarize_fever_baselines.py")
    metrics_path = write_evaluation_artifact(
        tmp_path,
        stem="bge_metrics",
        method="bge",
        metrics={"macro_f1": None},
    )
    summary = module.summarize([metrics_path], ["bge"])
    row = summary["methods"][0]
    assert row["status"] == "missing_metrics"
    assert row["macro_f1"] is None
    assert row["accuracy"] == 0.75
    assert row["missing_fields"] == ["macro_f1"]
    assert "macro_f1" in row["reason"]


@pytest.mark.parametrize(
    ("legacy_payload", "mapping"),
    [
        ({"f1_macro": 0.61}, "macro_f1<-f1_macro"),
        ({"metrics": {"f1_macro": 0.62}}, "macro_f1<-metrics.f1_macro"),
    ],
)
def test_summary_maps_legacy_and_nested_macro_f1(
    tmp_path: Path, legacy_payload: dict, mapping: str
) -> None:
    module = load_script("13_summarize_fever_baselines.py")
    metrics_path = write_evaluation_artifact(
        tmp_path,
        stem="naive_metrics",
        method="naive_topm",
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload.pop("macro_f1")
    payload.update(legacy_payload)
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    row = module.summarize([metrics_path], ["naive_topm"])["methods"][0]
    assert row["status"] == "completed"
    assert row["macro_f1"] in {0.61, 0.62}
    assert mapping in row["schema_mappings"]


def test_summary_preserves_no_evidence_zero_and_forces_oracle_metadata(
    tmp_path: Path,
) -> None:
    module = load_script("13_summarize_fever_baselines.py")
    no_evidence = write_evaluation_artifact(
        tmp_path,
        stem="no_evidence_metrics",
        method="no_evidence",
        metrics={"avg_num_docs": 0, "avg_evidence_chars": 0},
    )
    oracle = write_evaluation_artifact(
        tmp_path,
        stem="oracle_metrics",
        method="cbwdm_oracle",
        metrics={"deployable": True, "diagnostic_only": False},
    )
    rows = module.summarize(
        [no_evidence, oracle], ["no_evidence", "cbwdm_oracle"]
    )["methods"]
    assert rows[0]["avg_num_docs"] == 0.0
    assert rows[0]["status"] == "completed"
    assert rows[1]["deployable"] is False
    assert rows[1]["diagnostic_only"] is True


def test_summary_writes_only_fixed_output_directory_and_rejects_unfair_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script("13_summarize_fever_baselines.py")
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "artifacts" / "eval"
    for method in module.DEFAULT_METHODS:
        write_evaluation_artifact(
            eval_dir, stem=f"{method}_metrics", method=method
        )
    fairness = run_dir / "artifacts" / "baselines" / "baseline_fairness_audit.json"
    fairness.parent.mkdir(parents=True)
    fairness.write_text(
        json.dumps(
            {
                "status": "comparable",
                "methods": {
                    method: {"status": "comparable"} for method in module.DEFAULT_METHODS
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = run_dir / "artifacts" / "baselines" / "summary"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "13_summarize_fever_baselines.py",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--fairness-audit",
            str(fairness),
        ],
    )
    module.main()
    assert {path.name for path in output_dir.iterdir()} == set(module.SUMMARY_FILENAMES)

    wrong_output = tmp_path / "wrong-summary"
    with pytest.raises(ValueError, match="output must be"):
        module.fixed_output_dir(run_dir, str(wrong_output))

    fairness.write_text(
        json.dumps({"status": "not_comparable"}), encoding="utf-8"
    )
    for path in output_dir.iterdir():
        path.unlink()
    with pytest.raises(ValueError, match="Refusing formal baseline summary"):
        module.main()
    assert not any(output_dir.iterdir())


def test_runner_summary_validator_uses_real_summary_directory(tmp_path: Path) -> None:
    runner = load_script("run_fever_cbwdm.py")
    summary_dir = tmp_path / "artifacts" / "baselines" / "summary"
    summary_dir.mkdir(parents=True)
    methods = [
        {
            "method": method,
            "deployable": method != "cbwdm_oracle",
            "diagnostic_only": method == "cbwdm_oracle",
        }
        for method in (
            "no_evidence",
            "naive_topm",
            "bge",
            "infogain_fever",
            "rag_cbwdm",
            "cbwdm_oracle",
        )
    ]
    outputs = [
        summary_dir / "baseline_summary.json",
        summary_dir / "baseline_summary.csv",
        summary_dir / "baseline_summary.md",
    ]
    outputs[0].write_text(
        json.dumps({"status": "completed", "comparable": True, "methods": methods}),
        encoding="utf-8",
    )
    outputs[1].write_text("method\n", encoding="utf-8")
    outputs[2].write_text("# summary\n", encoding="utf-8")
    missing, invalid = runner.validate_stage_outputs(
        "summarize_baselines",
        outputs,
        {"baseline_summary_dir": summary_dir},
    )
    assert missing == []
    assert invalid == []

    other = tmp_path / "other"
    other.mkdir()
    wrong_outputs = [other / path.name for path in outputs]
    for source, destination in zip(outputs, wrong_outputs):
        destination.write_bytes(source.read_bytes())
    _, invalid = runner.validate_stage_outputs(
        "summarize_baselines",
        wrong_outputs,
        {"baseline_summary_dir": summary_dir},
    )
    assert any("must be read from" in reason for reason in invalid)


def test_oracle_is_diagnostic_and_requires_teacher(tmp_path: Path) -> None:
    oracle_module = load_script("09_select_cbwdm_oracle_from_teacher.py")
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(
        json.dumps(
            {
                "id": "q1",
                "query": "claim",
                "label": "SUPPORTS",
                "split": "dev",
                "teacher_selected_doc_ids": ["d2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = {"q1": source_row()}
    row = next(
        oracle_module.iter_oracle_rows(
            teacher, source, top_m=1, method_name="cbwdm_oracle"
        )
    )
    assert row["diagnostic_only"] is True
    assert row["uses_gold_at_test"] is True
    assert row["deployable"] is False
    assert row["selection_metadata"]["diagnostic_only"] is True
    with pytest.raises(FileNotFoundError):
        next(
            oracle_module.iter_oracle_rows(
                tmp_path / "missing_teacher.jsonl",
                source,
                top_m=1,
                method_name="cbwdm_oracle",
            )
        )


def test_runner_baseline_dry_run_lists_overrides(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_fever_cbwdm.py"),
        "--config",
        str(ROOT / "configs" / "fever2_server_smoke.yaml"),
        "--run-name",
        "baseline_dry_run_test",
        "--output-root",
        str(tmp_path),
        "--stages",
        "score_bge,select_bge,build_infogain_teacher,train_infogain,select_infogain",
        "--bge-model",
        "/models/local-bge",
        "--infogain-model",
        "/models/local-minilm",
        "--dry-run",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "[dry-run][score_bge]" in result.stdout
    assert "/models/local-bge" in result.stdout
    assert "/models/local-minilm" in result.stdout

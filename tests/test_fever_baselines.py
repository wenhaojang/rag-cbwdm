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


def test_summary_missing_and_oracle_excluded(tmp_path: Path) -> None:
    module = load_script("13_summarize_fever_baselines.py")
    deployable = tmp_path / "rag.json"
    oracle = tmp_path / "oracle.json"
    deployable.write_text(
        json.dumps(
            {
                "method": "rag_cbwdm",
                "accuracy": 0.8,
                "macro_f1": 0.7,
                "avg_num_docs": 2,
                "avg_evidence_chars": 20,
                "num_examples": 10,
                "deployable": True,
            }
        ),
        encoding="utf-8",
    )
    oracle.write_text(
        json.dumps(
            {
                "method": "cbwdm_oracle",
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "avg_num_docs": 2,
                "avg_evidence_chars": 20,
                "num_examples": 10,
                "deployable": False,
                "diagnostic_only": True,
            }
        ),
        encoding="utf-8",
    )
    summary = module.summarize(
        [deployable, oracle], ["naive_topm", "rag_cbwdm", "cbwdm_oracle"]
    )
    assert summary["methods"][0]["status"] == "missing"
    assert summary["methods"][0]["accuracy"] is None
    assert summary["deployable_best"] == "rag_cbwdm"


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

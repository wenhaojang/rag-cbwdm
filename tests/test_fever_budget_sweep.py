from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from src.baselines.infogain_protocol import budget_protocol_id
from src.diagnostics.signed_selector_v1 import greedy_select_without_gold
from src.fever_budget_sweep import (
    ALIAS_SCHEMA_VERSION,
    AUDIT_SCHEMA_VERSION,
    JOB_PLAN_SCHEMA_VERSION,
    BudgetSweepError,
    _bge_rows,
    _evaluation_command,
    _infogain_from_trace,
    _load_fingerprinted,
    _load_native_token_records,
    _rag_rows,
    _state_aware_inference_command,
    _write_alias,
    assert_isolated_artifact_root,
    build_plan,
    deduplicate_budget_vectors,
    evidence_vector,
    load_budget_config,
    render_evidence,
    replay_causal_prefix_row,
    summarize,
    validate_alias_manifest,
    validate_job_plan_inputs,
)
from src.run_manifest import atomic_write_json, sha256_file, stable_hash
from src.preformal.registry import resolve_inference_budget
from src.selection_schema import make_selection_row, normalize_selected_doc


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "formal" / "fever_budget_sweep_document_caps_234.seed13.yaml"
TOKEN_BASIS = "generator_tokenizer_rendered_evidence_block_before_prompt_truncation"
TOKEN_HISTOGRAM_BASIS = "observed_num_docs_values_only"
TOKEN_HISTOGRAM_COMPARISON = "zero_bins_elided_nonzero_counts_exact_v1"


def candidate(index: int) -> dict:
    return {
        "doc_id": f"d{index}",
        "title": f"Title {index}",
        "text": f"Text {index}",
        "rank": index,
        "retrieval_score": float(10 - index),
    }


def retrieval_row(*, label: str = "SUPPORTS") -> dict:
    return {
        "id": "q1",
        "query": "claim",
        "label": label,
        "gold": label,
        "split": "validation",
        "candidates": [candidate(index) for index in range(1, 5)],
    }


def native_token_artifacts(tmp_path: Path) -> tuple[dict, dict, dict, dict, dict]:
    datasets = {"fever_binary_v2": 5000, "fm2_official_closed_page_v1": 1169}
    generators_all = (
        "qwen2.5-0.5b-instruct",
        "qwen2.5-1.5b-instruct",
        "qwen2.5-7b-instruct",
        "mistral-7b-instruct-v0.3",
    )
    methods_all = (
        "no_evidence",
        "retrieval_topk",
        "bge",
        "infogain_fever",
        "rag_cbwdm_signed_v1",
        "kcbwdm_linear_gate_v2",
    )
    rows = []
    groups = []
    selections = {}
    evaluations = {}
    generators = {}
    for dataset_id, num_examples in datasets.items():
        for generator_id in generators_all:
            generator_identity = {
                "generator_id": generator_id,
                "tokenizer_name_or_path": f"/models/{generator_id}",
                "tokenizer_revision": None,
                "tokenizer_sha256": stable_hash(f"tokenizer:{generator_id}"),
                "trust_remote_code": False,
            }
            tokenizer_identity = dict(generator_identity)
            tokenizer_fingerprint = stable_hash(tokenizer_identity)
            generator_fingerprint = stable_hash(
                {"generator": generator_id, "identity": generator_identity}
            )
            for method_id in methods_all:
                key3 = (dataset_id, generator_id, method_id)
                group_id = "__".join(key3)
                source_sha256 = {
                    name: stable_hash(f"{group_id}:{name}")
                    for name in (
                        "selection",
                        "selection_manifest",
                        "predictions",
                        "metrics",
                        "evaluation_manifest",
                        "generator_manifest",
                    )
                }
                num_docs = 0 if method_id == "no_evidence" else 4
                histogram = {str(num_docs): num_examples}
                seed = 13 if method_id in {
                    "infogain_fever",
                    "rag_cbwdm_signed_v1",
                    "kcbwdm_linear_gate_v2",
                } else None
                paths = {
                    "selection_path": f"/formal/{group_id}/selection.jsonl",
                    "selection_manifest_path": f"/formal/{group_id}/selection.manifest.json",
                    "predictions_path": f"/formal/{group_id}/predictions.jsonl",
                    "metrics_path": f"/formal/{group_id}/metrics.json",
                    "evaluation_manifest_path": f"/formal/{group_id}/metrics.manifest.json",
                    "generator_manifest_path": f"/formal/{group_id}/generator.manifest.json",
                }
                group = {
                    "status": "PASS",
                    "group_id": group_id,
                    "dataset_id": dataset_id,
                    "generator_id": generator_id,
                    "method_id": method_id,
                    "experiment_type": "matched_main",
                    "seed": seed,
                    "split": "validation",
                    "accuracy": 0.75,
                    "macro_f1": 0.74,
                    "num_examples": num_examples,
                    "avg_num_docs": float(num_docs),
                    "avg_evidence_chars": float(num_docs * 100),
                    "avg_evidence_tokens": float(num_docs * 25),
                    "total_evidence_tokens": num_examples * num_docs * 25,
                    "num_docs_histogram": histogram,
                    "num_docs_histogram_basis": TOKEN_HISTOGRAM_BASIS,
                    "evidence_tokens_basis": TOKEN_BASIS,
                    **paths,
                    "source_sha256": source_sha256,
                    "tokenizer_identity": tokenizer_identity,
                    "tokenizer_identity_fingerprint": tokenizer_fingerprint,
                    "generator_identity_fingerprint": generator_fingerprint,
                }
                groups.append(group)
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "generator_id": generator_id,
                        "method_id": method_id,
                        "experiment_type": "matched_main",
                        "seed": seed,
                        "split": "validation",
                        "accuracy": group["accuracy"],
                        "macro_f1": group["macro_f1"],
                        "num_examples": num_examples,
                        "avg_num_docs": group["avg_num_docs"],
                        "avg_evidence_chars": group["avg_evidence_chars"],
                        "avg_evidence_tokens": group["avg_evidence_tokens"],
                        "total_evidence_tokens": group["total_evidence_tokens"],
                        "evidence_tokens_basis": TOKEN_BASIS,
                        **{f"num_docs_{value}": histogram.get(str(value), 0) for value in range(5)},
                        **paths,
                        **{f"{name}_sha256": value for name, value in source_sha256.items()},
                        "tokenizer_name_or_path": tokenizer_identity["tokenizer_name_or_path"],
                        "tokenizer_revision": tokenizer_identity["tokenizer_revision"],
                        "tokenizer_sha256": tokenizer_identity["tokenizer_sha256"],
                        "tokenizer_identity_fingerprint": tokenizer_fingerprint,
                        "generator_identity_fingerprint": generator_fingerprint,
                    }
                )
                if dataset_id == "fever_binary_v2" and generator_id in {
                    "qwen2.5-1.5b-instruct",
                    "qwen2.5-7b-instruct",
                } and method_id != "no_evidence":
                    key2 = (method_id, generator_id)
                    selections[key2] = {
                        "provenance": {
                            "selection_sha256": source_sha256["selection"],
                            "manifest_sha256": source_sha256["selection_manifest"],
                        }
                    }
                    evaluations[key2] = {
                        "predictions_sha256": source_sha256["predictions"],
                        "metrics_sha256": source_sha256["metrics"],
                        "manifest_sha256": source_sha256["evaluation_manifest"],
                        "metrics": {
                            "accuracy": group["accuracy"],
                            "macro_f1": group["macro_f1"],
                            "avg_num_docs": group["avg_num_docs"],
                        },
                    }
                    generators[key2] = {
                        "manifest_sha256": source_sha256["generator_manifest"],
                        "generator_identity_fingerprint": generator_fingerprint,
                        "generator_identity": generator_identity,
                    }

    results_path = tmp_path / "main_results_48.csv"
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "schema_version": "rag_cbwdm_evidence_token_audit.v1",
        "status": "PASS",
        "mode": "all",
        "num_groups": 48,
        "evidence_tokens_basis": TOKEN_BASIS,
        "num_docs_histogram_basis": TOKEN_HISTOGRAM_BASIS,
        "num_docs_histogram_comparison": TOKEN_HISTOGRAM_COMPARISON,
        "groups": groups,
        "error": None,
    }
    audit_path = tmp_path / "token_audit.json"
    atomic_write_json(audit_path, audit)
    manifest = {
        "schema_version": "rag_cbwdm_evidence_token_analysis.v1",
        "status": "completed",
        "completed": True,
        "mode": "all",
        "num_groups": 48,
        "evidence_tokens_basis": TOKEN_BASIS,
        "num_docs_histogram_basis": TOKEN_HISTOGRAM_BASIS,
        "num_docs_histogram_comparison": TOKEN_HISTOGRAM_COMPARISON,
        "outputs": {
            results_path.name: sha256_file(results_path),
            audit_path.name: sha256_file(audit_path),
        },
    }
    atomic_write_json(tmp_path / "analysis_manifest.json", manifest)
    config = {
        "native_token_results": str(results_path),
        "native_token_audit": str(audit_path),
    }
    return config, selections, generators, evaluations, {"audit": audit, "manifest": manifest}


def infogain_source(*, label: str = "SUPPORTS") -> dict:
    retrieval = retrieval_row(label=label)
    docs = [
        normalize_selected_doc(retrieval["candidates"][0], selector_score=4.0, selection_step=0),
        normalize_selected_doc(retrieval["candidates"][3], selector_score=1.0, selection_step=1),
    ]
    docs[0].update(filter_probability=0.9, min_docs_fallback=False, rank_score=4.0)
    docs[1].update(filter_probability=0.8, min_docs_fallback=False, rank_score=1.0)
    return make_selection_row(
        retrieval,
        method="infogain_fever",
        selected_docs=docs,
        selection_steps=[],
        stop_reason="filter_threshold_applied",
        max_docs=4,
        uses_gold_at_test=False,
        selection_metadata={
            "uses_gold_at_test": False,
            "candidate_score_audit": [
                {
                    "doc_id": f"d{index}",
                    "source_position": index - 1,
                    "rerank_position": index - 1,
                    "rank_score": float(5 - index),
                    "filter_probability": probability,
                    "within_top_m": True,
                    "passes_filter": probability >= 0.2,
                }
                for index, probability in ((1, 0.9), (2, 0.1), (3, 0.1), (4, 0.8))
            ],
        },
    )


def causal_source(*, stop_after: int | None = None) -> tuple[dict, dict[int, list[str]]]:
    retrieval = retrieval_row()

    def score(_query, selected, remaining):
        if stop_after is not None and len(selected) >= stop_after:
            return [-1.0 for _ in remaining]
        return [10.0 - int(item["doc_id"][1:]) for item in remaining]

    actions = {
        budget: greedy_select_without_gold(
            query="claim",
            candidates=retrieval["candidates"],
            score_remaining=score,
            top_m=budget,
            min_docs=0,
            score_threshold=0.0,
        )
        for budget in (2, 3, 4)
    }
    action = actions[4]
    docs = [
        normalize_selected_doc(
            item,
            selector_score=float(action["selection_steps"][index]["predicted_score"]),
            selection_step=index,
        )
        for index, item in enumerate(action["selected_docs"])
    ]
    row = make_selection_row(
        retrieval,
        method="rag_cbwdm_signed_v1",
        selected_docs=docs,
        selection_steps=action["selection_steps"],
        stop_reason=action["stop_reason"],
        max_docs=4,
        uses_gold_at_test=False,
        selection_metadata={
            "state_aware": True,
            "uses_gold_at_inference": False,
            "top_m": 4,
            "min_docs": 0,
            "score_threshold": 0.0,
        },
    )
    expected = {
        budget: [str(item["doc_id"]) for item in actions[budget]["selected_docs"]]
        for budget in actions
    }
    return row, expected


def test_frozen_plan_has_30_unique_points_and_16_new_selections(tmp_path: Path) -> None:
    config = load_budget_config(CONFIG)
    plan = build_plan(config, artifact_root=str(tmp_path / "budget"))
    assert len(plan["points"]) == 30
    assert len({point["point_id"] for point in plan["points"]}) == 30
    assert len(plan["selection_tasks"]) == 16
    assert sum(task["method_id"] == "retrieval_topk" for task in plan["selection_tasks"]) == 2
    assert sum(task["method_id"] == "bge" for task in plan["selection_tasks"]) == 2
    for method in ("infogain_fever", "rag_cbwdm_signed_v1", "kcbwdm_linear_gate_v2"):
        assert sum(task["method_id"] == method for task in plan["selection_tasks"]) == 4


def test_native_token_loader_accepts_completed_all_mode_schema(tmp_path: Path) -> None:
    config, selections, generators, evaluations, _payloads = native_token_artifacts(tmp_path)
    records = _load_native_token_records(
        config,
        selections=selections,
        generators=generators,
        evaluations=evaluations,
    )
    assert len(records) == 10
    assert records[("retrieval_topk", "qwen2.5-1.5b-instruct")][
        "avg_evidence_tokens"
    ] == 100.0
    assert records[("retrieval_topk", "qwen2.5-1.5b-instruct")][
        "source_analysis_manifest_sha256"
    ] == sha256_file(tmp_path / "analysis_manifest.json")


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    (
        ("audit", "status", "FAIL", "audit did not pass"),
        ("audit", "mode", "run", "audit is not a full analysis"),
        ("audit", "num_groups", 47, "audit does not cover 48 groups"),
        ("manifest", "status", "failed", "manifest is not completed"),
        ("manifest", "completed", False, "manifest is incomplete"),
        ("manifest", "mode", "validate-infogain", "manifest is not a full analysis"),
        ("manifest", "num_groups", 47, "manifest does not cover 48 groups"),
    ),
)
def test_native_token_loader_rejects_incomplete_or_failed_analysis(
    tmp_path: Path, target: str, field: str, value: object, message: str
) -> None:
    config, selections, generators, evaluations, payloads = native_token_artifacts(tmp_path)
    payloads[target][field] = value
    path = tmp_path / ("token_audit.json" if target == "audit" else "analysis_manifest.json")
    atomic_write_json(path, payloads[target])
    if target == "audit":
        payloads["manifest"]["outputs"]["token_audit.json"] = sha256_file(path)
        atomic_write_json(tmp_path / "analysis_manifest.json", payloads["manifest"])
    with pytest.raises(BudgetSweepError, match=message):
        _load_native_token_records(
            config,
            selections=selections,
            generators=generators,
            evaluations=evaluations,
        )


def test_native_token_loader_rejects_invalid_or_missing_groups(tmp_path: Path) -> None:
    config, selections, generators, evaluations, payloads = native_token_artifacts(tmp_path)
    payloads["audit"]["groups"][0]["status"] = "FAIL"
    atomic_write_json(tmp_path / "token_audit.json", payloads["audit"])
    payloads["manifest"]["outputs"]["token_audit.json"] = sha256_file(
        tmp_path / "token_audit.json"
    )
    atomic_write_json(tmp_path / "analysis_manifest.json", payloads["manifest"])
    with pytest.raises(BudgetSweepError, match="group did not pass"):
        _load_native_token_records(
            config,
            selections=selections,
            generators=generators,
            evaluations=evaluations,
        )

    payloads["audit"]["groups"].pop()
    atomic_write_json(tmp_path / "token_audit.json", payloads["audit"])
    payloads["manifest"]["outputs"]["token_audit.json"] = sha256_file(
        tmp_path / "token_audit.json"
    )
    atomic_write_json(tmp_path / "analysis_manifest.json", payloads["manifest"])
    with pytest.raises(BudgetSweepError, match="must contain 48 groups"):
        _load_native_token_records(
            config,
            selections=selections,
            generators=generators,
            evaluations=evaluations,
        )


def test_native_token_loader_rejects_analysis_output_sha_mismatch(tmp_path: Path) -> None:
    config, selections, generators, evaluations, _payloads = native_token_artifacts(tmp_path)
    with Path(config["native_token_results"]).open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(BudgetSweepError, match="CSV SHA differs"):
        _load_native_token_records(
            config,
            selections=selections,
            generators=generators,
            evaluations=evaluations,
        )


def test_rag_and_bge_are_fixed_ranking_prefixes() -> None:
    retrieval = retrieval_row()
    rag2 = next(iter(_rag_rows([retrieval], 2)))
    rag3 = next(iter(_rag_rows([retrieval], 3)))
    assert rag2["selected_doc_ids"] == ["d1", "d2"]
    assert rag3["selected_doc_ids"] == ["d1", "d2", "d3"]

    manifest = {"contract": {"model": {"model": "bge"}}}
    scores = {"q1": {"d1": 0.1, "d2": 0.9, "d3": 0.5, "d4": 0.4}}
    bge2 = next(iter(_bge_rows([retrieval], scores, manifest, 2)))
    bge3 = next(iter(_bge_rows([retrieval], scores, manifest, 3)))
    assert bge2["selected_doc_ids"] == ["d2", "d3"]
    assert bge3["selected_doc_ids"] == ["d2", "d3", "d4"]
    assert bge2["selection_metadata"]["budget_selection_mode"] == "full_reranking_prefix"


def test_infogain_filters_inside_each_budget_window_not_posthoc_truncation() -> None:
    retrieval = retrieval_row()
    source = infogain_source()
    _ranked4, chosen4 = _infogain_from_trace(retrieval, source, budget=4)
    _ranked2, chosen2 = _infogain_from_trace(retrieval, source, budget=2)
    assert [item["doc_id"] for item in chosen4] == ["d1", "d4"]
    assert [item["doc_id"] for item in chosen2] == ["d1", "d2"]
    assert [item["doc_id"] for item in chosen2] != [item["doc_id"] for item in chosen4][:2]
    assert chosen2[1]["min_docs_fallback"] is True


def test_infogain_k4_reconstruction_is_fail_closed() -> None:
    retrieval = retrieval_row()
    source = infogain_source()
    _infogain_from_trace(retrieval, source, budget=4)
    source["selected_doc_ids"] = ["d1", "d3"]
    with pytest.raises(BudgetSweepError, match="reconstruction mismatch"):
        _infogain_from_trace(retrieval, source, budget=4)


def test_infogain_budget_protocol_identity_is_frozen() -> None:
    assert budget_protocol_id(top_m=2, filter_threshold=0.2, min_docs=2).endswith("top2_then_filter02_min2_v1")
    assert budget_protocol_id(top_m=4, filter_threshold=0.2, min_docs=2).endswith("top4_then_filter02_min2_v1")
    with pytest.raises(ValueError, match="filter_threshold"):
        budget_protocol_id(top_m=3, filter_threshold=0.3, min_docs=2)
    path = ROOT / "scripts" / "12c_select_infogain_reranker.py"
    spec = importlib.util.spec_from_file_location("infogain_budget_selector_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.resolved_protocol_id(top_m=2, filter_threshold=0.2, min_docs=2).endswith(
        "top2_then_filter02_min2_v1"
    )
    assert module.resolved_protocol_id(top_m=2, filter_threshold=0.5, min_docs=1) is None


def test_causal_prefix_replay_matches_native_budget_greedy() -> None:
    source, expected = causal_source()
    for budget in (2, 3):
        replay = replay_causal_prefix_row(source, budget)
        assert replay["selected_doc_ids"] == expected[budget]
        assert replay["max_docs"] == budget
        assert replay["stop_reason"] == "inference_budget_reached"
        assert replay["selection_metadata"]["trained_max_docs"] == 4
        assert replay["selection_metadata"]["inference_budget"] == budget


def test_causal_prefix_replay_preserves_natural_early_stop() -> None:
    source, expected = causal_source(stop_after=1)
    assert expected[2] == expected[3] == expected[4] == ["d1"]
    for budget in (2, 3):
        replay = replay_causal_prefix_row(source, budget)
        assert replay["selected_doc_ids"] == ["d1"]
        assert replay["stop_reason"] == "score_below_threshold"
        assert replay["selection_steps"][-1]["stop"] is True


def test_train_top4_and_inference_budget_are_separate() -> None:
    assert resolve_inference_budget(4, None) == 4
    assert resolve_inference_budget(4, 2) == 2
    assert resolve_inference_budget(4, 3) == 3
    with pytest.raises(ValueError, match="frozen selector top_m"):
        resolve_inference_budget(4, 5)


def test_state_aware_fallback_command_reuses_checkpoint_without_training(tmp_path: Path) -> None:
    config = load_budget_config(CONFIG)
    task = {
        "method_id": "rag_cbwdm_signed_v1",
        "generator_id": "qwen2.5-1.5b-instruct",
        "budget_k": 2,
        "output_selection": str(tmp_path / "selection.jsonl"),
    }
    context = {
        "selections": {
            ("rag_cbwdm_signed_v1", "qwen2.5-1.5b-instruct"): {
                "provenance": {
                    "manifest": {
                        "contract": {
                            "inputs": {
                                "posteriors": {"path": "/source/posteriors.jsonl"},
                                "training_manifest": {"path": "/source/training_manifest.json"},
                            },
                            "model": {"checkpoint": "/source/checkpoint"},
                        }
                    }
                }
            }
        }
    }
    command = _state_aware_inference_command(task, config, context, resume=True)
    assert command[command.index("--inference-budget") + 1] == "2"
    assert command[command.index("--checkpoint-dir") + 1] == "/source/checkpoint"
    assert "--resume" in command
    assert not any("train_signed" in token for token in command)


def test_selection_is_independent_of_validation_gold() -> None:
    first = retrieval_row(label="SUPPORTS")
    second = retrieval_row(label="REFUTES")
    assert next(iter(_rag_rows([first], 2)))["selected_doc_ids"] == next(iter(_rag_rows([second], 2)))["selected_doc_ids"]
    _, chosen_first = _infogain_from_trace(first, infogain_source(label="SUPPORTS"), budget=2)
    _, chosen_second = _infogain_from_trace(second, infogain_source(label="REFUTES"), budget=2)
    assert [item["doc_id"] for item in chosen_first] == [item["doc_id"] for item in chosen_second]


def test_ordered_evidence_vector_detects_order_and_text_changes() -> None:
    first = next(iter(_rag_rows([retrieval_row()], 2)))
    reordered = json.loads(json.dumps(first))
    reordered["selected_docs"].reverse()
    reordered["selected_doc_ids"].reverse()
    text_changed = json.loads(json.dumps(first))
    text_changed["selected_docs"][0]["text"] = "changed"
    assert evidence_vector([first], expected_rows=1)[0] != evidence_vector([reordered], expected_rows=1)[0]
    assert evidence_vector([first], expected_rows=1)[0] != evidence_vector([text_changed], expected_rows=1)[0]


def test_qwen7_signed_fixture_all_budgets_share_one_vector() -> None:
    source, _expected = causal_source(stop_after=1)
    vectors = {
        budget: evidence_vector(
            [source if budget == 4 else replay_causal_prefix_row(source, budget)],
            expected_rows=1,
        )[0]
        for budget in (2, 3, 4)
    }
    assert len(set(vectors.values())) == 1


def test_server_precheck_pattern_computes_18_jobs_without_hardcoding() -> None:
    distinct = {2: "v2", 3: "v3", 4: "v4"}
    all_same = {2: "same", 3: "same", 4: "same"}
    groups = [distinct for _ in range(9)] + [all_same]
    job_count = sum(
        sum(canonical == budget and budget != 4 for budget, canonical in deduplicate_budget_vectors(group).items())
        for group in groups
    )
    assert job_count == 18


def test_artifact_root_must_be_physically_isolated() -> None:
    protected = ["/root/experiments/rag_cbwdm/formal_v2_full_development"]
    assert_isolated_artifact_root("/root/experiments/rag_cbwdm/new_budget", protected)
    with pytest.raises(BudgetSweepError, match="overlaps"):
        assert_isolated_artifact_root(protected[0] + "/budget", protected)


def test_k4_alias_binds_all_source_checksums_and_rejects_tamper(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    selection_manifest = tmp_path / "selection.manifest.json"
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    evaluation_manifest = tmp_path / "metrics.manifest.json"
    for path, text in (
        (selection, "selection\n"),
        (selection_manifest, "{}\n"),
        (predictions, "prediction\n"),
        (metrics, "{}\n"),
        (evaluation_manifest, "{}\n"),
    ):
        path.write_text(text, encoding="utf-8")
    alias_path = tmp_path / "out" / "alias.manifest.json"
    point = {
        "point_id": "q7.rag.k4",
        "generator_id": "qwen2.5-7b-instruct",
        "method_id": "retrieval_topk",
        "budget_k": 4,
        "selection_path": str(selection),
        "selection_manifest_path": str(selection_manifest),
        "output": {"alias_manifest": str(alias_path)},
    }
    source_eval = {
        "paths": {
            "predictions": str(predictions),
            "metrics": str(metrics),
            "manifest": str(evaluation_manifest),
        },
        "predictions_sha256": sha256_file(predictions),
        "metrics_sha256": sha256_file(metrics),
        "manifest_sha256": sha256_file(evaluation_manifest),
    }
    _write_alias(
        point,
        source_point=point,
        source_evaluation=source_eval,
        vector_sha256="a" * 64,
        generator_input_fingerprint="b" * 64,
        plan_fingerprint="c" * 64,
    )
    alias = validate_alias_manifest(alias_path, expected_point_id=point["point_id"])
    assert alias["schema_version"] == ALIAS_SCHEMA_VERSION
    metrics.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(BudgetSweepError, match="metrics SHA changed"):
        validate_alias_manifest(alias_path)


def test_frozen_gpu_job_plan_rejects_changed_inputs(tmp_path: Path) -> None:
    files = {}
    for name in ("selection", "selection_manifest", "generator_manifest", "generator_config"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        files[name] = path
    audit = {"schema_version": AUDIT_SCHEMA_VERSION, "status": "completed"}
    audit["audit_fingerprint"] = stable_hash(audit)
    audit_path = tmp_path / "audit.json"
    atomic_write_json(audit_path, audit)
    job = {
        "job_id": "job",
        "selection_path": str(files["selection"]),
        "selection_sha256": sha256_file(files["selection"]),
        "selection_manifest_path": str(files["selection_manifest"]),
        "selection_manifest_sha256": sha256_file(files["selection_manifest"]),
        "generator_manifest_path": str(files["generator_manifest"]),
        "generator_manifest_sha256": sha256_file(files["generator_manifest"]),
        "generator_config_path": str(files["generator_config"]),
        "generator_config_sha256": sha256_file(files["generator_config"]),
    }
    plan = {
        "schema_version": JOB_PLAN_SCHEMA_VERSION,
        "status": "frozen",
        "audit_path": str(audit_path),
        "audit_fingerprint": audit["audit_fingerprint"],
        "evaluation_job_count": 1,
        "jobs": [job],
    }
    plan["job_plan_fingerprint"] = stable_hash(plan)
    plan_path = tmp_path / "jobs.json"
    atomic_write_json(plan_path, plan)
    validate_job_plan_inputs(plan_path)
    files["selection"].write_text("changed", encoding="utf-8")
    with pytest.raises(BudgetSweepError, match="Frozen job input changed"):
        validate_job_plan_inputs(plan_path)


def test_evaluation_command_uses_native_budget_selection_without_max_docs(tmp_path: Path) -> None:
    config = load_budget_config(CONFIG)
    point = build_plan(config, artifact_root=str(tmp_path / "budget"))["points"][0]
    point = {**point, "budget_k": 2}
    generator = {"generator_identity": {"generator_id": point["generator_id"]}}
    command = _evaluation_command(config, point, generator)
    assert "--max-docs" not in command
    assert command[command.index("--experiment-type") + 1] == "budget_frontier"
    assert "--resume" in command


def test_rendering_matches_formal_evaluator_exactly() -> None:
    path = ROOT / "scripts" / "07_eval_rag_classification.py"
    spec = importlib.util.spec_from_file_location("formal_eval_for_budget_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    docs = [candidate(1), candidate(2)]
    assert render_evidence(docs) == module.build_evidence_context(docs)


def test_summary_writes_figure2_ready_30_row_csv(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    selection.write_text("{}\n", encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    atomic_write_json(
        metrics,
        {
            "num_examples": 5000,
            "accuracy": 0.75,
            "macro_f1": 0.74,
            "avg_num_docs": 2.5,
            "avg_evidence_tokens": 123.0,
            "evidence_tokens_basis": "generator_tokenizer_rendered_evidence_block_before_prompt_truncation",
        },
    )
    points = []
    for generator in ("qwen2.5-1.5b-instruct", "qwen2.5-7b-instruct"):
        for method in (
            "retrieval_topk",
            "bge",
            "infogain_fever",
            "rag_cbwdm_signed_v1",
            "kcbwdm_linear_gate_v2",
        ):
            for budget in (2, 3, 4):
                points.append(
                    {
                        "point_id": f"{generator}.{method}.k{budget}",
                        "generator_id": generator,
                        "method_id": method,
                        "budget_k": budget,
                        "selection_path": str(selection),
                        "resolved_metrics_path": str(metrics),
                        "resolved_metric_values": None,
                        "source_or_alias": "new_evaluation",
                        "evidence_vector_sha256": "a" * 64,
                        "alias_manifest": None,
                    }
                )
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "completed",
        "points": points,
    }
    audit["audit_fingerprint"] = stable_hash(audit)
    audit_path = tmp_path / "audit.json"
    atomic_write_json(audit_path, audit)
    output_csv = tmp_path / "fever_budget_accuracy_tokens.csv"
    output_manifest = tmp_path / "summary.manifest.json"
    result = summarize(
        audit_path,
        output_csv=output_csv,
        output_manifest=output_manifest,
    )
    assert result["row_count"] == 30
    lines = output_csv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 31
    assert lines[0].split(",") == [
        "dataset_id",
        "generator_id",
        "method_id",
        "budget_k",
        "accuracy",
        "macro_f1",
        "avg_num_docs",
        "avg_evidence_tokens",
        "selection_path",
        "evaluation_path",
        "source_or_alias",
        "analysis_status",
    ]


def test_fingerprinted_artifact_rejects_manifest_edit(tmp_path: Path) -> None:
    payload = {"schema_version": JOB_PLAN_SCHEMA_VERSION, "status": "frozen", "jobs": []}
    payload["job_plan_fingerprint"] = stable_hash(payload)
    path = tmp_path / "plan.json"
    atomic_write_json(path, payload)
    _load_fingerprinted(path, JOB_PLAN_SCHEMA_VERSION, "job_plan_fingerprint")
    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["status"] = "edited"
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(BudgetSweepError, match="Fingerprint mismatch"):
        _load_fingerprinted(path, JOB_PLAN_SCHEMA_VERSION, "job_plan_fingerprint")

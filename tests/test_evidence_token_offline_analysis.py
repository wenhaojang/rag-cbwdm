from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from src.run_manifest import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "formal" / "evidence_token_analysis_48.development.server.yaml"


@pytest.fixture(scope="module")
def analysis():
    path = ROOT / "scripts" / "analyze_formal_evidence_tokens.py"
    spec = importlib.util.spec_from_file_location("evidence_token_analysis", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CharacterTokenizer:
    def __init__(self) -> None:
        self.seen: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool):
        self.seen.append((text, add_special_tokens))
        return list(text)


def selection_row(*, docs: list[dict] | None = None, split: str = "validation") -> dict:
    selected = docs or []
    return {
        "id": "row-1",
        "query": "claim",
        "label": "SUPPORTS",
        "split": split,
        "method": "bge",
        "selected_doc_ids": [doc["doc_id"] for doc in selected],
        "selected_docs": selected,
    }


def prediction_row(*, doc_ids: list[str] | None = None) -> dict:
    selected = doc_ids or []
    return {
        "id": "row-1",
        "query": "claim",
        "gold": "SUPPORTS",
        "pred": "SUPPORTS",
        "method": "bge",
        "selected_doc_ids": selected,
        "num_docs": len(selected),
    }


def test_no_evidence_renders_empty_and_counts_zero_tokens(analysis) -> None:
    tokenizer = CharacterTokenizer()
    source = selection_row(
        docs=[{"doc_id": "ignored", "title": "Ignored", "text": "not used"}]
    )
    source["method"] = "no_evidence"
    prediction = prediction_row()
    prediction["method"] = "no_evidence"
    report = analysis.audit_evidence_rows(
        [source],
        [prediction],
        tokenizer=tokenizer,
        no_evidence=True,
        max_docs=None,
        expected_rows=1,
        group_id="no-evidence",
        expected_method="no_evidence",
    )
    assert report["avg_num_docs"] == 0
    assert report["avg_evidence_chars"] == 0
    assert report["avg_evidence_tokens"] == 0
    assert report["total_evidence_tokens"] == 0
    assert tokenizer.seen == []


def test_evidence_rendering_preserves_order_titles_and_separators(analysis) -> None:
    docs = [
        {"doc_id": "d2", "title": "Second first", "text": "alpha"},
        {"doc_id": "d1", "text": "beta"},
    ]
    rendered_docs, text = analysis.render_evidence(
        selection_row(docs=docs), no_evidence=False, max_docs=None
    )
    assert [doc["doc_id"] for doc in rendered_docs] == ["d2", "d1"]
    assert text == "[1] Title: Second first\nalpha\n\n[2] beta"


def test_max_docs_matches_evaluator_before_tokenization(analysis) -> None:
    docs = [
        {"doc_id": "d1", "title": "One", "text": "alpha"},
        {"doc_id": "d2", "title": "Two", "text": "beta"},
    ]
    tokenizer = CharacterTokenizer()
    report = analysis.audit_evidence_rows(
        [selection_row(docs=docs)],
        [prediction_row(doc_ids=["d1"])],
        tokenizer=tokenizer,
        no_evidence=False,
        max_docs=1,
        expected_rows=1,
        group_id="max-docs",
        expected_method="bge",
    )
    expected = "[1] Title: One\nalpha"
    assert tokenizer.seen == [(expected, False)]
    assert report["avg_num_docs"] == 1
    assert report["total_evidence_tokens"] == len(expected)


def test_tokenizer_loader_uses_frozen_tokenizer_identity_only(
    analysis, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []
    sentinel = object()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append((name, kwargs))
            return sentinel

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    identity = {
        "tokenizer_name_or_path": "/models/frozen-tokenizer",
        "tokenizer_revision": "tokenizer-rev",
        "model_revision": "model-rev",
        "trust_remote_code": True,
    }
    assert analysis.load_tokenizer_for_identity(identity) is sentinel
    assert calls == [
        (
            "/models/frozen-tokenizer",
            {"trust_remote_code": True, "revision": "tokenizer-rev"},
        )
    ]


def test_prediction_selected_doc_ids_mismatch_is_rejected(analysis) -> None:
    docs = [{"doc_id": "d1", "text": "alpha"}]
    with pytest.raises(ValueError, match="selected_doc_ids mismatch"):
        analysis.audit_evidence_rows(
            [selection_row(docs=docs)],
            [prediction_row(doc_ids=["other"])],
            tokenizer=CharacterTokenizer(),
            no_evidence=False,
            max_docs=None,
            expected_rows=1,
            group_id="identity",
            expected_method="bge",
        )


def test_recomputed_character_mismatch_blocks_summary(analysis) -> None:
    group = {"group_id": "chars", "method_id": "bge"}
    metrics = {
        "avg_num_docs": 1.0,
        "avg_evidence_chars": 999.0,
        "avg_evidence_tokens": None,
        "num_docs": {"histogram": {"1": 1}},
    }
    recomputed = {
        "avg_num_docs": 1.0,
        "avg_evidence_chars": 10.0,
        "avg_evidence_tokens": 4.0,
        "num_docs_histogram": {"1": 1},
    }
    with pytest.raises(ValueError, match="avg_evidence_chars mismatch"):
        analysis.validate_recomputed_metrics(group, metrics, recomputed)


def test_infogain_token_mismatch_blocks_summary_without_hardcoded_value(analysis) -> None:
    group = {
        "group_id": "infogain",
        "method_id": "infogain_fever",
        "expected_rows": 1,
    }
    metrics = {
        "avg_num_docs": 2.0,
        "avg_evidence_chars": 20.0,
        "avg_evidence_tokens": 12.5,
        "evidence_tokens_basis": analysis.TOKEN_BASIS,
        "num_docs": {"histogram": {"2": 1}},
    }
    recomputed = {
        "avg_num_docs": 2.0,
        "avg_evidence_chars": 20.0,
        "avg_evidence_tokens": 12.6,
        "num_docs_histogram": {"2": 1},
    }
    with pytest.raises(ValueError, match="avg_evidence_tokens mismatch"):
        analysis.validate_recomputed_metrics(group, metrics, recomputed)


def test_no_evidence_5000_rows_has_sparse_zero_only_histogram(analysis) -> None:
    selections = []
    predictions = []
    for index in range(5000):
        selection = selection_row(
            docs=[{"doc_id": f"ignored-{index}", "text": "not used"}]
        )
        selection.update({"id": f"row-{index}", "method": "no_evidence"})
        prediction = prediction_row()
        prediction.update({"id": f"row-{index}", "method": "no_evidence"})
        selections.append(selection)
        predictions.append(prediction)
    result = analysis.audit_evidence_rows(
        selections,
        predictions,
        tokenizer=CharacterTokenizer(),
        no_evidence=True,
        max_docs=None,
        expected_rows=5000,
        group_id="fever-no-evidence",
        expected_method="no_evidence",
    )
    assert result["num_docs_histogram"] == {"0": 5000}
    analysis.validate_recomputed_metrics(
        {
            "group_id": "fever-no-evidence",
            "method_id": "no_evidence",
            "expected_rows": 5000,
        },
        {
            "avg_num_docs": 0.0,
            "avg_evidence_chars": 0.0,
            "avg_evidence_tokens": None,
            "num_docs": {"histogram": {"0": 5000}},
        },
        result,
    )


def test_mixed_doc_counts_histogram_contains_only_observed_bins(analysis) -> None:
    selections = []
    predictions = []
    for index, count in enumerate((0, 1, 2)):
        docs = [
            {"doc_id": f"d{index}-{doc_index}", "text": f"text-{doc_index}"}
            for doc_index in range(count)
        ]
        selection = selection_row(docs=docs)
        selection["id"] = f"row-{index}"
        prediction = prediction_row(doc_ids=[doc["doc_id"] for doc in docs])
        prediction["id"] = f"row-{index}"
        selections.append(selection)
        predictions.append(prediction)
    result = analysis.audit_evidence_rows(
        selections,
        predictions,
        tokenizer=CharacterTokenizer(),
        no_evidence=False,
        max_docs=None,
        expected_rows=3,
        group_id="mixed-doc-counts",
        expected_method="bge",
    )
    assert result["num_docs_histogram"] == {"0": 1, "1": 1, "2": 1}


def test_infogain_zero_bins_normalize_before_exact_comparison(analysis) -> None:
    expected_rows = 5000
    avg_num_docs = (2 * 328 + 3 * 167 + 4 * 4505) / expected_rows
    group = {
        "group_id": "infogain-zero-bins",
        "method_id": "infogain_fever",
        "expected_rows": expected_rows,
    }
    metrics = {
        "avg_num_docs": avg_num_docs,
        "avg_evidence_chars": 10.0,
        "avg_evidence_tokens": 12.5,
        "evidence_tokens_basis": analysis.TOKEN_BASIS,
        "num_docs": {
            "histogram": {"0": 0, "1": 0, "2": 328, "3": 167, "4": 4505}
        },
    }
    recomputed = {
        "avg_num_docs": avg_num_docs,
        "avg_evidence_chars": 10.0,
        "avg_evidence_tokens": 12.5,
        "num_docs_histogram": {"2": 328, "3": 167, "4": 4505},
    }
    analysis.validate_recomputed_metrics(group, metrics, recomputed)


def test_nonzero_doc_distribution_mismatch_is_still_rejected(analysis) -> None:
    group = {"group_id": "histogram", "method_id": "bge", "expected_rows": 4}
    metrics = {
        "avg_num_docs": 2.0,
        "avg_evidence_chars": 10.0,
        "avg_evidence_tokens": None,
        "num_docs": {"histogram": {"0": 1, "2": 2, "4": 1}},
    }
    recomputed = {
        "avg_num_docs": 2.0,
        "avg_evidence_chars": 10.0,
        "avg_evidence_tokens": 4.0,
        "num_docs_histogram": {"1": 2, "3": 2},
    }
    with pytest.raises(ValueError, match="num_docs histogram mismatch"):
        analysis.validate_recomputed_metrics(group, metrics, recomputed)


@pytest.mark.parametrize(
    ("histogram", "message"),
    [
        (None, "must be a dictionary"),
        ({0: 1}, "invalid document-count key"),
        ({"5": 1}, "invalid document-count key"),
        ({"01": 1}, "invalid document-count key"),
        ({"0": -1, "1": 2}, "non-negative integer"),
        ({"0": 1.0}, "non-negative integer"),
        ({"0": True}, "non-negative integer"),
        ({"0": 2}, "frequency sum mismatch"),
    ],
)
def test_histogram_contract_rejects_invalid_keys_counts_and_totals(
    analysis, histogram, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        analysis.normalize_num_docs_histogram(
            histogram, expected_rows=1, label="test histogram"
        )


def test_train_split_is_rejected(analysis) -> None:
    with pytest.raises(ValueError, match="split is not validation"):
        analysis.audit_evidence_rows(
            [selection_row(split="train")],
            [prediction_row()],
            tokenizer=CharacterTokenizer(),
            no_evidence=False,
            max_docs=None,
            expected_rows=1,
            group_id="split",
            expected_method="bge",
        )


def test_old_infogain_root_is_rejected(analysis, tmp_path: Path) -> None:
    text = PLAN.read_text(encoding="utf-8")
    text = text.replace(
        "formal_v2_infogain_top4_filter02_final",
        "formal_v2_full_development",
    )
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="corrected InfoGain root mismatch"):
        analysis.load_plan(wrong)


def test_missing_file_and_wrong_sha_are_rejected(analysis, tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="Missing evaluation manifest"):
        analysis._load_json(missing, "evaluation manifest")
    artifact = tmp_path / "predictions.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        analysis.require_recorded_sha(artifact, "0" * 64, "predictions")
    assert analysis.require_recorded_sha(
        artifact, sha256_file(artifact), "predictions"
    ) == sha256_file(artifact)


def test_duplicate_ids_are_rejected(analysis) -> None:
    first = selection_row()
    second = dict(first)
    prediction = prediction_row()
    with pytest.raises(ValueError, match="duplicate selection ID"):
        analysis.audit_evidence_rows(
            [first, second],
            [prediction, dict(prediction)],
            tokenizer=CharacterTokenizer(),
            no_evidence=False,
            max_docs=None,
            expected_rows=2,
            group_id="duplicates",
            expected_method="bge",
        )


def test_plan_expands_to_unique_corrected_48_group_matrix(analysis) -> None:
    _, groups = analysis.load_plan(PLAN)
    assert len(groups) == 48
    analysis.validate_result_rows(groups, expected_count=48)
    infogain = [group for group in groups if group["method_id"] == "infogain_fever"]
    assert len(infogain) == 8
    assert all("formal_v2_infogain_top4_filter02_final" in group["selection"] for group in infogain)
    assert all("/infogain/seed13/selection/validation.jsonl" in group["selection"].replace("\\", "/") for group in infogain)


def _find_group(analysis, dataset_id: str, method_id: str) -> dict:
    _, groups = analysis.load_plan(PLAN)
    return next(
        group
        for group in groups
        if group["dataset_id"] == dataset_id
        and group["generator_id"] == "qwen2.5-0.5b-instruct"
        and group["method_id"] == method_id
    )


def test_fever_kcbwdm_uses_independent_generator_manifest(analysis) -> None:
    group = _find_group(analysis, "fever_binary_v2", "kcbwdm_linear_gate_v2")
    normalized = group["generator_manifest"].replace("\\", "/")
    assert group["generator_manifest_source_root"] == "kcbwdm_v2a"
    assert normalized == (
        "/root/experiments/rag_cbwdm/"
        "formal_v2_fever_kcbwdm_linear_gate_v2_full_development/"
        "fever_binary_v2/qwen2.5-0.5b-instruct/generator/manifest.json"
    )


def test_fm2_kcbwdm_uses_independent_generator_manifest(analysis) -> None:
    group = _find_group(
        analysis, "fm2_official_closed_page_v1", "kcbwdm_linear_gate_v2"
    )
    normalized = group["generator_manifest"].replace("\\", "/")
    assert group["generator_manifest_source_root"] == "kcbwdm_v2a"
    assert normalized == (
        "/root/experiments/rag_cbwdm/"
        "formal_v2_kcbwdm_linear_gate_v2_full_development/"
        "fm2_official_closed_page_v1/qwen2.5-0.5b-instruct/"
        "generator/manifest.json"
    )


def test_other_five_methods_keep_main_generator_manifest(analysis) -> None:
    _, groups = analysis.load_plan(PLAN)
    main_methods = set(analysis.EXPECTED_METHODS) - {"kcbwdm_linear_gate_v2"}
    checked = [group for group in groups if group["method_id"] in main_methods]
    assert len(checked) == 40
    assert all(group["generator_manifest_source_root"] == "main" for group in checked)
    assert all(
        "/formal_v2_full_development/" in group["generator_manifest"].replace("\\", "/")
        for group in checked
    )
    infogain = [group for group in checked if group["method_id"] == "infogain_fever"]
    assert all(
        "/formal_v2_infogain_top4_filter02_final/"
        in group["selection"].replace("\\", "/")
        for group in infogain
    )


def test_kcbwdm_generator_manifest_root_policy_is_frozen(
    analysis, tmp_path: Path
) -> None:
    text = PLAN.read_text(encoding="utf-8")
    text = text.replace(
        "  - method_id: kcbwdm_linear_gate_v2\n"
        "    seed: 13\n"
        "    shared_selection: false\n"
        "    source_root: kcbwdm_v2a\n"
        "    generator_manifest_root: kcbwdm_v2a",
        "  - method_id: kcbwdm_linear_gate_v2\n"
        "    seed: 13\n"
        "    shared_selection: false\n"
        "    source_root: kcbwdm_v2a\n"
        "    generator_manifest_root: main",
    )
    wrong = tmp_path / "wrong-generator-root.yaml"
    wrong.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="generator manifest root policy mismatch"):
        analysis.load_plan(wrong)


def test_generator_cache_is_isolated_by_real_manifest_path(
    analysis, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    identity = {
        "tokenizer_name_or_path": "/models/frozen-tokenizer",
        "tokenizer_revision": None,
    }

    def fake_validate(path, **kwargs):
        calls.append(str(Path(path).resolve()))
        return {
            "manifest": {"config_path": f"/configs/{len(calls)}.yaml"},
            "generator_identity": dict(identity),
        }

    monkeypatch.setattr(analysis, "validate_generator_manifest", fake_validate)
    monkeypatch.setattr(
        analysis,
        "load_yaml",
        lambda path: {"generator": {"model_name": "/models/frozen-tokenizer"}},
    )
    monkeypatch.setattr(
        analysis, "validate_generator_identity_against_config", lambda *args: None
    )
    common = {
        "group_id": "cache",
        "dataset_id": "fever_binary_v2",
        "generator_id": "qwen2.5-0.5b-instruct",
        "expected_tokenizer_name_or_path": "/models/frozen-tokenizer",
        "expected_tokenizer_revision": None,
    }
    main = {**common, "generator_manifest": "/main/generator/manifest.json"}
    kcbwdm = {**common, "generator_manifest": "/kcbwdm/generator/manifest.json"}
    cache = {}
    main_result, _ = analysis.validate_generator_source(main, generator_cache=cache)
    kcbwdm_result, _ = analysis.validate_generator_source(
        kcbwdm, generator_cache=cache
    )
    cached_main_result, _ = analysis.validate_generator_source(
        main, generator_cache=cache
    )
    assert len(calls) == 2
    assert len(cache) == 2
    assert main_result is cached_main_result
    assert main_result is not kcbwdm_result


def test_evaluation_binding_rejects_generator_manifest_path_deviation(
    analysis,
) -> None:
    expected = {
        "generator_manifest_path": "/frozen/kcbwdm/generator/manifest.json",
        "generator_manifest_sha256": "a" * 64,
        "selection_sha256": "b" * 64,
    }
    recorded = {
        **expected,
        "generator_manifest_path": "/unfrozen/main/generator/manifest.json",
    }
    with pytest.raises(ValueError, match="evaluation artifact binding mismatch"):
        analysis.require_exact_evaluation_binding("kcbwdm", recorded, expected)


def test_evaluation_binding_rejects_generator_manifest_sha_mismatch(
    analysis,
) -> None:
    expected = {
        "generator_manifest_path": "/frozen/kcbwdm/generator/manifest.json",
        "generator_manifest_sha256": "a" * 64,
        "selection_sha256": "b" * 64,
    }
    recorded = {**expected, "generator_manifest_sha256": "c" * 64}
    with pytest.raises(ValueError, match="evaluation artifact binding mismatch"):
        analysis.require_exact_evaluation_binding("kcbwdm", recorded, expected)


def test_output_safety_prevents_source_overwrite_and_requires_resume(
    analysis, tmp_path: Path
) -> None:
    protected = Path(
        "/root/experiments/rag_cbwdm/formal_v2_full_development/analysis"
    )
    with pytest.raises(ValueError, match="formal experiment root"):
        analysis._ensure_output_safety(protected, resume=False)
    output = tmp_path / "analysis"
    output.mkdir()
    (output / "token_audit.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="use --resume"):
        analysis._ensure_output_safety(output, resume=False)
    analysis._ensure_output_safety(output, resume=True)


def test_resume_rejects_identity_mismatched_group_cache(
    analysis, tmp_path: Path
) -> None:
    output = tmp_path / "analysis"
    cache = output / "group_audits" / "group.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(
        json.dumps(
            {
                "schema_version": analysis.AUDIT_SCHEMA,
                "status": "PASS",
                "source_fingerprint": "old-identity",
                "result": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity-mismatched"):
        analysis.analyze_or_resume_group(
            {"group_id": "group"},
            {"source_fingerprint": "new-identity"},
            tokenizer=None,
            output_dir=output,
            resume=True,
        )


def test_resume_rejects_previous_histogram_comparison_cache(
    analysis, tmp_path: Path
) -> None:
    output = tmp_path / "analysis"
    cache = output / "group_audits" / "group.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(
        json.dumps(
            {
                "schema_version": analysis.AUDIT_SCHEMA,
                "status": "PASS",
                "source_fingerprint": "same-identity",
                "num_docs_histogram_basis": analysis.NUM_DOCS_HISTOGRAM_BASIS,
                "result": {"num_docs_histogram": {"0": 1, "1": 0}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="histogram-semantics-mismatched"):
        analysis.analyze_or_resume_group(
            {"group_id": "group"},
            {"source_fingerprint": "same-identity"},
            tokenizer=None,
            output_dir=output,
            resume=True,
        )


def test_formal_outputs_have_48_rows_and_latex_escapes(analysis, tmp_path: Path) -> None:
    _, groups = analysis.load_plan(PLAN)
    results = []
    for group in groups:
        results.append(
            {
                **group,
                "accuracy": 0.75,
                "macro_f1": 0.7,
                "num_examples": group["expected_rows"],
                "avg_num_docs": 2.0,
                "avg_evidence_chars": 100.0,
                "avg_evidence_tokens": 25.0,
                "total_evidence_tokens": 25 * group["expected_rows"],
                "evidence_tokens_basis": analysis.TOKEN_BASIS,
                "selection_path": group["selection"],
                "selection_manifest_path": group["selection_manifest"],
                "predictions_path": group["predictions"],
                "metrics_path": group["metrics"],
                "evaluation_manifest_path": group["evaluation_manifest"],
                "generator_manifest_path": group["generator_manifest"],
                "num_docs_histogram": {"2": group["expected_rows"]},
                "source_sha256": {
                    "selection": "1" * 64,
                    "selection_manifest": "2" * 64,
                    "predictions": "3" * 64,
                    "metrics": "4" * 64,
                    "evaluation_manifest": "5" * 64,
                    "generator_manifest": "6" * 64,
                },
                "tokenizer_identity": {
                    "tokenizer_name_or_path": group[
                        "expected_tokenizer_name_or_path"
                    ],
                    "tokenizer_revision": None,
                    "tokenizer_sha256": "7" * 64,
                },
                "tokenizer_identity_fingerprint": "8" * 64,
                "generator_identity_fingerprint": "9" * 64,
            }
        )
    analysis.write_formal_outputs(tmp_path, results)
    main_csv = tmp_path / "main_results_48.csv"
    assert len(main_csv.read_text(encoding="utf-8").splitlines()) == 49
    with main_csv.open(encoding="utf-8", newline="") as handle:
        first = next(csv.DictReader(handle))
    assert first["num_docs_0"] == "0"
    assert first["num_docs_2"] == str(groups[0]["expected_rows"])
    assert len((tmp_path / "accuracy_token_plot_data.csv").read_text(encoding="utf-8").splitlines()) == 49
    latex = (tmp_path / "main_results_48.tex").read_text(encoding="utf-8")
    assert "fever\\_binary\\_v2" in latex
    assert len(latex.splitlines()) == 48

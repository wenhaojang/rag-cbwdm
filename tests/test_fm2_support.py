from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.baselines.bge_reranker import make_bge_selection
from src.baselines.infogain import infogain_multitask_loss, posterior_to_teacher_rows
from src.datasets.fm2 import (
    FM2_LABEL_MAPPING,
    adapt_raw_row,
    normalize_label,
    summarize_pool,
    validate_raw_row,
)
from src.diagnostics.signed_selector_v1 import greedy_select_without_gold
from src.diagnostics.signed_teacher_v1 import (
    build_signed_teacher_row,
    build_signed_training_groups,
)
from src.io_utils import write_jsonl
from src.metrics import ClassificationMetrics
from src.prompts import (
    FEVER_PROMPT_VERSION,
    FM2_PROMPT_VERSION,
    build_classification_prompt,
    build_fever_prompt,
    classification_prompt_hash,
    classification_prompt_version,
    fever_prompt_hash,
)
from src.selection_schema import make_selection_row
from src.selector_cross_encoder import cbwdm_multitask_loss

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["SUPPORTS", "REFUTES"]
VERBALIZERS = {
    "SUPPORTS": ["A", " A", "\nA"],
    "REFUTES": ["B", " B", "\nB"],
}


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location("fm2_test_" + name.replace(".", "_"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_row(*, identifier: str = "raw-1", label: str = "SUPPORTS", page: str = "Page A") -> dict:
    return {
        "category": "Science",
        "correct_votes": 2,
        "gold_evidence": [{"section_header": "History", "text": "Gold sentence."}],
        "id": identifier,
        "label": label,
        "retrieved_evidence": [
            {"section_header": "Lead", "text": "Distractor sentence."},
            {"section_header": "History", "text": "Gold sentence."},
            {"section_header": "Details", "text": "Another sentence."},
            {"section_header": "Later", "text": "Last sentence."},
        ],
        "text": "A claim about the page.",
        "total_likes": 1,
        "total_votes": 3,
        "wikipedia_page": page,
    }


def adapted(label: str = "SUPPORTS") -> tuple[dict, dict, dict]:
    return adapt_raw_row(raw_row(label=label), split="train", row_number=1)


def posterior_row(split: str = "train") -> dict:
    query, pool, _ = adapted()
    candidates = []
    etas = ([0.80, 0.20], [0.30, 0.70], [0.65, 0.35], [0.52, 0.48])
    for candidate, eta in zip(pool["candidates"], etas):
        candidates.append({**candidate, "eta": list(eta)})
    return {
        "schema_version": "rag_cbwdm_posteriors.v2",
        "id": query["id"],
        "query": query["query"],
        "label": query["label"],
        "split": split,
        "labels": LABELS,
        "eta0": [0.5, 0.5],
        "candidates": candidates,
    }


def signed_params() -> dict:
    return {
        "top_m": 4,
        "stop_threshold": 0.001,
        "alignment_eps": 0.0,
        "b_plus": 0.01,
        "b_minus": 0.001,
        "neutral_sample_policy": "negative",
        "ridge_lambda": 0.01,
        "eps_smooth": 0.001,
        "l_type": "euclidean_posterior_shift",
        "target_smoothing": "paper_mixture",
        "gain_tolerance": 1e-10,
    }


def test_fm2_schema_adapter_preserves_raw_metadata_and_separates_gold() -> None:
    query, pool, diagnostic = adapted()
    assert query["id"] == "fm2:train:raw-1"
    assert query["query"] == "A claim about the page."
    assert query["metadata"]["original_label"] == "SUPPORTS"
    assert query["metadata"]["wikipedia_page"] == "Page A"
    assert len(pool["candidates"]) == 4
    assert pool["candidates"][0]["rank"] == 1
    assert pool["candidates"][0]["title"] == "Page A | Lead"
    assert "gold_evidence" not in pool
    assert "gold_evidence_keys" not in pool
    assert diagnostic["all_gold_in_official_pool"] is True


def test_fm2_label_normalization_covers_both_official_labels() -> None:
    assert FM2_LABEL_MAPPING == {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES"}
    assert normalize_label("SUPPORTS") == "SUPPORTS"
    assert normalize_label("REFUTES") == "REFUTES"
    with pytest.raises(ValueError, match="Unknown FM2 label"):
        normalize_label("TRUE")


def test_fm2_real_schema_validation_rejects_missing_and_malformed_fields() -> None:
    validate_raw_row(raw_row(), split="dev", row_number=1)
    missing = raw_row(); missing.pop("retrieved_evidence")
    with pytest.raises(ValueError, match="missing fields"):
        validate_raw_row(missing, split="dev", row_number=1)
    malformed = raw_row(); malformed["gold_evidence"] = [{"text": "missing header"}]
    with pytest.raises(ValueError, match="section_header/text"):
        validate_raw_row(malformed, split="dev", row_number=1)


def test_fm2_candidate_pool_parsing_and_statistics() -> None:
    _, pool, _ = adapted()
    stats = summarize_pool([pool])
    assert stats["candidate_count_min"] == 4
    assert stats["candidate_count_max"] == 4
    assert stats["rows_with_fewer_than_4"] == 0
    assert stats["rows_with_fewer_than_20"] == 1
    assert [candidate["source_rank"] for candidate in pool["candidates"]] == [1, 2, 3, 4]


def test_prompt_registry_keeps_fever_bytes_and_gives_fm2_distinct_contract() -> None:
    expected = (
        "You are a fact verification model.\n\nEvidence:\nA source.\n\nClaim:\n"
        "The sky is blue.\n\nChoose the correct label:\nA. SUPPORTS\nB. REFUTES\n\nAnswer:"
    )
    frozen_hash = "c8e654d06f541cda5a6d58445d4e60132ee922ada37e9cd5c7105982396ed518"
    assert classification_prompt_version("fever2") == FEVER_PROMPT_VERSION
    assert classification_prompt_version("fm2") == FM2_PROMPT_VERSION
    assert build_fever_prompt("The sky is blue.", LABELS, VERBALIZERS, "A source.") == expected
    assert build_classification_prompt("fever2", "The sky is blue.", LABELS, VERBALIZERS, "A source.") == expected
    assert build_classification_prompt("fm2", "The sky is blue.", LABELS, VERBALIZERS, "A source.") == expected
    assert fever_prompt_hash(LABELS, VERBALIZERS) == frozen_hash
    assert classification_prompt_hash("fever2", LABELS, VERBALIZERS) == frozen_hash
    assert classification_prompt_hash("fm2", LABELS, VERBALIZERS) != frozen_hash


def test_fm2_posterior_one_row_smoke_uses_same_template_for_query_and_docs() -> None:
    module = load_script("03_compute_label_posteriors.py")
    _, pool, _ = adapted()

    class FakeScorer:
        def score_prompts(self, prompts, batch_size, labels, verbalizers):
            assert len(prompts) == 5
            assert prompts[0].startswith("You are a fact verification model.")
            assert "Evidence:" not in prompts[0]
            assert all("Evidence:" in prompt for prompt in prompts[1:])
            return np.tile(np.array([[0.6, 0.4]], dtype=np.float32), (len(prompts), 1))

    result, count = module.score_retrieval_row(
        pool, FakeScorer(), LABELS, VERBALIZERS, batch_size=4,
        max_candidates=None, dataset="fm2",
    )
    assert count == 4
    assert result["split"] == "train"
    assert result["eta0"] == pytest.approx([0.6, 0.4])
    assert len(result["candidates"]) == 4


def test_fm2_signed_teacher_and_training_group_smoke(tmp_path: Path) -> None:
    posterior = posterior_row()
    teacher = build_signed_teacher_row(posterior, signed_params())
    assert teacher["uses_gold_for_teacher"] is True
    assert teacher["variant"] == "signed_teacher_v1"
    teacher_path, posterior_path, retrieval_path = (
        tmp_path / "teacher.jsonl", tmp_path / "posteriors.jsonl", tmp_path / "retrieval.jsonl"
    )
    _, pool, _ = adapted()
    write_jsonl(teacher_path, [teacher]); write_jsonl(posterior_path, [posterior]); write_jsonl(retrieval_path, [pool])
    groups = build_signed_training_groups(
        teacher_path=teacher_path, posteriors_path=posterior_path, retrieval_path=retrieval_path
    )
    assert groups
    assert all(group.candidate_docs for group in groups)
    assert all(len(group.effective_gains) == len(group.candidate_docs) for group in groups)
    torch = pytest.importorskip("torch")
    group = groups[0]
    scores = torch.zeros(len(group.effective_gains), requires_grad=True)
    loss, details = cbwdm_multitask_loss(
        scores, group.effective_gains, b_plus=0.01, b_minus=0.001,
        gamma=1.0, beta=0.25, neutral_sample_policy="negative",
    )
    loss.backward()
    assert details["valid_ranking_group"] + details["skipped_ranking_group"] == 1


def test_fm2_signed_selector_inference_has_no_gold_argument() -> None:
    _, pool, _ = adapted()
    candidates = pool["candidates"]

    def score_state(query, selected, remaining):
        return [1.0 / int(candidate["rank"]) for candidate in remaining]

    first = greedy_select_without_gold(
        query=pool["query"], candidates=candidates, score_remaining=score_state,
        top_m=4, min_docs=0, score_threshold=0.0,
    )
    flipped = {**pool, "label": "REFUTES"}
    second = greedy_select_without_gold(
        query=flipped["query"], candidates=flipped["candidates"], score_remaining=score_state,
        top_m=4, min_docs=0, score_threshold=0.0,
    )
    assert [row["doc_id"] for row in first["selected_docs"]] == [
        row["doc_id"] for row in second["selected_docs"]
    ]


def test_fm2_infogain_teacher_train_select_smoke() -> None:
    teacher_rows = posterior_to_teacher_rows(posterior_row(), purpose="training")
    assert len(teacher_rows) == 4
    assert all(row["teacher_definition"] == "eta_gold_doc_minus_eta_gold_query" for row in teacher_rows)
    torch = pytest.importorskip("torch")
    rank_scores = torch.tensor([0.8, -0.2, 0.4, 0.1], requires_grad=True)
    filter_logits = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]], requires_grad=True)
    loss, details = infogain_multitask_loss(
        rank_scores, filter_logits, [row["dig"] for row in teacher_rows],
        b_pos=0.1, b_neg=-0.1, beta=0.75,
    )
    loss.backward()
    assert details["num_pairs"] > 0

    module = load_script("12c_select_infogain_reranker.py")
    _, pool, _ = adapted()

    class FakeModel:
        def score(self, texts, batch_size):
            return [0.1, 0.9, 0.3, 0.2], [0.9, 0.8, 0.7, 0.6]

    selected = module.select_row(
        pool, FakeModel(), batch_size=4, top_m=4, filter_threshold=0.5,
        min_docs=2, checkpoint_metadata={"fingerprint": "test"},
        method_name="infogain_adapter",
    )
    assert selected["method"] == "infogain_adapter"
    assert selected["uses_gold_at_test"] is False


def test_fm2_naive_bge_and_no_evidence_smoke(tmp_path: Path) -> None:
    _, pool, _ = adapted()
    naive = load_script("08_select_naive_topm.py")
    retrieval_path = tmp_path / "retrieval.jsonl"
    write_jsonl(retrieval_path, [pool])
    naive_rows = list(naive.iter_selection_rows(
        retrieval_path, top_m=4, method_name="naive_top4", min_docs=0
    ))
    assert [doc["source_rank"] for doc in naive_rows[0]["selected_docs"]] == [1, 2, 3, 4]
    scores = {candidate["doc_id"]: float(5 - candidate["rank"]) for candidate in pool["candidates"]}
    bge = make_bge_selection(
        pool, scores, method="bge", top_m=4, score_threshold=None,
        min_docs=0, model_metadata={"model": "fake"},
    )
    assert len(bge["selected_docs"]) == 4
    no_evidence = make_selection_row(
        pool, method="no_evidence", selected_docs=[], selection_steps=[],
        stop_reason="no_evidence", max_docs=0,
        selection_metadata={"uses_gold_at_inference": False},
    )
    assert no_evidence["selected_doc_ids"] == []
    assert no_evidence["uses_gold_at_test"] is False


def test_fm2_evaluator_smoke_outputs_full_metrics(tmp_path: Path) -> None:
    module = load_script("07_eval_rag_classification.py")
    query, pool, _ = adapted()
    selection = {
        "id": query["id"], "query": query["query"], "label": query["label"],
        "split": "train", "method": "naive_top4", "selected_docs": pool["candidates"][:2],
    }
    path = tmp_path / "selection.jsonl"
    write_jsonl(path, [selection])

    class FakeScorer:
        def score_prompt(self, prompt, labels, verbalizers):
            assert "Evidence:" in prompt
            return [0.8, 0.2]

    metrics = ClassificationMetrics(labels=LABELS)
    rows = list(module.iter_prediction_rows(
        path, FakeScorer(), LABELS, VERBALIZERS, metrics, "naive_top4",
        log_every=100, dataset="fm2",
    ))
    result = metrics.compute()
    assert rows[0]["correct"] is True
    assert result["accuracy"] == 1.0
    assert "macro_f1" in result and "per_class" in result and "confusion_matrix" in result
    assert result["avg_num_docs"] == 2.0
    assert result["avg_evidence_chars"] > 0
    assert result["prediction_distribution"]["SUPPORTS"] == 1

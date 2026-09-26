from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.diagnostics.train_signed_selector_v2 import (
    TRAINING_MANIFEST_SCHEMA,
    build_completed_manifest,
)
from src.diagnostics.signed_teacher_v1 import build_signed_teacher_row
from src.diagnostics.signed_selector_v2 import (
    build_signed_v2_training_groups,
    gate_separability_audit,
    gate_targets_from_alignments,
    greedy_select_dual_without_gold,
    harmful_selection_audit,
    select_row_without_gold,
    signed_v2_loss,
    stopping_and_budget_audit,
)
from src.diagnostics.signed_v2_model import (
    SIGNED_V2_ARCHITECTURE,
    SIGNED_V2_CHECKPOINT_SCHEMA,
    SIGNED_V2_METHOD,
    DualHeadLogits,
    SignedV2DualHeadSelector,
)


class DummyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=4,
            hidden_dropout_prob=0.0,
            initializer_range=0.02,
        )
        self.embedding = torch.nn.Embedding(32, 4)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=hidden[:, 0])

    def save_pretrained(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination / "dummy_encoder.pt")
        (destination / "config.json").write_text(
            json.dumps({"model_type": "dummy"}), encoding="utf-8"
        )


class DummyTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, max_length, return_tensors
        rows = [
            [len(text) % 31 + 1, sum(ord(char) for char in text) % 31 + 1]
            for text in texts
        ]
        input_ids = torch.tensor(rows, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def save_pretrained(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "DummyTokenizer"}), encoding="utf-8"
        )


class DummyAutoModel:
    @classmethod
    def from_pretrained(cls, path: str | Path) -> DummyEncoder:
        encoder = DummyEncoder()
        try:
            state = torch.load(
                Path(path) / "dummy_encoder.pt", map_location="cpu", weights_only=True
            )
        except TypeError:  # pragma: no cover
            state = torch.load(Path(path) / "dummy_encoder.pt", map_location="cpu")
        encoder.load_state_dict(state)
        return encoder


class DummyAutoTokenizer:
    @classmethod
    def from_pretrained(cls, path: str | Path) -> DummyTokenizer:
        if not (Path(path) / "tokenizer_config.json").is_file():
            raise FileNotFoundError(path)
        return DummyTokenizer()


def _selector(seed: int = 13) -> SignedV2DualHeadSelector:
    torch.manual_seed(7)
    return SignedV2DualHeadSelector(
        "dummy",
        device="cpu",
        head_seed=seed,
        encoder=DummyEncoder(),
        tokenizer=DummyTokenizer(),
    )


def _candidates() -> list[dict]:
    return [
        {"doc_id": "d1", "rank": 1, "title": "T1", "text": "one"},
        {"doc_id": "d2", "rank": 2, "title": "T2", "text": "two"},
        {"doc_id": "d3", "rank": 3, "title": "T3", "text": "three"},
    ]


class SignedSelectorV2Tests(unittest.TestCase):
    def test_dual_head_forward_shapes_and_independent_parameters(self) -> None:
        selector = _selector()
        output = selector(input_ids=torch.tensor([[1, 2], [3, 4]]))
        self.assertEqual(output.gate_logit.shape, (2,))
        self.assertEqual(output.utility_logit.shape, (2,))
        self.assertNotEqual(
            selector.gate_head.weight.data_ptr(), selector.utility_head.weight.data_ptr()
        )

    def test_gate_target_is_strict_authoritative_sign(self) -> None:
        self.assertEqual(
            gate_targets_from_alignments([0.2, 0.0, -0.1], alignment_eps=0.0),
            [1.0, 0.0, 0.0],
        )

    def test_training_group_reconstructs_authoritative_alignment(self) -> None:
        row = {
            "id": "q",
            "query": "claim",
            "label": "SUPPORTS",
            "split": "train_core",
            "labels": ["SUPPORTS", "REFUTES"],
            "eta0": [0.5, 0.5],
            "candidates": [
                {"doc_id": "d1", "rank": 1, "title": "T1", "text": "one", "eta": [0.7, 0.3]},
                {"doc_id": "d2", "rank": 2, "title": "T2", "text": "two", "eta": [0.3, 0.7]},
            ],
        }
        params = {
            "top_m": 1,
            "stop_threshold": 0.0,
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
        teacher = build_signed_teacher_row(row, params)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            posterior_path = root / "posteriors.jsonl"
            teacher_path = root / "teacher.jsonl"
            posterior_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            teacher_path.write_text(json.dumps(teacher) + "\n", encoding="utf-8")
            groups = build_signed_v2_training_groups(
                teacher_path=teacher_path,
                posteriors_path=posterior_path,
            )
        self.assertEqual(groups[0].gate_targets, [1.0, 0.0])
        self.assertGreater(groups[0].alignments[0], 0.0)
        self.assertLess(groups[0].alignments[1], 0.0)

    def test_harmful_candidate_is_excluded_from_utility_loss_and_ranking(self) -> None:
        gate = torch.tensor([-0.5, 0.5, 0.5])
        first, first_details = signed_v2_loss(
            gate,
            torch.tensor([100.0, 0.3, -0.2]),
            alignments=[-0.1, 0.2, 0.1],
            effective_gains=[0.0, 0.02, 0.0005],
        )
        second, second_details = signed_v2_loss(
            gate,
            torch.tensor([-100.0, 0.3, -0.2]),
            alignments=[-0.1, 0.2, 0.1],
            effective_gains=[0.0, 0.02, 0.0005],
        )
        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(
            torch.allclose(
                first_details["utility_loss"], second_details["utility_loss"]
            )
        )
        self.assertEqual(first_details["num_utility_candidates"], 2)
        self.assertTrue(first_details["valid_ranking_group"])

    def test_no_admissible_state_trains_gate_without_nan(self) -> None:
        gate = torch.tensor([0.2, -0.4], requires_grad=True)
        utility = torch.tensor([10.0, -10.0], requires_grad=True)
        loss, details = signed_v2_loss(
            gate,
            utility,
            alignments=[0.0, -0.1],
            effective_gains=[0.0, 0.0],
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(details["utility_loss"].detach()), 0.0)
        self.assertGreater(float(details["gate_loss"].detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(gate.grad)
        self.assertTrue(torch.all(utility.grad == 0))

    def test_inference_gates_before_utility_ranking(self) -> None:
        result = greedy_select_dual_without_gold(
            query="claim",
            candidates=_candidates(),
            score_remaining=lambda query, selected, remaining: (
                [-1.0, 1.0, 1.0][: len(remaining)],
                [100.0, 2.0, 1.0][: len(remaining)],
            ),
            top_m=1,
        )
        self.assertEqual(result["selected_docs"][0]["doc_id"], "d2")

    def test_no_predicted_admissible_stops(self) -> None:
        result = greedy_select_dual_without_gold(
            query="claim",
            candidates=_candidates()[:2],
            score_remaining=lambda query, selected, remaining: (
                [-0.1] * len(remaining),
                [10.0] * len(remaining),
            ),
        )
        self.assertEqual(result["selected_docs"], [])
        self.assertEqual(result["stop_reason"], "no_predicted_admissible_candidate")
        self.assertTrue(result["selection_steps"][0]["stop"])

    def test_utility_below_threshold_stops(self) -> None:
        result = greedy_select_dual_without_gold(
            query="claim",
            candidates=_candidates()[:2],
            score_remaining=lambda query, selected, remaining: (
                [0.1] * len(remaining),
                [-0.1, -0.2][: len(remaining)],
            ),
        )
        self.assertEqual(result["selected_docs"], [])
        self.assertEqual(result["stop_reason"], "utility_below_threshold")

    def test_top_m_stops_after_budget(self) -> None:
        result = greedy_select_dual_without_gold(
            query="claim",
            candidates=_candidates(),
            score_remaining=lambda query, selected, remaining: (
                [1.0] * len(remaining),
                list(reversed(range(1, len(remaining) + 1))),
            ),
            top_m=2,
        )
        self.assertEqual(len(result["selected_docs"]), 2)
        self.assertEqual(result["stop_reason"], "top_m_reached")

    def test_deterministic_tie_break_prefers_lower_source_rank(self) -> None:
        result = greedy_select_dual_without_gold(
            query="claim",
            candidates=[_candidates()[1], _candidates()[0]],
            score_remaining=lambda query, selected, remaining: (
                [1.0] * len(remaining),
                [1.0] * len(remaining),
            ),
            top_m=1,
        )
        self.assertEqual(result["selected_docs"][0]["doc_id"], "d1")

    def test_inference_feature_and_action_are_gold_free(self) -> None:
        class RecordingSelector:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def score_texts(self, texts, *, batch_size, requires_grad):
                del batch_size, requires_grad
                self.calls.append(list(texts))
                count = len(texts)
                return DualHeadLogits(
                    gate_logit=torch.ones(count),
                    utility_logit=torch.arange(count, 0, -1, dtype=torch.float32),
                )

        base = {
            "id": "q",
            "query": "claim",
            "split": "validation",
            "candidates": _candidates(),
        }
        first_selector = RecordingSelector()
        second_selector = RecordingSelector()
        supports = select_row_without_gold(
            {**base, "label": "SUPPORTS"}, first_selector, top_m=1
        )
        refutes = select_row_without_gold(
            {**base, "label": "REFUTES"}, second_selector, top_m=1
        )
        self.assertEqual(supports["selected_doc_ids"], refutes["selected_doc_ids"])
        self.assertEqual(supports["selection_steps"], refutes["selection_steps"])
        self.assertEqual(first_selector.calls, second_selector.calls)
        self.assertFalse(supports["uses_gold_at_test"])
        self.assertFalse(supports["selection_metadata"]["uses_gold_at_inference"])
        self.assertEqual(
            supports["selected_docs"][0]["selector_score_alias"], "utility_score"
        )

    def test_checkpoint_round_trip_preserves_both_logits(self) -> None:
        selector = _selector(seed=19)
        selector.eval()
        texts = ["alpha", "beta"]
        before = selector.score_texts(texts)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint"
            selector.save_checkpoint(
                checkpoint, training_contract={"alignment_eps": 0.0}
            )
            with patch(
                "src.diagnostics.signed_v2_model._import_transformers",
                return_value=(DummyAutoModel, DummyAutoTokenizer),
            ):
                loaded = SignedV2DualHeadSelector.load_checkpoint(
                    checkpoint, device="cpu"
                )
            loaded.eval()
            after = loaded.score_texts(texts)
            metadata = json.loads(
                (checkpoint / "signed_v2_config.json").read_text(encoding="utf-8")
            )
        self.assertTrue(torch.allclose(before.gate_logit, after.gate_logit))
        self.assertTrue(torch.allclose(before.utility_logit, after.utility_logit))
        self.assertEqual(metadata["schema_version"], SIGNED_V2_CHECKPOINT_SCHEMA)
        self.assertEqual(metadata["method"], SIGNED_V2_METHOD)
        self.assertEqual(metadata["architecture"], SIGNED_V2_ARCHITECTURE)

    def test_training_manifest_is_explicitly_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "training_config.json"
            output.write_text("{}\n", encoding="utf-8")
            manifest = build_completed_manifest(
                contract={"variant": "signed_selector_v2_training"},
                fingerprint="fingerprint",
                checkpoint=root / "checkpoint",
                checkpoint_sha256="checkpoint-sha",
                outputs=[output],
                num_groups=3,
            )
        self.assertEqual(manifest["schema_version"], TRAINING_MANIFEST_SCHEMA)
        self.assertEqual(manifest["method"], SIGNED_V2_METHOD)
        self.assertEqual(manifest["architecture"], SIGNED_V2_ARCHITECTURE)
        self.assertNotIn("signed_selector_v1", json.dumps(manifest))

    def test_posthoc_reports_gate_harm_and_budget_without_inference_gold(self) -> None:
        posterior = {
            "q": {
                "id": "q",
                "label": "SUPPORTS",
                "labels": ["SUPPORTS", "REFUTES"],
                "eta0": [0.5, 0.5],
                "candidates": [
                    {"doc_id": "d1", "eta": [0.7, 0.3]},
                    {"doc_id": "d2", "eta": [0.3, 0.7]},
                ],
            }
        }
        selection = {
            "q": {
                "id": "q",
                "selected_doc_ids": ["d1"],
                "num_docs": 1,
                "stop_reason": "utility_below_threshold",
                "selection_steps": [
                    {
                        "step": 0,
                        "all_candidate_scores": [
                            {"doc_id": "d1", "gate_score": 1.0},
                            {"doc_id": "d2", "gate_score": -1.0},
                        ],
                    }
                ],
            }
        }
        cbwdm = {
            "L_type": "euclidean_posterior_shift",
            "eps_smooth": 0.001,
            "target_smoothing": "paper_mixture",
        }
        separability = gate_separability_audit(
            selection_rows=selection,
            posterior_rows=posterior,
            cbwdm=cbwdm,
        )
        harmful = harmful_selection_audit(
            selection_rows=selection,
            posterior_rows=posterior,
            prediction_rows={"q": {"id": "q", "pred": "SUPPORTS"}},
            cbwdm=cbwdm,
        )
        budget = stopping_and_budget_audit(selection)
        self.assertEqual(
            separability["gold=SUPPORTS/step=0"]["authoritative_pos_vs_nonpos"]["auroc"],
            1.0,
        )
        self.assertEqual(harmful["gold=SUPPORTS/correct"]["raw_negative"], 0)
        self.assertEqual(budget["avg_docs"], 1.0)
        self.assertEqual(budget["stop_reason_counts"]["utility_below_threshold"], 1)


if __name__ == "__main__":
    unittest.main()

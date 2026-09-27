from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import scripts.diagnostics.train_signed_selector_v22 as train_v22
from scripts.diagnostics.analyze_signed_selector_v22 import (
    _add_later_only,
    _supports_error_probabilities,
)
from scripts.diagnostics.train_signed_selector_v22 import (
    TRAINING_MANIFEST_SCHEMA,
    build_completed_manifest,
    create_selector,
)
from src.diagnostics.signed_selector_v1 import (
    greedy_select_without_gold as v1_greedy_select_without_gold,
)
from src.diagnostics.signed_selector_v22 import (
    SIGNED_V22_ARCHITECTURE,
    SIGNED_V22_METHOD,
    SIGNED_V22_VARIANT,
    greedy_select_without_gold,
    select_row_without_gold,
    signed_v22_multitask_loss,
    state_top_ordinal_loss,
)
from src.selector_cross_encoder import CrossEncoderSelector, cbwdm_multitask_loss


def _loss(
    scores: torch.Tensor,
    gains: list[float],
    classes: list[str],
    *,
    lambda_top: float = 0.25,
):
    return signed_v22_multitask_loss(
        scores,
        gains,
        classes,
        b_plus=0.01,
        b_minus=0.001,
        beta=0.25,
        gamma=1.0,
        neutral_sample_policy="negative",
        lambda_top=lambda_top,
        gamma_top=1.0,
    )


def _candidates() -> list[dict]:
    return [
        {"doc_id": "d1", "rank": 1, "title": "T1", "text": "one"},
        {"doc_id": "d2", "rank": 2, "title": "T2", "text": "two"},
        {"doc_id": "d3", "rank": 3, "title": "T3", "text": "three"},
    ]


class DummySequenceClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 4)
        self.classifier = torch.nn.Linear(4, 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        del attention_mask
        return SimpleNamespace(logits=self.classifier(self.embedding(input_ids)[:, 0]))

    def save_pretrained(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination / "pytorch_model.bin")
        (destination / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["DummySequenceClassifier"],
                    "num_labels": 1,
                }
            ),
            encoding="utf-8",
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
        values = [
            [len(text) % 63 + 1, sum(ord(char) for char in text) % 63 + 1]
            for text in texts
        ]
        input_ids = torch.tensor(values, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def save_pretrained(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "DummyTokenizer"}), encoding="utf-8"
        )


class DummyAutoModelForSequenceClassification:
    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs) -> DummySequenceClassifier:
        del kwargs
        torch.manual_seed(41)
        model = DummySequenceClassifier()
        weights = Path(path) / "pytorch_model.bin"
        if weights.is_file():
            try:
                state = torch.load(weights, map_location="cpu", weights_only=True)
            except TypeError:  # pragma: no cover
                state = torch.load(weights, map_location="cpu")
            model.load_state_dict(state)
        return model


class DummyAutoTokenizer:
    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs) -> DummyTokenizer:
        del path, kwargs
        return DummyTokenizer()


class SignedSelectorV22Tests(unittest.TestCase):
    def test_lambda_zero_exactly_reduces_to_v1_base_loss(self) -> None:
        scores = torch.tensor([0.3, -0.2, -0.5])
        gains = [0.02, 0.0005, 0.0]
        classes = ["positive", "negative", "explicit_harmful_negative"]
        base, _ = cbwdm_multitask_loss(
            scores,
            gains,
            b_plus=0.01,
            b_minus=0.001,
            beta=0.25,
            gamma=1.0,
            neutral_sample_policy="negative",
        )
        total, details = _loss(scores, gains, classes, lambda_top=0.0)
        self.assertTrue(torch.allclose(total, base, rtol=0.0, atol=0.0))
        self.assertTrue(torch.allclose(details["base_loss"], base, rtol=0.0, atol=0.0))

    def test_positive_and_harmful_form_valid_top_competition(self) -> None:
        loss, details = state_top_ordinal_loss(
            torch.tensor([0.2, 1.0]),
            ["positive", "explicit_harmful_negative"],
        )
        self.assertGreater(float(loss), 0.0)
        self.assertTrue(details["top_valid_ranking_group"])

    def test_ordinary_negative_and_harmful_form_valid_top_competition(self) -> None:
        loss, details = state_top_ordinal_loss(
            torch.tensor([0.2, 1.0]),
            ["negative", "explicit_harmful_negative"],
        )
        self.assertGreater(float(loss), 0.0)
        self.assertEqual(details["top_competition_count"], 1)

    def test_neutral_and_harmful_form_valid_top_competition(self) -> None:
        loss, details = state_top_ordinal_loss(
            torch.tensor([0.2, 1.0]),
            ["neutral", "explicit_harmful_negative"],
        )
        self.assertGreater(float(loss), 0.0)
        self.assertEqual(details["admissible_count"], 1)

    def test_no_harmful_is_differentiable_zero(self) -> None:
        scores = torch.tensor([0.0, 1.0], requires_grad=True)
        loss, details = state_top_ordinal_loss(scores, ["positive", "neutral"])
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertFalse(details["top_valid_ranking_group"])
        self.assertIsNone(details["top_gap"])
        loss.backward()
        self.assertTrue(torch.all(scores.grad == 0))

    def test_no_admissible_is_differentiable_zero(self) -> None:
        scores = torch.tensor([0.0, 1.0], requires_grad=True)
        loss, details = state_top_ordinal_loss(
            scores,
            ["explicit_harmful_negative", "explicit_harmful_negative"],
        )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(details["top_skipped_ranking_group"])
        loss.backward()
        self.assertTrue(torch.all(scores.grad == 0))

    def test_empty_state_is_stable_differentiable_zero(self) -> None:
        scores = torch.empty(0, requires_grad=True)
        loss, details = state_top_ordinal_loss(scores, [])
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertFalse(details["top_valid_ranking_group"])
        loss.backward()
        self.assertEqual(scores.grad.numel(), 0)

    def test_best_harmful_gradient_is_positive(self) -> None:
        scores = torch.tensor([-1.0, 0.2, -0.5, 1.0], requires_grad=True)
        loss, _ = state_top_ordinal_loss(
            scores,
            ["negative", "positive", "explicit_harmful_negative", "explicit_harmful_negative"],
        )
        loss.backward()
        self.assertGreater(float(scores.grad[3]), 0.0)

    def test_best_admissible_gradient_is_negative(self) -> None:
        scores = torch.tensor([-1.0, 0.2, -0.5, 1.0], requires_grad=True)
        loss, _ = state_top_ordinal_loss(
            scores,
            ["negative", "positive", "explicit_harmful_negative", "explicit_harmful_negative"],
        )
        loss.backward()
        self.assertLess(float(scores.grad[1]), 0.0)

    def test_nonmax_harmful_auxiliary_gradient_is_zero(self) -> None:
        scores = torch.tensor([-1.0, 0.2, -0.5, 1.0], requires_grad=True)
        loss, _ = state_top_ordinal_loss(
            scores,
            ["negative", "positive", "explicit_harmful_negative", "explicit_harmful_negative"],
        )
        loss.backward()
        self.assertEqual(float(scores.grad[2]), 0.0)

    def test_nonmax_admissible_auxiliary_gradient_is_zero(self) -> None:
        scores = torch.tensor([-1.0, 0.2, -0.5, 1.0], requires_grad=True)
        loss, _ = state_top_ordinal_loss(
            scores,
            ["negative", "positive", "explicit_harmful_negative", "explicit_harmful_negative"],
        )
        loss.backward()
        self.assertEqual(float(scores.grad[0]), 0.0)

    def test_loss_is_finite(self) -> None:
        scores = torch.tensor([-100.0, 100.0, 0.0])
        total, details = _loss(
            scores,
            [0.0005, 0.0, 0.005],
            ["negative", "explicit_harmful_negative", "neutral"],
        )
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(details["top_rank_loss"]))

    def test_total_is_base_plus_weighted_top_loss(self) -> None:
        scores = torch.tensor([0.0, 1.0])
        total, details = _loss(
            scores,
            [0.0005, 0.0],
            ["negative", "explicit_harmful_negative"],
        )
        expected = details["base_loss"] + 0.25 * details["top_rank_loss"]
        self.assertTrue(torch.allclose(total, expected))

    def test_top_gap_is_best_admissible_minus_best_harmful(self) -> None:
        _, details = state_top_ordinal_loss(
            torch.tensor([-1.0, 0.2, -0.5, 1.0]),
            ["negative", "positive", "explicit_harmful_negative", "explicit_harmful_negative"],
        )
        self.assertAlmostEqual(details["best_admissible_score"], 0.2, places=6)
        self.assertAlmostEqual(details["best_harmful_score"], 1.0, places=6)
        self.assertAlmostEqual(details["top_gap"], -0.8, places=6)

    def test_permutation_invariance(self) -> None:
        scores = torch.tensor([-1.0, 0.2, -0.5, 1.0])
        classes = ["negative", "positive", "explicit_harmful_negative", "explicit_harmful_negative"]
        first, _ = state_top_ordinal_loss(scores, classes)
        permutation = torch.tensor([3, 0, 2, 1])
        second, _ = state_top_ordinal_loss(
            scores[permutation], [classes[index] for index in permutation.tolist()]
        )
        self.assertTrue(torch.allclose(first, second, rtol=0.0, atol=0.0))

    def test_monotonicity_in_both_top_scores(self) -> None:
        baseline, _ = state_top_ordinal_loss(
            torch.tensor([0.2, 1.0]), ["neutral", "explicit_harmful_negative"]
        )
        lower_harmful, _ = state_top_ordinal_loss(
            torch.tensor([0.2, 0.5]), ["neutral", "explicit_harmful_negative"]
        )
        higher_admissible, _ = state_top_ordinal_loss(
            torch.tensor([0.7, 1.0]), ["neutral", "explicit_harmful_negative"]
        )
        self.assertLess(float(lower_harmful), float(baseline))
        self.assertLess(float(higher_admissible), float(baseline))

    def test_unknown_supervision_class_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown signed supervision classes"):
            state_top_ordinal_loss(torch.tensor([0.0]), ["mystery"])

    def test_v22_inference_is_exact_v1_delegation(self) -> None:
        def score(query, selected, remaining):
            del query, selected
            return [3.0 - int(candidate["rank"]) for candidate in remaining]

        kwargs = {
            "query": "claim",
            "candidates": _candidates(),
            "score_remaining": score,
            "top_m": 2,
            "min_docs": 0,
            "score_threshold": 0.0,
        }
        self.assertEqual(
            greedy_select_without_gold(**kwargs),
            v1_greedy_select_without_gold(**kwargs),
        )

    def test_deterministic_tie_break_matches_v1(self) -> None:
        candidates = [_candidates()[1], _candidates()[0]]
        score = lambda query, selected, remaining: [1.0] * len(remaining)
        kwargs = {
            "query": "claim",
            "candidates": candidates,
            "score_remaining": score,
            "top_m": 1,
        }
        v1 = v1_greedy_select_without_gold(**kwargs)
        v22 = greedy_select_without_gold(**kwargs)
        self.assertEqual(v22, v1)
        self.assertEqual(v22["selected_docs"][0]["doc_id"], "d1")

    def test_score_threshold_stop_matches_v1(self) -> None:
        score = lambda query, selected, remaining: [-0.1] * len(remaining)
        kwargs = {
            "query": "claim",
            "candidates": _candidates(),
            "score_remaining": score,
            "top_m": 4,
            "min_docs": 0,
            "score_threshold": 0.0,
        }
        v1 = v1_greedy_select_without_gold(**kwargs)
        v22 = greedy_select_without_gold(**kwargs)
        self.assertEqual(v22, v1)
        self.assertEqual(v22["stop_reason"], "score_below_threshold")

    def test_gold_free_row_wrapper_changes_metadata_only(self) -> None:
        class RecordingSelector:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def score_texts(self, texts, batch_size, requires_grad):
                del batch_size, requires_grad
                self.calls.append(list(texts))
                return torch.arange(len(texts), 0, -1, dtype=torch.float32)

        base = {
            "id": "q",
            "query": "claim",
            "split": "validation",
            "candidates": _candidates(),
        }
        first_selector = RecordingSelector()
        second_selector = RecordingSelector()
        supports = select_row_without_gold(
            {**base, "label": "SUPPORTS"},
            first_selector,
            top_m=1,
            min_docs=0,
            score_threshold=0.0,
            batch_size=8,
            max_candidates=None,
        )
        refutes = select_row_without_gold(
            {**base, "label": "REFUTES"},
            second_selector,
            top_m=1,
            min_docs=0,
            score_threshold=0.0,
            batch_size=8,
            max_candidates=None,
        )
        self.assertEqual(supports["selected_doc_ids"], refutes["selected_doc_ids"])
        self.assertEqual(supports["selection_steps"], refutes["selection_steps"])
        self.assertEqual(first_selector.calls, second_selector.calls)
        self.assertEqual(supports["method"], SIGNED_V22_METHOD)
        self.assertEqual(
            supports["selection_metadata"]["variant"], SIGNED_V22_VARIANT
        )
        self.assertFalse(supports["selection_metadata"]["uses_gold_at_inference"])
        score_item = supports["selection_steps"][0]["all_candidate_scores"][0]
        self.assertEqual(set(score_item), {"doc_id", "score", "rank"})
        self.assertFalse(
            set(inspect.signature(greedy_select_without_gold).parameters)
            & {"gold", "label", "alignment", "eta", "oracle", "teacher"}
        )

    def test_train_factory_uses_only_pretrained_single_head_selector(self) -> None:
        sentinel = object()
        with patch.object(train_v22, "CrossEncoderSelector", return_value=sentinel) as model:
            result = create_selector(
                model_name="ms-marco-MiniLM-L-6-v2",
                max_length=512,
                device="cpu",
                seed=13,
            )
        self.assertIs(result, sentinel)
        model.assert_called_once_with(
            model_name="ms-marco-MiniLM-L-6-v2",
            max_length=512,
            device="cpu",
        )
        source = inspect.getsource(train_v22)
        self.assertNotIn("SignedV2DualHeadSelector", source)
        self.assertNotIn("torch.nn.Linear", source)

    def test_cross_encoder_checkpoint_round_trip_preserves_scalar_classifier(self) -> None:
        import src.selector_cross_encoder as cross_encoder_module

        fake_import = (
            DummyAutoModelForSequenceClassification,
            DummyAutoTokenizer,
        )
        with patch.object(
            cross_encoder_module, "_import_transformers", return_value=fake_import
        ):
            selector = CrossEncoderSelector("dummy-source", device="cpu")
            selector.model.eval()
            before = selector.score_texts(["alpha", "beta"])
            classifier_before = selector.model.classifier.weight.detach().clone()
            with tempfile.TemporaryDirectory() as temp:
                checkpoint = Path(temp) / "checkpoint"
                selector.save_checkpoint(
                    checkpoint,
                    extra_config={
                        "variant": SIGNED_V22_VARIANT,
                        "method": SIGNED_V22_METHOD,
                        "architecture": SIGNED_V22_ARCHITECTURE,
                    },
                )
                loaded = CrossEncoderSelector.load_checkpoint(checkpoint, device="cpu")
                loaded.model.eval()
                after = loaded.score_texts(["alpha", "beta"])
                metadata = json.loads(
                    (checkpoint / "selector_config.json").read_text(encoding="utf-8")
                )
        self.assertTrue(torch.allclose(before, after))
        self.assertTrue(
            torch.allclose(classifier_before, loaded.model.classifier.weight.detach())
        )
        self.assertEqual(metadata["variant"], SIGNED_V22_VARIANT)
        self.assertEqual(metadata["architecture"], SIGNED_V22_ARCHITECTURE)

    def test_training_manifest_identity_is_v22(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "training_config.json"
            output.write_text("{}\n", encoding="utf-8")
            manifest = build_completed_manifest(
                contract={"variant": "signed_selector_v22_training"},
                fingerprint="fingerprint",
                checkpoint=root / "checkpoint",
                checkpoint_sha256="checkpoint-sha",
                outputs=[output],
                num_groups=2,
            )
        self.assertEqual(manifest["schema_version"], TRAINING_MANIFEST_SCHEMA)
        self.assertEqual(manifest["method"], SIGNED_V22_METHOD)
        self.assertEqual(manifest["variant"], SIGNED_V22_VARIANT)
        self.assertEqual(manifest["architecture"], SIGNED_V22_ARCHITECTURE)

    def test_supports_posthoc_probabilities_and_later_only(self) -> None:
        groups = _add_later_only(
            {
                "gold=SUPPORTS": {
                    "queries": 10,
                    "queries_with_selected_docs": 8,
                    "queries_with_any_raw_negative": 6,
                    "first_selected_raw_negative": 4,
                },
                "gold=SUPPORTS/wrong": {
                    "queries": 5,
                    "queries_with_selected_docs": 4,
                    "queries_with_any_raw_negative": 4,
                    "first_selected_raw_negative": 3,
                },
            }
        )
        result = _supports_error_probabilities(groups)
        self.assertEqual(groups["gold=SUPPORTS"]["later_only_raw_negative"], 2)
        self.assertEqual(result["later_only_raw_negative"], 2)
        self.assertAlmostEqual(result["p_wrong_given_any_raw_negative"], 4 / 6)
        self.assertAlmostEqual(result["p_wrong_given_no_raw_negative"], 1 / 4)
        self.assertAlmostEqual(result["p_wrong_given_first_raw_negative"], 3 / 4)
        self.assertAlmostEqual(result["p_wrong_given_first_raw_nonnegative"], 1 / 4)


if __name__ == "__main__":
    unittest.main()

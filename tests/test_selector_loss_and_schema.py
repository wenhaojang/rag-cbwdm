from __future__ import annotations

import unittest
import torch

from src.selection_schema import (
    make_selection_row,
    normalize_selected_doc,
    validate_selection_row,
)
from src.selector_cross_encoder import cbwdm_multitask_loss, listwise_distillation_loss


class SelectorTests(unittest.TestCase):
 def test_multitask_loss_and_gradient(self) -> None:
    scores = torch.tensor([0.3, -0.2, 0.1], requires_grad=True)
    loss, stats = cbwdm_multitask_loss(
        scores,
        [0.2, 0.0001, 0.005],
        b_plus=0.01,
        b_minus=0.001,
        gamma=2.0,
        beta=0.5,
        neutral_sample_policy="ignore",
    )
    self.assertTrue(torch.isfinite(loss))
    self.assertEqual((stats["num_positive"], stats["num_negative"], stats["num_neutral"]), (1, 1, 1))
    self.assertTrue(stats["valid_ranking_group"])
    loss.backward()
    self.assertTrue(torch.isfinite(scores.grad).all())


 def test_multitask_missing_positive_or_negative_is_safe(self) -> None:
  for gains in [[0.2, 0.3], [0.0, 0.0001]]:
    scores = torch.tensor([0.0, 0.0], requires_grad=True)
    loss, stats = cbwdm_multitask_loss(
        scores, gains, b_plus=0.01, b_minus=0.001
    )
    self.assertTrue(torch.isfinite(loss))
    self.assertTrue(stats["skipped_ranking_group"])


 def test_invalid_thresholds_and_legacy_listwise(self) -> None:
    scores = torch.tensor([0.0, 1.0])
    with self.assertRaises(ValueError):
        cbwdm_multitask_loss(scores, [0.0, 1.0], b_plus=0.001, b_minus=0.01)
    self.assertTrue(torch.isfinite(listwise_distillation_loss(scores, [0.0, 1.0])))


 def test_unified_selection_schema(self) -> None:
    source = {"id": "q", "query": "c", "label": "SUPPORTS", "split": "dev"}
    doc = normalize_selected_doc(
        {"doc_id": "d", "title": "t", "text": "x", "rank": 1, "score": 2.0},
        selector_score=0.4,
        selection_step=0,
    )
    row = make_selection_row(
        source,
        method="selector",
        selected_docs=[doc],
        selection_steps=[{"step": 0, "selected_doc_id": "d"}],
        stop_reason="top_m_reached",
    )
    validate_selection_row(row)
    self.assertEqual(row["num_docs"], 1)
    self.assertTrue(row["deployable"])

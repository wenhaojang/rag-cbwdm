from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from src.label_logits import LabelLogitScorer
from src.run_manifest import stable_hash, validate_resume_manifest


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return {"A": [5], "B": [6], " A": [5]}.get(text, [2, 3])

    def __call__(self, prompts, padding=True, return_tensors="pt", **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        rows = [[1] + [2 + (ord(ch) % 2) for ch in prompt] for prompt in prompts]
        width = max(map(len, rows))
        ids, masks = [], []
        for row in rows:
            pad = width - len(row)
            ids.append(row + [0] * pad)
            masks.append([1] * len(row) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 2)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 10)
        logits[:, :, 5] = input_ids.float()
        logits[:, :, 6] = -input_ids.float()
        return SimpleNamespace(logits=logits)


def fake_scorer() -> LabelLogitScorer:
    scorer = LabelLogitScorer.__new__(LabelLogitScorer)
    scorer.torch = torch
    scorer.tokenizer = FakeTokenizer()
    scorer.model = FakeModel()
    scorer.max_length = None
    scorer._verbalizer_cache = {}
    return scorer


class LabelAndManifestTests(unittest.TestCase):
 def test_batched_equals_single_and_normalized(self) -> None:
    scorer = fake_scorer()
    prompts = ["x", "longer", "zz"]
    labels = ["SUPPORTS", "REFUTES"]
    verbalizers = {"SUPPORTS": ["A", " A"], "REFUTES": ["B"]}
    batched = scorer.score_prompts(prompts, 3, labels, verbalizers)
    singles = np.asarray(
        [scorer.score_prompt(prompt, labels, verbalizers) for prompt in prompts],
        dtype=np.float32,
    )
    np.testing.assert_allclose(batched, singles, atol=1e-7)
    np.testing.assert_allclose(batched.sum(axis=1), 1.0, atol=1e-6)
    self.assertEqual(batched.dtype, np.float32)


 def test_manifest_resume_fingerprint_validation(self) -> None:
    fingerprint = stable_hash({"model": "m", "prompt": "v1"})
    validate_resume_manifest({"fingerprint": fingerprint}, fingerprint, stage="posterior")
    with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
        validate_resume_manifest(
            {"fingerprint": fingerprint}, stable_hash({"model": "m2"}), stage="posterior"
        )

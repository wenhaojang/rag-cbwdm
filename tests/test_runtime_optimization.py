from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.baselines.infogain import infogain_multitask_loss


ROOT = Path(__file__).resolve().parents[1]


def load_signed_trainer():
    path = ROOT / "scripts/preformal/26_train_signed_v1.py"
    spec = importlib.util.spec_from_file_location("runtime_signed_trainer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_infogain_loss(
    rank_scores: torch.Tensor,
    filter_logits: torch.Tensor,
    digs: list[float],
    *,
    b_pos: float,
    b_neg: float,
    beta: float,
):
    pair_losses = []
    for left in range(len(digs)):
        for right in range(left + 1, len(digs)):
            if digs[left] == digs[right]:
                continue
            high, low = (left, right) if digs[left] > digs[right] else (right, left)
            pair_losses.append(F.softplus(-(rank_scores[high] - rank_scores[low])))
    rank_loss = (
        torch.stack(pair_losses).mean()
        if pair_losses
        else rank_scores.sum() * 0.0
    )
    indices = []
    targets = []
    for index, dig in enumerate(digs):
        if dig >= b_pos:
            indices.append(index)
            targets.append(1)
        elif dig <= b_neg:
            indices.append(index)
            targets.append(0)
    filter_loss = (
        F.cross_entropy(
            filter_logits[torch.tensor(indices)], torch.tensor(targets)
        )
        if indices
        else filter_logits.sum() * 0.0
    )
    return beta * rank_loss + (1.0 - beta) * filter_loss, rank_loss, filter_loss


@pytest.mark.parametrize("seed", range(20))
def test_vectorized_infogain_loss_matches_legacy_values_and_gradients(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    size = rng.randint(1, 14)
    digs = [rng.choice([-0.2, -0.05, 0.0, 0.0, 0.03, 0.2]) for _ in range(size)]
    rank_legacy = torch.randn(size, generator=generator, dtype=torch.float32, requires_grad=True)
    logits_legacy = torch.randn(size, 2, generator=generator, dtype=torch.float32, requires_grad=True)
    rank_new = rank_legacy.detach().clone().requires_grad_(True)
    logits_new = logits_legacy.detach().clone().requires_grad_(True)
    legacy = legacy_infogain_loss(
        rank_legacy,
        logits_legacy,
        digs,
        b_pos=0.1,
        b_neg=-0.1,
        beta=0.75,
    )
    optimized, details = infogain_multitask_loss(
        rank_new,
        logits_new,
        digs,
        b_pos=0.1,
        b_neg=-0.1,
        beta=0.75,
    )
    legacy_grads = torch.autograd.grad(legacy[0], (rank_legacy, logits_legacy))
    optimized_grads = torch.autograd.grad(optimized, (rank_new, logits_new))
    torch.testing.assert_close(optimized, legacy[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(details["rank_loss"], legacy[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(details["filter_loss"], legacy[2], atol=1e-6, rtol=1e-6)
    for actual, expected in zip(optimized_grads, legacy_grads):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_vectorized_infogain_no_valid_pairs_keeps_differentiable_zero() -> None:
    rank = torch.tensor([0.1, -0.2, 0.3], requires_grad=True)
    logits = torch.randn(3, 2, requires_grad=True)
    total, details = infogain_multitask_loss(
        rank, logits, [0.0, 0.0, 0.0], b_pos=0.1, b_neg=-0.1, beta=1.0
    )
    total.backward()
    assert details["num_pairs"] == 0
    assert rank.grad is not None
    torch.testing.assert_close(rank.grad, torch.zeros_like(rank))


class ToySelector:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(2, 1, bias=True)
        self.tokenizer_calls = 0
        self.forward_calls = 0

    def encode_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:
        self.tokenizer_calls += 1
        values = [
            [float(len(text)), float(sum(ord(char) for char in text) % 997) / 997.0]
            for text in texts
        ]
        return {"features": torch.tensor(values, dtype=torch.float32)}

    def forward_encoded(
        self,
        encoded: dict[str, torch.Tensor],
        batch_size: int,
        requires_grad: bool,
    ) -> torch.Tensor:
        outputs = []
        for start in range(0, len(encoded["features"]), batch_size):
            self.forward_calls += 1
            outputs.append(self.model(encoded["features"][start : start + batch_size]).squeeze(-1))
        return torch.cat(outputs)

    def score_texts(
        self, texts: list[str], batch_size: int, requires_grad: bool
    ) -> torch.Tensor:
        return self.forward_encoded(
            self.encode_texts(texts), batch_size=batch_size, requires_grad=requires_grad
        )


def toy_groups(count: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            example_id=f"q{index}",
            query=f"claim {index}",
            selected_docs=[],
            candidate_docs=[
                {"title": "positive", "text": f"evidence {index}"},
                {"title": "negative", "text": f"counter {index}"},
                {"title": "neutral", "text": f"other {index}"},
            ],
            effective_gains=[0.02, -0.01, 0.005],
        )
        for index in range(count)
    ]


def test_signed_block_matches_legacy_updates_and_incomplete_normalization() -> None:
    trainer = load_signed_trainer()
    groups = toy_groups(10)
    legacy = ToySelector()
    optimized = ToySelector()
    optimized.model.load_state_dict(legacy.model.state_dict())
    legacy_optimizer = torch.optim.SGD(legacy.model.parameters(), lr=1e-4)
    optimized_optimizer = torch.optim.SGD(optimized.model.parameters(), lr=1e-4)

    legacy_optimizer.zero_grad(set_to_none=True)
    legacy_steps = 0
    for index, group in enumerate(groups, start=1):
        texts = trainer.group_texts(group)
        scores = legacy.score_texts(texts, batch_size=len(texts), requires_grad=True)
        loss, _ = trainer.signed_group_loss(scores, group)
        (loss / trainer.OPTIMIZER_GROUP_BATCH_SIZE).backward()
        if index % trainer.OPTIMIZER_GROUP_BATCH_SIZE == 0 or index == len(groups):
            legacy_optimizer.step()
            legacy_optimizer.zero_grad(set_to_none=True)
            legacy_steps += 1

    optimized_optimizer.zero_grad(set_to_none=True)
    blocks = trainer.optimizer_blocks(groups)
    for block in blocks:
        block_loss, _, _ = trainer.block_forward_and_loss(
            optimized, block, forward_batch_size=5
        )
        block_loss.backward()
        optimized_optimizer.step()
        optimized_optimizer.zero_grad(set_to_none=True)

    assert [len(block) for block in blocks] == [8, 2]
    assert legacy_steps == len(blocks) == 2
    assert legacy.tokenizer_calls == 10
    assert optimized.tokenizer_calls == 2
    for actual, expected in zip(optimized.model.parameters(), legacy.model.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.infogain import infogain_multitask_loss
from src.run_manifest import atomic_write_json, git_state, utc_now


def load_signed_trainer():
    path = PROJECT_ROOT / "scripts/preformal/26_train_signed_v1.py"
    spec = importlib.util.spec_from_file_location("kernel_signed_trainer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KernelSelector:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model = torch.nn.Linear(2, 1).to(device)
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
        results = []
        for start in range(0, len(encoded["features"]), batch_size):
            self.forward_calls += 1
            batch = encoded["features"][start : start + batch_size].to(self.device)
            results.append(self.model(batch).squeeze(-1))
        return torch.cat(results)

    def score_texts(
        self, texts: list[str], batch_size: int, requires_grad: bool
    ) -> torch.Tensor:
        return self.forward_encoded(
            self.encode_texts(texts), batch_size=batch_size, requires_grad=requires_grad
        )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_reset(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_peak(device: torch.device) -> int | None:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None


def benchmark_infogain(
    count: int, implementation: str, device: torch.device
) -> dict[str, Any]:
    torch.manual_seed(13)
    rng = random.Random(13)
    rank = torch.nn.Parameter(torch.randn(10, device=device))
    logits = torch.nn.Parameter(torch.randn(10, 2, device=device))
    optimizer = torch.optim.SGD([rank, logits], lr=1e-4)
    sizes = [rng.randint(2, 10) for _ in range(count)]
    dig_pool = [-0.2, -0.05, 0.0, 0.0, 0.03, 0.2]
    dig_groups = [[rng.choice(dig_pool) for _ in range(size)] for size in sizes]
    memory_reset(device); synchronize(device); started = time.perf_counter()
    losses = []
    for size, digs in zip(sizes, dig_groups):
        loss, _ = infogain_multitask_loss(
            rank[:size], logits[:size], digs,
            b_pos=0.1, b_neg=-0.1, beta=0.75,
            rank_loss_implementation=implementation,
        )
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))
    synchronize(device); elapsed = time.perf_counter() - started
    return {
        "method": "infogain", "implementation": implementation, "groups": count,
        "wall_clock_seconds": elapsed, "groups_per_second": count / elapsed,
        "texts": sum(sizes), "texts_per_second": sum(sizes) / elapsed,
        "peak_cuda_memory_bytes": memory_peak(device), "gpu_utilization": None,
        "optimizer_steps": count, "forward_calls": count, "tokenizer_calls": count,
        "final_loss": losses[-1],
        "loss_trajectory": [losses[index] for index in range(999, count, 1000)],
    }


def signed_groups(count: int) -> list[SimpleNamespace]:
    rng = random.Random(13)
    result = []
    for index in range(count):
        size = rng.randint(2, 10)
        gains = [0.02, -0.01] + [rng.choice([-0.01, 0.005, 0.02]) for _ in range(size - 2)]
        result.append(SimpleNamespace(
            query=f"claim {index}", selected_docs=[],
            candidate_docs=[{"title": f"doc {item}", "text": f"evidence {index} {item}"} for item in range(size)],
            effective_gains=gains,
        ))
    return result


def benchmark_signed(
    count: int, implementation: str, forward_batch_size: int, device: torch.device
) -> dict[str, Any]:
    trainer = load_signed_trainer(); groups = signed_groups(count)
    torch.manual_seed(13); selector = KernelSelector(device)
    optimizer = torch.optim.SGD(selector.model.parameters(), lr=1e-6)
    optimizer.zero_grad(set_to_none=True); losses = []; steps = 0; texts = 0
    memory_reset(device); synchronize(device); started = time.perf_counter()
    if implementation == "legacy":
        for index, group in enumerate(groups, start=1):
            group_inputs = trainer.group_texts(group); texts += len(group_inputs)
            scores = selector.score_texts(group_inputs, len(group_inputs), True)
            loss, _ = trainer.signed_group_loss(scores, group); losses.append(float(loss.detach()))
            (loss / trainer.OPTIMIZER_GROUP_BATCH_SIZE).backward()
            if index % 8 == 0 or index == count:
                optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1
    else:
        for block in trainer.optimizer_blocks(groups):
            block_loss, results, block_texts = trainer.block_forward_and_loss(
                selector, block, forward_batch_size=forward_batch_size
            )
            block_loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1
            texts += block_texts; losses.extend(float(loss.detach()) for loss, _ in results)
    synchronize(device); elapsed = time.perf_counter() - started
    return {
        "method": "signed_v1", "implementation": implementation,
        "forward_batch_size": forward_batch_size if implementation != "legacy" else None,
        "groups": count, "wall_clock_seconds": elapsed,
        "groups_per_second": count / elapsed, "texts": texts,
        "texts_per_second": texts / elapsed, "peak_cuda_memory_bytes": memory_peak(device),
        "gpu_utilization": None, "optimizer_steps": steps,
        "forward_calls": selector.forward_calls, "tokenizer_calls": selector.tokenizer_calls,
        "final_loss": losses[-1],
        "loss_trajectory": [losses[index] for index in range(999, count, 1000)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic learned-selector runtime kernel benchmark.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-counts", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument("--forward-batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    results = []
    for count in args.group_counts:
        results.extend(benchmark_infogain(count, implementation, device) for implementation in ("legacy", "vectorized"))
        results.append(benchmark_signed(count, "legacy", 1, device))
        results.extend(benchmark_signed(count, "block_v1", size, device) for size in args.forward_batch_sizes)
    payload = {
        "schema_version": "rag_cbwdm_runtime_kernel_benchmark.v1",
        "created_at": utc_now(), "seed": 13, "device": str(device),
        "representative_transformer_benchmark": False,
        "note": "Synthetic kernel/control-flow benchmark; run the artifact-backed harness on RTX 4090 before choosing a production microbatch.",
        "git": git_state(PROJECT_ROOT), "results": results,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

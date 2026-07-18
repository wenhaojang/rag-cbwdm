"""Pointwise shared-encoder, dual-head reranker for InfoGain-FEVER."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _imports() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "InfoGain-FEVER training requires torch and transformers. "
            "Install: pip install -r requirements-baselines.txt"
        ) from exc
    return torch, AutoModel, AutoTokenizer


class InfoGainPointwiseReranker:
    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        torch, AutoModel, AutoTokenizer = _imports()
        self.torch = torch
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            ("cpu" if device == "auto" else device)
        )
        self.model_name = model_name
        self.revision = revision
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.encoder = AutoModel.from_pretrained(model_name, revision=revision).to(self.device)
        hidden = int(self.encoder.config.hidden_size)
        self.rank_head = torch.nn.Linear(hidden, 1).to(self.device)
        self.filter_head = torch.nn.Linear(hidden, 2).to(self.device)

    def parameters(self) -> list[Any]:
        return [
            *self.encoder.parameters(),
            *self.rank_head.parameters(),
            *self.filter_head.parameters(),
        ]

    def train(self) -> None:
        self.encoder.train()
        self.rank_head.train()
        self.filter_head.train()

    def eval(self) -> None:
        self.encoder.eval()
        self.rank_head.eval()
        self.filter_head.eval()

    def forward(self, texts: list[str]) -> tuple[Any, Any]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        hidden = self.encoder(**encoded).last_hidden_state[:, 0]
        return self.rank_head(hidden).squeeze(-1), self.filter_head(hidden)

    def score(self, texts: list[str], batch_size: int) -> tuple[list[float], list[float]]:
        self.eval()
        rank_values: list[float] = []
        probabilities: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                rank, logits = self.forward(texts[start : start + batch_size])
                rank_values.extend(float(value) for value in rank.cpu().tolist())
                probabilities.extend(
                    float(value) for value in logits.softmax(-1)[:, 1].cpu().tolist()
                )
        return rank_values, probabilities

    def save(self, path: str | Path, config: dict[str, Any]) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(target / "encoder")
        self.tokenizer.save_pretrained(target / "encoder")
        self.torch.save(
            {
                "rank_head": self.rank_head.state_dict(),
                "filter_head": self.filter_head.state_dict(),
            },
            target / "heads.pt",
        )
        (target / "infogain_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str = "auto") -> tuple["InfoGainPointwiseReranker", dict[str, Any]]:
        target = Path(path)
        config = json.loads((target / "infogain_config.json").read_text(encoding="utf-8"))
        model = cls(
            str(target / "encoder"),
            device=device,
            max_length=int(config["max_length"]),
        )
        state = model.torch.load(target / "heads.pt", map_location=model.device)
        model.rank_head.load_state_dict(state["rank_head"])
        model.filter_head.load_state_dict(state["filter_head"])
        return model, config

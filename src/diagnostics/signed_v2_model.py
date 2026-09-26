"""Shared-backbone dual-head cross-encoder for experimental signed-selector v2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.io_utils import ensure_dir
from src.selector_cross_encoder import resolve_device


SIGNED_V2_METHOD = "rag_cbwdm_signed_v2"
SIGNED_V2_ARCHITECTURE = "rag_cbwdm_signed_v2_dual_head_v1"
SIGNED_V2_CHECKPOINT_SCHEMA = "rag_cbwdm_signed_v2_checkpoint.v1"


def _import_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "signed-selector v2 requires transformers and torch. "
            "Install them with: pip install -r requirements.txt"
        ) from exc
    return AutoModel, AutoTokenizer


@dataclass
class DualHeadLogits:
    """Named dual-head logits, each with shape ``[batch]``."""

    gate_logit: torch.Tensor
    utility_logit: torch.Tensor


class SignedV2DualHeadSelector(torch.nn.Module):
    """One pretrained text encoder shared by independent gate and utility heads."""

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 512,
        device: str | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        head_seed: int = 13,
        encoder: torch.nn.Module | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        if not model_name:
            raise ValueError("model_name must be non-empty")
        self.model_name = str(model_name)
        self.max_length = int(max_length)
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        self.device = resolve_device(device)
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision or revision
        self.head_seed = int(head_seed)

        if encoder is None or tokenizer is None:
            AutoModel, AutoTokenizer = _import_transformers()
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, revision=self.tokenizer_revision
                )
            if encoder is None:
                kwargs: dict[str, Any] = {}
                if revision:
                    kwargs["revision"] = revision
                encoder = AutoModel.from_pretrained(model_name, **kwargs)

        self.encoder = encoder
        self.tokenizer = tokenizer
        hidden_size = int(getattr(self.encoder.config, "hidden_size"))
        dropout_probability = float(
            getattr(
                self.encoder.config,
                "classifier_dropout",
                None,
            )
            or getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        )
        self.dropout = torch.nn.Dropout(dropout_probability)
        self.gate_head = torch.nn.Linear(hidden_size, 1)
        self.utility_head = torch.nn.Linear(hidden_size, 1)
        self._initialize_heads()
        self.to(self.device)

    def _initialize_heads(self) -> None:
        """Deterministically initialize only the two new heads."""
        initializer_range = float(getattr(self.encoder.config, "initializer_range", 0.02))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.head_seed)
            for head in (self.gate_head, self.utility_head):
                torch.nn.init.normal_(head.weight, mean=0.0, std=initializer_range)
                torch.nn.init.zeros_(head.bias)

    def _encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, **encoded: torch.Tensor) -> DualHeadLogits:
        outputs = self.encoder(**encoded)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is None:
                last_hidden_state = outputs[0]
            pooled = last_hidden_state[:, 0]
        shared = self.dropout(pooled)
        return DualHeadLogits(
            gate_logit=self.gate_head(shared).squeeze(-1),
            utility_logit=self.utility_head(shared).squeeze(-1),
        )

    def score_texts(
        self,
        texts: list[str],
        *,
        batch_size: int = 8,
        requires_grad: bool = False,
    ) -> DualHeadLogits:
        if not texts:
            empty = torch.empty(0, device=self.device)
            return DualHeadLogits(gate_logit=empty, utility_logit=empty)
        gate_scores: list[torch.Tensor] = []
        utility_scores: list[torch.Tensor] = []
        size = max(int(batch_size), 1)
        context = torch.enable_grad() if requires_grad else torch.inference_mode()
        with context:
            for start in range(0, len(texts), size):
                encoded = self._encode(texts[start : start + size])
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self(**encoded)
                gate_scores.append(output.gate_logit)
                utility_scores.append(output.utility_logit)
        return DualHeadLogits(
            gate_logit=torch.cat(gate_scores, dim=0),
            utility_logit=torch.cat(utility_scores, dim=0),
        )

    def checkpoint_metadata(
        self, training_contract: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {
            "schema_version": SIGNED_V2_CHECKPOINT_SCHEMA,
            "method": SIGNED_V2_METHOD,
            "architecture": SIGNED_V2_ARCHITECTURE,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "head_seed": self.head_seed,
            "head_initialization": "normal(mean=0,std=encoder.config.initializer_range); bias=zeros",
            "utility_head_initialized_from_v1_classifier": False,
            "training_contract": dict(training_contract or {}),
        }

    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        training_contract: dict[str, Any] | None = None,
    ) -> None:
        checkpoint = ensure_dir(checkpoint_dir)
        encoder_dir = ensure_dir(checkpoint / "encoder")
        tokenizer_dir = ensure_dir(checkpoint / "tokenizer")
        self.encoder.save_pretrained(encoder_dir)
        self.tokenizer.save_pretrained(tokenizer_dir)
        torch.save(
            {
                "schema_version": SIGNED_V2_CHECKPOINT_SCHEMA,
                "architecture": SIGNED_V2_ARCHITECTURE,
                "gate_head": self.gate_head.state_dict(),
                "utility_head": self.utility_head.state_dict(),
            },
            checkpoint / "dual_heads.pt",
        )
        with (checkpoint / "signed_v2_config.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                self.checkpoint_metadata(training_contract),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        max_length: int | None = None,
        device: str | None = None,
    ) -> "SignedV2DualHeadSelector":
        checkpoint = Path(checkpoint_dir)
        config_path = checkpoint / "signed_v2_config.json"
        heads_path = checkpoint / "dual_heads.pt"
        if not config_path.is_file() or not heads_path.is_file():
            raise FileNotFoundError(f"Incomplete signed-v2 checkpoint: {checkpoint}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("schema_version") != SIGNED_V2_CHECKPOINT_SCHEMA:
            raise ValueError("Checkpoint schema is not signed-v2")
        if config.get("method") != SIGNED_V2_METHOD:
            raise ValueError("Checkpoint method is not rag_cbwdm_signed_v2")
        if config.get("architecture") != SIGNED_V2_ARCHITECTURE:
            raise ValueError("Unsupported signed-v2 architecture")

        AutoModel, AutoTokenizer = _import_transformers()
        encoder = AutoModel.from_pretrained(checkpoint / "encoder")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
        selector = cls(
            model_name=str(config.get("model_name") or checkpoint),
            max_length=(
                int(max_length)
                if max_length is not None
                else int(config.get("max_length", 512))
            ),
            device=device,
            revision=config.get("revision"),
            tokenizer_revision=config.get("tokenizer_revision"),
            head_seed=int(config.get("head_seed", 13)),
            encoder=encoder,
            tokenizer=tokenizer,
        )
        try:
            payload = torch.load(heads_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older torch
            payload = torch.load(heads_path, map_location="cpu")
        if payload.get("schema_version") != SIGNED_V2_CHECKPOINT_SCHEMA:
            raise ValueError("Dual-head weights schema is not signed-v2")
        selector.gate_head.load_state_dict(payload["gate_head"])
        selector.utility_head.load_state_dict(payload["utility_head"])
        selector.to(selector.device)
        return selector

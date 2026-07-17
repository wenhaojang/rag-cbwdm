"""Utilities for computing label posteriors from next-token logits."""

from __future__ import annotations

from typing import Any

import numpy as np


def _import_torch_and_transformers() -> tuple[Any, Any, Any]:
    """Import heavy model dependencies with a clear installation hint."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Label posterior scoring requires torch, transformers, and accelerate. "
            "Install them with: pip install -r requirements.txt"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


class LabelLogitScorer:
    """Score discrete labels by aggregating verbalizer next-token logits."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        device_map: str = "auto",
        trust_remote_code: bool = False,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        max_length: int | None = None,
    ) -> None:
        """Load a HuggingFace causal LM and tokenizer for next-token scoring."""
        if not model_name:
            raise ValueError("model_name must be a non-empty string")

        self.torch, AutoModelForCausalLM, AutoTokenizer = _import_torch_and_transformers()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            revision=tokenizer_revision or revision,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token nor eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if device_map:
            model_kwargs["device_map"] = device_map
        if dtype:
            model_kwargs["torch_dtype"] = self._resolve_dtype(dtype)

        if revision:
            model_kwargs["revision"] = revision
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        self.model_name = model_name
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision or revision
        self.max_length = int(max_length) if max_length is not None else None
        self._verbalizer_cache: dict[tuple[str, tuple[str, ...]], list[int]] = {}

    def _resolve_dtype(self, dtype: str) -> Any:
        """Map config dtype strings to torch dtype objects or Transformers auto mode."""
        normalized = str(dtype).lower()
        if normalized == "auto":
            return "auto"
        mapping = {
            "float16": self.torch.float16,
            "fp16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "bf16": self.torch.bfloat16,
            "float32": self.torch.float32,
            "fp32": self.torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Use auto, float16, bfloat16, or float32."
            )
        return mapping[normalized]

    def _single_token_ids(self, label: str, verbalizers: list[str]) -> list[int]:
        """Return token ids for single-token verbalizers for one label."""
        cache_key = (label, tuple(str(value) for value in verbalizers))
        if cache_key in self._verbalizer_cache:
            return self._verbalizer_cache[cache_key]
        token_ids: list[int] = []
        skipped: list[str] = []
        for verbalizer in verbalizers:
            ids = self.tokenizer.encode(str(verbalizer), add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])
            else:
                skipped.append(str(verbalizer))

        if not token_ids:
            raise ValueError(
                f"All verbalizers for label '{label}' are multi-token or empty for this "
                f"tokenizer. Verbalizers tried: {verbalizers}. Add at least one "
                "single-token verbalizer to the config."
            )
        self._verbalizer_cache[cache_key] = token_ids
        return token_ids

    def _input_device(self) -> Any:
        """Return the embedding device, which is safe for device_map-sharded models."""
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, StopIteration):
            return next(self.model.parameters()).device

    def score_prompts(
        self,
        prompts: list[str],
        batch_size: int,
        labels: list[str],
        verbalizers: dict[str, list[str]],
    ) -> np.ndarray:
        """Return float32 posteriors with shape ``[len(prompts), len(labels)]``."""
        if not labels:
            raise ValueError("labels must not be empty")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
            raise ValueError("every prompt must be a non-empty string")
        if not prompts:
            return np.empty((0, len(labels)), dtype=np.float32)

        token_ids_by_label = []
        for label in labels:
            if label not in verbalizers:
                raise KeyError(f"Missing verbalizers for label: {label}")
            token_ids_by_label.append(self._single_token_ids(label, verbalizers[label]))

        batches: list[np.ndarray] = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            tokenize_kwargs: dict[str, Any] = {
                "padding": True,
                "return_tensors": "pt",
            }
            if self.max_length is not None:
                tokenize_kwargs.update({"truncation": True, "max_length": self.max_length})
            inputs = self.tokenizer(batch_prompts, **tokenize_kwargs)
            inputs = {key: value.to(self._input_device()) for key, value in inputs.items()}
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                raise ValueError("Tokenizer output must include attention_mask for batched scoring")
            positions = self.torch.arange(attention_mask.shape[1], device=attention_mask.device)
            last_indices = (attention_mask * positions.unsqueeze(0)).max(dim=1).values
            try:
                with self.torch.inference_mode():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[
                        self.torch.arange(len(batch_prompts), device=outputs.logits.device),
                        last_indices.to(outputs.logits.device),
                        :,
                    ].float()
                    label_scores = []
                    for ids in token_ids_by_label:
                        ids_tensor = self.torch.tensor(ids, device=logits.device)
                        label_scores.append(
                            self.torch.logsumexp(logits.index_select(1, ids_tensor), dim=1)
                        )
                    probs = self.torch.softmax(self.torch.stack(label_scores, dim=1), dim=1)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        f"OOM while scoring posterior batch of size {len(batch_prompts)}. "
                        "Reduce --batch-size (and optionally max context length)."
                    ) from exc
                raise
            array = probs.detach().cpu().numpy().astype(np.float32, copy=False)
            if not np.all(np.isfinite(array)):
                raise FloatingPointError("Posterior batch contains NaN or Inf")
            if not np.allclose(array.sum(axis=1), 1.0, atol=1e-5):
                raise FloatingPointError("Posterior batch is not normalized")
            batches.append(array)
        return np.concatenate(batches, axis=0)

    def score_prompt(
        self,
        prompt: str,
        labels: list[str],
        verbalizers: dict[str, list[str]],
    ) -> list[float]:
        """Return a probability distribution over labels for the next answer token."""
        if not labels:
            raise ValueError("labels must not be empty")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")

        probs = self.score_prompts(
            [prompt],
            batch_size=1,
            labels=labels,
            verbalizers=verbalizers,
        )
        return [float(value) for value in probs[0]]

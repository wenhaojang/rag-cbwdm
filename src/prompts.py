"""Prompt builders for fixed-generator label posterior estimation."""

from __future__ import annotations

import hashlib
import json

FEVER_PROMPT_VERSION = "fever_classification_v1"


def fever_prompt_hash(labels: list[str], verbalizers: dict[str, list[str]]) -> str:
    """Hash the cache-defining prompt template contract."""
    payload = {
        "version": FEVER_PROMPT_VERSION,
        "template": (
            "system=fact verification; optional Evidence; Claim; ordered label menu; Answer:"
        ),
        "labels": labels,
        "verbalizers": verbalizers,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def display_label(label: str) -> str:
    """Return a human-readable label string for prompt display."""
    return label.replace("_", " ")


def build_fever_prompt(
    claim: str,
    labels: list[str],
    verbalizers: dict[str, list[str]],
    evidence: str | None = None,
) -> str:
    """Build a FEVER-style classification prompt for query-only or single evidence scoring."""
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")
    if not labels:
        raise ValueError("labels must not be empty")

    lines = ["You are a fact verification model.", ""]
    if evidence is not None:
        lines.extend(["Evidence:", evidence.strip(), ""])

    lines.extend(["Claim:", claim.strip(), "", "Choose the correct label:"])
    for label in labels:
        label_verbalizers = verbalizers.get(label)
        if not label_verbalizers:
            raise KeyError(f"Missing verbalizer for label: {label}")
        answer_token = str(label_verbalizers[0]).strip()
        if not answer_token:
            raise ValueError(f"Empty first verbalizer for label: {label}")
        lines.append(f"{answer_token}. {display_label(label)}")

    lines.extend(["", "Answer:"])
    return "\n".join(lines)

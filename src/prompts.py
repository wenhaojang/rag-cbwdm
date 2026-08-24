"""Prompt builders for fixed-generator label posterior estimation."""

from __future__ import annotations

import hashlib
import json

FEVER_PROMPT_VERSION = "fever_classification_v1"
FM2_PROMPT_VERSION = "fm2_classification_v1"


def classification_prompt_version(dataset: str) -> str:
    """Resolve a dataset prompt version while preserving the frozen FEVER contract."""
    normalized = str(dataset).strip().lower()
    if normalized in {"fever", "fever2", "fever3"}:
        return FEVER_PROMPT_VERSION
    if normalized in {"fm2", "foolmetwice", "fool-me-twice", "fool_me_twice"}:
        return FM2_PROMPT_VERSION
    raise ValueError(f"No classification prompt registered for dataset={dataset!r}")


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


def classification_prompt_hash(
    dataset: str, labels: list[str], verbalizers: dict[str, list[str]]
) -> str:
    """Hash the dataset prompt contract; FEVER delegates to its frozen v1 hash."""
    version = classification_prompt_version(dataset)
    if version == FEVER_PROMPT_VERSION:
        return fever_prompt_hash(labels, verbalizers)
    payload = {
        "version": version,
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


def build_classification_prompt(
    dataset: str,
    query: str,
    labels: list[str],
    verbalizers: dict[str, list[str]],
    evidence: str | None = None,
) -> str:
    """Build a registered prompt for query-only or query-document scoring.

    FM2 v1 intentionally shares FEVER's task text and verbalizer menu for a
    controlled comparison, while using a distinct cache/manifest version.
    """
    classification_prompt_version(dataset)
    return build_fever_prompt(query, labels, verbalizers, evidence=evidence)

"""FoolMeTwice (FM2) adapter for the shared RAG-CBWDM schemas.

The official ``retrieved_evidence`` field is a closed-page evidence pool.  Gold
evidence is intentionally emitted through a separate diagnostic record so it
cannot accidentally become a selector feature.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable

FM2_SOURCE_COMMIT = "d9db753e5acf91c0d9bf543db327ab655661eb94"
FM2_SOURCE_REPOSITORY = "https://github.com/google-research/fool-me-twice"
FM2_SOURCE_URLS = {
    split: (
        "https://raw.githubusercontent.com/google-research/fool-me-twice/"
        f"{FM2_SOURCE_COMMIT}/dataset/{split}.jsonl"
    )
    for split in ("train", "dev", "test")
}
FM2_EXPECTED_SHA256 = {
    "train": "1f47e035650aa5734301afb36a94afa610af73ac908fd4c4009e281b9956a311",
    "dev": "eeb36a4757fb86412f1d7ad5197ffd6ffa837c1cf7fb3c0067d82d0b65257229",
    "test": "4a45aa8edd45ea8ff54eb66be06e8c6113873c22496b209925dd5b529ce00721",
}
FM2_EXPECTED_ROWS = {"train": 10419, "dev": 1169, "test": 1380}
FM2_LABEL_MAPPING = {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES"}
FM2_INTERNAL_LABELS = ("SUPPORTS", "REFUTES")
FM2_REQUIRED_FIELDS = frozenset(
    {
        "category",
        "correct_votes",
        "gold_evidence",
        "id",
        "label",
        "retrieved_evidence",
        "text",
        "total_likes",
        "total_votes",
        "wikipedia_page",
    }
)


def normalize_label(raw_label: Any) -> str:
    """Map an official FM2 label to the internal binary task labels."""
    if not isinstance(raw_label, str) or raw_label not in FM2_LABEL_MAPPING:
        raise ValueError(
            f"Unknown FM2 label {raw_label!r}; expected {sorted(FM2_LABEL_MAPPING)}"
        )
    return FM2_LABEL_MAPPING[raw_label]


def _validate_evidence(item: Any, *, where: str) -> dict[str, str]:
    if not isinstance(item, dict):
        raise ValueError(f"{where}: evidence must be an object")
    if set(item) != {"section_header", "text"}:
        raise ValueError(
            f"{where}: evidence fields must be section_header/text, got {sorted(item)}"
        )
    header, text = item["section_header"], item["text"]
    if not isinstance(header, str) or not isinstance(text, str) or not text.strip():
        raise ValueError(f"{where}: invalid section_header/text")
    return {"section_header": header.strip(), "text": text.strip()}


def validate_raw_row(row: Any, *, split: str, row_number: int) -> dict[str, Any]:
    """Validate the real official JSONL schema without guessing optional aliases."""
    if split not in FM2_EXPECTED_ROWS:
        raise ValueError(f"Unsupported FM2 split: {split!r}")
    if not isinstance(row, dict):
        raise ValueError(f"{split} row {row_number}: expected object")
    missing = FM2_REQUIRED_FIELDS - set(row)
    if missing:
        raise ValueError(f"{split} row {row_number}: missing fields {sorted(missing)}")
    for key in ("id", "text", "wikipedia_page", "category"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"{split} row {row_number}: invalid {key}")
    normalize_label(row["label"])
    for key in ("retrieved_evidence", "gold_evidence"):
        if not isinstance(row[key], list) or not row[key]:
            raise ValueError(f"{split} row {row_number}: {key} must be non-empty")
        for index, evidence in enumerate(row[key], start=1):
            _validate_evidence(evidence, where=f"{split} row {row_number}.{key}[{index}]")
    return row


def canonical_id(split: str, original_id: str) -> str:
    return f"fm2:{split}:{original_id}"


def evidence_key(item: dict[str, Any]) -> str:
    normalized = _validate_evidence(item, where="evidence")
    return normalized["section_header"] + "\n" + normalized["text"]


def adapt_raw_row(
    row: dict[str, Any], *, split: str, row_number: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return canonical query, deployable pool, and separate gold diagnostic rows."""
    validate_raw_row(row, split=split, row_number=row_number)
    original_id = row["id"].strip()
    identifier = canonical_id(split, original_id)
    raw_label = row["label"]
    label = normalize_label(raw_label)
    page = row["wikipedia_page"].strip()
    retrieved = [
        _validate_evidence(item, where=f"{identifier}.retrieved_evidence[{index}]")
        for index, item in enumerate(row["retrieved_evidence"], start=1)
    ]
    gold = [
        _validate_evidence(item, where=f"{identifier}.gold_evidence[{index}]")
        for index, item in enumerate(row["gold_evidence"], start=1)
    ]
    candidates = []
    for rank, item in enumerate(retrieved, start=1):
        header = item["section_header"]
        candidates.append(
            {
                "doc_id": f"{identifier}:official:{rank:02d}",
                "rank": rank,
                "source_rank": rank,
                "score": None,
                "retrieval_score": None,
                "title": f"{page} | {header}" if header else page,
                "text": item["text"],
                "section_header": header,
                "wikipedia_page": page,
                "candidate_source": "official_retrieved_evidence",
            }
        )
    metadata = {
        "original_id": original_id,
        "original_label": raw_label,
        "wikipedia_page": page,
        "category": row["category"],
        "correct_votes": row["correct_votes"],
        "total_likes": row["total_likes"],
        "total_votes": row["total_votes"],
        "gold_evidence": gold,
    }
    canonical = {
        "schema_version": "rag_cbwdm_query.v2",
        "id": identifier,
        "query": row["text"].strip(),
        "label": label,
        "split": split,
        "metadata": metadata,
    }
    pool = {
        "schema_version": "rag_cbwdm_retrieval.v1",
        "id": identifier,
        "query": canonical["query"],
        "label": label,
        "split": split,
        "candidates": candidates,
        "candidate_pool": {
            "protocol": "fm2_official_closed_page_v1",
            "source_field": "retrieved_evidence",
            "wikipedia_page": page,
            "gold_evidence_in_selector_payload": False,
            "construction_gold_free": True,
            "known_source_page_required": True,
            "source_order_preserved": True,
        },
    }
    retrieved_keys = {evidence_key(item) for item in retrieved}
    gold_keys = [evidence_key(item) for item in gold]
    diagnostic = {
        "schema_version": "rag_cbwdm_fm2_gold_diagnostic.v1",
        "id": identifier,
        "split": split,
        "gold_evidence_keys": gold_keys,
        "gold_evidence": gold,
        "any_gold_in_official_pool": any(key in retrieved_keys for key in gold_keys),
        "all_gold_in_official_pool": all(key in retrieved_keys for key in gold_keys),
    }
    return canonical, pool, diagnostic


def summarize_pool(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts = [len(row.get("candidates", [])) for row in rows]
    histogram = Counter(counts)
    return {
        "num_rows": len(counts),
        "candidate_count_min": min(counts) if counts else 0,
        "candidate_count_max": max(counts) if counts else 0,
        "candidate_count_average": sum(counts) / len(counts) if counts else 0.0,
        "candidate_count_histogram": {
            str(key): histogram[key] for key in sorted(histogram)
        },
        "rows_with_fewer_than_4": sum(value < 4 for value in counts),
        "rows_with_fewer_than_20": sum(value < 20 for value in counts),
    }


def stable_candidate_pool_sha(rows: Iterable[dict[str, Any]]) -> str:
    """Content hash for tests/tools that need a stream-independent pool identity."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["id"]).encode("utf-8"))
        for candidate in row.get("candidates", []):
            digest.update(str(candidate["doc_id"]).encode("utf-8"))
            digest.update(str(candidate["text"]).encode("utf-8"))
    return digest.hexdigest()

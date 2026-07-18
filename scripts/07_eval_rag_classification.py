from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_yaml, read_jsonl, require_keys
from src.label_logits import LabelLogitScorer
from src.metrics import ClassificationMetrics
from src.prompts import build_fever_prompt, fever_prompt_hash
from src.run_manifest import atomic_write_json, git_state, sha256_file, stable_hash, utc_now


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RAG classification evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate RAG classification from selected evidence.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"], help="Data split.")
    parser.add_argument("--selection", required=True, help="Selection JSONL path.")
    parser.add_argument("--output", required=True, help="Prediction JSONL output path.")
    parser.add_argument("--metrics-output", required=True, help="Metrics JSON output path.")
    parser.add_argument("--model-name", default=None, help="Override generator.model_name.")
    parser.add_argument("--limit", type=int, default=None, help="Max selection rows to evaluate.")
    parser.add_argument("--max-docs", type=int, default=None, help="Use at most first K selected docs.")
    parser.add_argument("--method-name", default=None, help="Method name written to outputs.")
    parser.add_argument("--no-evidence", action="store_true", help="Evaluate query-only baseline.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def validate_config(config: Dict[str, Any]) -> None:
    """Validate config fields used by evaluation."""
    require_keys(config, ["dataset", "task", "generator"], "config")
    require_keys(config["task"], ["labels", "verbalizers"], "config.task")
    require_keys(config["generator"], ["model_name"], "config.generator")


def recover_selected_docs(row: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return selected docs with text, falling back to candidates + selected_doc_ids."""
    selected_docs = row.get("selected_docs")
    if isinstance(selected_docs, list):
        docs = selected_docs
    else:
        selected_ids = row.get("selected_doc_ids")
        candidates = row.get("candidates")
        if not isinstance(selected_ids, list) or not isinstance(candidates, list):
            raise ValueError(
                f"Selection row {row.get('id')} lacks selected_docs and cannot recover "
                "evidence from candidates + selected_doc_ids."
            )
        candidate_by_id = {candidate.get("doc_id"): candidate for candidate in candidates}
        docs = []
        for doc_id in selected_ids:
            if doc_id not in candidate_by_id:
                raise ValueError(f"Selection row {row.get('id')} selected doc_id not found: {doc_id}")
            docs.append(candidate_by_id[doc_id])

    for doc in docs:
        if not isinstance(doc, dict) or not doc.get("text"):
            raise ValueError(f"Selection row {row.get('id')} contains selected doc without text: {doc}")
    return docs


def build_evidence_context(docs: list[Dict[str, Any]]) -> str:
    """Concatenate selected documents into a simple evidence context."""
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        title = doc.get("title")
        text = doc.get("text", "")
        if title:
            blocks.append(f"[{idx}] Title: {title}\n{text}")
        else:
            blocks.append(f"[{idx}] {text}")
    return "\n\n".join(blocks)


def argmax_label(labels: list[str], probs: list[float]) -> str:
    """Return the label with maximum probability."""
    if len(labels) != len(probs):
        raise ValueError(f"labels/probs length mismatch: {len(labels)} vs {len(probs)}")
    return labels[max(range(len(probs)), key=lambda idx: probs[idx])]


def iter_prediction_rows(
    selection_path: Path,
    scorer: LabelLogitScorer,
    labels: list[str],
    verbalizers: dict[str, list[str]],
    metrics: ClassificationMetrics,
    method_name: str | None,
    no_evidence: bool = False,
    max_docs: int | None = None,
    limit: int | None = None,
    log_every: int = 10,
) -> Iterator[Dict[str, Any]]:
    """Stream prediction rows while updating metrics."""
    for row_index, row in enumerate(read_jsonl(selection_path, limit=limit), start=1):
        require_keys(row, ["id", "query", "label"], f"selection row {row_index}")
        selected_docs = [] if no_evidence else recover_selected_docs(row)
        if max_docs is not None:
            if max_docs < 0:
                raise ValueError(f"max_docs must be non-negative, got {max_docs}")
            selected_docs = selected_docs[:max_docs]

        evidence_text = None if no_evidence or not selected_docs else build_evidence_context(selected_docs)
        prompt = build_fever_prompt(
            claim=row["query"],
            labels=labels,
            verbalizers=verbalizers,
            evidence=evidence_text,
        )
        probs = scorer.score_prompt(prompt, labels=labels, verbalizers=verbalizers)
        pred = argmax_label(labels, probs)
        gold = row["label"]
        evidence_chars = len(evidence_text or "")
        num_docs = len(selected_docs)
        metrics.update(
            gold=gold,
            pred=pred,
            num_docs=num_docs,
            evidence_chars=evidence_chars,
            probs=probs,
            original_bm25_ranks=[
                float(doc.get("source_rank", doc.get("rank")))
                for doc in selected_docs
                if doc.get("source_rank", doc.get("rank")) is not None
            ],
            used_min_docs_fallback=any(
                bool(doc.get("min_docs_fallback")) for doc in selected_docs
            ),
        )

        if row_index % log_every == 0:
            print(f"[eval_rag] processed={row_index}", file=sys.stderr)

        yield {
            "id": row["id"],
            "query": row["query"],
            "gold": gold,
            "pred": pred,
            "correct": pred == gold,
            "labels": labels,
            "probs": probs,
            "selected_doc_ids": [doc.get("doc_id") for doc in selected_docs],
            "num_docs": num_docs,
            "method": method_name or row.get("method", "rag_classification"),
            "source_ranks": [
                doc.get("source_rank", doc.get("rank")) for doc in selected_docs
            ],
        }


def atomic_write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    count = 0
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return count


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_config(config)

    generator_config = config["generator"]
    model_name = args.model_name or generator_config["model_name"]
    labels = list(config["task"]["labels"])
    verbalizers = dict(config["task"]["verbalizers"])
    selection_path = resolve_project_path(args.selection)
    output_path = resolve_project_path(args.output)
    metrics_path = resolve_project_path(args.metrics_output)
    manifest_path = metrics_path.with_suffix(".manifest.json")
    evaluation_contract = {
        "stage": "evaluation",
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "split": args.split,
        "method_override": args.method_name,
        "no_evidence": args.no_evidence,
        "limit": args.limit,
        "max_docs": args.max_docs,
        "generator_model": model_name,
        "generator_revision": generator_config.get("revision"),
        "tokenizer_revision": generator_config.get("tokenizer_revision"),
        "max_context_tokens": generator_config.get("max_context_tokens"),
        "prompt_hash": fever_prompt_hash(labels, verbalizers),
        "verbalizer_hash": stable_hash(verbalizers),
    }
    fingerprint = stable_hash(evaluation_contract)
    if args.resume and output_path.exists() and metrics_path.exists() and manifest_path.exists() and not args.overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "completed"
            and manifest.get("fingerprint") == fingerprint
            and manifest.get("predictions_sha256") == sha256_file(output_path)
            and manifest.get("metrics_sha256") == sha256_file(metrics_path)
        ):
            print(f"[eval_rag] reused=true output={output_path} metrics={metrics_path}")
            return
        raise ValueError("Cannot resume evaluation: manifest fingerprint/checksum mismatch")
    if any(path.exists() for path in (output_path, metrics_path, manifest_path)) and not args.overwrite:
        raise FileExistsError("Evaluation artifacts exist; use --resume or --overwrite")

    scorer = LabelLogitScorer(
        model_name=model_name,
        dtype=generator_config.get("dtype", "auto"),
        device_map=generator_config.get("device_map", "auto"),
        trust_remote_code=bool(generator_config.get("trust_remote_code", False)),
        revision=generator_config.get("revision"),
        tokenizer_revision=generator_config.get("tokenizer_revision"),
        max_length=generator_config.get("max_context_tokens"),
    )
    metrics_acc = ClassificationMetrics(labels=labels)
    selection_metadata = list(read_jsonl(selection_path, limit=args.limit))
    detected_methods = {
        str(row.get("method"))
        for row in selection_metadata
        if row.get("method")
    }
    selection_is_diagnostic = any(
        bool(row.get("diagnostic_only")) or row.get("deployable") is False
        for row in selection_metadata
    ) and not args.no_evidence
    detected_method = next(iter(detected_methods)) if len(detected_methods) == 1 else None
    method_name = (
        "no_evidence"
        if args.no_evidence
        else (args.method_name or detected_method or "rag_classification")
    )

    written = atomic_write_jsonl(
        output_path,
        iter_prediction_rows(
            selection_path=selection_path,
            scorer=scorer,
            labels=labels,
            verbalizers=verbalizers,
            metrics=metrics_acc,
            method_name=method_name,
            no_evidence=args.no_evidence,
            max_docs=args.max_docs,
            limit=args.limit,
        ),
    )
    metrics = metrics_acc.compute()
    metrics.update(
        {
            "dataset": config.get("dataset"),
            "split": args.split,
            "method": method_name or "rag_classification",
            "model_name": model_name,
            "diagnostic_only": selection_is_diagnostic,
            "deployable": not selection_is_diagnostic,
        }
    )
    atomic_write_json(metrics_path, metrics)
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "rag_cbwdm_evaluation_manifest.v1",
            "stage": "evaluation",
            "status": "completed",
            "completed": True,
            "method": method_name,
            "num_docs": 0 if args.no_evidence else None,
            "fingerprint": fingerprint,
            "contract": evaluation_contract,
            "num_rows": written,
            "predictions_sha256": sha256_file(output_path),
            "metrics_sha256": sha256_file(metrics_path),
            "git": git_state(PROJECT_ROOT),
            "end_time": utc_now(),
        },
    )
    print(
        f"[eval_rag][{config.get('dataset')}][{args.split}] rows={written} "
        f"accuracy={metrics['accuracy']:.6f} output={output_path} "
        f"metrics={metrics_path}"
    )


if __name__ == "__main__":
    main()

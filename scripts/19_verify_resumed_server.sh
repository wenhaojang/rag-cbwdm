#!/usr/bin/env bash
set -euo pipefail

# Read-only acceptance gate. It prints only PASS, or BLOCKED plus every blocker.
REPO="${REPO:-/root/rag-cbwdm}"
EXP_ROOT="${EXP_ROOT:-/root/experiments/rag_cbwdm}"
RUN_NAME="${RUN_NAME:-fever2_formal_pilot_5000_500_seed13}"
RUN="${RUN:-$EXP_ROOT/$RUN_NAME}"
MODEL_ROOT="${MODEL_ROOT:-/root/models}"
RETRIEVAL_ENV="${RETRIEVAL_ENV:-rag-cbwdm-retrieval}"
BASELINE_ENV="${BASELINE_ENV:-rag-cbwdm-baselines}"
EXPECTED_GIT_HEAD="${EXPECTED_GIT_HEAD:-}"
CONFIG="${CONFIG:-$REPO/configs/fever2_server_pilot_5000_500.yaml}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$REPO/outputs/formal_splits/fever2_seed13/fever2_formal_splits.manifest.json}"
RUN_MANIFEST="$RUN/run_manifest.json"

blockers=()

block() {
  blockers+=("$1")
}

json_manifest_path() {
  local manifest="$1"
  local key="$2"
  local fallback="$3"
  if [[ ! -f "$manifest" ]] || ! command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "$fallback"
    return
  fi
  python3 - "$manifest" "$key" "$fallback" 2>/dev/null <<'PY'
import json
import sys

manifest_path, dotted_key, fallback = sys.argv[1:]
try:
    value = json.loads(open(manifest_path, encoding="utf-8").read())
    for part in dotted_key.split("."):
        value = value[part]
    print(value if isinstance(value, str) and value else fallback)
except Exception:
    print(fallback)
PY
}

if [[ ! -d "$REPO/.git" ]]; then
  block "repository missing or not a Git worktree: $REPO"
elif ! command -v git >/dev/null 2>&1; then
  block "git command is unavailable"
else
  actual_head="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
  if [[ -z "$EXPECTED_GIT_HEAD" ]]; then
    block "EXPECTED_GIT_HEAD is not set"
  elif [[ "$actual_head" != "$EXPECTED_GIT_HEAD" ]]; then
    block "Git HEAD mismatch: expected=$EXPECTED_GIT_HEAD actual=${actual_head:-MISSING}"
  fi
  worktree_status="$(git -C "$REPO" status --porcelain 2>/dev/null || true)"
  if [[ -n "$worktree_status" ]]; then
    while IFS= read -r status_line; do
      [[ -n "$status_line" ]] && block "worktree is not clean: $status_line"
    done <<<"$worktree_status"
  fi
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  block "nvidia-smi is unavailable"
elif ! nvidia-smi >/dev/null 2>&1; then
  block "NVIDIA driver/GPU query failed"
fi

if ! command -v conda >/dev/null 2>&1; then
  block "conda command is unavailable"
else
  if ! conda run -n "$RETRIEVAL_ENV" python -c \
    'import importlib.metadata as m; import pyserini; assert m.version("pyserini")' \
    >/dev/null 2>&1; then
    block "$RETRIEVAL_ENV cannot import Pyserini or report its package version"
  fi
  if ! conda run -n "$RETRIEVAL_ENV" java -version >/dev/null 2>&1; then
    block "$RETRIEVAL_ENV cannot execute Java"
  fi
  if ! conda run -n "$BASELINE_ENV" python -c \
    'import torch, transformers; assert torch.cuda.is_available(); assert torch.cuda.get_device_name(0)' \
    >/dev/null 2>&1; then
    block "$BASELINE_ENV lacks importable torch/transformers or CUDA availability"
  fi
  if conda run -n "$BASELINE_ENV" python -c 'import pyserini' \
    >/dev/null 2>&1; then
    block "$BASELINE_ENV unexpectedly imports Pyserini; dual-environment boundary is not reproduced"
  fi
fi

MODEL_PATHS=(
  "$MODEL_ROOT/Qwen2.5-1.5B-Instruct"
  "$MODEL_ROOT/Qwen2.5-7B-Instruct"
  "$MODEL_ROOT/ms-marco-MiniLM-L-6-v2"
  "$MODEL_ROOT/bge-reranker-base"
  "$MODEL_ROOT/bge-reranker-large"
)
for model_path in "${MODEL_PATHS[@]}"; do
  if [[ ! -d "$model_path" ]]; then
    block "model directory missing: $model_path"
  elif [[ -z "$(find "$model_path" -type f -print -quit 2>/dev/null || true)" ]]; then
    block "model directory is empty: $model_path"
  fi
done

if [[ ! -f "$CONFIG" ]]; then
  block "pilot config missing: $CONFIG"
fi
for raw_path in \
  "$REPO/data/raw/fever/train.jsonl" \
  "$REPO/data/raw/fever/dev.jsonl"; do
  [[ -f "$raw_path" ]] || block "raw FEVER file missing: $raw_path"
done
if [[ ! -d "$REPO/data/raw/fever/wiki-pages" ]] \
  || [[ -z "$(
    find "$REPO/data/raw/fever/wiki-pages" -type f -print -quit 2>/dev/null \
      || true
  )" ]]; then
  block "raw FEVER wiki-pages directory is missing or empty"
fi
if [[ ! -f "$RUN_MANIFEST" ]]; then
  block "run manifest missing: $RUN_MANIFEST"
fi
if [[ ! -f "$SPLIT_MANIFEST" ]]; then
  block "formal split manifest missing: $SPLIT_MANIFEST"
elif command -v conda >/dev/null 2>&1; then
  if ! env PYTHONPATH="$REPO" conda run -n "$BASELINE_ENV" python -c \
    'import sys; from src.formal_splits import validate_split_manifest; validate_split_manifest(sys.argv[1])' \
    "$SPLIT_MANIFEST" >/dev/null 2>&1; then
    block "formal split manifest or its split/checksum/overlap contract is invalid"
  fi
fi

TRAIN_RETRIEVAL="$(json_manifest_path "$RUN_MANIFEST" paths.formal_retrieval.train_core "$RUN/artifacts/formal/fever2_train_core_bm25_top20.jsonl")"
VALIDATION_RETRIEVAL="$(json_manifest_path "$RUN_MANIFEST" paths.formal_retrieval.validation "$RUN/artifacts/formal/fever2_validation_bm25_top20.jsonl")"
HELD_OUT_RETRIEVAL="$(json_manifest_path "$RUN_MANIFEST" paths.formal_retrieval.held_out_test "$RUN/artifacts/formal/fever2_held_out_test_bm25_top20.jsonl")"
TRAIN_POSTERIOR="$(json_manifest_path "$RUN_MANIFEST" paths.formal_posteriors.train_core "$RUN/artifacts/formal/fever2_train_core_posteriors.jsonl")"
VALIDATION_POSTERIOR="$(json_manifest_path "$RUN_MANIFEST" paths.formal_posteriors.validation "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl")"
HELD_OUT_POSTERIOR="$(json_manifest_path "$RUN_MANIFEST" paths.formal_posteriors.held_out_test "$RUN/artifacts/formal/fever2_held_out_test_posteriors.jsonl")"
CORPUS_PATH="$(json_manifest_path "$RUN_MANIFEST" paths.corpus "$EXP_ROOT/_shared/MISSING_CORPUS")"
CORPUS_MANIFEST="$(json_manifest_path "$RUN_MANIFEST" paths.corpus_manifest "${CORPUS_PATH%.jsonl}.manifest.json")"
INDEX_PATH="$(json_manifest_path "$RUN_MANIFEST" paths.index_path "$EXP_ROOT/_shared/indexes/MISSING_INDEX")"
INDEX_MANIFEST="$(json_manifest_path "$RUN_MANIFEST" paths.index_manifest "$INDEX_PATH/index_manifest.json")"

if [[ ! -f "$CORPUS_PATH" ]] || [[ ! -f "$CORPUS_MANIFEST" ]]; then
  block "shared corpus or corpus manifest is missing"
elif command -v python3 >/dev/null 2>&1; then
  if ! python3 - "$CORPUS_PATH" "$CORPUS_MANIFEST" >/dev/null 2>&1 <<'PY'
import hashlib
import json
import sys

corpus, manifest_path = sys.argv[1:]
manifest = json.loads(open(manifest_path, encoding="utf-8").read())
hasher = hashlib.sha256()
with open(corpus, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        hasher.update(chunk)
digest = hasher.hexdigest()
assert manifest.get("status") == "completed"
assert manifest.get("output_sha256") == digest
PY
  then
    block "shared corpus manifest is incomplete or its output SHA-256 differs"
  fi
fi

if [[ ! -d "$INDEX_PATH" ]] || [[ ! -f "$INDEX_MANIFEST" ]]; then
  block "Lucene v2 index directory or manifest is missing"
elif command -v conda >/dev/null 2>&1; then
  if ! env PYTHONPATH="$REPO" conda run -n "$RETRIEVAL_ENV" python -c \
    'import sys; from src.retrieval.pyserini_bm25 import INDEX_CONTRACT_VERSION, validate_index; m=validate_index(sys.argv[1]); assert m["index_contract_schema_version"] == INDEX_CONTRACT_VERSION' \
    "$INDEX_PATH" >/dev/null 2>&1; then
    block "Lucene index fails the v2 contract, inventory, checksum, or metadata validation"
  fi
fi

expected_hashes=""
if [[ -f "$CONFIG" ]] && command -v conda >/dev/null 2>&1; then
  expected_hashes="$(
    env PYTHONPATH="$REPO" conda run -n "$BASELINE_ENV" python -c \
      'import sys; from src.io_utils import load_yaml; from src.prompts import fever_prompt_hash; from src.run_manifest import stable_hash; c=load_yaml(sys.argv[1]); print(fever_prompt_hash(list(c["task"]["labels"]), dict(c["task"]["verbalizers"])), stable_hash(c["task"]["verbalizers"]), sep="\t")' \
      "$CONFIG" 2>/dev/null | tail -n 1 || true
  )"
fi
if [[ -z "$expected_hashes" ]]; then
  block "could not compute expected prompt/verbalizer hashes from the pilot config"
fi
EXPECTED_PROMPT_HASH="${expected_hashes%%$'\t'*}"
EXPECTED_VERBALIZER_HASH="${expected_hashes#*$'\t'}"

if command -v python3 >/dev/null 2>&1; then
  artifact_blockers="$(
    python3 - \
      "$TRAIN_RETRIEVAL" 5000 train_core \
      "$VALIDATION_RETRIEVAL" 500 validation \
      "$TRAIN_POSTERIOR" 5000 train_core \
      "$VALIDATION_POSTERIOR" 500 validation \
      "$EXPECTED_PROMPT_HASH" "$EXPECTED_VERBALIZER_HASH" \
      "$RUN/artifacts/formal/calibration_candidates.json" 2>/dev/null <<'PY'
import hashlib
import json
import pathlib
import sys

(
    train_retrieval,
    train_retrieval_rows,
    train_role,
    validation_retrieval,
    validation_retrieval_rows,
    validation_role,
    train_posterior,
    train_posterior_rows,
    train_posterior_role,
    validation_posterior,
    validation_posterior_rows,
    validation_posterior_role,
    expected_prompt_hash,
    expected_verbalizer_hash,
    candidates_path,
) = sys.argv[1:]

issues = []

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_manifest(path):
    manifest_path = pathlib.Path(path).with_suffix(".manifest.json")
    if not manifest_path.is_file():
        issues.append(f"manifest missing: {manifest_path}")
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(f"manifest invalid: {manifest_path}: {exc}")
        return None

def validate_retrieval(path, expected_rows, role):
    path = pathlib.Path(path)
    expected_rows = int(expected_rows)
    if not path.is_file():
        issues.append(f"retrieval missing: {path}")
        return
    manifest = load_manifest(path)
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except Exception as exc:
        issues.append(f"retrieval JSONL invalid: {path}: {exc}")
        return
    if len(rows) != expected_rows:
        issues.append(f"retrieval row count mismatch: {path}: expected={expected_rows} actual={len(rows)}")
    if any(str(row.get("split")) != role for row in rows):
        issues.append(f"retrieval role mismatch: {path}: expected={role}")
    if any(len(row.get("candidates", [])) > 20 for row in rows):
        issues.append(f"retrieval candidate count exceeds top-N=20: {path}")
    if manifest:
        if manifest.get("completed") is not True:
            issues.append(f"retrieval manifest not completed: {path}")
        if manifest.get("split") != role:
            issues.append(f"retrieval manifest role mismatch: {path}")
        if manifest.get("top_n") != 20:
            issues.append(f"retrieval top-N mismatch: {path}: {manifest.get('top_n')!r}")
        if manifest.get("num_output_rows") != expected_rows:
            issues.append(f"retrieval manifest row count mismatch: {path}")
        if manifest.get("output_sha256") != sha256(path):
            issues.append(f"retrieval SHA-256 mismatch: {path}")

def validate_posterior(path, expected_rows, role, retrieval_path):
    path = pathlib.Path(path)
    expected_rows = int(expected_rows)
    if not path.is_file():
        issues.append(f"posterior missing: {path}")
        return
    manifest = load_manifest(path)
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except Exception as exc:
        issues.append(f"posterior JSONL invalid: {path}: {exc}")
        return
    if len(rows) != expected_rows:
        issues.append(f"posterior row count mismatch: {path}: expected={expected_rows} actual={len(rows)}")
    if any(str(row.get("split")) != role for row in rows):
        issues.append(f"posterior role mismatch: {path}: expected={role}")
    if manifest:
        provenance = manifest.get("provenance", {})
        if manifest.get("status") != "completed":
            issues.append(f"posterior manifest not completed: {path}")
        if manifest.get("expected_rows") != expected_rows or manifest.get("completed_rows") != expected_rows:
            issues.append(f"posterior manifest row count mismatch: {path}")
        if manifest.get("output_sha256") != sha256(path):
            issues.append(f"posterior SHA-256 mismatch: {path}")
        if provenance.get("split") != role:
            issues.append(f"posterior manifest role mismatch: {path}")
        retrieval = pathlib.Path(retrieval_path)
        if retrieval.is_file() and provenance.get("input_sha256") != sha256(retrieval):
            issues.append(f"posterior input retrieval SHA-256 mismatch: {path}")
        if provenance.get("prompt_template_hash") != expected_prompt_hash:
            issues.append(f"posterior prompt hash mismatch: {path}")
        if provenance.get("verbalizers_hash") != expected_verbalizer_hash:
            issues.append(f"posterior verbalizer hash mismatch: {path}")

validate_retrieval(train_retrieval, train_retrieval_rows, train_role)
validate_retrieval(validation_retrieval, validation_retrieval_rows, validation_role)
validate_posterior(train_posterior, train_posterior_rows, train_posterior_role, train_retrieval)
validate_posterior(validation_posterior, validation_posterior_rows, validation_posterior_role, validation_retrieval)

candidates_file = pathlib.Path(candidates_path)
if not candidates_file.is_file():
    issues.append(f"calibration candidates missing: {candidates_file}")
else:
    try:
        payload = json.loads(candidates_file.read_text(encoding="utf-8"))
        candidates = payload.get("candidates", payload if isinstance(payload, list) else [])
        for method in ("infogain_fever", "rag_cbwdm"):
            records = [row for row in candidates if isinstance(row, dict) and row.get("method") == method]
            for row in records:
                metrics = row.get("metrics")
                if row.get("status") != "completed":
                    issues.append(f"{method} smoke candidate not completed: {row.get('candidate_fingerprint')}")
                if not isinstance(metrics, dict) or metrics.get("accuracy") is None or metrics.get("macro_f1") is None:
                    issues.append(f"{method} smoke candidate metrics missing: {row.get('candidate_fingerprint')}")
            completed_metrics = {
                str(row.get("metrics_path"))
                for row in records
                if row.get("status") == "completed"
                and isinstance(row.get("metrics"), dict)
                and row["metrics"].get("accuracy") is not None
                and row["metrics"].get("macro_f1") is not None
            }
            grid_root = candidates_file.parent / "calibration_grid" / method
            if grid_root.is_dir():
                for metrics_path in grid_root.glob("*/evaluations/*/metrics.json"):
                    manifest_path = metrics_path.with_suffix(".manifest.json")
                    try:
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if (
                        manifest.get("status") == "completed"
                        and metrics.get("accuracy") is not None
                        and metrics.get("macro_f1") is not None
                    ):
                        completed_metrics.add(str(metrics_path.resolve()))
            if len(completed_metrics) < 2:
                issues.append(
                    f"{method} completed smoke evaluations fewer than 2: "
                    f"actual={len(completed_metrics)}"
                )
    except Exception as exc:
        issues.append(f"calibration candidates invalid: {candidates_file}: {exc}")

print("\n".join(issues))
PY
  )"
  if [[ -n "$artifact_blockers" ]]; then
    while IFS= read -r issue; do
      [[ -n "$issue" ]] && block "$issue"
    done <<<"$artifact_blockers"
  fi
else
  block "python3 is unavailable for artifact validation"
fi

# Do not open or parse held-out artifacts. Their mere existence is a blocker.
held_out_paths=(
  "$HELD_OUT_RETRIEVAL"
  "${HELD_OUT_RETRIEVAL%.jsonl}.manifest.json"
  "$HELD_OUT_POSTERIOR"
  "${HELD_OUT_POSTERIOR%.jsonl}.manifest.json"
)
for held_out_path in "${held_out_paths[@]}"; do
  if [[ -e "$held_out_path" ]]; then
    block "held-out artifact exists before readiness=ready: $held_out_path"
  fi
done

if (( ${#blockers[@]} == 0 )); then
  printf 'PASS\n'
else
  printf 'BLOCKED\n'
  printf -- '- %s\n' "${blockers[@]}"
  exit 2
fi

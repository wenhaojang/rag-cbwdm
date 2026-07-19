#!/usr/bin/env bash
set -euo pipefail

# Read-only with respect to source artifacts. Writes a new SNAPSHOT_ROOT and
# atomically refreshes the small, Git-safe metadata files under CAPTURED_DIR.
REPO="${REPO:-/root/rag-cbwdm}"
EXP_ROOT="${EXP_ROOT:-/root/experiments/rag_cbwdm}"
RUN_NAME="${RUN_NAME:-fever2_formal_pilot_5000_500_seed13}"
RUN="${RUN:-$EXP_ROOT/$RUN_NAME}"
MODEL_ROOT="${MODEL_ROOT:-/root/models}"
HF_HOME="${HF_HOME:-/root/huggingface}"
RETRIEVAL_ENV="${RETRIEVAL_ENV:-rag-cbwdm-retrieval}"
BASELINE_ENV="${BASELINE_ENV:-rag-cbwdm-baselines}"
TIMESTAMP="${SNAPSHOT_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-$EXP_ROOT/server_state_snapshots/$TIMESTAMP}"
CAPTURED_DIR="${CAPTURED_DIR:-$REPO/environment/server/captured}"

if [[ -e "$SNAPSHOT_ROOT" ]]; then
  printf 'Refusing to modify existing snapshot directory: %s\n' \
    "$SNAPSHOT_ROOT" >&2
  exit 2
fi
mkdir -p \
  "$SNAPSHOT_ROOT/system" \
  "$SNAPSHOT_ROOT/git" \
  "$SNAPSHOT_ROOT/environments" \
  "$SNAPSHOT_ROOT/models" \
  "$SNAPSHOT_ROOT/data" \
  "$SNAPSHOT_ROOT/artifacts" \
  "$SNAPSHOT_ROOT/processes" \
  "$SNAPSHOT_ROOT/storage" \
  "$SNAPSHOT_ROOT/checkpoints"

missing() {
  local output="$1"
  local message="$2"
  printf 'MISSING: %s\n' "$message" >"$output"
}

capture_command() {
  local output="$1"
  shift
  if ! "$@" >"$output" 2>&1; then
    printf '\nMISSING_OR_FAILED: command=%q' "$1" >>"$output"
    printf ' %q' "${@:2}" >>"$output"
    printf '\n' >>"$output"
  fi
}

file_fact() {
  local category="$1"
  local path="$2"
  local output="$3"
  if [[ -f "$path" ]]; then
    printf '%s\t%s\t%s\t%s\n' \
      "$category" \
      "$path" \
      "$(stat -c '%s' "$path")" \
      "$(sha256sum "$path" | awk '{print $1}')" >>"$output"
  else
    printf '%s\t%s\tMISSING\tMISSING\n' "$category" "$path" >>"$output"
  fi
}

json_manifest_path() {
  local manifest="$1"
  local key="$2"
  local fallback="$3"
  if [[ ! -f "$manifest" ]] || ! command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "$fallback"
    return
  fi
  python3 - "$manifest" "$key" "$fallback" <<'PY'
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

{
  printf 'captured_at_utc=%s\n' "$(date -u --iso-8601=seconds)"
  printf 'hostname=%s\n' "$(hostname 2>/dev/null || printf MISSING)"
  printf 'repo=%s\n' "$REPO"
  printf 'exp_root=%s\n' "$EXP_ROOT"
  printf 'run=%s\n' "$RUN"
  printf 'model_root=%s\n' "$MODEL_ROOT"
  printf 'hf_home=%s\n' "$HF_HOME"
} >"$SNAPSHOT_ROOT/system/identity.txt"

capture_command "$SNAPSHOT_ROOT/system/date.txt" date --iso-8601=seconds
capture_command "$SNAPSHOT_ROOT/system/hostnamectl.txt" hostnamectl
if [[ -r /etc/os-release ]]; then
  cp /etc/os-release "$SNAPSHOT_ROOT/system/os-release.txt"
else
  missing "$SNAPSHOT_ROOT/system/os-release.txt" "/etc/os-release"
fi
capture_command "$SNAPSHOT_ROOT/system/kernel.txt" uname -a
capture_command "$SNAPSHOT_ROOT/system/cpu.txt" lscpu
capture_command "$SNAPSHOT_ROOT/system/memory.txt" free -h
capture_command "$SNAPSHOT_ROOT/system/disks.txt" df -hT
capture_command "$SNAPSHOT_ROOT/system/block-devices.txt" lsblk -o NAME,SIZE,FSTYPE,TYPE,MOUNTPOINTS,MODEL,SERIAL

if command -v nvidia-smi >/dev/null 2>&1; then
  capture_command "$SNAPSHOT_ROOT/system/nvidia-smi.txt" nvidia-smi
  capture_command "$SNAPSHOT_ROOT/system/nvidia-smi-query.csv" \
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,compute_cap --format=csv
else
  missing "$SNAPSHOT_ROOT/system/nvidia-smi.txt" "nvidia-smi command"
  missing "$SNAPSHOT_ROOT/system/nvidia-smi-query.csv" "nvidia-smi command"
fi
if command -v nvcc >/dev/null 2>&1; then
  capture_command "$SNAPSHOT_ROOT/system/nvcc.txt" nvcc --version
else
  missing "$SNAPSHOT_ROOT/system/nvcc.txt" "nvcc is not installed or not on PATH"
fi

if [[ -d "$REPO/.git" ]] && command -v git >/dev/null 2>&1; then
  capture_command "$SNAPSHOT_ROOT/git/head.txt" git -C "$REPO" rev-parse HEAD
  capture_command "$SNAPSHOT_ROOT/git/branch.txt" git -C "$REPO" branch --show-current
  capture_command "$SNAPSHOT_ROOT/git/status.txt" git -C "$REPO" status --short --branch
  capture_command "$SNAPSHOT_ROOT/git/tags.txt" git -C "$REPO" tag --points-at HEAD
  if ! git -C "$REPO" remote -v 2>/dev/null \
    | sed -E 's#(https?://)[^/@[:space:]]+@#\1<redacted>@#g' \
    >"$SNAPSHOT_ROOT/git/remotes.txt"; then
    missing "$SNAPSHOT_ROOT/git/remotes.txt" "git remote inventory"
  fi
else
  for name in head branch status tags remotes; do
    missing "$SNAPSHOT_ROOT/git/$name.txt" "repository or git command"
  done
fi

if command -v conda >/dev/null 2>&1; then
  capture_command "$SNAPSHOT_ROOT/environments/conda-env-list.txt" conda env list
  for env_name in "$RETRIEVAL_ENV" "$BASELINE_ENV"; do
    env_dir="$SNAPSHOT_ROOT/environments/$env_name"
    mkdir -p "$env_dir"
    capture_command "$env_dir/python-version.txt" \
      conda run -n "$env_name" python --version
    capture_command "$env_dir/conda-no-builds.yml" \
      conda env export -n "$env_name" --no-builds
    capture_command "$env_dir/conda-from-history.yml" \
      conda env export -n "$env_name" --from-history
    capture_command "$env_dir/pip-freeze.txt" \
      conda run -n "$env_name" python -m pip freeze
  done
  capture_command "$SNAPSHOT_ROOT/environments/$RETRIEVAL_ENV/java-version.txt" \
    conda run -n "$RETRIEVAL_ENV" java -version
  capture_command "$SNAPSHOT_ROOT/environments/$RETRIEVAL_ENV/pyserini-version.txt" \
    conda run -n "$RETRIEVAL_ENV" python -c \
    'import importlib.metadata as m; import pyserini; print(m.version("pyserini"))'
  capture_command "$SNAPSHOT_ROOT/environments/$BASELINE_ENV/gpu-python-stack.json" \
    conda run -n "$BASELINE_ENV" python -c \
    'import json, torch, transformers; print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers.__version__, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, sort_keys=True))'
  if conda run -n "$BASELINE_ENV" python -c 'import pyserini' >/dev/null 2>&1; then
    printf 'PRESENT: baseline environment can import pyserini\n' \
      >"$SNAPSHOT_ROOT/environments/$BASELINE_ENV/pyserini-presence.txt"
  else
    printf 'ABSENT: baseline environment cannot import pyserini (expected)\n' \
      >"$SNAPSHOT_ROOT/environments/$BASELINE_ENV/pyserini-presence.txt"
  fi
else
  missing "$SNAPSHOT_ROOT/environments/conda-env-list.txt" "conda command"
  for env_name in "$RETRIEVAL_ENV" "$BASELINE_ENV"; do
    mkdir -p "$SNAPSHOT_ROOT/environments/$env_name"
    missing "$SNAPSHOT_ROOT/environments/$env_name/environment.txt" "conda command"
  done
fi

MODEL_PATHS=(
  "$MODEL_ROOT/Qwen2.5-1.5B-Instruct"
  "$MODEL_ROOT/Qwen2.5-7B-Instruct"
  "$MODEL_ROOT/ms-marco-MiniLM-L-6-v2"
  "$MODEL_ROOT/bge-reranker-base"
  "$MODEL_ROOT/bge-reranker-large"
)
printf 'model\tpath\tfile_count\tbytes\n' >"$SNAPSHOT_ROOT/models/inventory.tsv"
for model_path in "${MODEL_PATHS[@]}"; do
  model_name="$(basename "$model_path")"
  model_dir="$SNAPSHOT_ROOT/models/$model_name"
  mkdir -p "$model_dir"
  if [[ ! -d "$model_path" ]]; then
    printf '%s\t%s\tMISSING\tMISSING\n' "$model_name" "$model_path" \
      >>"$SNAPSHOT_ROOT/models/inventory.tsv"
    missing "$model_dir/config-tokenizer.sha256" "$model_path"
    missing "$model_dir/weights.sha256" "$model_path"
    continue
  fi
  file_count="$(find "$model_path" -type f | wc -l | tr -d ' ')"
  bytes="$(du -sb "$model_path" | awk '{print $1}')"
  printf '%s\t%s\t%s\t%s\n' "$model_name" "$model_path" "$file_count" "$bytes" \
    >>"$SNAPSHOT_ROOT/models/inventory.tsv"
  find "$model_path" -maxdepth 2 -type f \
    \( -name 'config.json' -o -name 'generation_config.json' \
       -o -name 'tokenizer.json' -o -name 'tokenizer_config.json' \
       -o -name 'special_tokens_map.json' -o -name 'vocab.json' \
       -o -name 'merges.txt' -o -name 'sentence_*.json' \
       -o -name 'modules.json' \) \
    -print0 | sort -z | xargs -0 -r sha256sum \
    >"$model_dir/config-tokenizer.sha256"
  find "$model_path" -maxdepth 2 -type f \
    \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \
       -o -name '*.pth' -o -name '*.ckpt' \) \
    -print0 | sort -z | xargs -0 -r sha256sum \
    >"$model_dir/weights.sha256"
done

printf 'category\tpath\tbytes\tsha256\n' >"$SNAPSHOT_ROOT/data/files.tsv"
file_fact raw_fever "$REPO/data/raw/fever/train.jsonl" "$SNAPSHOT_ROOT/data/files.tsv"
file_fact raw_fever "$REPO/data/raw/fever/dev.jsonl" "$SNAPSHOT_ROOT/data/files.tsv"
file_fact raw_fever "$REPO/data/raw/fever/shared_task_test.jsonl" "$SNAPSHOT_ROOT/data/files.tsv"
if [[ -d "$REPO/data/raw/fever/wiki-pages" ]]; then
  find "$REPO/data/raw/fever/wiki-pages" -type f -print0 \
    | sort -z | xargs -0 -r sha256sum \
    >"$SNAPSHOT_ROOT/data/wiki-pages.sha256"
else
  missing "$SNAPSHOT_ROOT/data/wiki-pages.sha256" "FEVER wiki-pages directory"
fi

RUN_MANIFEST="$RUN/run_manifest.json"
SPLIT_MANIFEST="$REPO/outputs/formal_splits/fever2_seed13/fever2_formal_splits.manifest.json"
CORPUS_PATH="$(json_manifest_path "$RUN_MANIFEST" paths.corpus "$EXP_ROOT/_shared/MISSING_CORPUS")"
CORPUS_MANIFEST="$(json_manifest_path "$RUN_MANIFEST" paths.corpus_manifest "${CORPUS_PATH%.jsonl}.manifest.json")"
INDEX_PATH="$(json_manifest_path "$RUN_MANIFEST" paths.index_path "$EXP_ROOT/_shared/indexes/MISSING_INDEX")"
INDEX_MANIFEST="$(json_manifest_path "$RUN_MANIFEST" paths.index_manifest "$INDEX_PATH/index_manifest.json")"
TRAIN_RETRIEVAL="$(json_manifest_path "$RUN_MANIFEST" paths.formal_retrieval.train_core "$RUN/artifacts/formal/fever2_train_core_bm25_top20.jsonl")"
VALIDATION_RETRIEVAL="$(json_manifest_path "$RUN_MANIFEST" paths.formal_retrieval.validation "$RUN/artifacts/formal/fever2_validation_bm25_top20.jsonl")"
TRAIN_POSTERIOR="$(json_manifest_path "$RUN_MANIFEST" paths.formal_posteriors.train_core "$RUN/artifacts/formal/fever2_train_core_posteriors.jsonl")"
VALIDATION_POSTERIOR="$(json_manifest_path "$RUN_MANIFEST" paths.formal_posteriors.validation "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl")"

printf 'category\tpath\tbytes\tsha256\n' >"$SNAPSHOT_ROOT/artifacts/files.tsv"
file_fact formal_split_manifest "$SPLIT_MANIFEST" "$SNAPSHOT_ROOT/artifacts/files.tsv"
file_fact shared_corpus "$CORPUS_PATH" "$SNAPSHOT_ROOT/artifacts/files.tsv"
file_fact shared_corpus_manifest "$CORPUS_MANIFEST" "$SNAPSHOT_ROOT/artifacts/files.tsv"
file_fact lucene_v2_manifest "$INDEX_MANIFEST" "$SNAPSHOT_ROOT/artifacts/files.tsv"
file_fact run_manifest "$RUN_MANIFEST" "$SNAPSHOT_ROOT/artifacts/files.tsv"
for artifact in \
  "$TRAIN_RETRIEVAL" "${TRAIN_RETRIEVAL%.jsonl}.manifest.json" \
  "$VALIDATION_RETRIEVAL" "${VALIDATION_RETRIEVAL%.jsonl}.manifest.json" \
  "$TRAIN_POSTERIOR" "${TRAIN_POSTERIOR%.jsonl}.manifest.json" \
  "$VALIDATION_POSTERIOR" "${VALIDATION_POSTERIOR%.jsonl}.manifest.json" \
  "$RUN/artifacts/formal/calibration_candidates.json" \
  "$RUN/artifacts/formal/calibration_grid_manifest.json"; do
  file_fact run_artifact "$artifact" "$SNAPSHOT_ROOT/artifacts/files.tsv"
done

for manifest in \
  "$SPLIT_MANIFEST" "$CORPUS_MANIFEST" "$INDEX_MANIFEST" "$RUN_MANIFEST" \
  "${TRAIN_RETRIEVAL%.jsonl}.manifest.json" \
  "${VALIDATION_RETRIEVAL%.jsonl}.manifest.json" \
  "${TRAIN_POSTERIOR%.jsonl}.manifest.json" \
  "${VALIDATION_POSTERIOR%.jsonl}.manifest.json" \
  "$RUN/artifacts/formal/calibration_grid_manifest.json"; do
  if [[ -f "$manifest" ]]; then
    cp "$manifest" "$SNAPSHOT_ROOT/artifacts/$(basename "$(dirname "$manifest")")--$(basename "$manifest")"
  fi
done

if [[ -f "$INDEX_MANIFEST" ]] && command -v python3 >/dev/null 2>&1; then
  python3 - "$INDEX_MANIFEST" >"$SNAPSHOT_ROOT/artifacts/lucene-v2-summary.json" <<'PY'
import json
import sys

manifest = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(json.dumps({
    "schema_version": manifest.get("schema_version"),
    "contract_version": manifest.get("index_contract_schema_version"),
    "fingerprint": manifest.get("fingerprint"),
    "index_path": manifest.get("index_path"),
    "index_file_inventory": manifest.get("index_file_inventory"),
}, indent=2, sort_keys=True))
PY
  sha256sum "$SNAPSHOT_ROOT/artifacts/lucene-v2-summary.json" \
    >"$SNAPSHOT_ROOT/artifacts/lucene-v2-summary.sha256"
else
  missing "$SNAPSHOT_ROOT/artifacts/lucene-v2-summary.json" "$INDEX_MANIFEST"
fi

if [[ -f "$RUN/artifacts/formal/calibration_candidates.json" ]] \
  && command -v python3 >/dev/null 2>&1; then
  python3 - "$RUN/artifacts/formal/calibration_candidates.json" \
    >"$SNAPSHOT_ROOT/artifacts/calibration-candidate-status-and-metrics.json" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
rows = payload.get("candidates", payload if isinstance(payload, list) else [])
result = [{
    "method": row.get("method"),
    "candidate_fingerprint": row.get("candidate_fingerprint"),
    "training_fingerprint": row.get("training_fingerprint"),
    "selection_fingerprint": row.get("selection_fingerprint"),
    "status": row.get("status"),
    "metrics": row.get("metrics"),
    "reason": row.get("reason"),
} for row in rows if isinstance(row, dict)]
print(json.dumps(result, indent=2, sort_keys=True))
PY
else
  missing "$SNAPSHOT_ROOT/artifacts/calibration-candidate-status-and-metrics.json" \
    "calibration_candidates.json"
fi

if [[ -d "$RUN/artifacts/formal/calibration_grid" ]] \
  && command -v python3 >/dev/null 2>&1; then
  python3 - "$RUN/artifacts/formal/calibration_grid" \
    >"$SNAPSHOT_ROOT/artifacts/all-grid-evaluation-status-and-metrics.json" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
records = []
for metrics_path in sorted(root.glob("*/*/evaluations/*/metrics.json")):
    manifest_path = metrics_path.with_suffix(".manifest.json")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        records.append({
            "method": metrics_path.relative_to(root).parts[0],
            "metrics_path": str(metrics_path),
            "status": "invalid",
            "reason": str(exc),
        })
        continue
    records.append({
        "method": metrics_path.relative_to(root).parts[0],
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "status": manifest.get("status"),
        "fingerprint": manifest.get("fingerprint"),
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "avg_num_docs": metrics.get("avg_num_docs"),
            "avg_evidence_chars": metrics.get("avg_evidence_chars"),
        },
    })
print(json.dumps(records, indent=2, sort_keys=True))
PY
else
  missing "$SNAPSHOT_ROOT/artifacts/all-grid-evaluation-status-and-metrics.json" \
    "calibration_grid directory"
fi

if [[ -d "$RUN/artifacts/formal/calibration_grid" ]]; then
  while IFS= read -r -d '' checkpoint_dir; do
    relative="${checkpoint_dir#"$RUN/artifacts/formal/calibration_grid/"}"
    safe_name="${relative//\//--}"
    {
      printf 'checkpoint_dir=%s\n' "$checkpoint_dir"
      find "$checkpoint_dir" -type f -print0 \
        | sort -z | xargs -0 -r sha256sum
    } >"$SNAPSHOT_ROOT/checkpoints/$safe_name.sha256"
  done < <(find "$RUN/artifacts/formal/calibration_grid" -type d -name checkpoint -print0)
else
  missing "$SNAPSHOT_ROOT/checkpoints/MISSING.txt" "calibration_grid directory"
fi

if command -v tmux >/dev/null 2>&1; then
  capture_command "$SNAPSHOT_ROOT/processes/tmux-ls.txt" tmux ls
else
  missing "$SNAPSHOT_ROOT/processes/tmux-ls.txt" "tmux command"
fi
if command -v pgrep >/dev/null 2>&1; then
  if ! pgrep -af 'python|run_fever_cbwdm|15a_run_fever_calibration_grid' \
    | sed -E 's/((token|api[_-]?key|password)[= ]+)[^ ]+/\1<redacted>/Ig' \
    >"$SNAPSHOT_ROOT/processes/python-processes.txt"; then
    printf 'NONE\n' >"$SNAPSHOT_ROOT/processes/python-processes.txt"
  fi
else
  missing "$SNAPSHOT_ROOT/processes/python-processes.txt" "pgrep command"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  capture_command "$SNAPSHOT_ROOT/processes/nvidia-compute-apps.csv" \
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
else
  missing "$SNAPSHOT_ROOT/processes/nvidia-compute-apps.csv" "nvidia-smi command"
fi

{
  du -sh "$REPO" 2>&1 || true
  du -sh "$EXP_ROOT" 2>&1 || true
  du -sh "$RUN" 2>&1 || true
  du -sh "$EXP_ROOT/_shared" 2>&1 || true
  du -sh "$MODEL_ROOT" 2>&1 || true
  du -sh "$HF_HOME" 2>&1 || true
} >"$SNAPSHOT_ROOT/storage/key-directory-sizes.txt"

GIT_HEAD="MISSING"
if [[ -f "$SNAPSHOT_ROOT/git/head.txt" ]]; then
  GIT_HEAD="$(head -n 1 "$SNAPSHOT_ROOT/git/head.txt")"
fi

# Publish only small, credential-sanitized reconstruction metadata into the
# repository. No raw data, model weights, experiment artifacts, or large logs
# are copied into environment/server/captured.
CAPTURED_PARENT="$(dirname "$CAPTURED_DIR")"
mkdir -p "$CAPTURED_PARENT"
CAPTURED_TMP="$(mktemp -d "$CAPTURED_PARENT/.captured.XXXXXX")"
trap 'rm -rf "$CAPTURED_TMP"' EXIT

cp "$SNAPSHOT_ROOT/environments/$RETRIEVAL_ENV/conda-no-builds.yml" \
  "$CAPTURED_TMP/retrieval.conda.no-builds.yml"
cp "$SNAPSHOT_ROOT/environments/$RETRIEVAL_ENV/pip-freeze.txt" \
  "$CAPTURED_TMP/retrieval.pip-freeze.txt"
cp "$SNAPSHOT_ROOT/environments/$BASELINE_ENV/conda-no-builds.yml" \
  "$CAPTURED_TMP/baselines.conda.no-builds.yml"
cp "$SNAPSHOT_ROOT/environments/$BASELINE_ENV/pip-freeze.txt" \
  "$CAPTURED_TMP/baselines.pip-freeze.txt"
cp "$SNAPSHOT_ROOT/git/head.txt" "$CAPTURED_TMP/git_head.txt"
cp "$SNAPSHOT_ROOT/environments/$BASELINE_ENV/gpu-python-stack.json" \
  "$CAPTURED_TMP/cuda_torch_info.json"

{
  printf 'captured_at_utc=%s\n' "$(date -u --iso-8601=seconds)"
  printf 'hostname=%s\n' "$(hostname 2>/dev/null || printf MISSING)"
  printf '\n[os-release]\n'
  cat "$SNAPSHOT_ROOT/system/os-release.txt"
  printf '\n[kernel]\n'
  cat "$SNAPSHOT_ROOT/system/kernel.txt"
  printf '\n[cpu]\n'
  cat "$SNAPSHOT_ROOT/system/cpu.txt"
  printf '\n[memory]\n'
  cat "$SNAPSHOT_ROOT/system/memory.txt"
  printf '\n[disks]\n'
  cat "$SNAPSHOT_ROOT/system/disks.txt"
  printf '\n[nvidia-smi]\n'
  sed -E 's/GPU-[0-9a-fA-F-]+/<redacted-gpu-uuid>/g' \
    "$SNAPSHOT_ROOT/system/nvidia-smi-query.csv"
  printf '\n[nvcc]\n'
  cat "$SNAPSHOT_ROOT/system/nvcc.txt"
} >"$CAPTURED_TMP/system_info.txt"

python3 - \
  "$SNAPSHOT_ROOT/environments/$RETRIEVAL_ENV/java-version.txt" \
  "$SNAPSHOT_ROOT/environments/$RETRIEVAL_ENV/pyserini-version.txt" \
  >"$CAPTURED_TMP/java_pyserini_info.json" <<'PY'
import json
import pathlib
import sys

java_path, pyserini_path = map(pathlib.Path, sys.argv[1:])
print(json.dumps({
    "java_version_output": java_path.read_text(encoding="utf-8", errors="replace").strip(),
    "pyserini_version_output": pyserini_path.read_text(encoding="utf-8", errors="replace").strip(),
}, indent=2, sort_keys=True))
PY

python3 - "$MODEL_ROOT" >"$CAPTURED_TMP/model_inventory.json" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
models = {
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-large": "BAAI/bge-reranker-large",
}

def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

inventory = []
for directory_name, repo_id in models.items():
    path = root / directory_name
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else []
    config = {}
    config_path = path / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}
    selected = [
        item for item in files
        if item.name in {
            "config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json",
        }
        or item.suffix in {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    ]
    inventory.append({
        "directory_name": directory_name,
        "path": str(path),
        "repo_id": repo_id,
        "revision": config.get("_commit_hash"),
        "configured_name_or_path": config.get("_name_or_path"),
        "exists": path.is_dir(),
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "required_file_sha256": {
            str(item.relative_to(path)): digest(item) for item in selected
        },
        "revision_blocker": (
            None if config.get("_commit_hash")
            else "immutable revision is not recorded in config.json; recover it from the HF cache/snapshot metadata"
        ),
    })
print(json.dumps({"models": inventory}, indent=2, sort_keys=True))
PY

python3 - "$REPO/data/raw/fever" "${FEVER_DATASET_REVISION:-}" \
  >"$CAPTURED_TMP/data_inventory.json" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
dataset_revision = sys.argv[2] or None

def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def line_count(path):
    with path.open("rb") as handle:
        return sum(1 for _ in handle)

files = []
for path in [root / "train.jsonl", root / "dev.jsonl"]:
    files.append({
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "rows": line_count(path) if path.is_file() else None,
        "sha256": digest(path) if path.is_file() else None,
    })
wiki = sorted((root / "wiki-pages").glob("*.jsonl")) if (root / "wiki-pages").is_dir() else []
files.extend({
    "path": str(path),
    "exists": True,
    "bytes": path.stat().st_size,
    "rows": line_count(path),
    "sha256": digest(path),
} for path in wiki)
print(json.dumps({
    "dataset_id": "fever",
    "dataset_revision": dataset_revision,
    "dataset_revision_blocker": None if dataset_revision else (
        "scripts/download_fever_hf.py requires an immutable revision; set "
        "FEVER_DATASET_REVISION when capturing the source server"
    ),
    "files": files,
}, indent=2, sort_keys=True))
PY

python3 - "$CAPTURED_TMP" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
credential_url = re.compile(r"(https?://)([^/@\s]+)@", re.I)
token_path = re.compile(r"(/t/)[A-Za-z0-9._-]{12,}(/)", re.I)
assignment = re.compile(
    r"(?i)\b(token|password|passwd|secret|api[_-]?key|hf_token|access[_-]?key)"
    r"(\s*[:=]\s*)([^\s,}\]]+)"
)
for path in root.iterdir():
    if not path.is_file():
        continue
    if path.stat().st_size > 5 * 1024 * 1024:
        raise SystemExit(f"captured file exceeds 5 MiB Git safety limit: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    text = credential_url.sub(r"\1<redacted>@", text)
    text = token_path.sub(r"\1<redacted>\2", text)
    text = assignment.sub(r"\1\2<redacted>", text)
    if "-----BEGIN " in text and "PRIVATE KEY-----" in text:
        raise SystemExit(f"private key marker found in captured file: {path}")
    path.write_text(text, encoding="utf-8", newline="\n")
PY

mkdir -p "$CAPTURED_DIR"
for captured_file in "$CAPTURED_TMP"/*; do
  captured_name="$(basename "$captured_file")"
  captured_target="$CAPTURED_DIR/$captured_name"
  captured_staging="$CAPTURED_DIR/.$captured_name.tmp.$$"
  install -m 0644 "$captured_file" "$captured_staging"
  mv -f "$captured_staging" "$captured_target"
done
printf '%s\n' "$CAPTURED_DIR" >"$SNAPSHOT_ROOT/git/captured-environment-directory.txt"

cat >"$SNAPSHOT_ROOT/README.txt" <<EOF
RAG-CBWDM server resume-state snapshot
captured_at_utc=$(date -u --iso-8601=seconds)
git_head=$GIT_HEAD
repo=$REPO
run=$RUN

This snapshot was collected read-only. It did not run retrieval, posterior,
training, selection, evaluation, or model loading. It did not modify existing
experiment artifacts or manifests. It did not parse held_out_test JSONL content;
held-out paths are represented only through existing manifest/path metadata.

MISSING and MISSING_OR_FAILED records are intentional audit evidence and must
be reviewed before stopping, releasing, deleting, or migrating the server.

Small credential-sanitized reconstruction files were also published to:
$CAPTURED_DIR
Review them for secrets and size before git add/commit.
EOF

printf '%s\n' "$SNAPSHOT_ROOT"

#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/rag-cbwdm}"
MODEL_ROOT="${MODEL_ROOT:-/root/models}"
HF_HOME="${HF_HOME:-/root/huggingface}"
DATA_ROOT="${DATA_ROOT:-$REPO/data/raw/fever}"
ASSET_MANIFEST="${ASSET_MANIFEST:-$DATA_ROOT/project_assets.manifest.json}"
CAPTURED_DIR="${CAPTURED_DIR:-$REPO/environment/server/captured}"
MINICONDA_ROOT="${MINICONDA_ROOT:-/root/miniconda3}"
BASELINE_ENV="${BASELINE_ENV:-rag-cbwdm-baselines}"
CONDA_BIN="${CONDA_BIN:-$MINICONDA_ROOT/bin/conda}"

CHECK_ONLY=0
MODELS=1
DATA=1
RESUME=1
ALLOW_UNPINNED_ASSETS="${ALLOW_UNPINNED_ASSETS:-0}"
RESOLVED_MODELS_JSON=""

usage() {
  cat <<'EOF'
Usage: bash scripts/21_download_project_assets.sh [OPTIONS]

Options:
  --check-only   Validate local assets and SHA-256 values; never use network.
  --models-only  Download/validate only five model directories.
  --data-only    Download/materialize only FEVER train/dev/wiki pages.
  --resume       Explicitly request resumable downloads (default behavior).
  --allow-unpinned
                 Explicitly allow functional recovery from current upstream
                 defaults when immutable revisions are unavailable.

Required immutable revision environment variables:
  QWEN15_REVISION, QWEN7_REVISION, MINILM_REVISION
  BGE_BASE_REVISION, BGE_LARGE_REVISION, FEVER_DATASET_REVISION

Paths may be overridden with MODEL_ROOT, HF_HOME, DATA_ROOT, ASSET_MANIFEST,
CAPTURED_DIR, MINICONDA_ROOT, and BASELINE_ENV.
ALLOW_UNPINNED_ASSETS=1 is equivalent to --allow-unpinned.
EOF
}

while (( $# )); do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      ;;
    --models-only)
      MODELS=1
      DATA=0
      ;;
    --data-only)
      MODELS=0
      DATA=1
      ;;
    --resume)
      RESUME=1
      ;;
    --allow-unpinned)
      ALLOW_UNPINNED_ASSETS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

fail() {
  printf 'BLOCKED: %s\n' "$1" >&2
  exit 2
}

[[ "$ALLOW_UNPINNED_ASSETS" == "0" || "$ALLOW_UNPINNED_ASSETS" == "1" ]] \
  || fail "ALLOW_UNPINNED_ASSETS must be 0 or 1"

cleanup() {
  if [[ -n "$RESOLVED_MODELS_JSON" && -f "$RESOLVED_MODELS_JSON" ]]; then
    rm -f -- "$RESOLVED_MODELS_JSON"
  fi
}
trap cleanup EXIT

[[ -x "$CONDA_BIN" ]] || fail "conda executable missing: $CONDA_BIN"
[[ -d "$REPO/.git" ]] || fail "repository missing: $REPO"

captured_revision() {
  local directory_name="$1"
  local inventory="$CAPTURED_DIR/model_inventory.json"
  [[ -f "$inventory" ]] || return 0
  "$CONDA_BIN" run -n "$BASELINE_ENV" python -c \
    'import json,sys; rows=json.load(open(sys.argv[1])).get("models", []); print(next((r.get("revision") or "" for r in rows if r.get("directory_name") == sys.argv[2]), ""))' \
    "$inventory" "$directory_name" 2>/dev/null | tail -n 1
}

manifest_revision() {
  local repo_id="$1"
  [[ -f "$ASSET_MANIFEST" ]] || return 0
  "$CONDA_BIN" run -n "$BASELINE_ENV" python -c \
    'import json,sys; rows=json.load(open(sys.argv[1])).get("models", []); print(next((r.get("revision") or "" for r in rows if r.get("repo_id") == sys.argv[2]), ""))' \
    "$ASSET_MANIFEST" "$repo_id" 2>/dev/null | tail -n 1
}

QWEN15_REVISION="${QWEN15_REVISION:-$(captured_revision Qwen2.5-1.5B-Instruct)}"
QWEN15_REVISION="${QWEN15_REVISION:-$(manifest_revision Qwen/Qwen2.5-1.5B-Instruct)}"
QWEN7_REVISION="${QWEN7_REVISION:-$(captured_revision Qwen2.5-7B-Instruct)}"
QWEN7_REVISION="${QWEN7_REVISION:-$(manifest_revision Qwen/Qwen2.5-7B-Instruct)}"
MINILM_REVISION="${MINILM_REVISION:-$(captured_revision ms-marco-MiniLM-L-6-v2)}"
MINILM_REVISION="${MINILM_REVISION:-$(manifest_revision cross-encoder/ms-marco-MiniLM-L-6-v2)}"
BGE_BASE_REVISION="${BGE_BASE_REVISION:-$(captured_revision bge-reranker-base)}"
BGE_BASE_REVISION="${BGE_BASE_REVISION:-$(manifest_revision BAAI/bge-reranker-base)}"
BGE_LARGE_REVISION="${BGE_LARGE_REVISION:-$(captured_revision bge-reranker-large)}"
BGE_LARGE_REVISION="${BGE_LARGE_REVISION:-$(manifest_revision BAAI/bge-reranker-large)}"
if [[ -z "${FEVER_DATASET_REVISION:-}" ]] \
  && [[ -f "$CAPTURED_DIR/data_inventory.json" ]]; then
  FEVER_DATASET_REVISION="$(
    "$CONDA_BIN" run -n "$BASELINE_ENV" python -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("dataset_revision") or "")' \
      "$CAPTURED_DIR/data_inventory.json" 2>/dev/null | tail -n 1
  )"
fi
if [[ -z "${FEVER_DATASET_REVISION:-}" ]] \
  && [[ -f "$ASSET_MANIFEST" ]]; then
  FEVER_DATASET_REVISION="$(
    "$CONDA_BIN" run -n "$BASELINE_ENV" python -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("data", {}).get("revision") or "")' \
      "$ASSET_MANIFEST" 2>/dev/null | tail -n 1
  )"
fi

MODELS_REVISION_MODE="pinned"
if (( MODELS == 1 )); then
  for assignment in \
    "QWEN15_REVISION=$QWEN15_REVISION" \
    "QWEN7_REVISION=$QWEN7_REVISION" \
    "MINILM_REVISION=$MINILM_REVISION" \
    "BGE_BASE_REVISION=$BGE_BASE_REVISION" \
    "BGE_LARGE_REVISION=$BGE_LARGE_REVISION"; do
    value="${assignment#*=}"
    if [[ -z "$value" || "$value" == "null" ]]; then
      if (( ALLOW_UNPINNED_ASSETS == 0 )); then
        fail "${assignment%%=*} is unknown; provide an immutable revision or explicitly use --allow-unpinned"
      fi
      MODELS_REVISION_MODE="unpinned"
    fi
  done
fi

DATA_REVISION_MODE="pinned"
if (( DATA == 1 )); then
  if [[ -z "${FEVER_DATASET_REVISION:-}" \
    || "${FEVER_DATASET_REVISION:-}" == "null" ]]; then
    if (( ALLOW_UNPINNED_ASSETS == 0 )); then
      fail "FEVER_DATASET_REVISION is unknown; provide an immutable revision or explicitly use --allow-unpinned"
    fi
    DATA_REVISION_MODE="unpinned"
  fi
fi

if [[ "$MODELS_REVISION_MODE" == "unpinned" \
  || "$DATA_REVISION_MODE" == "unpinned" ]]; then
  printf '%s\n' \
    '[assets] WARNING revision_mode=unpinned: functional reproduction only; byte-for-byte reproduction is not guaranteed.' >&2
fi

validate_manifest() {
  local require_models="$1"
  local require_data="$2"
  [[ -f "$ASSET_MANIFEST" ]] || return 1
  "$CONDA_BIN" run --no-capture-output -n "$BASELINE_ENV" python - \
    "$ASSET_MANIFEST" "$require_models" "$require_data" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
require_models = sys.argv[2] == "1"
require_data = sys.argv[3] == "1"
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
assert payload.get("status") == "completed"
revision_mode = payload.get("revision_mode", "pinned")
assert revision_mode in {"pinned", "unpinned"}

def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

sections = []
if require_models:
    for model in payload.get("models", []):
        assert model.get("repo_id") and model.get("revision")
        assert model.get("revision_mode", "pinned") in {"pinned", "unpinned"}
        if model.get("revision_mode") == "unpinned":
            assert model.get("resolved_revision")
        sections.extend(model.get("files", []))
if require_data:
    data = payload.get("data", {})
    assert data.get("revision_mode", "pinned") in {"pinned", "unpinned"}
    if data.get("revision_mode") == "unpinned":
        assert data.get("dataset_fingerprint")
    sections.extend(data.get("files", []))
assert sections
for item in sections:
    path = pathlib.Path(item["path"])
    assert path.is_file(), path
    assert path.stat().st_size == item["bytes"], path
    assert digest(path) == item["sha256"], path
PY
}

if (( CHECK_ONLY == 1 )); then
  validate_manifest "$MODELS" "$DATA" \
    || fail "asset manifest validation failed: $ASSET_MANIFEST"
  if (( MODELS == 1 )); then
    "$CONDA_BIN" run --no-capture-output -n "$BASELINE_ENV" python - \
      "$MODEL_ROOT" <<'PY'
import pathlib
import sys
from transformers import AutoConfig, AutoTokenizer

root = pathlib.Path(sys.argv[1])
for name in (
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-7B-Instruct",
    "ms-marco-MiniLM-L-6-v2",
    "bge-reranker-base",
    "bge-reranker-large",
):
    path = root / name
    AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
PY
  fi
  printf '[assets] PASS mode=check-only manifest=%s\n' "$ASSET_MANIFEST"
  exit 0
fi

mkdir -p "$MODEL_ROOT" "$HF_HOME" "$DATA_ROOT"
export HF_HOME

if (( MODELS == 1 )); then
  RESOLVED_MODELS_JSON="$(mktemp /tmp/rag-cbwdm-model-revisions.XXXXXX)"
  "$CONDA_BIN" run --no-capture-output -n "$BASELINE_ENV" python - \
    "$MODEL_ROOT" "$HF_HOME" "$RESOLVED_MODELS_JSON" \
    "Qwen/Qwen2.5-1.5B-Instruct=$QWEN15_REVISION" \
    "Qwen/Qwen2.5-7B-Instruct=$QWEN7_REVISION" \
    "cross-encoder/ms-marco-MiniLM-L-6-v2=$MINILM_REVISION" \
    "BAAI/bge-reranker-base=$BGE_BASE_REVISION" \
    "BAAI/bge-reranker-large=$BGE_LARGE_REVISION" <<'PY'
import pathlib
import json
import sys
from huggingface_hub import HfApi, snapshot_download

root = pathlib.Path(sys.argv[1])
cache = sys.argv[2]
resolved_path = pathlib.Path(sys.argv[3])
api = HfApi()
resolved = []
for assignment in sys.argv[4:]:
    repo_id, revision = assignment.rsplit("=", 1)
    requested_revision = revision or None
    info = api.model_info(repo_id, revision=requested_revision)
    resolved_revision = info.sha
    if not resolved_revision:
        raise RuntimeError(f"Hub did not return a commit hash for {repo_id}")
    target = root / repo_id.rsplit("/", 1)[-1]
    snapshot_download(
        repo_id=repo_id,
        revision=resolved_revision,
        local_dir=target,
        cache_dir=cache,
    )
    resolved.append({
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "revision_mode": "pinned" if requested_revision else "unpinned",
    })
resolved_path.write_text(
    json.dumps({"models": resolved}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
fi

if (( DATA == 1 )); then
  FEVER_DOWNLOAD_REVISION="${FEVER_DATASET_REVISION:-}"
  if [[ -z "$FEVER_DOWNLOAD_REVISION" \
    || "$FEVER_DOWNLOAD_REVISION" == "null" ]]; then
    FEVER_DOWNLOAD_REVISION="main"
  fi
  "$CONDA_BIN" run --no-capture-output -n "$BASELINE_ENV" python \
    "$REPO/scripts/download_fever_hf.py" \
    --output-root "$DATA_ROOT" \
    --cache-dir "$HF_HOME/datasets" \
    --revision "$FEVER_DOWNLOAD_REVISION"
fi

"$CONDA_BIN" run --no-capture-output -n "$BASELINE_ENV" python - \
  "$ASSET_MANIFEST" "$MODEL_ROOT" "$DATA_ROOT" "$MODELS" "$DATA" \
  "$QWEN15_REVISION" "$QWEN7_REVISION" "$MINILM_REVISION" \
  "$BGE_BASE_REVISION" "$BGE_LARGE_REVISION" \
  "${FEVER_DATASET_REVISION:-}" "$RESUME" \
  "$RESOLVED_MODELS_JSON" "$MODELS_REVISION_MODE" \
  "$DATA_REVISION_MODE" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from datetime import datetime, timezone
from transformers import AutoConfig, AutoTokenizer

(
    manifest_arg,
    model_root_arg,
    data_root_arg,
    models_flag,
    data_flag,
    qwen15_revision,
    qwen7_revision,
    minilm_revision,
    bge_base_revision,
    bge_large_revision,
    fever_revision,
    resume_flag,
    resolved_models_arg,
    models_revision_mode,
    data_revision_mode,
) = sys.argv[1:]
manifest_path = pathlib.Path(manifest_arg)
model_root = pathlib.Path(model_root_arg)
data_root = pathlib.Path(data_root_arg)
include_models = models_flag == "1"
include_data = data_flag == "1"
assert models_revision_mode in {"pinned", "unpinned"}
assert data_revision_mode in {"pinned", "unpinned"}

def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def identity(path):
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }

payload = {}
if manifest_path.is_file():
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
payload.update({
    "schema_version": "rag_cbwdm_project_assets.v1",
    "status": "completed",
    "resume_enabled": resume_flag == "1",
    "updated_at": datetime.now(timezone.utc).isoformat(),
})

if include_models:
    resolved_payload = json.loads(
        pathlib.Path(resolved_models_arg).read_text(encoding="utf-8")
    )
    resolved_by_repo = {
        item["repo_id"]: item for item in resolved_payload["models"]
    }
    model_specs = [
        ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct", qwen15_revision),
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", qwen7_revision),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "ms-marco-MiniLM-L-6-v2", minilm_revision),
        ("BAAI/bge-reranker-base", "bge-reranker-base", bge_base_revision),
        ("BAAI/bge-reranker-large", "bge-reranker-large", bge_large_revision),
    ]
    model_records = []
    for repo_id, directory_name, requested_revision in model_specs:
        resolved = resolved_by_repo[repo_id]
        resolved_revision = resolved["resolved_revision"]
        revision_mode = resolved["revision_mode"]
        path = model_root / directory_name
        AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        configs = [
            item for item in path.rglob("*")
            if item.is_file() and item.name in {
                "config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json",
            }
        ]
        weights = [
            item for item in path.rglob("*")
            if item.is_file() and item.suffix in {
                ".safetensors", ".bin", ".pt", ".pth", ".ckpt"
            }
        ]
        assert configs, f"config/tokenizer files missing: {path}"
        assert weights, f"weight files missing: {path}"
        required = sorted({*configs, *weights})
        model_records.append({
            "repo_id": repo_id,
            "revision": resolved_revision,
            "requested_revision": requested_revision or None,
            "resolved_revision": resolved_revision,
            "revision_mode": revision_mode,
            "directory": str(path.resolve()),
            "config_and_tokenizer_file_count": len(configs),
            "weight_file_count": len(weights),
            "total_bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file()),
            "files": [identity(item) for item in required],
        })
    payload["models"] = model_records

if include_data:
    files = [data_root / "train.jsonl", data_root / "dev.jsonl"]
    files.extend(sorted((data_root / "wiki-pages").glob("wiki-*.jsonl")))
    assert len(files) > 2, "FEVER wiki pages are missing"
    for path in files:
        assert path.is_file() and path.stat().st_size > 0, path
    data_records = []
    for path in files:
        record = identity(path)
        with path.open("rb") as handle:
            record["rows"] = sum(1 for _ in handle)
        data_records.append(record)
    fingerprint = hashlib.sha256()
    for record in data_records:
        relative_path = pathlib.Path(record["path"]).relative_to(
            data_root.resolve()
        )
        fingerprint.update(relative_path.as_posix().encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(record["sha256"].encode("ascii"))
        fingerprint.update(b"\n")
    payload["data"] = {
        "dataset_id": "fever",
        "revision": fever_revision or None,
        "requested_revision": fever_revision or None,
        "revision_mode": data_revision_mode,
        "dataset_fingerprint": fingerprint.hexdigest(),
        "files": data_records,
    }

section_modes = []
section_modes.extend(
    item.get("revision_mode", "pinned")
    for item in payload.get("models", [])
)
if payload.get("data"):
    section_modes.append(payload["data"].get("revision_mode", "pinned"))
payload["revision_mode"] = (
    "unpinned" if "unpinned" in section_modes else "pinned"
)
payload["reproduction_scope"] = (
    "functional_reproduction_only_not_byte_identical"
    if payload["revision_mode"] == "unpinned"
    else "pinned_assets_with_content_sha256"
)

manifest_path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".assets-", dir=manifest_path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, manifest_path)
PY

validate_manifest "$MODELS" "$DATA" \
  || fail "post-download asset manifest validation failed"
printf '[assets] PASS mode=download manifest=%s\n' "$ASSET_MANIFEST"

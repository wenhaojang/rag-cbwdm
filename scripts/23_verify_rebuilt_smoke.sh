#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/rag-cbwdm}"
EXP_ROOT="${EXP_ROOT:-/root/experiments/rag_cbwdm}"
RUN_NAME="${RUN_NAME:-fever2_formal_pilot_5000_500_seed13}"
RUN="${RUN:-$EXP_ROOT/$RUN_NAME}"
MODEL_ROOT="${MODEL_ROOT:-/root/models}"
HF_HOME="${HF_HOME:-/root/huggingface}"
DATA_ROOT="${DATA_ROOT:-$REPO/data/raw/fever}"
ASSET_MANIFEST="${ASSET_MANIFEST:-$DATA_ROOT/project_assets.manifest.json}"
CAPTURED_DIR="${CAPTURED_DIR:-$REPO/environment/server/captured}"
MINICONDA_ROOT="${MINICONDA_ROOT:-/root/miniconda3}"
RETRIEVAL_ENV="${RETRIEVAL_ENV:-rag-cbwdm-retrieval}"
BASELINE_ENV="${BASELINE_ENV:-rag-cbwdm-baselines}"
EXPECTED_GIT_HEAD="${EXPECTED_GIT_HEAD:-}"

blockers=()

append_output_blockers() {
  local prefix="$1"
  local output="$2"
  local found=0
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == "PASS" ]] && continue
    if [[ "$line" == "BLOCKED" ]]; then
      found=1
      continue
    fi
    line="${line#- }"
    blockers+=("$prefix: $line")
    found=1
  done <<<"$output"
  if (( found == 0 )); then
    blockers+=("$prefix failed without diagnostic output")
  fi
}

if [[ -z "$EXPECTED_GIT_HEAD" ]]; then
  blockers+=("EXPECTED_GIT_HEAD is not set to the final pushed recovery commit")
fi

export REPO EXP_ROOT RUN_NAME RUN MODEL_ROOT HF_HOME DATA_ROOT ASSET_MANIFEST
export CAPTURED_DIR MINICONDA_ROOT RETRIEVAL_ENV BASELINE_ENV EXPECTED_GIT_HEAD

set +e
server_output="$(bash "$REPO/scripts/19_verify_resumed_server.sh" 2>&1)"
server_exit=$?
set -e
if (( server_exit != 0 )); then
  append_output_blockers "smoke state" "$server_output"
fi

set +e
asset_output="$(
  bash "$REPO/scripts/21_download_project_assets.sh" --check-only 2>&1
)"
asset_exit=$?
set -e
if (( asset_exit != 0 )); then
  append_output_blockers "asset validation" "$asset_output"
fi

if (( ${#blockers[@]} == 0 )); then
  printf 'PASS\n'
else
  printf 'BLOCKED\n'
  printf -- '- %s\n' "${blockers[@]}"
  exit 2
fi

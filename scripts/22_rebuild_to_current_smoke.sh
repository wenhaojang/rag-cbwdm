#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/rag-cbwdm}"
EXP_ROOT="${EXP_ROOT:-/root/experiments/rag_cbwdm}"
RUN_NAME="${RUN_NAME:-fever2_formal_pilot_5000_500_seed13}"
RUN="${RUN:-$EXP_ROOT/$RUN_NAME}"
HF_HOME="${HF_HOME:-/root/huggingface}"
MODEL_ROOT="${MODEL_ROOT:-/root/models}"
CONFIG="${CONFIG:-$REPO/configs/fever2_server_pilot_5000_500.yaml}"
MINICONDA_ROOT="${MINICONDA_ROOT:-/root/miniconda3}"
CONDA_BIN="${CONDA_BIN:-$MINICONDA_ROOT/bin/conda}"
RETRIEVAL_ENV="${RETRIEVAL_ENV:-rag-cbwdm-retrieval}"
BASELINE_ENV="${BASELINE_ENV:-rag-cbwdm-baselines}"
EXPECTED_GIT_HEAD="${EXPECTED_GIT_HEAD:-}"
REPORT_PATH="${REPORT_PATH:-$RUN/artifacts/formal/rebuild_to_current_smoke_report.json}"

DRY_RUN=0
VERIFY_ONLY=0
STOP_AFTER=""

STAGES=(
  prepare_formal_splits
  corpus
  index
  retrieve_train_core
  retrieve_validation
  posterior_train_core
  posterior_validation
  infogain_smoke
  rag_cbwdm_smoke
)

usage() {
  cat <<'EOF'
Usage: bash scripts/22_rebuild_to_current_smoke.sh [OPTIONS]

Options:
  --stop-after STAGE  Stop after one of:
                      prepare_formal_splits, corpus, index,
                      retrieve_train_core, retrieve_validation,
                      posterior_train_core, posterior_validation,
                      infogain_smoke, rag_cbwdm_smoke
  --dry-run           Print exact conda/runner commands; execute nothing.
  --verify-only       Run scripts/23_verify_rebuilt_smoke.sh only.

All production stages use current manifest/resume validation. Held-out stages
are intentionally absent.
EOF
}

while (( $# )); do
  case "$1" in
    --stop-after)
      [[ $# -ge 2 ]] || { printf '%s\n' "--stop-after requires a value" >&2; exit 2; }
      STOP_AFTER="$2"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --verify-only)
      VERIFY_ONLY=1
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

if [[ -n "$STOP_AFTER" ]]; then
  valid=0
  for stage in "${STAGES[@]}"; do
    [[ "$stage" == "$STOP_AFTER" ]] && valid=1
  done
  (( valid == 1 )) || {
    printf 'Unknown --stop-after stage: %s\n' "$STOP_AFTER" >&2
    exit 2
  }
fi

[[ -x "$CONDA_BIN" ]] || {
  printf 'Conda executable missing: %s\n' "$CONDA_BIN" >&2
  exit 2
}
[[ -d "$REPO/.git" ]] || {
  printf 'Repository missing: %s\n' "$REPO" >&2
  exit 2
}

[[ -n "$EXPECTED_GIT_HEAD" ]] || {
  printf 'EXPECTED_GIT_HEAD is required (use the final pushed recovery commit)\n' >&2
  exit 2
}
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXPECTED_GIT_HEAD" ]] || {
  printf 'Git HEAD does not match EXPECTED_GIT_HEAD\n' >&2
  exit 2
}

run_command() {
  local stage="$1"
  shift
  if (( DRY_RUN == 1 )); then
    printf '[rebuild][dry-run][%s] ' "$stage"
    printf '%q ' "$@"
    printf '\n'
  else
    printf '[rebuild][%s] starting\n' "$stage"
    "$@"
    printf '[rebuild][%s] finished\n' "$stage"
  fi
}

runner_command() {
  local env_name="$1"
  local stage="$2"
  shift 2
  run_command "$stage" \
    "$CONDA_BIN" run --no-capture-output -n "$env_name" \
    python "$REPO/scripts/run_fever_cbwdm.py" \
    --config "$CONFIG" \
    --run-name "$RUN_NAME" \
    --stages "$stage" \
    --output-root "$EXP_ROOT" \
    --cache-root "$HF_HOME" \
    --resume \
    "$@"
}

if (( VERIFY_ONLY == 1 )); then
  export REPO EXP_ROOT RUN_NAME RUN HF_HOME MODEL_ROOT
  export MINICONDA_ROOT RETRIEVAL_ENV BASELINE_ENV EXPECTED_GIT_HEAD
  exec bash "$REPO/scripts/23_verify_rebuilt_smoke.sh"
fi

completed_stages=()
should_stop=0

runner_command "$RETRIEVAL_ENV" prepare_formal_splits
completed_stages+=(prepare_formal_splits)
[[ "$STOP_AFTER" == "prepare_formal_splits" ]] && should_stop=1

if (( should_stop == 0 )); then
  runner_command "$RETRIEVAL_ENV" corpus
  completed_stages+=(corpus)
  [[ "$STOP_AFTER" == "corpus" ]] && should_stop=1
fi

if (( should_stop == 0 )); then
  runner_command "$RETRIEVAL_ENV" index
  completed_stages+=(index)
  [[ "$STOP_AFTER" == "index" ]] && should_stop=1
fi

if (( should_stop == 0 )); then
  runner_command "$RETRIEVAL_ENV" retrieve_train_core
  completed_stages+=(retrieve_train_core)
  [[ "$STOP_AFTER" == "retrieve_train_core" ]] && should_stop=1
fi

if (( should_stop == 0 )); then
  runner_command "$RETRIEVAL_ENV" retrieve_validation
  completed_stages+=(retrieve_validation)
  [[ "$STOP_AFTER" == "retrieve_validation" ]] && should_stop=1
fi

if (( should_stop == 0 )); then
  runner_command "$BASELINE_ENV" posterior_train_core \
    --generator-model "$MODEL_ROOT/Qwen2.5-1.5B-Instruct"
  completed_stages+=(posterior_train_core)
  [[ "$STOP_AFTER" == "posterior_train_core" ]] && should_stop=1
fi

if (( should_stop == 0 )); then
  runner_command "$BASELINE_ENV" posterior_validation \
    --generator-model "$MODEL_ROOT/Qwen2.5-1.5B-Instruct"
  completed_stages+=(posterior_validation)
  [[ "$STOP_AFTER" == "posterior_validation" ]] && should_stop=1
fi

if (( should_stop == 0 )); then
  runner_command "$BASELINE_ENV" run_calibration_grid \
    --methods infogain_fever \
    --candidate-limit 2 \
    --max-training-candidates 1 \
    --generator-model "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" \
    --infogain-model "$MODEL_ROOT/ms-marco-MiniLM-L-6-v2" \
    --selector-device cuda \
    --infogain-device cuda \
    --skip-completed \
    --continue-on-error
  completed_stages+=(infogain_smoke)
  [[ "$STOP_AFTER" == "infogain_smoke" ]] && should_stop=1
fi

if (( should_stop == 0 )); then
  runner_command "$BASELINE_ENV" run_calibration_grid \
    --methods rag_cbwdm \
    --candidate-limit 2 \
    --max-training-candidates 1 \
    --generator-model "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" \
    --selector-model "$MODEL_ROOT/ms-marco-MiniLM-L-6-v2" \
    --selector-device cuda \
    --infogain-device cuda \
    --skip-completed \
    --continue-on-error
  completed_stages+=(rag_cbwdm_smoke)
  [[ "$STOP_AFTER" == "rag_cbwdm_smoke" ]] && should_stop=1
fi

if (( DRY_RUN == 1 )); then
  exit 0
fi

report_status="stopped"
verification_output=""
if [[ -z "$STOP_AFTER" || "$STOP_AFTER" == "rag_cbwdm_smoke" ]]; then
  export REPO EXP_ROOT RUN_NAME RUN HF_HOME MODEL_ROOT
  export MINICONDA_ROOT RETRIEVAL_ENV BASELINE_ENV EXPECTED_GIT_HEAD
  set +e
  verification_output="$(bash "$REPO/scripts/23_verify_rebuilt_smoke.sh" 2>&1)"
  verification_exit=$?
  set -e
  if (( verification_exit == 0 )); then
    report_status="completed"
  else
    report_status="blocked"
  fi
else
  report_status="stopped_after_${STOP_AFTER}"
fi

"$CONDA_BIN" run -n "$BASELINE_ENV" python - \
  "$REPORT_PATH" "$report_status" "$EXPECTED_GIT_HEAD" \
  "$verification_output" "${completed_stages[@]}" <<'PY'
import json
import os
import pathlib
import sys
import tempfile
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
status = sys.argv[2]
git_head = sys.argv[3]
verification = sys.argv[4]
stages = sys.argv[5:]
payload = {
    "schema_version": "rag_cbwdm_rebuild_to_smoke.v1",
    "status": status,
    "git_head": git_head,
    "completed_invocations": stages,
    "held_out_used": False,
    "verification_output": verification.splitlines(),
    "created_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".rebuild-report-", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

printf '%s\n' "$verification_output"
printf '[rebuild] status=%s report=%s\n' "$report_status" "$REPORT_PATH"
[[ "$report_status" != "blocked" ]] || exit 2

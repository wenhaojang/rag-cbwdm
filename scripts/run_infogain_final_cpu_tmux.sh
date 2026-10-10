#!/usr/bin/env bash
set -u

REPO=/root/rag-cbwdm
PYTHON=/root/miniconda3/envs/rag-cbwdm-mistral/bin/python
PLAN="$REPO/configs/formal/infogain_top4_filter02_final.development.server.yaml"
RUN_ROOT=/root/experiments/rag_cbwdm/formal_v2_runs/infogain_top4_filter02_final_20261011
SESSION=infogain-final-cpu-selection
SCRIPT="$REPO/scripts/run_infogain_final_cpu_tmux.sh"

mkdir -p "$RUN_ROOT"

if [[ "${INFOGAIN_TMUX_WORKER:-0}" == "1" ]]; then
  set +e
  "$PYTHON" "$REPO/scripts/infogain_final_protocol.py" --plan "$PLAN" --stage selection
  rc=$?
  if [[ $rc -eq 0 ]]; then
    "$PYTHON" "$REPO/scripts/infogain_final_protocol.py" --plan "$PLAN" --stage audit
    rc=$?
  fi
  printf '%s\n' "$rc" > "$RUN_ROOT/cpu_selection_and_audit.exitcode"
  exit "$rc"
fi

set -e
command -v tmux >/dev/null
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "env INFOGAIN_TMUX_WORKER=1 bash '$SCRIPT'"
echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"

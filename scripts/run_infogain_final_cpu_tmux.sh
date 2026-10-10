#!/usr/bin/env bash
set -u

REPO=/root/rag-cbwdm
PYTHON=/root/miniconda3/envs/rag-cbwdm-mistral/bin/python
PLAN="$REPO/configs/formal/infogain_top4_filter02_final.development.server.yaml"
RUN_ROOT=/root/experiments/rag_cbwdm/formal_v2_runs/infogain_top4_filter02_final_20261011
SESSION=infogain-final-cpu-selection

if ! command -v tmux >/dev/null; then
  echo "tmux is required" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
set -e

ATTEMPT_ID="cpu_$(date -u +%Y%m%dT%H%M%SZ)_$$"
ATTEMPT_DIR="$RUN_ROOT/attempts/$ATTEMPT_ID"
mkdir -p "$ATTEMPT_DIR"
printf '%s\n' "$ATTEMPT_DIR" > "$RUN_ROOT/latest_cpu_attempt.txt"
{
  printf 'attempt_id=%s\n' "$ATTEMPT_ID"
  printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo=%s\n' "$REPO"
  printf 'git_commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
  printf 'plan=%s\n' "$PLAN"
} > "$ATTEMPT_DIR/attempt.meta"

# Validate every source, checkpoint binding, and output destination in the
# foreground. A failure is visible immediately and no tmux session is created.
set +e
"$PYTHON" "$REPO/scripts/infogain_final_protocol.py" \
  --plan "$PLAN" \
  --stage source-preflight \
  --audit-output "$ATTEMPT_DIR/source_preflight.foreground.json" \
  2>&1 | tee -a "$ATTEMPT_DIR/pre_tmux_source_preflight.log" "$ATTEMPT_DIR/launcher.console.log"
preflight_status=("${PIPESTATUS[@]}")
preflight_rc=${preflight_status[0]}
if [[ ${preflight_status[1]} -ne 0 && $preflight_rc -eq 0 ]]; then
  preflight_rc=${preflight_status[1]}
fi
set -e
printf '%s\n' "$preflight_rc" > "$ATTEMPT_DIR/pre_tmux_source_preflight.exitcode"
if [[ $preflight_rc -ne 0 ]]; then
  printf '%s\n' "$preflight_rc" > "$ATTEMPT_DIR/overall.exitcode"
  echo "source preflight failed; CPU tmux was not created (attempt: $ATTEMPT_DIR)" >&2
  exit "$preflight_rc"
fi

# Recheck after preflight so two launchers cannot race into duplicate work.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf '1\n' > "$ATTEMPT_DIR/overall.exitcode"
  echo "tmux session appeared during preflight: $SESSION" >&2
  exit 1
fi

set +e
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "\"$PYTHON\" \"$REPO/scripts/infogain_final_protocol.py\" --plan \"$PLAN\" --stage cpu-pipeline --attempt-dir \"$ATTEMPT_DIR\" >> \"$ATTEMPT_DIR/launcher.console.log\" 2>&1"
launch_rc=$?
set -e
if [[ $launch_rc -ne 0 ]]; then
  printf '%s\n' "$launch_rc" > "$ATTEMPT_DIR/overall.exitcode"
  echo "failed to create tmux session: $SESSION (attempt: $ATTEMPT_DIR)" >&2
  exit "$launch_rc"
fi

echo "started tmux session: $SESSION"
echo "attempt: $ATTEMPT_DIR"
echo "attach: tmux attach -t $SESSION"
echo "log: tail -F $ATTEMPT_DIR/launcher.log"

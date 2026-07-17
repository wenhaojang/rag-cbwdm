#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
exec "${PYTHON:-python}" "${PROJECT_ROOT}/scripts/run_fever_cbwdm.py" "$@"

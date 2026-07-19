#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/rag-cbwdm}"
SERVER_ENV_DIR="${SERVER_ENV_DIR:-$REPO/environment/server}"
CAPTURED_DIR="${CAPTURED_DIR:-$SERVER_ENV_DIR/captured}"
MINICONDA_ROOT="${MINICONDA_ROOT:-/root/miniconda3}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
MINICONDA_SHA256="${MINICONDA_SHA256:-}"
RETRIEVAL_ENV="${RETRIEVAL_ENV:-rag-cbwdm-retrieval}"
BASELINE_ENV="${BASELINE_ENV:-rag-cbwdm-baselines}"
RETRIEVAL_YAML="${RETRIEVAL_YAML:-}"
BASELINE_YAML="${BASELINE_YAML:-}"
RETRIEVAL_PIP_FREEZE="${RETRIEVAL_PIP_FREEZE:-}"
BASELINE_PIP_FREEZE="${BASELINE_PIP_FREEZE:-}"
RETRIEVAL_PIP_EXTRA_INDEX_URL="${RETRIEVAL_PIP_EXTRA_INDEX_URL:-}"
BASELINE_PIP_EXTRA_INDEX_URL="${BASELINE_PIP_EXTRA_INDEX_URL:-}"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-24000}"
MIN_RAM_GIB="${MIN_RAM_GIB:-64}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-500}"

CHECK_ONLY=0
SKIP_APT=0

usage() {
  cat <<'EOF'
Usage: bash scripts/20_bootstrap_new_server.sh [--check-only] [--skip-apt]

Environment overrides:
  REPO, SERVER_ENV_DIR, CAPTURED_DIR
  MINICONDA_ROOT, MINICONDA_URL, MINICONDA_SHA256
  RETRIEVAL_ENV, BASELINE_ENV
  RETRIEVAL_YAML, BASELINE_YAML
  RETRIEVAL_PIP_FREEZE, BASELINE_PIP_FREEZE
  RETRIEVAL_PIP_EXTRA_INDEX_URL, BASELINE_PIP_EXTRA_INDEX_URL
  MIN_GPU_MEMORY_MIB, MIN_RAM_GIB, MIN_FREE_DISK_GIB

MINICONDA_SHA256 is mandatory when Miniconda must be installed.
Non-template Linux YAML files are discovered by their top-level name field.
EOF
}

while (( $# )); do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      ;;
    --skip-apt)
      SKIP_APT=1
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

installer=""
env_spec_tmp=""
cleanup() {
  if [[ -n "$installer" && -f "$installer" ]]; then
    rm -f -- "$installer"
  fi
  if [[ -n "$env_spec_tmp" && -d "$env_spec_tmp" ]]; then
    rm -rf -- "$env_spec_tmp"
  fi
}
trap cleanup EXIT

[[ -r /etc/os-release ]] || fail "/etc/os-release is unavailable"
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || fail "Ubuntu is required; detected ID=${ID:-MISSING}"
ubuntu_major="${VERSION_ID%%.*}"
[[ "$ubuntu_major" =~ ^[0-9]+$ ]] || fail "invalid Ubuntu VERSION_ID=${VERSION_ID:-MISSING}"
(( ubuntu_major >= 22 )) || fail "Ubuntu 22.04 or newer is required"

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"
nvidia-smi >/dev/null
gpu_memory="$(
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
    | sort -nr | head -n 1 | tr -d '[:space:]'
)"
[[ "$gpu_memory" =~ ^[0-9]+$ ]] || fail "cannot determine GPU memory"
(( gpu_memory >= MIN_GPU_MEMORY_MIB )) \
  || fail "GPU memory ${gpu_memory} MiB is below ${MIN_GPU_MEMORY_MIB} MiB"

ram_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
required_ram_kib=$(( MIN_RAM_GIB * 1024 * 1024 ))
(( ram_kib >= required_ram_kib )) \
  || fail "RAM is below ${MIN_RAM_GIB} GiB"

disk_target="$(dirname "$MINICONDA_ROOT")"
while [[ ! -d "$disk_target" && "$disk_target" != "/" ]]; do
  disk_target="$(dirname "$disk_target")"
done
free_disk_kib="$(df -Pk "$disk_target" | awk 'NR==2 {print $4}')"
required_disk_kib=$(( MIN_FREE_DISK_GIB * 1024 * 1024 ))
(( free_disk_kib >= required_disk_kib )) \
  || fail "free disk at $disk_target is below ${MIN_FREE_DISK_GIB} GiB"

APT_PACKAGES=(
  git
  git-lfs
  wget
  curl
  ca-certificates
  tmux
  rsync
  jq
  build-essential
  openjdk-21-jre-headless
)
if (( CHECK_ONLY == 0 && SKIP_APT == 0 )); then
  if (( EUID == 0 )); then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${APT_PACKAGES[@]}"
  else
    command -v sudo >/dev/null 2>&1 || fail "sudo is required for apt installation"
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${APT_PACKAGES[@]}"
  fi
  git lfs install --system
fi
for command_name in git git-lfs wget curl tmux rsync gcc java; do
  command -v "$command_name" >/dev/null 2>&1 \
    || fail "required system command missing: $command_name"
done

CONDA_BIN="$MINICONDA_ROOT/bin/conda"
if [[ ! -x "$CONDA_BIN" ]]; then
  (( CHECK_ONLY == 0 )) || fail "Miniconda missing at $MINICONDA_ROOT"
  [[ "$MINICONDA_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] \
    || fail "set verified MINICONDA_SHA256 before installing Miniconda"
  installer="$(mktemp /tmp/miniconda.XXXXXX.sh)"
  curl -fL --retry 5 --retry-delay 5 "$MINICONDA_URL" -o "$installer"
  actual_sha="$(sha256sum "$installer" | awk '{print $1}')"
  [[ "${actual_sha,,}" == "${MINICONDA_SHA256,,}" ]] \
    || fail "Miniconda installer SHA-256 mismatch"
  bash "$installer" -b -p "$MINICONDA_ROOT"
fi
[[ -x "$CONDA_BIN" ]] || fail "conda executable missing: $CONDA_BIN"

[[ -d "$SERVER_ENV_DIR" ]] \
  || fail "server environment directory missing: $SERVER_ENV_DIR"

yaml_environment_name() {
  awk '
    NR == 1 {
      sub(/^\357\273\277/, "", $0)
    }
    /^name:[[:space:]]*/ {
      value=$0
      sub(/^name:[[:space:]]*/, "", value)
      sub(/\r$/, "", value)
      gsub(/^["\047]|["\047]$/, "", value)
      print value
      exit
    }
  ' "$1"
}

validate_linux_yaml() {
  local yaml_path="$1"
  local expected_name="$2"
  [[ -s "$yaml_path" ]] || fail "conda YAML missing or empty: $yaml_path"
  [[ "$yaml_path" != *".template"* ]] \
    || fail "template YAML cannot be used for deployment: $yaml_path"
  [[ "$(yaml_environment_name "$yaml_path")" == "$expected_name" ]] \
    || fail "conda YAML name does not match $expected_name: $yaml_path"
  if grep -Eq '<[^>]+>' "$yaml_path"; then
    fail "placeholder found in conda YAML: $yaml_path"
  fi
  if grep -Eqi '(^|[/=:_-])(win-64|pywin32|pywinpty)([/=:_-]|$)|^[[:space:]]*-[[:space:]]*vc=' \
    "$yaml_path"; then
    fail "Windows dependency marker found in Linux conda YAML: $yaml_path"
  fi
  if ! grep -Eq 'ld_impl_linux|libgcc|linux-64' "$yaml_path"; then
    fail "Linux dependency marker missing from conda YAML: $yaml_path"
  fi
  if grep -Eqi \
    '(BEGIN [A-Z ]*PRIVATE KEY|hf_[A-Za-z0-9]{20,}|https?://[^/@[:space:]]+@|(^|[[:space:]])(token|password|passwd|secret|api[_-]?key)[[:space:]]*[:=])' \
    "$yaml_path"; then
    fail "possible credential found in conda YAML: $yaml_path"
  fi
}

discover_yaml() {
  local expected_name="$1"
  local explicit_path="$2"
  local candidate
  local candidate_name
  local matches=()
  if [[ -n "$explicit_path" ]]; then
    validate_linux_yaml "$explicit_path" "$expected_name"
    printf '%s\n' "$explicit_path"
    return
  fi
  while IFS= read -r -d '' candidate; do
    [[ "$candidate" == *".template"* ]] && continue
    candidate_name="$(yaml_environment_name "$candidate")"
    [[ "$candidate_name" == "$expected_name" ]] && matches+=("$candidate")
  done < <(
    find "$SERVER_ENV_DIR" -maxdepth 1 -type f \
      \( -iname '*.yml' -o -iname '*.yaml' \) -print0
  )
  if (( ${#matches[@]} == 0 )); then
    while IFS= read -r -d '' candidate; do
      [[ "$candidate" == *".template"* ]] && continue
      candidate_name="$(yaml_environment_name "$candidate")"
      [[ "$candidate_name" == "$expected_name" ]] && matches+=("$candidate")
    done < <(
      find "$SERVER_ENV_DIR" -mindepth 2 -type f \
        \( -iname '*.yml' -o -iname '*.yaml' \) -print0
    )
  fi
  (( ${#matches[@]} > 0 )) \
    || fail "no non-template YAML with name=$expected_name under $SERVER_ENV_DIR"
  (( ${#matches[@]} == 1 )) \
    || fail "multiple YAML files match name=$expected_name; set an explicit YAML override"
  validate_linux_yaml "${matches[0]}" "$expected_name"
  printf '%s\n' "${matches[0]}"
}

discover_pip_freeze() {
  local role="$1"
  local explicit_path="$2"
  local candidate
  local lower_name
  local matches=()
  if [[ -n "$explicit_path" ]]; then
    [[ -s "$explicit_path" ]] || fail "pip-freeze audit file missing: $explicit_path"
    printf '%s\n' "$explicit_path"
    return
  fi
  while IFS= read -r -d '' candidate; do
    lower_name="${candidate,,}"
    if [[ "$lower_name" == *"$role"* \
      && "$lower_name" == *"pip"* \
      && "$lower_name" == *"freeze"* ]]; then
      matches+=("$candidate")
    fi
  done < <(
    find "$SERVER_ENV_DIR" -maxdepth 1 -type f -iname '*.txt' -print0
  )
  if (( ${#matches[@]} == 0 )); then
    while IFS= read -r -d '' candidate; do
      lower_name="${candidate,,}"
      if [[ "$lower_name" == *"$role"* \
        && "$lower_name" == *"pip"* \
        && "$lower_name" == *"freeze"* ]]; then
        matches+=("$candidate")
      fi
    done < <(
      find "$SERVER_ENV_DIR" -mindepth 2 -type f -iname '*.txt' -print0
    )
  fi
  (( ${#matches[@]} > 0 )) \
    || fail "no $role pip-freeze audit file found under $SERVER_ENV_DIR"
  (( ${#matches[@]} == 1 )) \
    || fail "multiple $role pip-freeze files found; set an explicit override"
  printf '%s\n' "${matches[0]}"
}

RETRIEVAL_YAML="$(discover_yaml "$RETRIEVAL_ENV" "$RETRIEVAL_YAML")"
BASELINE_YAML="$(discover_yaml "$BASELINE_ENV" "$BASELINE_YAML")"
RETRIEVAL_PIP_FREEZE="$(
  discover_pip_freeze retrieval "$RETRIEVAL_PIP_FREEZE"
)"
BASELINE_PIP_FREEZE="$(
  discover_pip_freeze baseline "$BASELINE_PIP_FREEZE"
)"

infer_pytorch_index_url() {
  local yaml_path="$1"
  local torch_requirement
  local build_tag
  torch_requirement="$(
    grep -E '^[[:space:]]*-[[:space:]]*torch==' "$yaml_path" \
      | head -n 1 | tr -d '\r' || true
  )"
  if [[ "$torch_requirement" =~ \+([A-Za-z0-9._-]+)$ ]]; then
    build_tag="${BASH_REMATCH[1]}"
    case "$build_tag" in
      cpu|cu[0-9]*)
        printf 'https://download.pytorch.org/whl/%s\n' "$build_tag"
        ;;
      *)
        fail "unsupported Torch build tag +$build_tag; set an explicit pip extra index"
        ;;
    esac
  fi
}

if [[ -z "$RETRIEVAL_PIP_EXTRA_INDEX_URL" ]]; then
  RETRIEVAL_PIP_EXTRA_INDEX_URL="$(infer_pytorch_index_url "$RETRIEVAL_YAML")"
fi
if [[ -z "$BASELINE_PIP_EXTRA_INDEX_URL" ]]; then
  BASELINE_PIP_EXTRA_INDEX_URL="$(infer_pytorch_index_url "$BASELINE_YAML")"
fi
for index_url in \
  "$RETRIEVAL_PIP_EXTRA_INDEX_URL" "$BASELINE_PIP_EXTRA_INDEX_URL"; do
  if [[ -n "$index_url" && "$index_url" =~ https?://[^/@[:space:]]+@ ]]; then
    fail "embedded credentials are forbidden in pip index URLs"
  fi
done

for audit_file in "$RETRIEVAL_PIP_FREEZE" "$BASELINE_PIP_FREEZE"; do
  [[ -s "$audit_file" ]] || fail "pip-freeze audit file missing: $audit_file"
  if grep -Eqi \
    '(BEGIN [A-Z ]*PRIVATE KEY|hf_[A-Za-z0-9]{20,}|https?://[^/@[:space:]]+@|(^|[[:space:]])(token|password|passwd|secret|api[_-]?key)[[:space:]]*[:=])' \
    "$audit_file"; then
    fail "possible credential found in pip-freeze audit file: $audit_file"
  fi
done

# Never reuse an old absolute prefix. Conda receives sanitized temporary YAML
# files and the desired environment name explicitly.
env_spec_tmp="$(mktemp -d /tmp/rag-cbwdm-conda-yaml.XXXXXX)"
RETRIEVAL_INSTALL_YAML="$env_spec_tmp/retrieval.yml"
BASELINE_INSTALL_YAML="$env_spec_tmp/baselines.yml"
awk 'NR == 1 {sub(/^\357\273\277/, "", $0)} !/^prefix:[[:space:]]*/' \
  "$RETRIEVAL_YAML" >"$RETRIEVAL_INSTALL_YAML"
awk 'NR == 1 {sub(/^\357\273\277/, "", $0)} !/^prefix:[[:space:]]*/' \
  "$BASELINE_YAML" >"$BASELINE_INSTALL_YAML"

printf '[bootstrap] retrieval_yaml=%s\n' "$RETRIEVAL_YAML"
printf '[bootstrap] baseline_yaml=%s\n' "$BASELINE_YAML"
printf '[bootstrap] retrieval_pip_audit=%s\n' "$RETRIEVAL_PIP_FREEZE"
printf '[bootstrap] baseline_pip_audit=%s\n' "$BASELINE_PIP_FREEZE"

env_exists() {
  "$CONDA_BIN" env list --json \
    | "$MINICONDA_ROOT/bin/python" -c \
      'import json, pathlib, sys; name=sys.argv[1]; print(any(pathlib.Path(p).name == name for p in json.load(sys.stdin)["envs"]))' \
      "$1" \
    | grep -qx True
}

if (( CHECK_ONLY == 0 )); then
  if ! env_exists "$RETRIEVAL_ENV"; then
    if [[ -n "$RETRIEVAL_PIP_EXTRA_INDEX_URL" ]]; then
      env PIP_EXTRA_INDEX_URL="$RETRIEVAL_PIP_EXTRA_INDEX_URL" \
        "$CONDA_BIN" env create -n "$RETRIEVAL_ENV" \
        -f "$RETRIEVAL_INSTALL_YAML"
    else
      "$CONDA_BIN" env create -n "$RETRIEVAL_ENV" \
        -f "$RETRIEVAL_INSTALL_YAML"
    fi
  fi
  if ! env_exists "$BASELINE_ENV"; then
    if [[ -n "$BASELINE_PIP_EXTRA_INDEX_URL" ]]; then
      env PIP_EXTRA_INDEX_URL="$BASELINE_PIP_EXTRA_INDEX_URL" \
        "$CONDA_BIN" env create -n "$BASELINE_ENV" \
        -f "$BASELINE_INSTALL_YAML"
    else
      "$CONDA_BIN" env create -n "$BASELINE_ENV" \
        -f "$BASELINE_INSTALL_YAML"
    fi
  fi
fi

env_exists "$RETRIEVAL_ENV" || fail "conda environment missing: $RETRIEVAL_ENV"
env_exists "$BASELINE_ENV" || fail "conda environment missing: $BASELINE_ENV"

"$CONDA_BIN" run -n "$RETRIEVAL_ENV" python -c \
  'import importlib.metadata as m, pyserini; assert m.version("pyserini") == "2.3.0"'
retrieval_java_version="$(
  "$CONDA_BIN" run -n "$RETRIEVAL_ENV" java -version 2>&1
)"
grep -Eq 'version "21([."]|$)' <<<"$retrieval_java_version" \
  || fail "$RETRIEVAL_ENV must provide Java 21"

"$CONDA_BIN" run -n "$BASELINE_ENV" python -c \
  'import torch, transformers; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.get_device_name(0))'
if "$CONDA_BIN" run -n "$BASELINE_ENV" python -c 'import pyserini' \
  >/dev/null 2>&1; then
  fail "$BASELINE_ENV must not depend on Pyserini"
fi

compare_pip_freeze() {
  local env_name="$1"
  local captured="$2"
  if ! diff -u \
    <(sed '/^[[:space:]]*$/d' "$captured" | sort) \
    <("$CONDA_BIN" run -n "$env_name" python -m pip freeze \
        | sed '/^[[:space:]]*$/d' | sort) >/dev/null; then
    fail "$env_name pip freeze differs from captured Linux environment"
  fi
}
compare_pip_freeze "$RETRIEVAL_ENV" "$RETRIEVAL_PIP_FREEZE"
compare_pip_freeze "$BASELINE_ENV" "$BASELINE_PIP_FREEZE"

printf '[bootstrap] PASS mode=%s retrieval_env=%s baseline_env=%s\n' \
  "$([[ "$CHECK_ONLY" == 1 ]] && printf check-only || printf install)" \
  "$RETRIEVAL_ENV" \
  "$BASELINE_ENV"

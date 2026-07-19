# RAG-CBWDM disaster recovery: new server from zero

## 1. Recovery objective

This runbook assumes the old instance has been destroyed and none of these
paths or environments exists:

```text
/root/rag-cbwdm
/root/models
/root/huggingface
/root/experiments/rag_cbwdm
rag-cbwdm-retrieval
rag-cbwdm-baselines
Lucene index, retrieval, posterior, checkpoint, or calibration artifacts
```

The objective is functional reproduction, not byte-for-byte recovery of old
artifacts. The new server must rebuild the software and assets, rerun the formal
pilot, and reach:

- cleaned FEVER-2 formal split;
- full FEVER Wiki corpus;
- Lucene v2 full index;
- 5000 train-core and 500 validation BM25 top-20 rows;
- 5000 train-core and 500 validation posterior rows;
- two completed InfoGain smoke evaluations from one training candidate;
- two completed RAG-CBWDM smoke evaluations from one training candidate;
- non-null accuracy and macro-F1 for every smoke evaluation;
- no held-out retrieval or posterior artifact.

Only then may the full calibration grids continue.

## 2. Exact versus functional reproduction

The target here is reliable functional reproduction:

- exact code commit;
- same seed and formal protocol;
- pinned model/data revisions;
- same major and captured Python dependencies;
- equivalent 24 GB or larger CUDA GPU;
- newly generated, internally checksum-valid artifacts.

This does not promise byte-identical posterior probabilities or checkpoints.
GPU model, driver, CUDA runtime, cuDNN/cuBLAS kernels, PyTorch build, compiler,
thread scheduling, and nondeterministic operations can cause small numerical
differences.

Byte-identical recovery would additionally require the same hardware model,
driver, CUDA libraries, package builds, deterministic-algorithm settings,
environment variables, and input bytes. Even then, every operation would need
a deterministic audit. The current goal is to rerun safely and continue the
formal experiment, not reproduce destroyed artifact bytes.

## 3. Repository audit and unresolved blockers

The following model IDs are explicitly recorded by current configs:

| Local directory | Audited model ID | Repository evidence |
|---|---|---|
| `Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | pilot config |
| `Qwen2.5-7B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` | formal config |
| `ms-marco-MiniLM-L-6-v2` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | selector and InfoGain configs |
| `bge-reranker-base` | `BAAI/bge-reranker-base` | smoke config |
| `bge-reranker-large` | `BAAI/bge-reranker-large` | pilot/formal configs |

The repository does **not** currently pin:

- immutable revisions for any of the five models (`revision: null`);
- the immutable Hugging Face `fever` dataset revision;
- the Miniconda installer version/checksum that created the old environments.

These remain disaster-recovery blockers. Do not replace them with guessed
commit hashes or moving `main` branches.

The imported real Linux exports now record:

- Python 3.12.13 in both environments;
- retrieval Pyserini 2.3.0 and conda OpenJDK 21.0.11;
- baseline Torch 2.6.0+cu124 and Transformers 4.57.6;
- an RTX 3090 24 GB snapshot with driver 535.216.01 and reported CUDA 12.2;
- no Pyserini dependency in the baseline export.

The standalone `java-version.txt` records that Java was not on the old login
shell PATH. Deployment therefore validates Java inside the reconstructed
retrieval conda environment.

---

# Phase 0 — minimum server recommendation

Use:

- Ubuntu 22.04 LTS;
- NVIDIA GPU with at least 24 GB VRAM; RTX 3090 24 GB is the known reference;
- 16 vCPUs or more;
- 64 GB RAM minimum, 128 GB recommended for filesystem cache and safer data
  preparation;
- at least 500 GiB free persistent storage before downloads;
- 1 TiB persistent storage recommended for five models, duplicated HF cache,
  FEVER raw/wiki data, full corpus, Lucene index, posteriors, grid checkpoints,
  evaluation artifacts, logs, and safety margin.

Use a persistent data disk. Do not place the only copy of models, raw data,
experiment output, or asset manifests on ephemeral instance NVMe.

The exact minimum NVIDIA driver cannot be determined from this repository
because the Torch/CUDA build is not pinned. The driver must support the CUDA
runtime recorded in `cuda_torch_info.json`. `nvidia-smi` must work before
bootstrap, and `torch.cuda.is_available()` must pass afterward. A separately
installed CUDA toolkit/`nvcc` is not required unless a captured dependency
needs compilation.

---

# Phase 1 — system dependencies

On the empty server:

```bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git git-lfs wget curl ca-certificates tmux rsync jq \
  build-essential openjdk-21-jre-headless

git lfs install --system
nvidia-smi
free -h
df -hT
lsblk -o NAME,SIZE,FSTYPE,TYPE,MOUNTPOINTS,MODEL,SERIAL
java -version
```

Install Miniconda only after obtaining the official installer SHA-256:

```bash
export MINICONDA_ROOT=/root/miniconda3
export MINICONDA_URL='https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'
export MINICONDA_SHA256='<verified official SHA-256>'

cd /tmp
curl -fL --retry 5 --retry-delay 5 \
  "$MINICONDA_URL" -o miniconda-installer.sh
printf '%s  %s\n' "$MINICONDA_SHA256" miniconda-installer.sh \
  | sha256sum -c -
bash miniconda-installer.sh -b -p "$MINICONDA_ROOT"
source "$MINICONDA_ROOT/etc/profile.d/conda.sh"
conda --version
```

Do not use an unverified “latest” installer in an auditable recovery.

---

# Phase 2 — exact code recovery

The final recovery commit is the commit pushed after the old-server environment
capture. It must be recorded outside the destroyed server.

```bash
export RECOVERY_GIT_HEAD='<final pushed recovery commit SHA>'

git clone https://github.com/wenhaojang/rag-cbwdm.git /root/rag-cbwdm
cd /root/rag-cbwdm
git fetch --all --tags --prune
git fetch origin feature/fever-formal-readiness
git checkout feature/fever-formal-readiness
git checkout --detach "$RECOVERY_GIT_HEAD"

test "$(git rev-parse HEAD)" = "$RECOVERY_GIT_HEAD"
test -z "$(git status --porcelain)"
```

The captured `environment/server/captured/git_head.txt` records the experiment
code HEAD at capture time. It cannot self-record the later commit that contains
the capture files. `RECOVERY_GIT_HEAD` must therefore be the separately recorded
final pushed commit.

Run code-only validation with any temporary Python that has test requirements,
or after Phase 4 with the baseline environment:

```bash
python -m compileall -q src scripts tests
python -m pytest -q
python -m unittest discover
bash -n scripts/run_fever_cbwdm.sh
bash -n scripts/18_capture_server_resume_state.sh
bash -n scripts/20_bootstrap_new_server.sh
bash -n scripts/21_download_project_assets.sh
bash -n scripts/22_rebuild_to_current_smoke.sh
bash -n scripts/23_verify_rebuilt_smoke.sh
git diff --check
```

---

# Phase 3 — retrieval environment

The authoritative inputs are:

```text
environment/server/rag-cbwdm-retrieval.yml
environment/server/rag-cbwdm-retrieval-pip-freeze.txt
```

These are real Linux exports imported from the old server. The bootstrap
discovers the YAML by its top-level `name:` field, ignores every template, and
removes any old top-level `prefix:` in a temporary copy. Never substitute
Windows packages.

The unified bootstrap command creates the environment if absent and verifies
it if present:

```bash
cd /root/rag-cbwdm
export MINICONDA_ROOT=/root/miniconda3
export MINICONDA_SHA256='<only needed if Miniconda is not installed>'
bash scripts/20_bootstrap_new_server.sh --skip-apt
```

The retrieval environment must:

- use the captured Python version;
- import Pyserini 2.3.0;
- execute Java 21;
- match the captured pip freeze;
- not require CUDA Torch or a GPU training stack.

Manual acceptance:

```bash
/root/miniconda3/bin/conda run -n rag-cbwdm-retrieval \
  python -c 'import importlib.metadata as m, pyserini; assert m.version("pyserini") == "2.3.0"'
/root/miniconda3/bin/conda run -n rag-cbwdm-retrieval java -version
```

---

# Phase 4 — baseline/GPU environment

Authoritative inputs:

```text
environment/server/rag-cbwdm-baselines.yml
environment/server/rag-cbwdm-baselines-pip-freeze.txt
```

`20_bootstrap_new_server.sh` creates this environment from the imported Linux
YAML. The pip-freeze file is an audit input, not the conda environment spec. It
then requires:

```bash
/root/miniconda3/bin/conda run -n rag-cbwdm-baselines \
  python -c 'import torch, transformers; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.get_device_name(0))'

if /root/miniconda3/bin/conda run -n rag-cbwdm-baselines \
  python -c 'import pyserini'
then
  echo 'BLOCKED: Pyserini must not be a baseline dependency' >&2
  exit 2
fi
```

This imports frameworks only. It does not load Qwen, BGE, or MiniLM weights.

Run bootstrap in audit-only mode at any later time:

```bash
bash scripts/20_bootstrap_new_server.sh --check-only --skip-apt
```

---

# Phase 5 — download five pinned models

Required environment variables:

```bash
export QWEN15_REVISION='<immutable Qwen 1.5B commit>'
export QWEN7_REVISION='<immutable Qwen 7B commit>'
export MINILM_REVISION='<immutable cross-encoder MiniLM commit>'
export BGE_BASE_REVISION='<immutable BGE base commit>'
export BGE_LARGE_REVISION='<immutable BGE large commit>'
```

If the capture contains revisions, the asset script reads them automatically.
If any remains empty, the script stops with `BLOCKED`; it never downloads a
moving revision.

Download with resumable Hugging Face snapshots:

```bash
cd /root/rag-cbwdm
export MODEL_ROOT=/root/models
export HF_HOME=/root/huggingface

bash scripts/21_download_project_assets.sh --models-only --resume
```

The script maps the audited model IDs to:

```text
Qwen/Qwen2.5-1.5B-Instruct
  -> /root/models/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-7B-Instruct
  -> /root/models/Qwen2.5-7B-Instruct
cross-encoder/ms-marco-MiniLM-L-6-v2
  -> /root/models/ms-marco-MiniLM-L-6-v2
BAAI/bge-reranker-base
  -> /root/models/bge-reranker-base
BAAI/bge-reranker-large
  -> /root/models/bge-reranker-large
```

Acceptance loads only config and tokenizer metadata with
`local_files_only=True`. It requires nonempty config/tokenizer files, at least
one weight file, sizes, and SHA-256 values. It does not instantiate a model.

Offline recheck:

```bash
bash scripts/21_download_project_assets.sh --models-only --check-only
```

---

# Phase 6 — download and materialize pinned FEVER data

The actual project entry point is `scripts/download_fever_hf.py`. It loads:

```text
dataset ID: fever
config:     v1.0
splits:     train, labelled_dev
config:     wiki_pages
split:      wikipedia_pages
```

The script now requires an immutable dataset revision.

```bash
export FEVER_DATASET_REVISION='<immutable Hugging Face fever dataset commit>'
export DATA_ROOT=/root/rag-cbwdm/data/raw/fever
export HF_HOME=/root/huggingface

cd /root/rag-cbwdm
bash scripts/21_download_project_assets.sh --data-only --resume
```

Expected layout:

```text
/root/rag-cbwdm/data/raw/fever/train.jsonl
/root/rag-cbwdm/data/raw/fever/dev.jsonl
/root/rag-cbwdm/data/raw/fever/wiki-pages/wiki-*.jsonl
```

The asset manifest records every raw file's row count, size, and SHA-256:

```text
/root/rag-cbwdm/data/raw/fever/project_assets.manifest.json
```

Validate without network:

```bash
bash scripts/21_download_project_assets.sh --data-only --check-only
```

Do not edit, deduplicate, or relabel raw FEVER files. Do not use official dev
gold for parameter selection outside the formal split/held-out protocol.

---

# Phase 7 — rebuild to the current smoke state

Set the final pushed commit explicitly:

```bash
cd /root/rag-cbwdm
export EXPECTED_GIT_HEAD="$RECOVERY_GIT_HEAD"
export REPO=/root/rag-cbwdm
export EXP_ROOT=/root/experiments/rag_cbwdm
export RUN_NAME=fever2_formal_pilot_5000_500_seed13
export RUN="$EXP_ROOT/$RUN_NAME"
export MODEL_ROOT=/root/models
export HF_HOME=/root/huggingface
```

Preview every command without executing:

```bash
bash scripts/22_rebuild_to_current_smoke.sh --dry-run
```

Main recovery command:

```bash
bash scripts/22_rebuild_to_current_smoke.sh
```

The script executes these stages in order:

| Order | Stage | Environment | Resume behavior |
|---:|---|---|---|
| 1 | `prepare_formal_splits` | retrieval | validates/reuses completed split manifest |
| 2 | `corpus` | retrieval | validates/reuses full corpus manifest |
| 3 | `index` | retrieval | validates/reuses Lucene v2 fingerprint/inventory |
| 4 | `retrieve_train_core` | retrieval | requires 5000 completed rows |
| 5 | `retrieve_validation` | retrieval | requires 500 completed rows |
| 6 | `posterior_train_core` | baseline/GPU | resumes completed/partial posterior contract |
| 7 | `posterior_validation` | baseline/GPU | resumes completed/partial posterior contract |
| 8 | InfoGain smoke | baseline/GPU | `--candidate-limit 2 --max-training-candidates 1` |
| 9 | RAG-CBWDM smoke | baseline/GPU | `--candidate-limit 2 --max-training-candidates 1` |

The exact InfoGain request is:

```text
--methods infogain_fever
--candidate-limit 2
--max-training-candidates 1
--resume
--skip-completed
--continue-on-error
```

The RAG-CBWDM request substitutes `--methods rag_cbwdm`.

Stop safely after a stage:

```bash
bash scripts/22_rebuild_to_current_smoke.sh \
  --stop-after retrieve_validation
```

Resume later by running the same main command. Current runner/grid manifests
decide reuse. A failed grid candidate is retried at the same fingerprint;
checksum-valid completed children are skipped. No `retrieve_test`,
`posterior_test`, or held-out command exists in this orchestrator.

The final report is:

```text
$RUN/artifacts/formal/rebuild_to_current_smoke_report.json
```

## Manual per-stage equivalents

Retrieval environment:

```bash
conda run --no-capture-output -n rag-cbwdm-retrieval \
  python scripts/run_fever_cbwdm.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-name fever2_formal_pilot_5000_500_seed13 \
  --stages prepare_formal_splits,corpus,index,retrieve_train_core,retrieve_validation \
  --output-root /root/experiments/rag_cbwdm \
  --cache-root /root/huggingface \
  --resume
```

Baseline/GPU environment:

```bash
conda run --no-capture-output -n rag-cbwdm-baselines \
  python scripts/run_fever_cbwdm.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-name fever2_formal_pilot_5000_500_seed13 \
  --stages posterior_train_core,posterior_validation \
  --output-root /root/experiments/rag_cbwdm \
  --cache-root /root/huggingface \
  --generator-model /root/models/Qwen2.5-1.5B-Instruct \
  --resume
```

The smoke commands are emitted exactly by `22_rebuild_to_current_smoke.sh` and
may be stopped/resumed independently.

---

# Phase 8 — acceptance

Run the read-only verifier:

```bash
cd /root/rag-cbwdm
export EXPECTED_GIT_HEAD="$RECOVERY_GIT_HEAD"
bash scripts/23_verify_rebuilt_smoke.sh
```

It checks:

- exact final Git commit and clean worktree;
- both environments and their dependency boundary;
- all five models and the asset manifest;
- FEVER raw data inventory;
- formal split contract;
- corpus and Lucene v2 manifest/inventory;
- retrieval roles, top-N, 5000/500 counts, and output SHA;
- posterior roles, 5000/500 counts, output/input SHA, prompt hash, and
  verbalizer hash;
- at least two completed smoke evaluations for each trainable method;
- non-null accuracy and macro-F1;
- no held-out retrieval or posterior files.

Final output is exactly `PASS`, or `BLOCKED` plus all discovered blockers.

Verifier-only orchestration:

```bash
bash scripts/22_rebuild_to_current_smoke.sh --verify-only
```

---

# Phase 9 — continue formal work

After Phase 8 returns `PASS`:

1. run the full InfoGain grid without either candidate limit;
2. run the full RAG-CBWDM grid without either candidate limit;
3. run a combined `infogain_fever,rag_cbwdm` resume pass to publish the full
   aggregate;
4. run `calibrate_methods`;
5. run fixed validation No-evidence, Naive, BGE, and Oracle diagnostic
   baselines;
6. run fairness audit and canonical summary;
7. run CBWDM diagnostics;
8. freeze the formal config with immutable revisions;
9. run formal readiness with artifact rehashing;
10. access held-out stages only after `formal_readiness.status=ready`.

The audited exact commands are in
`RAG_CBWDM_SERVER_REPRODUCTION_AND_RESUME_RUNBOOK.md`, section 10. Do not use
`--skip-artifact-rehash` for server readiness.

---

# Old-server final actions before destruction

These actions must happen while the real old server still exists.

## 1. Capture environment and inventories

```bash
cd /root/rag-cbwdm
export REPO=/root/rag-cbwdm
export EXP_ROOT=/root/experiments/rag_cbwdm
export RUN_NAME=fever2_formal_pilot_5000_500_seed13
export MODEL_ROOT=/root/models
export HF_HOME=/root/huggingface

# Must be the immutable HF revision that produced the existing raw files.
export FEVER_DATASET_REVISION='<verified fever dataset commit>'

bash scripts/18_capture_server_resume_state.sh
```

This generates the ten required small files under:

```text
environment/server/captured/
```

## 2. Resolve every blocker

Inspect:

```bash
jq . environment/server/captured/model_inventory.json
jq . environment/server/captured/data_inventory.json
cat environment/server/captured/cuda_torch_info.json
cat environment/server/captured/java_pyserini_info.json
```

Every model must have a non-null immutable revision. The FEVER dataset revision
must be non-null. If local `config.json` lacks `_commit_hash`, recover the
revision from the Hugging Face cache snapshot metadata or original download
records before destruction, then rerun capture. Do not type a guessed revision
into the inventory.

## 3. Secret and size audit

```bash
find environment/server/captured -type f -size +5M -print
test -z "$(find environment/server/captured -type f -size +5M -print)"

if grep -RInE \
  '(BEGIN .*PRIVATE KEY|hf_[A-Za-z0-9]{20,}|token[=:]|password[=:]|api[_-]?key[=:]|secret[=:])' \
  environment/server/captured
then
  echo 'BLOCKED: possible secret detected' >&2
  exit 2
fi

git diff --check
```

Review URLs and conda channels manually even if the scan is clean. The captured
directory must not contain model weights, raw FEVER JSONL, experiment
artifacts, caches, SSH keys, tokens, or large logs.

## 4. Commit and push from the old server

These commands are instructions for the user; they are not executed by Codex:

```bash
git add \
  scripts/18_capture_server_resume_state.sh \
  scripts/19_verify_resumed_server.sh \
  scripts/20_bootstrap_new_server.sh \
  scripts/21_download_project_assets.sh \
  scripts/22_rebuild_to_current_smoke.sh \
  scripts/23_verify_rebuilt_smoke.sh \
  scripts/download_fever_hf.py \
  environment/server \
  NEW_SERVER_FROM_ZERO_RUNBOOK.md \
  RAG_CBWDM_SERVER_REPRODUCTION_AND_RESUME_RUNBOOK.md

git diff --cached --check
git status --short
git commit -m "Add auditable new-server disaster recovery"
git push origin HEAD:feature/fever-formal-readiness
```

Verify from GitHub, not only local Git:

```bash
git fetch origin feature/fever-formal-readiness
FINAL_RECOVERY_COMMIT="$(git rev-parse HEAD)"
test "$(git rev-parse origin/feature/fever-formal-readiness)" = \
  "$FINAL_RECOVERY_COMMIT"

git ls-tree -r --name-only "$FINAL_RECOVERY_COMMIT" \
  environment/server/captured
```

Record `FINAL_RECOVERY_COMMIT` in at least two locations outside the old
instance. A file inside its own commit cannot contain that commit's final SHA,
so this external record is mandatory.

## 5. Confirm idle state, then destroy

```bash
tmux ls || true
pgrep -af 'python|run_fever_cbwdm|15a_run_fever_calibration_grid' || true
nvidia-smi
```

Confirm no experiment process is running, GitHub contains the capture files,
the final commit is recorded externally, and any required credentials for
redownloading licensed/private assets are available. Only then may the old
instance be destroyed.

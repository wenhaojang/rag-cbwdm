# RAG-CBWDM server reproduction and resume runbook

## 1. Scope and authority

This runbook covers the FEVER-2 formal pilot:

```text
repository:     /root/rag-cbwdm
branch:         feature/fever-formal-readiness
experiment:     /root/experiments/rag_cbwdm
run name:       fever2_formal_pilot_5000_500_seed13
retrieval env:  rag-cbwdm-retrieval
baseline env:   rag-cbwdm-baselines
```

Commands below were audited against the current CLI definitions in
`scripts/run_fever_cbwdm.py`, `scripts/15a_run_fever_calibration_grid.py`,
`scripts/15_calibrate_fever_methods.py`,
`scripts/16_freeze_fever_formal_config.py`,
`scripts/16a_diagnose_cbwdm_pilot.py`,
`scripts/17_check_fever_formal_readiness.py`, and the fixed-baseline entry
points.

The experiment description is not itself restoration evidence. The source of
truth is the exact Git commit plus completed manifests, artifact SHA-256 values,
row counts, role contracts, model inventories, and environment snapshot.

`PROJECT_STATUS_REPORT.md` is absent from the current local worktree. Its Git
HEAD version is historical and predates the formal protocol, so it must not be
used as a current server command source. `FEVER_FORMAL_EXPERIMENT_READINESS_REPORT.md`
and current code are authoritative.

This runbook does not authorize held-out work. Until
`formal_readiness.json.status=ready`, do not run:

```text
retrieve_test
posterior_test
any held_out_test teacher
any held_out_test evaluation
```

## 2. Current checkpoint

The reported source-server checkpoint is:

- cleaned FEVER-2 formal split completed;
- full FEVER Wiki sentence corpus completed;
- Lucene v2 full index completed;
- train-core BM25 top-20 retrieval: 5000 rows;
- validation BM25 top-20 retrieval: 500 rows;
- train-core posterior: 5000 rows;
- validation posterior: 500 rows;
- one InfoGain training candidate and two InfoGain selection/evaluation smoke
  candidates completed;
- one RAG-CBWDM training candidate and two RAG-CBWDM
  selection/evaluation smoke candidates completed;
- failed-candidate retry, completed-child reuse, and top-level grid resume
  verified on the server.

Still pending:

1. full InfoGain calibration grid;
2. full RAG-CBWDM calibration grid;
3. combined grid aggregate reconciliation;
4. `calibrate_methods`;
5. fixed validation baselines, fairness audit, and canonical summary;
6. CBWDM pilot diagnostics;
7. frozen formal configuration;
8. formal readiness;
9. held-out formal execution, only after readiness is `ready`.

Run `scripts/19_verify_resumed_server.sh` before accepting this checkpoint. A
reported item whose manifest or SHA fails verification is incomplete.

## 3. Storage and reproducibility layers

| Layer | Examples | Restore rule |
|---|---|---|
| System environment | Ubuntu, kernel, NVIDIA driver, mounts, RAM | Recreate from snapshot; do not copy a system disk image blindly across incompatible GPU hosts |
| Python environments | Two conda envs, pip freezes, Java | Rebuild separately from captured Linux histories |
| Code | Git repository and exact commit | Clone/fetch, checkout detached exact commit, then verify clean worktree |
| Raw data | FEVER train/dev/wiki pages | Restore verified bytes and SHA inventories; never edit raw FEVER files |
| Models | Five `/root/models/...` directories | Restore exact directory inventories or redownload pinned revisions and verify hashes |
| Shared artifacts | `$EXP_ROOT/_shared/corpora`, `$EXP_ROOT/_shared/indexes` | Preserve paths and manifests; included when the entire experiment root is backed up |
| Run-specific artifacts | `$RUN/artifacts`, logs, commands, `run_manifest.json` | Must be restored at the identical path for safe resume |
| Redownloadable cache | Hugging Face cache blobs that have immutable upstream revisions | May be redownloaded, but is not a substitute for local model/checkpoint backups |
| Must-back-up state | exact Git SHA, experiment root, split directory, environment snapshot, raw-data inventory, unique checkpoints | Loss forces recomputation or makes exact reproduction impossible |

The formal split manifest alone is insufficient: its referenced
`train_core.jsonl`, `validation.jsonl`, `held_out_test.jsonl`, conflict and
cross-source audit files must stay together under
`outputs/formal_splits/fever2_seed13/`.

## 4. Cloud lifecycle warning

Cloud-provider terminology is not portable:

- **stop/pause** may preserve attached persistent disks but still release the
  GPU allocation;
- **release/delete/terminate** commonly destroys the VM and may destroy the
  system disk;
- a temporary/local/NVMe instance disk may be erased on stop, host migration,
  release, or hardware maintenance;
- a separately attached data disk may still have an independent
  `delete-on-termination` policy;
- snapshots can be crash-consistent rather than application-consistent.

Before shutdown, confirm in the provider console, in writing or screenshots:

1. whether the system disk survives stop;
2. whether the data disk survives stop and instance deletion;
3. every disk's `delete on release/termination` setting;
4. whether `/root`, `/root/experiments`, `/root/models`, and
   `/root/huggingface` are on persistent or ephemeral storage;
5. snapshot completion, region/account, encryption-key retention, and restore
   permissions;
6. GPU instance reservation/restart availability.

Do not infer disk retention from the word “stop”.

---

# Scenario A — restart the same preserved server

## A1. Before stopping

SSH in and establish variables:

```bash
ssh root@SERVER_HOST
cd /root/rag-cbwdm

export REPO=/root/rag-cbwdm
export EXP_ROOT=/root/experiments/rag_cbwdm
export RUN_NAME=fever2_formal_pilot_5000_500_seed13
export RUN="$EXP_ROOT/$RUN_NAME"
export MODEL_ROOT=/root/models
export HF_HOME=/root/huggingface
```

Inspect active work before stopping it:

```bash
tmux ls || true
pgrep -af 'python|run_fever_cbwdm|15a_run_fever_calibration_grid' || true
nvidia-smi
```

Do not kill a process until its log and manifest status are understood.
`status=running` is not resumable proof; completed reuse requires the final
manifest and matching outputs.

Capture the state:

```bash
SNAPSHOT_DIR="$(
  bash scripts/18_capture_server_resume_state.sh
)"
printf 'snapshot=%s\n' "$SNAPSHOT_DIR"
```

The script is read-only with respect to the repository, raw data, models, and
experiment artifacts. It creates only a new timestamped directory under:

```text
/root/experiments/rag_cbwdm/server_state_snapshots/
```

Record the commit and run the acceptance gate:

```bash
export EXPECTED_GIT_HEAD="$(git rev-parse HEAD)"
bash scripts/19_verify_resumed_server.sh
```

Proceed only on `PASS`. On `BLOCKED`, preserve the output and resolve every
blocker without deleting or overwriting existing experiment artifacts.

Create an external backup according to section 12. Verify the backup from the
destination, then confirm provider disk-retention settings. The capture script
is not a backup by itself.

## A2. After restarting

SSH in and check the physical host:

```bash
ssh root@SERVER_HOST
nvidia-smi
free -h
df -hT
lsblk -o NAME,SIZE,FSTYPE,TYPE,MOUNTPOINTS,MODEL,SERIAL
```

Check code:

```bash
cd /root/rag-cbwdm
git rev-parse HEAD
git branch --show-current
git status --short
git remote -v
git tag --points-at HEAD
```

Use the commit from the pre-stop snapshot, not a branch name alone:

```bash
export EXPECTED_GIT_HEAD='<64-or-40-character HEAD from snapshot>'
test "$(git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(git status --porcelain)"
```

Validate environment separation:

```bash
conda activate rag-cbwdm-retrieval
python --version
java -version
python -c 'import importlib.metadata as m, pyserini; print(m.version("pyserini"))'
conda deactivate

conda activate rag-cbwdm-baselines
python -c 'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__); print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
python -c 'import pyserini' && {
  echo 'ERROR: Pyserini unexpectedly exists in baseline env' >&2
  exit 2
} || true
conda deactivate
```

Check model directories without loading models:

```bash
for model in \
  Qwen2.5-1.5B-Instruct \
  Qwen2.5-7B-Instruct \
  ms-marco-MiniLM-L-6-v2 \
  bge-reranker-base \
  bge-reranker-large
do
  test -d "/root/models/$model"
  find "/root/models/$model" -type f -print -quit
done
```

Run the single acceptance command:

```bash
cd /root/rag-cbwdm
export REPO=/root/rag-cbwdm
export EXP_ROOT=/root/experiments/rag_cbwdm
export RUN_NAME=fever2_formal_pilot_5000_500_seed13
export RUN="$EXP_ROOT/$RUN_NAME"
export MODEL_ROOT=/root/models
export EXPECTED_GIT_HEAD='<HEAD from the pre-stop snapshot>'
bash scripts/19_verify_resumed_server.sh
```

On `PASS`, do not recompute:

- formal splits;
- full corpus;
- Lucene v2 index;
- 5000/500 retrieval;
- 5000/500 posterior;
- completed smoke teachers, checkpoints, selections, or evaluations.

Resume at the full calibration grids using section 10.

---

# Scenario B — reconstruct a newly rented server

## B1. Recommended machine

Use:

- Ubuntu 22.04 LTS to match the source server;
- NVIDIA GPU with at least 24 GB VRAM; RTX 3090 equivalence is the minimum
  known working target;
- 16 or more vCPUs;
- at least 64 GB RAM;
- a persistent system volume of at least 100 GB;
- a separate persistent data volume sized from the captured `du -sh` results,
  with additional free space for checkpoints, full-grid logs, temporary files,
  and backups. Do not choose a disk size from an unverified estimate.

Mount persistent storage before restoring absolute paths. If `/root` is on the
system disk, confirm that disk is persistent. Do not put the experiment root,
models, or only copy of raw data on ephemeral NVMe.

## B2. System bootstrap

After installing the NVIDIA driver supported by the provider image:

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl ca-certificates rsync jq tmux build-essential \
  openssh-client pciutils

nvidia-smi
```

`nvcc` is optional for this project unless a dependency must compile CUDA
extensions. Its absence must be recorded; it is not equivalent to a missing
driver.

Install Miniconda from the official Linux x86-64 installer and verify the
published checksum before executing it:

```bash
cd /tmp
curl -fsSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sha256sum Miniconda3-latest-Linux-x86_64.sh
# Compare with the checksum published by repo.anaconda.com before continuing.
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
/root/miniconda3/bin/conda init bash
source /root/miniconda3/etc/profile.d/conda.sh
conda --version
```

## B3. Restore exact code

Clone, fetch the named branch for orientation, then detach the exact captured
commit:

```bash
git clone https://github.com/wenhaojang/rag-cbwdm.git /root/rag-cbwdm
cd /root/rag-cbwdm
git fetch --all --tags --prune
git fetch origin feature/fever-formal-readiness

export EXPECTED_GIT_HEAD='<HEAD from source-server snapshot>'
git checkout --detach "$EXPECTED_GIT_HEAD"
test "$(git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(git status --porcelain)"
```

The branch may advance after capture. Do not substitute its newest tip for the
captured commit.

## B4. Rebuild the two Python environments

Copy the source snapshot to the new host first. Inspect:

```bash
export SNAPSHOT=/path/to/restored/server_state_snapshots/TIMESTAMP
sed -n '1,200p' "$SNAPSHOT/environments/rag-cbwdm-retrieval/conda-from-history.yml"
sed -n '1,200p' "$SNAPSHOT/environments/rag-cbwdm-baselines/conda-from-history.yml"
```

Create the retrieval environment from its Linux history:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda env create \
  -f "$SNAPSHOT/environments/rag-cbwdm-retrieval/conda-from-history.yml"
conda activate rag-cbwdm-retrieval
python -m pip install -r /root/rag-cbwdm/requirements-retrieval.txt
python -c 'import importlib.metadata as m, pyserini; print(m.version("pyserini"))'
java -version
conda deactivate
```

The captured environment must show Java 21 and Pyserini 2.3.0. If Java was
installed outside conda on the source server, reproduce that provenance
explicitly and record it.

Create the baseline environment from its separate Linux history:

```bash
conda env create \
  -f "$SNAPSHOT/environments/rag-cbwdm-baselines/conda-from-history.yml"
conda activate rag-cbwdm-baselines

# Install the captured CUDA-compatible Torch build from the same official
# conda channel or pip index recorded by the snapshot. Do not use an
# unqualified `pip install torch` when the source build was CUDA-specific.

python -m pip install -r /root/rag-cbwdm/requirements-baselines.txt
python -m pip install \
  pyyaml numpy accelerate datasets pytest

python -c 'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python -c 'import pyserini' && {
  echo 'ERROR: keep Pyserini out of the baseline environment' >&2
  exit 2
} || true
conda deactivate
```

Compare both reconstructed environments with `conda-no-builds.yml` and
`pip-freeze.txt`. Do not export Windows `.venv` package state as a Linux lock.
See `environment/server/README.md`.

## B5. Restore models

Preferred: restore `/root/models` from the verified backup at the same absolute
path. Then compare file counts, byte totals, config/tokenizer hashes, and every
weight-file hash with the source snapshot.

If models must be redownloaded, use the exact upstream revisions recorded in
the model metadata/snapshot, download into a staging directory, verify hashes,
then move the verified complete directory to:

```text
/root/models/Qwen2.5-1.5B-Instruct
/root/models/Qwen2.5-7B-Instruct
/root/models/ms-marco-MiniLM-L-6-v2
/root/models/bge-reranker-base
/root/models/bge-reranker-large
```

Do not replace a pinned local model with a moving model-hub branch.

## B6. Restore raw FEVER data

Restore to:

```text
/root/rag-cbwdm/data/raw/fever/train.jsonl
/root/rag-cbwdm/data/raw/fever/dev.jsonl
/root/rag-cbwdm/data/raw/fever/wiki-pages/
```

Compare all file SHA-256 values with the snapshot. Never manually remove,
deduplicate, relabel, or edit FEVER records. The cleaned protocol is implemented
by the formal split builder and its exclusion artifacts.

If raw data must be redownloaded, use the same source and materialization
procedure, then require byte/inventory equivalence. A semantically similar
dataset export with different bytes does not satisfy existing manifest
fingerprints.

## B7. Restore experiment artifacts from backup

Restore the complete tree at the original path:

```text
/root/experiments/rag_cbwdm/
  _shared/corpora/...
  _shared/indexes/<Lucene-v2-fingerprint>/...
  fever2_formal_pilot_5000_500_seed13/...
  server_state_snapshots/...
```

Also restore:

```text
/root/rag-cbwdm/outputs/formal_splits/fever2_seed13/
```

Example rsync restore:

```bash
rsync -aHAX --numeric-ids --info=progress2 \
  BACKUP_HOST:/backup/rag_cbwdm/experiments/ \
  /root/experiments/rag_cbwdm/

rsync -aHAX --numeric-ids --info=progress2 \
  BACKUP_HOST:/backup/rag_cbwdm/formal_splits/ \
  /root/rag-cbwdm/outputs/formal_splits/fever2_seed13/

rsync -aHAX --numeric-ids --info=progress2 \
  BACKUP_HOST:/backup/rag_cbwdm/models/ \
  /root/models/
```

Use a trailing slash consistently. First run with `--dry-run`; never use
`--delete` against a partially restored or active experiment tree.

## B8. If no artifact backup exists

The following work is irrecoverable and must be recomputed:

1. cleaned formal split and its conflict/cross-source audits;
2. full sentence corpus;
3. Lucene v2 index;
4. 5000/500 retrieval;
5. 5000/500 posterior;
6. smoke and full calibration teachers/checkpoints/selections/evaluations.

After raw data, models, code, and environments are restored, reconstruction
starts in the retrieval environment:

```bash
conda activate rag-cbwdm-retrieval
cd /root/rag-cbwdm

python scripts/run_fever_cbwdm.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-name fever2_formal_pilot_5000_500_seed13 \
  --stages prepare_formal_splits,corpus,index,retrieve_train_core,retrieve_validation \
  --output-root /root/experiments/rag_cbwdm \
  --cache-root /root/huggingface \
  --resume
```

Then in the baseline environment:

```bash
conda activate rag-cbwdm-baselines
cd /root/rag-cbwdm

python scripts/run_fever_cbwdm.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-name fever2_formal_pilot_5000_500_seed13 \
  --stages posterior_train_core,posterior_validation \
  --output-root /root/experiments/rag_cbwdm \
  --cache-root /root/huggingface \
  --generator-model /root/models/Qwen2.5-1.5B-Instruct \
  --resume
```

This recomputation is not “resume to the old state”; it is a new execution whose
new manifests and SHA values must be captured and audited. Never fabricate old
manifests or copy only `status=completed`.

## B9. Acceptance after restore

Run:

```bash
cd /root/rag-cbwdm
export REPO=/root/rag-cbwdm
export EXP_ROOT=/root/experiments/rag_cbwdm
export RUN_NAME=fever2_formal_pilot_5000_500_seed13
export RUN="$EXP_ROOT/$RUN_NAME"
export MODEL_ROOT=/root/models
export EXPECTED_GIT_HEAD='<captured source-server HEAD>'
bash scripts/19_verify_resumed_server.sh
```

The script checks code, environments, models, formal split, shared
corpus/index, 5000/500 retrieval and posterior contracts, prompt/verbalizer
hashes, smoke candidate completion, and absence of premature held-out
retrieval/posterior artifacts. Continue only on `PASS`.

---

# 10. Exact continuation commands

All commands in this section run in `rag-cbwdm-baselines` and reuse the existing
5000/500 retrieval and posterior artifacts. They do not require Pyserini.

```bash
conda activate rag-cbwdm-baselines
cd /root/rag-cbwdm

export CONFIG=configs/fever2_server_pilot_5000_500.yaml
export RUN_NAME=fever2_formal_pilot_5000_500_seed13
export EXP_ROOT=/root/experiments/rag_cbwdm
export RUN="$EXP_ROOT/$RUN_NAME"
export HF_HOME=/root/huggingface
```

## 10.1 Full InfoGain-FEVER grid

Do not pass `--candidate-limit` or `--max-training-candidates`:

```bash
python scripts/run_fever_cbwdm.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --stages run_calibration_grid \
  --methods infogain_fever \
  --output-root "$EXP_ROOT" \
  --cache-root "$HF_HOME" \
  --generator-model /root/models/Qwen2.5-1.5B-Instruct \
  --infogain-model /root/models/ms-marco-MiniLM-L-6-v2 \
  --selector-device cuda \
  --infogain-device cuda \
  --resume \
  --skip-completed \
  --continue-on-error
```

Completed smoke artifacts are reused only when their fingerprints and output
SHA values match.

## 10.2 Full RAG-CBWDM grid

```bash
python scripts/run_fever_cbwdm.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --stages run_calibration_grid \
  --methods rag_cbwdm \
  --output-root "$EXP_ROOT" \
  --cache-root "$HF_HOME" \
  --generator-model /root/models/Qwen2.5-1.5B-Instruct \
  --selector-model /root/models/ms-marco-MiniLM-L-6-v2 \
  --selector-device cuda \
  --infogain-device cuda \
  --resume \
  --skip-completed \
  --continue-on-error
```

## 10.3 Reconcile the combined aggregate

Method-filtered runs publish the aggregate for their exact request. After both
full method grids complete, make one combined resume pass so
`calibration_candidates.json` contains the exact combined plan required by
`calibrate_methods`:

```bash
python scripts/run_fever_cbwdm.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --stages run_calibration_grid \
  --methods infogain_fever,rag_cbwdm \
  --output-root "$EXP_ROOT" \
  --cache-root "$HF_HOME" \
  --generator-model /root/models/Qwen2.5-1.5B-Instruct \
  --selector-model /root/models/ms-marco-MiniLM-L-6-v2 \
  --infogain-model /root/models/ms-marco-MiniLM-L-6-v2 \
  --selector-device cuda \
  --infogain-device cuda \
  --resume \
  --skip-completed \
  --continue-on-error
```

Every child should be checksum-reused. The aggregate is reusable only if all
108 planned candidates validate as completed.

## 10.4 Select calibrated methods

```bash
python scripts/run_fever_cbwdm.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --stages calibrate_methods \
  --output-root "$EXP_ROOT" \
  --cache-root "$HF_HOME" \
  --resume
```

## 10.5 Resolve selected trainable candidates

```bash
export FORMAL="$RUN/artifacts/formal"
export CANDIDATES="$FORMAL/calibration_candidates.json"
export CALIBRATION="$FORMAL/calibration/calibration.manifest.json"
export FIXED="$FORMAL/fixed_baselines"
export VALIDATION_SPLIT=/root/rag-cbwdm/outputs/formal_splits/fever2_seed13/validation.jsonl
export VALIDATION_RETRIEVAL="$FORMAL/fever2_validation_bm25_top20.jsonl"
export VALIDATION_POSTERIOR="$FORMAL/fever2_validation_posteriors.jsonl"
mkdir -p "$FIXED"

export INFOGAIN_SELECTION="$(
python - "$CALIBRATION" "$CANDIDATES" infogain_fever selection_path <<'PY'
import json, sys
calibration, candidates, method, field = sys.argv[1:]
c = json.load(open(calibration))
fp = c["selected"][method]["candidate_fingerprint"]
rows = json.load(open(candidates))["candidates"]
matches = [r for r in rows if r.get("method") == method and r.get("candidate_fingerprint") == fp and r.get("status") == "completed"]
assert len(matches) == 1
print(matches[0][field])
PY
)"
export INFOGAIN_METRICS="$(
python - "$CALIBRATION" "$CANDIDATES" infogain_fever metrics_path <<'PY'
import json, sys
calibration, candidates, method, field = sys.argv[1:]
c = json.load(open(calibration))
fp = c["selected"][method]["candidate_fingerprint"]
rows = json.load(open(candidates))["candidates"]
matches = [r for r in rows if r.get("method") == method and r.get("candidate_fingerprint") == fp and r.get("status") == "completed"]
assert len(matches) == 1
print(matches[0][field])
PY
)"
export RAG_SELECTION="$(
python - "$CALIBRATION" "$CANDIDATES" rag_cbwdm selection_path <<'PY'
import json, sys
calibration, candidates, method, field = sys.argv[1:]
c = json.load(open(calibration))
fp = c["selected"][method]["candidate_fingerprint"]
rows = json.load(open(candidates))["candidates"]
matches = [r for r in rows if r.get("method") == method and r.get("candidate_fingerprint") == fp and r.get("status") == "completed"]
assert len(matches) == 1
print(matches[0][field])
PY
)"
export RAG_METRICS="$(
python - "$CALIBRATION" "$CANDIDATES" rag_cbwdm metrics_path <<'PY'
import json, sys
calibration, candidates, method, field = sys.argv[1:]
c = json.load(open(calibration))
fp = c["selected"][method]["candidate_fingerprint"]
rows = json.load(open(candidates))["candidates"]
matches = [r for r in rows if r.get("method") == method and r.get("candidate_fingerprint") == fp and r.get("status") == "completed"]
assert len(matches) == 1
print(matches[0][field])
PY
)"
export INFOGAIN_EVAL_MANIFEST="${INFOGAIN_METRICS%.json}.manifest.json"
export RAG_EVAL_MANIFEST="${RAG_METRICS%.json}.manifest.json"
```

## 10.6 Fixed validation baselines

No-evidence:

```bash
python scripts/07_eval_rag_classification.py \
  --config "$CONFIG" \
  --split validation \
  --selection "$VALIDATION_SPLIT" \
  --output "$FIXED/no_evidence_predictions.jsonl" \
  --metrics-output "$FIXED/no_evidence_metrics.json" \
  --model-name /root/models/Qwen2.5-1.5B-Instruct \
  --method-name no_evidence \
  --no-evidence \
  --resume
```

Naive top-M:

```bash
python scripts/08_select_naive_topm.py \
  --config "$CONFIG" \
  --retrieval "$VALIDATION_RETRIEVAL" \
  --output "$FIXED/naive_topm_selection.jsonl" \
  --top-m 4 \
  --min-docs 4 \
  --method-name naive_topm \
  --resume

python scripts/07_eval_rag_classification.py \
  --config "$CONFIG" \
  --split validation \
  --selection "$FIXED/naive_topm_selection.jsonl" \
  --output "$FIXED/naive_topm_predictions.jsonl" \
  --metrics-output "$FIXED/naive_topm_metrics.json" \
  --model-name /root/models/Qwen2.5-1.5B-Instruct \
  --method-name naive_topm \
  --resume
```

BGE:

```bash
python scripts/12_select_bge_reranker.py \
  --retrieval "$VALIDATION_RETRIEVAL" \
  --output "$FIXED/bge_selection.jsonl" \
  --score-cache "$FIXED/bge_scores.jsonl" \
  --model-name-or-path /root/models/bge-reranker-large \
  --device cuda \
  --dtype auto \
  --batch-size 8 \
  --max-length 512 \
  --top-m 4 \
  --min-docs 4 \
  --local-files-only \
  --resume

python scripts/07_eval_rag_classification.py \
  --config "$CONFIG" \
  --split validation \
  --selection "$FIXED/bge_selection.jsonl" \
  --output "$FIXED/bge_predictions.jsonl" \
  --metrics-output "$FIXED/bge_metrics.json" \
  --model-name /root/models/Qwen2.5-1.5B-Instruct \
  --method-name bge \
  --resume
```

Validation-only Oracle diagnostic:

```bash
python scripts/04_build_cbwdm_teacher.py \
  --config "$CONFIG" \
  --split validation \
  --posteriors "$VALIDATION_POSTERIOR" \
  --output "$FIXED/cbwdm_validation_teacher.jsonl" \
  --resume

python scripts/09_select_cbwdm_oracle_from_teacher.py \
  --config "$CONFIG" \
  --teacher "$FIXED/cbwdm_validation_teacher.jsonl" \
  --retrieval "$VALIDATION_RETRIEVAL" \
  --posteriors "$VALIDATION_POSTERIOR" \
  --output "$FIXED/cbwdm_oracle_selection.jsonl" \
  --top-m 4 \
  --method-name cbwdm_oracle \
  --resume

python scripts/07_eval_rag_classification.py \
  --config "$CONFIG" \
  --split validation \
  --selection "$FIXED/cbwdm_oracle_selection.jsonl" \
  --output "$FIXED/cbwdm_oracle_predictions.jsonl" \
  --metrics-output "$FIXED/cbwdm_oracle_metrics.json" \
  --model-name /root/models/Qwen2.5-1.5B-Instruct \
  --method-name cbwdm_oracle \
  --resume
```

The Oracle is `deployable=false`, `diagnostic_only=true`. It cannot select
deployable parameters.

## 10.7 Fairness audit and canonical summary

```bash
export BASELINES="$FORMAL/baselines"
mkdir -p "$BASELINES/summary"

python scripts/14_audit_fever_baselines.py \
  --retrieval "$VALIDATION_RETRIEVAL" \
  --selection "naive_topm=$FIXED/naive_topm_selection.jsonl" \
  --selection "bge=$FIXED/bge_selection.jsonl" \
  --selection "infogain_fever=$INFOGAIN_SELECTION" \
  --selection "rag_cbwdm=$RAG_SELECTION" \
  --selection "cbwdm_oracle=$FIXED/cbwdm_oracle_selection.jsonl" \
  --evaluation-manifest "no_evidence=$FIXED/no_evidence_metrics.manifest.json" \
  --evaluation-manifest "naive_topm=$FIXED/naive_topm_metrics.manifest.json" \
  --evaluation-manifest "bge=$FIXED/bge_metrics.manifest.json" \
  --evaluation-manifest "infogain_fever=$INFOGAIN_EVAL_MANIFEST" \
  --evaluation-manifest "rag_cbwdm=$RAG_EVAL_MANIFEST" \
  --evaluation-manifest "cbwdm_oracle=$FIXED/cbwdm_oracle_metrics.manifest.json" \
  --expected-top-m 4 \
  --output "$BASELINES/baseline_fairness_audit.json"

python scripts/13_summarize_fever_baselines.py \
  --run-dir "$RUN" \
  --output-dir "$BASELINES/summary" \
  --fairness-audit "$BASELINES/baseline_fairness_audit.json" \
  --evaluation-manifest "no_evidence=$FIXED/no_evidence_metrics.manifest.json" \
  --evaluation-manifest "naive_topm=$FIXED/naive_topm_metrics.manifest.json" \
  --evaluation-manifest "bge=$FIXED/bge_metrics.manifest.json" \
  --evaluation-manifest "infogain_fever=$INFOGAIN_EVAL_MANIFEST" \
  --evaluation-manifest "rag_cbwdm=$RAG_EVAL_MANIFEST" \
  --evaluation-manifest "cbwdm_oracle=$FIXED/cbwdm_oracle_metrics.manifest.json"
```

The summary refuses formal publication unless the fairness audit is
`comparable`.

## 10.8 CBWDM diagnostics

The diagnostics script resolves the selected RAG-CBWDM candidate from
calibration provenance:

```bash
python scripts/16a_diagnose_cbwdm_pilot.py \
  --config "$CONFIG" \
  --no-evidence "$FIXED/no_evidence_predictions.jsonl" \
  --naive "$FIXED/naive_topm_predictions.jsonl" \
  --oracle "$FIXED/cbwdm_oracle_predictions.jsonl" \
  --naive-selection "$FIXED/naive_topm_selection.jsonl" \
  --oracle-selection "$FIXED/cbwdm_oracle_selection.jsonl" \
  --retrieval "$VALIDATION_RETRIEVAL" \
  --posteriors "$VALIDATION_POSTERIOR" \
  --calibration-manifest "$CALIBRATION" \
  --calibration-candidates "$CANDIDATES" \
  --output-dir "$FORMAL/diagnostics"
```

## 10.9 Record required tests

Run:

```bash
python -m compileall -q src scripts tests
python -m pytest -q
python -m unittest discover
bash -n scripts/run_fever_cbwdm.sh
bash -n scripts/18_capture_server_resume_state.sh
bash -n scripts/19_verify_resumed_server.sh
git diff --check
```

After all commands pass on the exact formal commit, record them without claiming
model results:

```bash
python - "$FORMAL/tests_status.json" "$(git rev-parse HEAD)" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
path, head = sys.argv[1:]
payload = {
    "status": "passed",
    "git_head": head,
    "commands": [
        "python -m compileall -q src scripts tests",
        "python -m pytest -q",
        "python -m unittest discover",
        "bash -n scripts/run_fever_cbwdm.sh",
        "bash -n scripts/18_capture_server_resume_state.sh",
        "bash -n scripts/19_verify_resumed_server.sh",
        "git diff --check",
    ],
    "recorded_at": datetime.now(timezone.utc).isoformat(),
}
os.makedirs(os.path.dirname(path), exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".tests-status-", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
```

## 10.10 Freeze the formal configuration

Resolve corpus/index manifests from the run manifest:

```bash
export CORPUS_MANIFEST="$(
python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["corpus_manifest"])' \
  "$RUN/run_manifest.json"
)"
export INDEX_MANIFEST="$(
python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["index_manifest"])' \
  "$RUN/run_manifest.json"
)"
```

Resolve selected checkpoint directories and immutable training fingerprints:

```bash
readarray -t SELECTED_MODEL_VALUES < <(
python - "$CALIBRATION" "$CANDIDATES" <<'PY'
import json, pathlib, sys
calibration, candidates = map(lambda p: json.load(open(p)), sys.argv[1:])
for method in ("infogain_fever", "rag_cbwdm"):
    fp = calibration["selected"][method]["candidate_fingerprint"]
    matches = [r for r in candidates["candidates"] if r.get("method") == method and r.get("candidate_fingerprint") == fp and r.get("status") == "completed"]
    assert len(matches) == 1
    row = matches[0]
    checkpoint = pathlib.Path(row["checkpoint_manifest"]).parent / "checkpoint"
    print(checkpoint)
    print(row["training_fingerprint"])
PY
)
export INFOGAIN_CHECKPOINT="${SELECTED_MODEL_VALUES[0]}"
export INFOGAIN_REVISION="${SELECTED_MODEL_VALUES[1]}"
export RAG_CHECKPOINT="${SELECTED_MODEL_VALUES[2]}"
export RAG_REVISION="${SELECTED_MODEL_VALUES[3]}"
```

Use immutable model revisions from the server snapshot or model-hub commit. Do
not use placeholders or `main`:

```bash
export GENERATOR_REVISION='<verified immutable Qwen revision/hash>'
export TOKENIZER_REVISION='<verified immutable tokenizer revision/hash>'
export BGE_REVISION='<verified immutable BGE revision/hash>'

python scripts/16_freeze_fever_formal_config.py \
  --base-config configs/fever2_server_formal.yaml \
  --split-manifest /root/rag-cbwdm/outputs/formal_splits/fever2_seed13/fever2_formal_splits.manifest.json \
  --calibration-manifest "$CALIBRATION" \
  --corpus "$CORPUS_MANIFEST" \
  --index "$INDEX_MANIFEST" \
  --output-dir "$FORMAL/frozen_config" \
  --model generator=/root/models/Qwen2.5-1.5B-Instruct \
  --revision "generator=$GENERATOR_REVISION" \
  --model tokenizer=/root/models/Qwen2.5-1.5B-Instruct \
  --revision "tokenizer=$TOKENIZER_REVISION" \
  --model bge=/root/models/bge-reranker-large \
  --revision "bge=$BGE_REVISION" \
  --model "infogain=$INFOGAIN_CHECKPOINT" \
  --revision "infogain=$INFOGAIN_REVISION" \
  --model "rag_cbwdm=$RAG_CHECKPOINT" \
  --revision "rag_cbwdm=$RAG_REVISION"
```

Do not add `--overwrite` to a resume attempt. A changed fingerprint must publish
to its own generated filename.

## 10.11 Formal readiness

Identify the single frozen manifest:

```bash
export FROZEN_MANIFEST="$(
find "$FORMAL/frozen_config" -maxdepth 1 -type f \
  -name 'fever2_formal_frozen_*.manifest.json' -print -quit
)"
test -n "$FROZEN_MANIFEST"
```

Run the P0 gate with full artifact rehashing:

```bash
python scripts/17_check_fever_formal_readiness.py \
  --split-manifest /root/rag-cbwdm/outputs/formal_splits/fever2_seed13/fever2_formal_splits.manifest.json \
  --calibration-manifest "$CALIBRATION" \
  --frozen-manifest "$FROZEN_MANIFEST" \
  --fairness-audit "$BASELINES/baseline_fairness_audit.json" \
  --baseline-summary "$BASELINES/summary/baseline_summary.json" \
  --cbwdm-diagnostics "$FORMAL/diagnostics/cbwdm_pilot_diagnostics.json" \
  --tests-status "$FORMAL/tests_status.json" \
  --output-dir "$FORMAL/readiness"
```

Do not use `--skip-artifact-rehash` on the server.

Require:

```bash
python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "ready" and not p["blockers"]' \
  "$FORMAL/readiness/formal_readiness.json"
```

Only after this assertion passes may a separately reviewed held-out runbook be
executed.

---

# 11. Audit checks performed by the scripts

## `18_capture_server_resume_state.sh`

The capture script records:

- UTC time, hostname, OS, kernel, CPU, RAM, filesystems, and block devices;
- NVIDIA driver/GPU state and optional `nvcc`;
- Git remotes (credential-redacted), branch, HEAD, status, and tags;
- conda environment list;
- no-build and from-history exports plus pip freeze for both environments;
- Java/Pyserini in retrieval;
- Torch/Transformers/CUDA/GPU and Pyserini absence/presence in baselines;
- five model paths, file counts, sizes, config/tokenizer SHA values, and weight
  inventories;
- raw FEVER train/dev/test metadata and wiki-page SHA inventory;
- formal split, corpus, Lucene v2, run, retrieval, posterior, and grid
  manifest/file identities;
- Lucene fingerprint and inventory summary SHA;
- candidate status/metrics without model execution;
- checkpoint file inventories and SHA values;
- tmux, relevant Python processes, GPU compute processes, and key directory
  sizes.

Missing optional commands/files are recorded as `MISSING` or
`MISSING_OR_FAILED`. The script never parses `held_out_test` JSONL records and
never runs a model.

## `19_verify_resumed_server.sh`

The acceptance script checks:

- repository, exact `EXPECTED_GIT_HEAD`, and clean worktree;
- NVIDIA driver/GPU query;
- both conda environments and their intended dependency separation;
- all five model directories;
- formal split validity, checksums, exclusions, and overlap contract;
- shared corpus manifest and Lucene v2 contract/inventory;
- exactly 5000/500 retrieval rows and 5000/500 posterior rows;
- output SHA values, roles, top-N=20, and posterior input SHA provenance;
- prompt and verbalizer hashes against the current pilot config;
- at least two completed smoke evaluations per trainable method with non-null
  accuracy and macro-F1;
- absence, without content inspection, of held-out retrieval and posterior
  artifacts.

Its only final output is `PASS`, or `BLOCKED` followed by every blocker.

---

# 12. Backup tiers

These are examples only. Review paths and destination capacity first; do not
run them from Codex.

## Minimum

Must include:

- exact Git commit SHA and remote;
- `/root/experiments/rag_cbwdm`;
- `/root/rag-cbwdm/outputs/formal_splits/fever2_seed13`;
- run manifest;
- latest environment snapshot;
- verified raw-data SHA inventory and either raw FEVER backup or a tested
  immutable redownload source.

Example:

```bash
tar --acls --xattrs --numeric-owner -cpf \
  /persistent-backup/rag_cbwdm-minimum.tar \
  -C /root \
  experiments/rag_cbwdm \
  rag-cbwdm/outputs/formal_splits/fever2_seed13

sha256sum /persistent-backup/rag_cbwdm-minimum.tar \
  > /persistent-backup/rag_cbwdm-minimum.tar.sha256
```

## Recommended

Add:

```text
/root/models
/root/rag-cbwdm/data/raw/fever
```

Example rsync:

```bash
rsync -aHAX --numeric-ids --info=progress2 \
  /root/experiments/rag_cbwdm/ \
  BACKUP_HOST:/backup/rag_cbwdm/experiments/

rsync -aHAX --numeric-ids --info=progress2 \
  /root/models/ \
  BACKUP_HOST:/backup/rag_cbwdm/models/

rsync -aHAX --numeric-ids --info=progress2 \
  /root/rag-cbwdm/data/raw/fever/ \
  BACKUP_HOST:/backup/rag_cbwdm/raw_fever/
```

## Complete

Also add:

```text
/root/huggingface
/root/miniconda3/pkgs
```

The conda package cache accelerates reconstruction but does not replace
environment exports. The Hugging Face cache is often redownloadable, but only
when every dependency is pinned and still available.

For every tier:

1. write a SHA-256 manifest at the destination;
2. verify a sample restore or full archive listing;
3. keep at least one copy outside the source server and outside its provider
   lifecycle;
4. protect model licenses, data access rules, and encryption keys;
5. never use `rsync --delete` on the only backup.

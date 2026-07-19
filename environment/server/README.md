# Server environment reconstruction

This directory now contains real Linux exports imported from the old server:

```text
rag-cbwdm-retrieval.yml
rag-cbwdm-retrieval-pip-freeze.txt
rag-cbwdm-baselines.yml
rag-cbwdm-baselines-pip-freeze.txt
java-version.txt
nvidia-smi.txt
git-head.txt
```

The two `*.from-history.yml.template` files remain documentation-only
placeholders. They are not authoritative package locks and
`20_bootstrap_new_server.sh` explicitly excludes every `*.template*` file.
Never replace the real Linux exports with files from the Windows development
environment.

The bootstrap script discovers a non-template YAML by its top-level `name:`
value instead of assuming a filename. It copies the selected YAML to a
temporary directory and removes a top-level `prefix:` before calling
`conda env create -n ...`. The pip-freeze files are used only as a
post-installation version audit; they are not treated as conda environment
specifications.

Before stopping or releasing the source server, run:

```bash
cd /root/rag-cbwdm
export EXP_ROOT=/root/experiments/rag_cbwdm
export RUN_NAME=fever2_formal_pilot_5000_500_seed13
bash scripts/18_capture_server_resume_state.sh
```

The snapshot records, for each Linux conda environment:

- `conda-no-builds.yml`: portable audit export;
- `conda-from-history.yml`: primary-dependency reconstruction history;
- `pip-freeze.txt`: exact Python package audit;
- Python and framework versions;
- Java/Pyserini details for retrieval;
- Torch/Transformers/CUDA details for baselines.

The same capture command also publishes a small, credential-sanitized recovery
set under `environment/server/captured/`:

```text
retrieval.conda.no-builds.yml
retrieval.pip-freeze.txt
baselines.conda.no-builds.yml
baselines.pip-freeze.txt
system_info.txt
cuda_torch_info.json
java_pyserini_info.json
model_inventory.json
data_inventory.json
git_head.txt
```

These files must be generated on the real Linux source server. The repository
does not contain fabricated Windows substitutes. Before committing them, scan
for credentials/private keys, inspect every URL/channel, and confirm that no
file exceeds the intended small-metadata scope. Model weights, FEVER records,
experiment artifacts, caches, and large logs must never be copied into this
directory.

Use the captured `conda-from-history.yml` as the starting point for reconstructing
major dependencies. Use `conda-no-builds.yml` and `pip-freeze.txt` to compare the
result with the source server. A blind `pip install -r pip-freeze.txt` is not
guaranteed to reproduce CUDA wheels or conda-provided native libraries.

For CUDA PyTorch:

1. inspect the captured NVIDIA driver, Torch version, `torch.version.cuda`, and
   conda/pip provenance;
2. choose the official PyTorch installation channel/index compatible with that
   driver;
3. install the same Torch family before the remaining baseline packages;
4. require `torch.cuda.is_available()` and the expected GPU name to pass;
5. never infer a Linux CUDA build from the Windows `.venv`.

For retrieval, keep Java and Pyserini confined to `rag-cbwdm-retrieval`. For
posterior/training/evaluation, keep CUDA Torch and Transformers in
`rag-cbwdm-baselines`; that environment intentionally does not require
Pyserini.

The two `.template` files only document the intended separation and historical
replacement fields. Do not pass them to `conda env create` while a real Linux
export is available. Keep the imported exports in a secret-free repository or
operations archive; never invent their contents locally.

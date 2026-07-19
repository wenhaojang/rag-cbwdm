# FEVER Calibration Grid Implementation Report

## Status

The 5000/500 pilot now has an executable validation-only grid between shared posterior generation and `calibrate_methods`. No grid, model, retrieval, posterior, or generator evaluation was run locally.

## Entry points

- Runner stage: `run_calibration_grid`
- Executor: `scripts/15a_run_fever_calibration_grid.py`
- Planner/executor module: `src/calibration/grid.py`
- Downstream selector: `scripts/15_calibrate_fever_methods.py`

## YAML-derived grid

The sole grid source is `configs/fever2_server_pilot_5000_500.yaml`.

| Method | Teachers | Training | Selection/evaluation |
|---|---:|---:|---:|
| InfoGain-FEVER | 3 | 6 | 36 |
| RAG-CBWDM | 3 | 18 | 72 |
| Total | 6 | 24 | 108 |

No Python fallback grid is defined.

## Parameter dependencies

| Method | Rebuild teacher/checkpoint | Retrain only | Selection only |
|---|---|---|---|
| InfoGain-FEVER | teacher quantiles | beta | filter threshold, min_docs, top_m |
| RAG-CBWDM | stop threshold and top_m | beta, gamma, b_plus, b_minus | score threshold, min_docs |

RAG-CBWDM `top_m` also controls validation selection, so it is represented in both the training dependency and selection contract. Margins are training-loss parameters, not teacher parameters.

## Shared artifacts

Every candidate reuses the same:

- train-core and validation split manifests;
- train-core and validation Lucene retrieval;
- train-core and validation generator posteriors;
- fixed generator, prompt, verbalizer, top-N, and evidence-budget contracts.

The grid contains no retrieval or posterior command.

## Fingerprints and reuse

Teacher, training, selection, evaluation, and final candidate fingerprints are separate. Canonical sorted parameter JSON is stored for every candidate. Same config, input hashes, and Git HEAD produce the same fingerprints.

Changing only a selection threshold does not rebuild a teacher or checkpoint. Completed reuse requires an exact stage fingerprint plus unchanged output checksums.

## Candidate schema and outputs

The executor publishes:

```text
artifacts/formal/calibration_candidates.json
artifacts/formal/calibration_candidates.csv
artifacts/formal/calibration_grid_manifest.json
artifacts/formal/calibration_grid_report.md
```

Each candidate records method, validation role, candidate/training/selection fingerprints, canonical parameters, metrics, status/reason, artifact manifests, split/retrieval/posterior SHAs, generator contract, and Git HEAD.

Failed candidates preserve:

```json
{
  "accuracy": null,
  "macro_f1": null,
  "avg_num_docs": null,
  "avg_evidence_chars": null
}
```

No failed metric is converted to zero.

## Resume and resource protection

Supported controls:

- `--methods`
- `--candidate-limit`
- `--candidate-fingerprint`
- `--max-training-candidates`
- `--skip-completed`
- `--fail-fast`
- `--continue-on-error`
- `--resume`
- `--dry-run`

Execution is sequential. CUDA cache cleanup runs between GPU candidates. Per-candidate logs record commands and elapsed time. Candidate directories and manifests isolate failures.

## Dry-run

```bash
python scripts/run_fever_cbwdm.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-name fever2_formal_pilot_5000_500_seed13 \
  --stages run_calibration_grid \
  --generator-model /models/Qwen2.5-1.5B-Instruct \
  --selector-model /models/ms-marco-MiniLM-L-6-v2 \
  --infogain-model /models/ms-marco-MiniLM-L-6-v2 \
  --dry-run
```

This prints the plan and does not load a model.

## Server execution

```bash
python scripts/run_fever_cbwdm.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-name fever2_formal_pilot_5000_500_seed13 \
  --stages run_calibration_grid \
  --generator-model /models/Qwen2.5-1.5B-Instruct \
  --selector-model /models/ms-marco-MiniLM-L-6-v2 \
  --infogain-model /models/ms-marco-MiniLM-L-6-v2 \
  --selector-device cuda \
  --infogain-device cuda \
  --skip-completed \
  --continue-on-error \
  --resume

python scripts/run_fever_cbwdm.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-name fever2_formal_pilot_5000_500_seed13 \
  --stages calibrate_methods \
  --resume
```

## Test coverage

Regression tests cover deterministic YAML expansion, stage dependency classification, teacher/checkpoint reuse, threshold-only selection, shared retrieval/posterior use, held-out rejection, null failed metrics, candidate schema, resume fingerprints, no-model dry-run, Oracle exclusion, deterministic tie-breaking, runner stage presence, and direct `calibrate_methods` consumption.

Final local results: `53 passed` under pytest and `38 tests ... OK` under unittest. Compileall, shell syntax validation, and `git diff --check` also passed.

## Remaining risks

- Full execution is intentionally not tested locally because it requires 24 training candidates and 108 validation generator evaluations.
- Fixed validation baselines must exist for the final CBWDM diagnostic gate.
- Model revisions remain to be frozen on the server.
- Diagnostics must follow the selected winner fingerprint from calibration output.

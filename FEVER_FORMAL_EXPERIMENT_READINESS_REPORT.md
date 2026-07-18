# FEVER Formal Experiment Readiness Report

## 1. Executive status

The offline engineering work required to make a formal FEVER experiment auditable is implemented. The repository now has deterministic formal splits, validation-only parameter selection, immutable configuration freezing, CBWDM pilot diagnostics, and an aggregate readiness gate.

Current status is deliberately **blocked**, not `ready`: no 5000/500 server pilot, calibration artifact, frozen model fingerprint set, or CBWDM diagnostic artifact has been produced in this workspace. No model was run and no existing experiment artifact was modified.

Status levels:

- Engineering ready: **yes**, subject to the test evidence in section 12.
- Experiment protocol ready: **implemented but not yet instantiated on the server**.
- Scientific conclusion ready: **no**. A readiness check is not a held-out result.

## 2. Current validated baseline state

The preceding FEVER baseline work records a comparable six-method suite:

1. `no_evidence`
2. `naive_topm`
3. `bge`
4. `infogain_fever`
5. `rag_cbwdm`
6. `cbwdm_oracle`

The summary path is manifest-driven and the Oracle is non-deployable and diagnostic-only. The 100-query smoke result remains a scientific warning: RAG-CBWDM and Oracle underperformed simpler baselines. This report does not reinterpret that result as sampling noise.

## 3. Split protocol

`scripts/01a_build_fever_formal_splits.py` implements:

- official train → `train_core` plus `validation`;
- official dev → `held_out_test`;
- `NOT ENOUGH INFO` filtering before partitioning;
- SUPPORTS/REFUTES stratification;
- a SHA-256 ordering keyed by `(seed, original FEVER id)`;
- output order independent of source row order;
- stable IDs derived from original FEVER IDs;
- duplicate original-ID and normalized-claim rejection;
- all limits applied only after the full legal split exists.

Pilot limits are 5000 train-core rows and 500 validation rows. Held-out test is not part of pilot parameter selection. Formal mode removes these limits while retaining the same frozen validation definition.

## 4. Split manifest schema

The split builder emits:

- `train_core.jsonl`
- `validation.jsonl`
- `held_out_test.jsonl`
- `fever2_formal_splits.manifest.json`

The completed manifest records source paths and SHA-256 values, filtering and partition contracts, seed, pre-limit and post-limit row counts, label counts, per-role file SHA, ID-set SHA, six overlap checks, Git state, and creation time. The manifest is published last. Resume verifies source SHA, all split parameters, output SHA, row counts, ID-set SHA, role labels, and overlap results.

Schema version: `rag_cbwdm_fever_split_manifest.v1`.

## 5. Leakage prevention

The implementation fails closed on:

- duplicate IDs within either official source;
- an original ID present in both official train and official dev;
- duplicate normalized claims within a source;
- normalized-claim overlap between official train and official dev;
- any ID or normalized-claim overlap after splitting;
- a changed source or split parameter on resume;
- calibration records whose role is not exactly `validation`;
- calibration paths or records that reference `held_out_test`;
- InfoGain teacher construction from `test` or `held_out_test`;
- non-diagnostic CBWDM teacher construction on held-out test.

CBWDM held-out teacher construction is allowed only with the explicit `--diagnostic-oracle` flag. Its resulting Oracle remains diagnostic-only and must never select deployable parameters.

## 6. Validation-only calibration

`scripts/15_calibrate_fever_methods.py` consumes precomputed candidate metrics and verifies every record against:

- the completed split manifest;
- the exact validation artifact SHA;
- the parameter grid declared in config;
- the canonical method name;
- `split=validation`;
- absence of held-out references.

The InfoGain grid contains only parameters consumed by current code: negative/positive teacher quantiles, inference filter threshold, `min_docs`, `top_m`, and training `beta`.

The RAG-CBWDM grid contains only consumed parameters: teacher stop threshold, selector score threshold, `min_docs`, `top_m`, `beta`, `gamma`, `b_plus`, and `b_minus`. Gain normalization is explicitly reported as not implemented and therefore is not exposed as a fake tunable parameter.

Selection is deterministic: maximize validation `macro_f1` or accuracy, then minimize average document count, then average evidence characters, then canonical parameter JSON. A missing metric remains `null` with a reason and is never replaced with zero.

Outputs:

- `calibration_results.json`
- `calibration_results.csv`
- `calibration_report.md`
- `frozen_parameters.yaml`
- `calibration.manifest.json`

## 7. Frozen formal config

`scripts/16_freeze_fever_formal_config.py` requires:

- base formal config;
- completed split and calibration manifests;
- corpus and Lucene index artifacts;
- local paths and immutable revisions for generator, tokenizer, BGE, InfoGain, and RAG-CBWDM models.

Files and directory trees are SHA-256 fingerprinted. The frozen YAML records all split SHAs, calibration SHA, retrieval contract, model paths/revisions/weights, corpus and index identity, prompt/verbalizer hashes, context and truncation settings, calibrated parameters, batch size, dtype, seed, Git state, and Python/torch/transformers environment.

Outputs are fingerprint-named:

- `configs/generated/fever2_formal_frozen_<fingerprint>.yaml`
- `configs/generated/fever2_formal_frozen_<fingerprint>.manifest.json`
- `configs/generated/fever2_formal_frozen_<fingerprint>.diff.md`

Formal runner use requires the sidecar manifest. It re-hashes frozen artifacts and rejects CLI overrides for models, limits, scientific batch settings, or seed.

## 8. 5000/500 pilot

`configs/fever2_server_pilot_5000_500.yaml` fixes:

- train-core limit: 5000;
- validation limit: 500;
- held-out test: excluded from the pilot;
- full FEVER corpus;
- Lucene top-N: 20;
- selected-document budget: 4;
- Qwen2.5-1.5B-Instruct generator;
- BGE reranker large;
- MiniLM backbones for InfoGain and RAG-CBWDM;
- seed 13;
- explicit small calibration grids.

The existing runner understands `prepare_formal_splits`, the three role-specific retrieval stages, the three role-specific posterior stages, and `calibrate_methods`. The `pilot` alias expands only through validation posterior construction and intentionally excludes test retrieval and posterior computation.

## 9. CBWDM diagnostic gate

`scripts/16a_diagnose_cbwdm_pilot.py` emits:

- `cbwdm_pilot_diagnostics.json`
- `cbwdm_pilot_diagnostics.md`

It reports aligned per-example flips, RAG-CBWDM/Oracle selection Jaccard, teacher and selector document-count distributions, marginal-gain distributions and signs, stop reasons and rate, candidate and selected evidence recall, per-label accuracy/F1, query-only and selected-evidence confidence, and Oracle-versus-Naive risk.

The new corpus/index contract stores FEVER page and sentence metadata so validation gold recall can be computed. An older Lucene index without this metadata must not be silently reused; its contract fingerprint is incompatible and it must be rebuilt once. Held-out retrieval artifacts omit gold evidence keys.

The gate blocks on schema mismatch, Oracle/teacher disagreement, all-zero or negative gains, budget violation, missing/abnormal evidence recall, label/verbalizer mismatch, insufficient pilot scale, or an Oracle accuracy more than the configured tolerance below Naive.

## 10. Formal readiness checker

`scripts/17_check_fever_formal_readiness.py` emits `formal_readiness.json` and `formal_readiness.md`.

It returns `ready` only when all P0 checks pass:

- completed and checksum-valid split manifest;
- zero ID and normalized-claim overlap;
- completed validation-only calibration;
- no held-out reference in calibration;
- frozen config and unchanged model/corpus/index hashes;
- frozen prompt and verbalizer hashes;
- comparable fairness audit for all six canonical methods;
- complete baseline summary with diagnostic Oracle metadata;
- passed CBWDM pilot gate with no scientific-risk flag;
- all required test commands recorded as passed;
- Git commit provenance present.

Any exception or malformed input becomes a named blocker rather than crashing the checker.

## 11. Changed files

Core additions:

- `src/formal_splits.py`
- `src/calibration/__init__.py`
- `src/calibration/fever.py`
- `src/formal_provenance.py`
- `src/formal_config.py`
- `src/cbwdm_diagnostics.py`
- `src/formal_readiness.py`

Entry points and configuration:

- `scripts/01a_build_fever_formal_splits.py`
- `scripts/15_calibrate_fever_methods.py`
- `scripts/16_freeze_fever_formal_config.py`
- `scripts/16a_diagnose_cbwdm_pilot.py`
- `scripts/17_check_fever_formal_readiness.py`
- `configs/fever2_server_pilot_5000_500.yaml`
- `configs/fever2_server_formal.yaml`

Integration changes:

- `scripts/run_fever_cbwdm.py`
- `scripts/02_retrieve_bm25.py`
- `scripts/03_compute_label_posteriors.py`
- `scripts/04_build_cbwdm_teacher.py`
- `scripts/07_eval_rag_classification.py`
- `scripts/12a_build_infogain_teacher.py`
- `src/retrieval/memory_bm25.py`
- `src/retrieval/pyserini_bm25.py`
- `src/run_manifest.py`
- `tests/test_fever_formal_protocol.py`

## 12. Tests

The formal-protocol regression suite covers:

- deterministic and stratified splitting;
- filtering before limits;
- source-order independence;
- duplicate ID/claim and cross-source leakage rejection;
- manifest checksum and changed-source resume refusal;
- validation-only calibration and held-out rejection;
- missing metric preservation;
- deterministic grids and cost tie-breaking;
- split/calibration/model SHA freezing;
- changed-model rejection;
- critical formal CLI override rejection;
- gain signs, stopping, document selection, prediction flips, gold recall, and label mapping;
- blocked readiness aggregation;
- explicit runner roles and exclusion of test stages from the pilot alias.

Required commands are:

```bash
python -m compileall -q src scripts tests
python -m pytest -q
python -m unittest discover
bash -n scripts/run_fever_cbwdm.sh
git diff --check
```

On this Windows workspace, the same Python commands run through `.venv/Scripts/python.exe`; Git Bash is used for `bash -n`.

## 13. Server commands

Do not execute these until local model revisions and paths are filled.

Retrieval environment:

```bash
conda activate rag-cbwdm-retrieval
cd /root/rag-cbwdm
CONFIG=configs/fever2_server_pilot_5000_500.yaml
RUN_NAME=fever2_formal_pilot_5000_500_seed13

python scripts/run_fever_cbwdm.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --stages prepare_formal_splits,corpus,index,retrieve_train_core,retrieve_validation \
  --resume
```

This is resumable. It uses the full corpus. A Lucene index built with the new metadata contract is reusable; an older contract must be rebuilt under a new compatible index fingerprint.

Baseline environment:

```bash
conda activate rag-cbwdm-baselines
cd /root/rag-cbwdm
CONFIG=configs/fever2_server_pilot_5000_500.yaml
RUN_NAME=fever2_formal_pilot_5000_500_seed13

python scripts/run_fever_cbwdm.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --stages posterior_train_core,posterior_validation \
  --resume
```

This computes 5000 train-core and 500 validation posterior rows. Resume reuses completed posterior manifests with matching model, prompt, input, and output hashes.

After candidate training and validation evaluation have produced `artifacts/formal/calibration_candidates.json`:

```bash
python scripts/run_fever_cbwdm.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --stages calibrate_methods \
  --resume
```

Diagnostic command:

```bash
RUN=/root/experiments/rag_cbwdm/$RUN_NAME
python scripts/16a_diagnose_cbwdm_pilot.py \
  --config "$CONFIG" \
  --no-evidence "$RUN/artifacts/eval/no_evidence_predictions.jsonl" \
  --naive "$RUN/artifacts/eval/naive_top4_predictions.jsonl" \
  --cbwdm "$RUN/artifacts/eval/rag_cbwdm_predictions.jsonl" \
  --oracle "$RUN/artifacts/eval/cbwdm_oracle_top4_predictions.jsonl" \
  --naive-selection "$RUN/artifacts/selections/naive_top4.jsonl" \
  --cbwdm-selection "$RUN/artifacts/fever2_validation_cross_encoder_selection.jsonl" \
  --oracle-selection "$RUN/artifacts/selections/cbwdm_oracle_top4.jsonl" \
  --teacher "$RUN/artifacts/formal/fever2_validation_teacher.jsonl" \
  --retrieval "$RUN/artifacts/formal/fever2_validation_bm25_top20.jsonl" \
  --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" \
  --output-dir "$RUN/artifacts/formal/diagnostics"
```

Freeze command template:

```bash
python scripts/16_freeze_fever_formal_config.py \
  --base-config configs/fever2_server_formal.yaml \
  --split-manifest outputs/formal_splits/fever2_seed13/fever2_formal_splits.manifest.json \
  --calibration-manifest "$RUN/artifacts/formal/calibration/calibration.manifest.json" \
  --corpus <corpus_manifest_from_run_manifest> \
  --index <index_manifest_from_run_manifest> \
  --model generator=/models/Qwen2.5-1.5B-Instruct \
  --revision generator=<immutable_revision> \
  --model tokenizer=/models/Qwen2.5-1.5B-Instruct \
  --revision tokenizer=<immutable_revision> \
  --model bge=/models/bge-reranker-large \
  --revision bge=<immutable_revision> \
  --model infogain="$RUN/artifacts/baselines/infogain_reranker/checkpoint" \
  --revision infogain=<checkpoint_revision> \
  --model rag_cbwdm="$RUN/artifacts/cross_encoder/checkpoint" \
  --revision rag_cbwdm=<checkpoint_revision>
```

The concrete server corpus/index manifest paths must come from the run manifest; do not guess them.

Readiness command:

```bash
python scripts/17_check_fever_formal_readiness.py \
  --split-manifest <split_manifest> \
  --calibration-manifest <calibration_manifest> \
  --frozen-manifest <frozen_manifest> \
  --fairness-audit <validation_fairness_audit> \
  --baseline-summary <validation_baseline_summary> \
  --cbwdm-diagnostics <cbwdm_pilot_diagnostics.json> \
  --tests-status <tests_status.json> \
  --output-dir "$RUN/artifacts/formal/readiness"
```

Only after this returns `ready` may held-out inference start with the generated frozen config and its sidecar:

```bash
python scripts/run_fever_cbwdm.py \
  --config <fever2_formal_frozen_fingerprint.yaml> \
  --frozen-manifest <fever2_formal_frozen_fingerprint.manifest.json> \
  --run-name fever2_formal_qwen15b_seed13 \
  --stages retrieve_test,posterior_test \
  --resume
```

Formal stability runs use seeds 13, 21, and 42 for InfoGain and RAG-CBWDM. Deterministic baselines run once. The Qwen2.5-7B robustness check is limited to Naive, BGE, InfoGain, and RAG-CBWDM after the main formal run.

## 14. P0 blockers

At report generation time:

1. The server split manifest has not been materialized and audited.
2. Immutable local model revisions and weight hashes are not filled in the base configs.
3. The 5000/500 candidate grid has not been executed.
4. Validation calibration and a frozen formal config do not yet exist.
5. CBWDM gain, stopping, recall, and Oracle-versus-Naive gates have not run on 500 validation examples.
6. The server test-status evidence file has not been recorded.
7. `RAG_CBWDM_SERVER_E2E100_RUNBOOK.md`, named by the task, is absent from both the worktree and current HEAD; this report does not fabricate or restore it.

## 15. P1 improvements

- Add a resource-aware server executor that expands every declared calibration grid candidate and writes the canonical candidate-metrics file automatically.
- Add confidence intervals and paired significance tests to pilot diagnostics.
- Add FEVER evidence-set recall in addition to sentence-level recall.
- Package the formal stage commands into scheduler-specific job templates after the actual server layout is confirmed.
- Add a small checked-in tests-status schema helper so CI can publish readiness evidence directly.

## 16. Exact conditions for `ready`

`status=ready` is permitted only if the readiness checker receives checksum-valid artifacts and every named P0 check passes. A comparable smoke baseline audit is not sufficient. A frozen config without validation calibration is not sufficient. A passing engineering test suite with a failed CBWDM scientific diagnostic is not sufficient.

No CLI flag can waive a P0 check. `--skip-artifact-rehash` exists only for checker unit/debug use and must never appear in a formal server command.

## 17. Remaining scientific risks

- The current smoke result suggests that the learned selector and even the gold-dependent Oracle may be selecting evidence that hurts the fixed generator.
- A larger pilot can reduce variance but cannot by itself explain a systematic Oracle deficit.
- Gold evidence recall may expose retrieval limitations, but high retrieval recall does not guarantee evidence usefulness under the generator prompt.
- Validation calibration can overfit a small grid; held-out test must remain untouched until freezing.
- Three seeds characterize training instability but do not make deterministic baselines stochastic.
- A `ready` verdict authorizes protocol execution; it does not imply RAG-CBWDM will outperform Naive or BGE.

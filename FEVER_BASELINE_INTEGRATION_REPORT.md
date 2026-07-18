# FEVER Baseline Integration Report

## 1. Executive status

This change integrates a common FEVER comparison pipeline for no evidence,
Naive BM25 top-M, a Hugging Face BGE-compatible reranker, the explicitly named
InfoGain-FEVER adaptation, the existing state-aware RAG-CBWDM selector, and the
gold-label CBWDM oracle diagnostic.

| Method/component | Status |
|---|---|
| No evidence | READY |
| Naive BM25 | READY |
| BGE | NEEDS SERVER MODEL TEST |
| InfoGain-FEVER | NEEDS SERVER TRAINING TEST |
| RAG-CBWDM | READY |
| CBWDM Oracle | DIAGNOSTIC ONLY |
| Unified summary/fairness audit | READY |

No model or dataset was downloaded, no heavy experiment was run, and no
push/commit/merge/tag was performed. The existing CBWDM posterior-shift,
teacher, selector, and evaluation mathematics were not changed.

Git safety baseline:

```text
branch: fix/disk-bm25-and-limit-semantics
HEAD: a433c08cce38104517324a65eb099b239f43be33
```

Pre-existing tracked deletions and untracked reports/task files were preserved.
They are not part of this implementation.

## 2. Changed files

Implementation:

```text
src/baselines/__init__.py
src/baselines/common.py
src/baselines/bge_reranker.py
src/baselines/infogain_fever.py
src/baselines/infogain_selector.py
src/selection_schema.py
src/metrics.py
scripts/07_eval_rag_classification.py
scripts/08_select_naive_topm.py
scripts/09_select_cbwdm_oracle_from_teacher.py
scripts/11_select_with_cross_encoder.py
scripts/12_select_bge_reranker.py
scripts/12a_build_infogain_teacher.py
scripts/12b_train_infogain_reranker.py
scripts/12c_select_infogain_reranker.py
scripts/13_summarize_fever_baselines.py
scripts/14_audit_fever_baselines.py
scripts/run_fever_cbwdm.py
configs/fever2_server_smoke.yaml
configs/fever2_server_pilot.yaml
configs/fever2_server_formal.yaml
requirements-baselines.txt
tests/test_fever_baselines.py
FEVER_BASELINE_INTEGRATION_REPORT.md
```

## 3. Existing baseline reuse

- No evidence continues to use `scripts/07_eval_rag_classification.py`; it now
  receives prepared query rows directly, produces zero-document metrics, and
  publishes an evaluation manifest.
- Naive and Oracle reuse and harden scripts 08 and 09.
- RAG-CBWDM still uses the existing state-aware cross-encoder implementation.
  Only its method name, metadata, atomic selection artifact, and manifest were
  unified.
- Every evidence method calls the same evaluator and therefore shares prompt,
  verbalizers, generator configuration, evidence ordering, and truncation.

## 4. BGE implementation

`src/baselines/bge_reranker.py` uses delayed imports of:

```text
transformers.AutoTokenizer
transformers.AutoModelForSequenceClassification
torch
```

The output scalar is interpreted as follows:

- one-logit heads: the single raw logit;
- multi-class heads: the last logit;
- direction: higher means more relevant;
- `--normalize-score`: optional sigmoid.

The production script supports model/revision/device/dtype/batch/max-length,
top-M, threshold, min-doc fallback, normalization, limit, local-only loading,
resume, and overwrite. Candidate scores are atomically cached independently of
top-M; the cache fingerprint includes the retrieval SHA, model/revision,
dtype, max length, normalization flag, input-template version, and limit.
Every cache row keys scores by query ID and document ID. An incompatible cache
causes a clear error under `--resume`; it is not silently overwritten.

OOM errors advise reducing batch size or max length. No automatic fallback
changes experiment settings.

The FlagEmbedding API is not required. This avoids forcing an additional
package into No-evidence/Naive/Oracle runs. A real local BGE checkpoint must
still be tested on the Linux GPU server.

## 5. InfoGain official-code audit

The read-only `_external/InfoGain-RAG/` tree was inspected and not modified.
Its Apache-2.0 license, shared RoBERTa/two-head idea, RankNet-style objective,
and pointwise inference were used only as design references. No upstream code
or hard-coded path was imported or copied.

Audit conclusion:

```text
baseline integration: GO WITH ADAPTER
official end-to-end reproduction: NOT READY
InfoGain-RAG official reproduction: NO
InfoGain-RAG FEVER adaptation: YES
```

The upstream tree does not provide a complete reproducible FEVER path for
sequence confidence, early-token weighting, final scalar DIG construction,
`b1/b2` data generation, the paper margin, or the FM2 adapter.

## 6. InfoGain-FEVER mathematical definition

The primary teacher is probability difference:

```text
gold_index = labels.index(gold_label)
DIG_prob(q, d; y) = eta_doc[gold_index] - eta0[gold_index]
```

Teacher construction accepts train/validation/dev posterior rows only, checks
label order, probability lengths/sums, non-negativity, and finiteness, and
records posterior SHA plus generator/prompt/verbalizer provenance.

Threshold supervision is:

```text
positive: DIG >= b_pos
negative: DIG <= b_neg
neutral:  b_neg < DIG < b_pos
```

Explicit and train/validation quantile modes are supported. Empty positive or
negative classes fail clearly. Threshold source, quantiles, values, and label
distribution are saved.

The reranker is pointwise:

```text
Claim + Candidate evidence
```

It never includes already-selected evidence. A shared HF encoder feeds a scalar
rank head and a two-logit filter head. Within-query ranking uses pairwise
logistic/RankNet; filter CE uses positive/negative examples and ignores
neutral examples:

```text
L = beta * L_rank + (1 - beta) * L_filter
```

Inference ranks by rank score, applies the independently calibrated filter
probability threshold, fills `min_docs` if necessary, and caps at top-M.
Scoring works on rows without a gold field and never sends gold to the model.

## 7. Paper / official code / current adapter

| Item | Paper description | Read-only official tree | Current adapter |
|---|---|---|---|
| Task | Open-QA information gain | Open-QA-oriented scripts | FEVER label classification |
| Teacher confidence | Answer-sequence confidence | Partial/hard-coded pipeline | Saved label posteriors |
| DIG | Answer-confidence change | Missing complete scalar provenance | `eta_doc[y] - eta0[y]` |
| Encoder | Large RoBERTa reranker | Shared backbone/two heads | Configurable HF AutoModel |
| State | Pointwise | Pointwise | Pointwise, `state_aware=false` |
| Rank objective | Ranking objective | RankNet-style code | Same-query pairwise logistic |
| Filter objective | Relevance filtering | Binary head | Positive/negative CE; neutral ignored |
| Thresholds | `b1/b2` not fully reproducible | Generation values incomplete | Explicit or train/validation quantiles |
| Fidelity claim | Paper method | Incomplete runnable reproduction | `official_fidelity=adapted` |

## 8. Unified schema and manifests

All evidence selection rows now use:

```text
schema_version = rag_cbwdm_selection.v2
```

They preserve `source_rank` and `source_score`, retain legacy `rank` and
`retrieval_score`, align `selected_doc_ids` with `selected_docs`, record
`num_docs`, `max_docs`, method-specific scores, steps, and metadata.

Selection output is written as:

```text
output.jsonl.partial -> flush -> fsync -> os.replace(output.jsonl)
```

Only after row validation and output publication is a completed manifest
atomically published. Its contract includes method, all input SHA-256 values,
model/revision/checkpoint inputs, top-M, min-docs, thresholds, limit, output
SHA, row count, timestamps, and Git state.

Evaluation now atomically publishes predictions and metrics and then a
`rag_cbwdm_evaluation_manifest.v1` containing selection SHA and generator,
revision, prompt, verbalizer, context, and output checksums. No-evidence
manifests explicitly record method `no_evidence` and `num_docs=0`.

## 9. Runner stages

New stages/aliases:

```text
select_naive_topm -> naive_topm
score_bge
select_bge
build_infogain_teacher
train_infogain
select_infogain
select_oracle -> oracle_diagnostic
eval_no_evidence -> no_evidence
eval_naive_topm
eval_bge
eval_infogain
eval_cbwdm
eval_oracle
fairness_audit
summarize_baselines
baselines
```

Artifacts are isolated under:

```text
artifacts/selections/
artifacts/eval/
artifacts/baselines/
```

Old run manifests are extended with missing stage entries. If an existing
e2e100 config differs only by the newly added `baselines` section, the runner
allows the compatible extension. Baseline resume stages deliberately invoke
their lightweight stage-specific validator instead of trusting only prior
runner status. Valid artifacts return before model initialization; mismatched
model/checkpoint/input contracts fail or require explicit overwrite/new paths.

Retrieval and GPU work can remain in separate Conda invocations because all
artifacts are filesystem contracts.

## 10. Config

Smoke, pilot, and formal configs now define:

```text
baselines.common
baselines.naive
baselines.bge
baselines.infogain_fever
baselines.oracle
```

Common top-M equals `cbwdm.top_m`; the runner rejects an unfair mismatch.
Naive/BGE main-table defaults select the full shared budget. InfoGain defaults
to min-docs 2 and a train-quantile teacher. Model paths remain CLI-overridable.
Formal local execution must supply frozen local paths and revisions before
results are reportable.

## 11. Dependencies

`requirements-baselines.txt` intentionally adds only an API compatibility
constraint:

```text
transformers>=4.45,<5
```

It does not list or upgrade torch. Use the already validated GPU torch build
from the main environment. The implemented BGE backend does not require
FlagEmbedding, pandas, or scikit-learn. Optional model imports are delayed, so
help, No evidence, Naive, and Oracle do not require BGE weights or imports.

Server prerequisites:

```text
Python 3.12
existing compatible torch 2.x GPU build
transformers >=4.45,<5
PyYAML
local generator checkpoint
local BGE sequence-classification reranker checkpoint
local InfoGain backbone checkpoint
```

## 12. Tests

Added regression coverage for:

- selection v2, source rank/score, atomic publication, manifest checksum, and
  no-iteration resume;
- Naive rank order and too-few-candidate behavior;
- BGE pair ordering, score direction, top-M, threshold, min-doc fallback, and
  score-cache reuse;
- FEVER DIG gold-index order, `0.7 - 0.4 = 0.3`, invalid probabilities,
  threshold distribution, same-query grouping, finite RankNet/filter loss,
  neutral CE exclusion, and beta;
- InfoGain gold-free pointwise inference, rank order, filtering, fallback, and
  `state_aware=false`;
- Oracle diagnostic flags and missing teacher;
- macro-F1/per-class precision/recall/F1 and zero-document metrics;
- summary missing values and Oracle exclusion from deployable best;
- runner dry-run model overrides.

Local results:

```text
python -m compileall -q src scripts tests: PASS
bash -n scripts/run_fever_cbwdm.sh: PASS
git diff --check: PASS
python -m unittest tests.test_cbwdm_score tests.test_disk_bm25_and_limits -q:
  PASS, 13 tests
python -m pytest -q:
  NOT RUNNABLE LOCALLY, pytest is absent
python -m unittest discover:
  13 executable tests passed; discovery then failed because local .venv lacks
  pytest and torch (the existing torch tests also cannot import)
```

This Windows workspace therefore does not prove real BGE inference,
InfoGain training, CUDA behavior, or the full dependency-enabled suite.

## 13. Local unverified items

- Real BGE output shape/score semantics for the selected server checkpoint.
- Local-only loading and peak VRAM for BGE large.
- Real InfoGain MiniLM/RoBERTa training, save/reload, and CUDA inference.
- Qwen evaluation of every baseline.
- End-to-end fairness audit on the real e2e100 candidate pool.
- Full pytest/unittest suite in the server GPU environment.

## 14. Server smoke commands

Common setup in the GPU environment:

```bash
cd /root/rag_cbwdm
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rag-cbwdm
export EXP_ROOT=/root/experiments/rag_cbwdm
export HF_HOME=/root/huggingface
export GENERATOR_MODEL=/root/models/Qwen2.5-1.5B-Instruct
export SELECTOR_MODEL=/root/models/ms-marco-MiniLM-L-6-v2
export BGE_MODEL=/root/models/bge-reranker-large
export INFOGAIN_MODEL=/root/models/ms-marco-MiniLM-L-6-v2
python -m pip install -r requirements-baselines.txt
```

The install must not upgrade/replace torch. Verify `python -m pip check` and the
existing CUDA smoke afterward.

### 14.1 Existing e2e100: Naive, Oracle, no evidence, RAG-CBWDM evaluation

This reuses retrieval/posteriors/teacher/checkpoint and does not recompute
Qwen posteriors. The selector selection is regenerated once to selection v2.

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_pilot.yaml \
  --run-name fever2_full_corpus_e2e100_v02_seed13 \
  --stages select_cross_encoder,naive_topm,oracle_diagnostic,no_evidence,eval_naive_topm,eval_cbwdm,eval_oracle,fairness_audit,summarize_baselines \
  --train-limit 100 --dev-limit 100 \
  --seed 13 --resume \
  --overwrite-stage select_cross_encoder \
  --output-root "$EXP_ROOT" --cache-root "$HF_HOME" \
  --generator-model "$GENERATOR_MODEL" \
  --selector-model "$SELECTOR_MODEL"
```

Environment: `rag-cbwdm`. Local models: Qwen and selector checkpoint/model.
Expected resource: one small-selector inference plus repeated 1.5B generator
evaluation; use one GPU, approximately 8 GB VRAM headroom and 16 GB host RAM,
then record measured peaks. Resume: yes; valid eval artifacts return before
loading Qwen.

### 14.2 BGE 100-query smoke

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_pilot.yaml \
  --run-name fever2_full_corpus_e2e100_v02_seed13 \
  --stages score_bge,select_bge,eval_bge,fairness_audit,summarize_baselines \
  --train-limit 100 --dev-limit 100 \
  --seed 13 --resume \
  --output-root "$EXP_ROOT" --cache-root "$HF_HOME" \
  --generator-model "$GENERATOR_MODEL" \
  --bge-model "$BGE_MODEL"
```

Environment: `rag-cbwdm`. Local models: BGE reranker and Qwen. Expected
resource: one GPU; allow roughly 6-10 GB VRAM and 16 GB host RAM until measured.
Resume: yes; score cache is reused across top-M values and evaluator resume
returns before Qwen initialization.

### 14.3 InfoGain-FEVER 100-train / 100-dev smoke

This reuses existing train/dev posterior and retrieval artifacts; it does not
run the posterior stage.

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_pilot.yaml \
  --run-name fever2_full_corpus_e2e100_v02_seed13 \
  --stages build_infogain_teacher,train_infogain,select_infogain,eval_infogain,fairness_audit,summarize_baselines \
  --train-limit 100 --dev-limit 100 \
  --seed 13 --resume \
  --output-root "$EXP_ROOT" --cache-root "$HF_HOME" \
  --generator-model "$GENERATOR_MODEL" \
  --infogain-model "$INFOGAIN_MODEL"
```

Environment: `rag-cbwdm`. Local models: MiniLM (or chosen frozen backbone) and
Qwen. Expected resource: one GPU, approximately 4-8 GB VRAM and 16 GB host RAM
for the 100-row smoke; measure rather than relying on estimates. Resume: yes;
teacher/checkpoint/selection/eval fingerprints are independently validated.

### 14.4 Unified evaluation/summary only

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_pilot.yaml \
  --run-name fever2_full_corpus_e2e100_v02_seed13 \
  --stages no_evidence,eval_naive_topm,eval_bge,eval_infogain,eval_cbwdm,eval_oracle,fairness_audit,summarize_baselines \
  --train-limit 100 --dev-limit 100 \
  --seed 13 --resume \
  --output-root "$EXP_ROOT" --cache-root "$HF_HOME" \
  --generator-model "$GENERATOR_MODEL"
```

Environment: `rag-cbwdm`. Local model: Qwen. Valid evaluation artifacts are
checksum-resumed; otherwise this can load Qwen once per evaluation process.

Retrieval/index regeneration, if required, remains in `rag-cbwdm-retrieval`;
none of the four commands above invokes retrieval or Pyserini.

## 15. Fairness audit

`scripts/14_audit_fever_baselines.py` writes
`baseline_fairness_audit.json`. It verifies:

- identical retrieval/prepared query ID order;
- every selected document belongs to that query's shared candidate pool;
- identical maximum document budget;
- evaluation selection checksum;
- identical generator model/revision, prompt hash, verbalizer hash, and maximum
  context;
- method state-awareness and teacher-role metadata.

Missing methods are reported as missing and do not fabricate zeros. Any actual
contract mismatch sets the audit and summary to `not_comparable`; the audit
process exits nonzero. InfoGain is recorded as pointwise DIG/state-unaware and
RAG-CBWDM as set-dependent marginal gain/state-aware.

## 16. Formal experiment blockers

1. Freeze exact local generator, BGE, selector, and InfoGain model revisions.
2. Run the dependency-enabled full test suite on Linux.
3. Validate real BGE score direction/output and record package/model versions.
4. Run InfoGain training/reload smoke and inspect positive/negative/neutral
   counts before pilot.
5. Calibrate InfoGain and CBWDM thresholds on train/validation only.
6. Measure VRAM, RAM, wall time, truncation, and score distributions.
7. Produce a fully comparable fairness audit on all six methods.
8. Run multiple formal seeds; do not claim a paper main result from the smoke.

## 17. P0 / P1

P0:

- Execute sections 14.1-14.4 on the real Linux GPU server.
- Run `python -m pytest -q` and `python -m unittest discover` with project
  dependencies installed.
- Freeze revisions and validation-derived thresholds before pilot/formal.
- Require `baseline_fairness_audit.json: status=comparable`.

P1:

- Add a tested FlagEmbedding backend only if it materially improves throughput.
- Add multi-seed mean/std fixtures and optional LaTeX export.
- Add wall-time/VRAM telemetry to manifests.
- Consider a separately named InfoGain-RoBERTa capacity comparison.

## 18. Exact method naming for paper

Use:

```text
No evidence
Naive BM25 top-M
BGE reranker (HF sequence-classification backend)
InfoGain-FEVER (probability-difference adaptation)
RAG-CBWDM (state-aware cross-encoder; Euclidean posterior-shift variant)
CBWDM Oracle† (diagnostic only; uses gold at test)
```

Do not use:

```text
official InfoGain-RAG
exact InfoGain-RAG reproduction
paper-faithful InfoGain implementation
full categorical BW Hessian
deployable CBWDM Oracle
```

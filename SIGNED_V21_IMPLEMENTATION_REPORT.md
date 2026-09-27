# Signed-v2.1 Ordinal Ranking Implementation Report

## 1. Scope

This change adds the independent development method `rag_cbwdm_signed_v21`. It keeps the signed-v1 pretrained scalar cross-encoder and its inference policy, while adding one training-only ordinal loss that separates ordinary low-utility candidates from explicitly harmful candidates. No server training or evaluation was run locally, so signed/base validation counts remain `UNAVAILABLE UNTIL SERVER RUN`.

## 2. Why v2.1 differs from v2

The earlier v2 direction introduced a dual-head model and a different inference design. That expands both the learned representation and serving contract, and can collapse when the new heads do not learn stable, independently calibrated signals. V2.1 instead targets the narrower missing relation in the existing scalar score: an ordinary admissible low-utility document should rank above an explicitly harmful document. It therefore preserves the pretrained scalar classifier and leaves deployment behavior identical to v1.

## 3. Files Added / Modified

Added:

- `src/diagnostics/signed_selector_v21.py`
- `scripts/diagnostics/train_signed_selector_v21.py`
- `scripts/diagnostics/select_signed_selector_v21.py`
- `scripts/diagnostics/analyze_signed_selector_v21.py`
- `tests/test_signed_selector_v21.py`
- `SIGNED_V21_IMPLEMENTATION_REPORT.md`

Modified existing files: none. The pre-existing dirty allowlist was not edited.

## 4. Preserved v1 Model Architecture

Training instantiates the existing `CrossEncoderSelector`, which wraps `AutoModelForSequenceClassification` with `num_labels=1`. Its pretrained scalar classification head is retained and optimized together with the encoder. No model head, raw gate, hard gate, or step-balancing module was added. Checkpoints use the existing `CrossEncoderSelector.save_checkpoint` and `CrossEncoderSelector.load_checkpoint` contract.

## 5. Existing v1 Supervision

V2.1 reuses `build_signed_training_groups` and passes each group's existing effective gains to `cbwdm_multitask_loss`. An `explicit_harmful_negative` already has effective gain zero, contributes as a negative example to the base BCE term, and is eligible for the base positive-versus-negative ranking term. Those semantics are unchanged.

## 6. Missing Ordinal Relation

The base objective does not explicitly distinguish an ordinary low-utility admissible candidate from an explicitly harmful candidate when both are on the negative side of the base target. V2.1 adds exactly that within-group ordering:

- `A_low = {j | supervision_class_j in {negative, neutral}}`
- `H = {j | supervision_class_j = explicit_harmful_negative}`
- desired order: `s_a > s_h` for every `(a, h) in A_low x H`

Positive candidates do not enter this auxiliary relation.

## 7. Mathematical Objective

For scalar selector logits `s`, fixed `gamma_signed = 1.0`, and fixed `lambda_signed = 0.25`:

```text
L_signed = mean_{h in H, a in A_low} softplus(gamma_signed * (s_h - s_a))
L_base   = beta * L_CE + (1 - beta) * L_rank_base
         = cbwdm_multitask_loss(scores, effective_gains, ...)
L_total  = L_base + lambda_signed * L_signed
```

When either set is empty, `L_signed = scores.sum() * 0.0`, which is a differentiable zero on the correct device and dtype. With `lambda_signed = 0`, tests verify that `L_total` is exactly equal to the directly returned base loss.

## 8. Gradient Semantics

For a valid pair, minimizing `softplus(s_h - s_a)` gives a positive gradient on `s_h` and a negative gradient on `s_a`; gradient descent therefore lowers the harmful score and raises the admissible-low score. Tests cover direction, finiteness, pair counting, empty-pair behavior, and additive loss composition.

## 9. Training Contract

The training entry point is `scripts/diagnostics/train_signed_selector_v21.py`. It directly calls the existing `cbwdm_multitask_loss` through `signed_v21_multitask_loss`, then adds only the signed ordinal auxiliary. The frozen experiment parameters enforced by the CLI are:

```text
epochs=3
lr=2e-5
batch_size=8
max_length=512
b_plus=0.01
b_minus=0.001
beta=0.25
gamma=1.0
lambda_signed=0.25
gamma_signed=1.0
neutral_sample_policy=negative
seed=13
```

Training records base, auxiliary, and total losses; valid/skipped ranking groups; signed pair counts; and supervision coverage. Artifacts are restricted to `artifacts/diagnostics/method_failure_audit/signed_v21/training`. Safe reuse requires `--resume` plus an identical fingerprint and matching output checksums.

The training history and static coverage block report `signed_valid_ranking_groups` beside `base_valid_ranking_groups`. The two sets are not assumed to be nested: the signed auxiliary is specifically able to supervise a state with no utility-positive candidate when it contains both an ordinary negative/neutral candidate and a harmful candidate. Real train-core counts are `UNAVAILABLE UNTIL SERVER RUN`; the implementation does not invent them from local fixtures.

## 10. Inference Contract

Inference remains the signed-v1 scalar greedy policy. The v2.1 `greedy_select_without_gold` directly delegates to signed-v1, and the row wrapper reuses signed-v1 feature construction and selection before changing only method metadata. Fixed deployment values are `score_threshold=0.0`, `top_m=4`, and `min_docs=0`. There is no second score, raw gate, hard gate, or altered stopping rule.

## 11. Gold Leakage Audit

The inference call signature accepts query text, candidates, selector scores, and fixed selection controls only. It does not accept labels, gold evidence, posterior alignment, teacher classes, or oracle decisions. The selection test asserts v1-equivalent outputs, the gold-free signature, and the scalar score schema `{doc_id, score, rank}`.

## 12. Backward Compatibility

No signed-v1, signed-v2, shared model, shared loss, configuration, or production file was modified. V1 behavior is unchanged. V2 behavior is unchanged. V2.1 has a distinct method name, checkpoint metadata, selection metadata, and artifact subtree under `signed_v21`.

## 13. Tests Run

Focused suite:

```text
.venv\Scripts\python.exe -m pytest -q tests/test_signed_selector_v21.py
17 passed in 1.30s
```

Required compatibility suite:

```text
.venv\Scripts\python.exe -m pytest -q tests/test_signed_teacher_v1.py tests/test_signed_selector_v21.py tests/test_signed_selector_v2.py tests/test_selector_loss_and_schema.py tests/test_method_failure_diagnostics.py tests/test_cbwdm_score.py
58 passed in 1.59s
```

All five new Python files also passed bytecode compilation, and the patch passed whitespace validation.

## 14. Known Limitations

- No server training, five-query smoke run, FEVER 500 selection, generator evaluation, or signed/base comparison was executed in this source implementation round.
- Seed 21 and seed 42 were not run, and no preformal experiment was run or registered.
- `lambda_signed=0.25` is the frozen first-round setting and was not tuned.
- Training still depends on the existing authoritative signed-v1 teacher; no alternative raw-sign teacher was introduced.
- A raw gate and step balancing were intentionally not implemented.
- The auxiliary constrains within-group ordering only; it does not guarantee an absolute zero threshold calibration.
- Groups without both `A_low` and `H` receive no auxiliary gradient.
- The number of pair terms grows as `|A_low| * |H|`, although the current candidate-group sizes are small.
- Post-hoc quality depends on matching selection, posterior, prediction, metric, and oracle artifacts from the same run.

## 15. Exact Proposed Server Commands

These proposals were verified against the implemented CLIs' real `--help` output. Set paths and choose a tag containing the actual new commit's short SHA after committing this implementation; replace `PENDING` below before execution.

```bash
REPO=/root/rag-cbwdm
RUN=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13
PY=/root/miniconda3/envs/rag-cbwdm-baselines/bin/python
MODEL=/root/models/ms-marco-MiniLM-L-6-v2
V21="$RUN/artifacts/diagnostics/method_failure_audit/signed_v21"
TAG=head_PENDING_v21_seed13
cd "$REPO"
```

Fresh training (intentionally no `--resume`):

```bash
"$PY" scripts/diagnostics/train_signed_selector_v21.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-dir "$RUN" \
  --teacher "$RUN/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher/head6586_train_core/teacher.jsonl" \
  --posteriors "$RUN/artifacts/formal/fever2_train_core_posteriors.jsonl" \
  --retrieval "$RUN/artifacts/formal/fever2_train_core_bm25_top20.jsonl" \
  --output-dir "$V21/training/$TAG" \
  --model-name "$MODEL" \
  --epochs 3 \
  --lr 2e-5 \
  --batch-size 8 \
  --max-length 512 \
  --b-plus 0.01 \
  --b-minus 0.001 \
  --beta 0.25 \
  --gamma 1.0 \
  --lambda-signed 0.25 \
  --gamma-signed 1.0 \
  --neutral-sample-policy negative \
  --seed 13 \
  --device auto
```

Fresh five-query smoke selection (intentionally no `--resume`):

```bash
"$PY" scripts/diagnostics/select_signed_selector_v21.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-dir "$RUN" \
  --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" \
  --checkpoint-dir "$V21/training/$TAG/checkpoint" \
  --output "$V21/selections/${TAG}_smoke5/selection.jsonl" \
  --top-m 4 \
  --min-docs 0 \
  --score-threshold 0.0 \
  --validation-limit 5 \
  --device auto
```

Fresh 500-query selection (intentionally no `--resume`):

```bash
"$PY" scripts/diagnostics/select_signed_selector_v21.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-dir "$RUN" \
  --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" \
  --checkpoint-dir "$V21/training/$TAG/checkpoint" \
  --output "$V21/selections/${TAG}_full500/selection.jsonl" \
  --top-m 4 \
  --min-docs 0 \
  --score-threshold 0.0 \
  --validation-limit 500 \
  --device auto
```

Evaluate the isolated v2.1 selection before analysis:

```bash
"$PY" scripts/07_eval_rag_classification.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --split validation \
  --selection "$V21/selections/${TAG}_full500/selection.jsonl" \
  --output "$V21/evaluation/${TAG}_full500/predictions.jsonl" \
  --metrics-output "$V21/evaluation/${TAG}_full500/metrics.json" \
  --model-name /root/models/Qwen2.5-1.5B-Instruct \
  --method-name rag_cbwdm_signed_v21 \
  --limit 500
```

Then run the read-only post-hoc analyzer:

```bash
"$PY" scripts/diagnostics/analyze_signed_selector_v21.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-dir "$RUN" \
  --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" \
  --selection "$V21/selections/${TAG}_full500/selection.jsonl" \
  --predictions "$V21/evaluation/${TAG}_full500/predictions.jsonl" \
  --metrics "$V21/evaluation/${TAG}_full500/metrics.json" \
  --oracle-selection "$RUN/artifacts/diagnostics/method_failure_audit/signed_gate/full500/selection.jsonl" \
  --output-dir "$V21/reports/${TAG}_full500" \
  --alignment-eps 0
```

Only already-complete training or selection artifacts may be reused, by repeating the identical corresponding command with `--resume`. The analyzer has no resume mode and must receive a new output directory.

## HANDOFF_TO_CHATGPT

STATUS: COMPLETED
GIT_HEAD: 9c89f5cebe937b3ee18124f0bcae1e6d385c6c76
METHOD: rag_cbwdm_signed_v21
VARIANT: signed_selector_v21
MODEL_CLASS: CrossEncoderSelector
PRETRAINED_CLASSIFIER_PRESERVED: YES
NEW_MODEL_HEAD_ADDED: NO
BASE_LOSS: cbwdm_multitask_loss
SIGNED_AUXILIARY: mean softplus(gamma_signed * (s_h - s_a)) over H x A_low
A_LOW: supervision_class in {negative, neutral}
HARMFUL_SET: supervision_class == explicit_harmful_negative
LAMBDA_SIGNED: 0.25
GAMMA_SIGNED: 1.0
INFERENCE_CHANGED: NO
SCORE_THRESHOLD: 0.0
TOP_M: 4
MIN_DOCS: 0
RAW_GATE_IMPLEMENTED: NO
HARD_GATE_IMPLEMENTED: NO
STEP_BALANCING_IMPLEMENTED: NO
V1_BEHAVIOR_CHANGED: NO
V2_BEHAVIOR_CHANGED: NO
GOLD_USED_AT_INFERENCE: NO
TESTS: 17 focused passed; 58 required compatibility tests passed
FILES_ADDED: src/diagnostics/signed_selector_v21.py, scripts/diagnostics/train_signed_selector_v21.py, scripts/diagnostics/select_signed_selector_v21.py, scripts/diagnostics/analyze_signed_selector_v21.py, tests/test_signed_selector_v21.py, SIGNED_V21_IMPLEMENTATION_REPORT.md
FILES_MODIFIED: NONE
SERVER_TRAIN_COMMAND: "$PY" scripts/diagnostics/train_signed_selector_v21.py --config configs/fever2_server_pilot_5000_500.yaml --run-dir "$RUN" --teacher "$RUN/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher/head6586_train_core/teacher.jsonl" --posteriors "$RUN/artifacts/formal/fever2_train_core_posteriors.jsonl" --retrieval "$RUN/artifacts/formal/fever2_train_core_bm25_top20.jsonl" --output-dir "$V21/training/$TAG" --model-name "$MODEL" --epochs 3 --lr 2e-5 --batch-size 8 --max-length 512 --b-plus 0.01 --b-minus 0.001 --beta 0.25 --gamma 1.0 --lambda-signed 0.25 --gamma-signed 1.0 --neutral-sample-policy negative --seed 13 --device auto
SERVER_SMOKE_COMMAND: "$PY" scripts/diagnostics/select_signed_selector_v21.py --config configs/fever2_server_pilot_5000_500.yaml --run-dir "$RUN" --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" --checkpoint-dir "$V21/training/$TAG/checkpoint" --output "$V21/selections/${TAG}_smoke5/selection.jsonl" --top-m 4 --min-docs 0 --score-threshold 0.0 --validation-limit 5 --device auto
SERVER_SELECT500_COMMAND: "$PY" scripts/diagnostics/select_signed_selector_v21.py --config configs/fever2_server_pilot_5000_500.yaml --run-dir "$RUN" --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" --checkpoint-dir "$V21/training/$TAG/checkpoint" --output "$V21/selections/${TAG}_full500/selection.jsonl" --top-m 4 --min-docs 0 --score-threshold 0.0 --validation-limit 500 --device auto
SERVER_ANALYZE_COMMAND: "$PY" scripts/diagnostics/analyze_signed_selector_v21.py --config configs/fever2_server_pilot_5000_500.yaml --run-dir "$RUN" --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" --selection "$V21/selections/${TAG}_full500/selection.jsonl" --predictions "$V21/evaluation/${TAG}_full500/predictions.jsonl" --metrics "$V21/evaluation/${TAG}_full500/metrics.json" --oracle-selection "$RUN/artifacts/diagnostics/method_failure_audit/signed_gate/full500/selection.jsonl" --output-dir "$V21/reports/${TAG}_full500" --alignment-eps 0
REPORT_PATH: C:\Users\wenhao\Desktop\CBWDM\rag_cbwdm\SIGNED_V21_IMPLEMENTATION_REPORT.md
ALLOWLIST_UNCHANGED: YES
GIT_DIFF_STAT: task delta is 6 new files, 1854 insertions(+); no existing file modified by this task

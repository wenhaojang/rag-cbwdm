# Signed-v2.2 State-Level Top-of-List Implementation Report

## 1. Scope

This change adds the independent development method `rag_cbwdm_signed_v22`. It preserves signed-v1's pretrained single-score model, base loss, greedy inference, and zero STOP threshold. Relative to v2.1, v2.2 replaces mean pairwise ordering over `A_low = {negative, neutral}` with a new hard-max, state-level top-of-list objective over `A = {positive, negative, neutral}`; both the aggregation and the admissible-set definition therefore change. No server experiment or model/data download was performed.

## 2. Empirical Motivation

Signed-v2.1 reduced the overall selected raw-negative ratio from about `0.3246` to `0.2991` and authoritative-negative documents from `186/570` to `165/545`, without selection collapse. However, the critical SUPPORTS first-selected raw-negative count changed only from `104` to `103`. SUPPORTS step-0 raw sign AUROC also changed from about `0.8177` to `0.8094`. The evidence therefore suggests that v2.1 moved some later or non-leading harmful candidates but did not improve the top-1 competition that greedy inference actually consumes.

## 3. Why Mean Pairwise Is Insufficient

V2.1 averages over every `A_low x H` pair. It can reduce its loss by improving many pairs in the middle or tail of the score list, even if the highest-scoring harmful candidate still outranks every admissible candidate. Test-time selection uses only `argmax_j s_j`; the first harmful decision is determined by `max_h s_h` versus `max_a s_a`. V2.2 introduces a new state-level objective and correspondingly expands the admissible set from v2.1's low-utility classes to all non-harmful teacher classes: `positive`, `negative`, and `neutral`. V2.2 is therefore not a strict single-factor ablation that changes only the aggregation operator; if performance improves, the full effect cannot be attributed to hard max alone because admissible-set expansion is a simultaneous change.

## 4. Files Added / Modified

Added:

- `src/diagnostics/signed_selector_v22.py`
- `scripts/diagnostics/train_signed_selector_v22.py`
- `scripts/diagnostics/select_signed_selector_v22.py`
- `scripts/diagnostics/analyze_signed_selector_v22.py`
- `tests/test_signed_selector_v22.py`
- `SIGNED_V22_IMPLEMENTATION_REPORT.md`

Modified existing files: none. The three pre-existing dirty allowlist files were not edited.

## 5. Preserved Model Architecture

Training instantiates the existing `CrossEncoderSelector`, which uses `AutoModelForSequenceClassification` with `num_labels=1`. This preserves the pretrained MS-MARCO scalar classification head from `ms-marco-MiniLM-L-6-v2`. Optimization remains over `selector.model.parameters()`. No fresh classifier, gate head, sign head, utility head, dual-head model, or custom Hugging Face architecture was introduced.

## 6. Preserved Base Loss

`signed_v22_multitask_loss` directly calls the existing `cbwdm_multitask_loss` with `group.effective_gains` and the unchanged `b_plus`, `b_minus`, `beta`, `gamma`, and neutral policy. It does not reproduce or alter the internal BCE and base ranking mathematics:

```text
L_base = cbwdm_multitask_loss(scores, group.effective_gains, ...)
```

The `lambda_top=0` regression test verifies exact numerical reduction to this base loss.

## 7. Admissible and Harmful Sets

Sets are derived only from the existing teacher's `group.supervision_classes`:

```text
A = {j | class_j in {positive, negative, neutral}}
H = {j | class_j == explicit_harmful_negative}
```

Harmfulness is not inferred from effective gain or a gold label. Unknown classes raise `ValueError` instead of being silently categorized.

## 8. State-Level Top Objective

For a valid group containing both sets:

```text
s_A_max = max_{a in A} s_a
s_H_max = max_{h in H} s_h
L_top   = softplus(gamma_top * (s_H_max - s_A_max))
L_total = L_base + lambda_top * L_top
```

The implementation uses `torch.max` directly: no mean pairwise loss, margin, LogSumExp max, smoothed max, or top-k average. If either set is empty, `L_top = scores.sum() * 0.0`, preserving dtype, device, differentiability, and finite behavior.

## 9. Gradient Semantics

For a valid state without tied maxima, the auxiliary gives a positive gradient to the highest harmful score and a negative gradient to the highest admissible score. Gradient descent therefore lowers the former and raises the latter. Non-maximum candidates receive zero gradient from this hard-max auxiliary, while the unchanged base loss can still train them. Tests use strictly distinct maxima because tied-max gradient allocation is not part of the v2.2 contract.

## 10. Why Positive Is Included in A

V2.1 excluded positive candidates from `A_low` because it targeted the missing ordinary-low-above-harmful pair type. V2.2 targets the greedy state decision itself. Any admissible candidate—positive, negative, or neutral—may be the best admissible alternative that prevents a harmful document from becoming top-1. Utility ordering inside the admissible set remains the base loss's responsibility.

## 11. STOP Calibration

The top auxiliary is relative: it encourages `s_A_max > s_H_max`, but does not guarantee `s_A_max > 0`. It therefore does not replace or recalibrate the existing `score_threshold=0.0`. No absolute admissibility BCE, harmful gate threshold, or top-loss-specific inference rule was added.

## 12. Training Contract

The training CLI hard-checks the complete frozen first-round setting:

```text
epochs=3
lr=2e-5
batch_size=8
max_length=512
b_plus=0.01
b_minus=0.001
beta=0.25
gamma=1.0
lambda_top=0.25
gamma_top=1.0
neutral_sample_policy=negative
seed=13
```

Training reuses `build_signed_training_groups` and records total/base/CE/base-rank/top-rank loss, base and top valid/skipped groups, top competition count, supervision counts, average top gap, and dynamic correct/incorrect top order counts. The latter are training-time observations, not independent validation metrics.

Static coverage records total groups, base-valid groups, and top-valid groups. Known references are base-valid `4515/14961` and v2.1 all-pair-valid `9651/14961`; v2.2 top-valid coverage is not assumed and is `UNAVAILABLE UNTIL SERVER RUN`.

The checkpoint remains a standard `CrossEncoderSelector.save_checkpoint` artifact. Metadata records method, variant, architecture, loss type, all frozen parameters, and group count. Training reuse requires a completed manifest, identical fingerprint, checkpoint checksum, and output checksums.

## 13. Inference Contract

V2.2 inference directly delegates to signed-v1's `greedy_select_without_gold` and row wrapper. Given identical query, state, candidates, scalar scores, and controls, selected IDs, steps, STOP behavior, and deterministic tie-break are identical to v1. Fixed controls remain:

```text
score_threshold=0.0
top_m=4
min_docs=0
```

Selection artifacts contain only `doc_id`, scalar `score`, and `rank` for each candidate. No gate, sign, utility, or predicted-admissible field is introduced.

## 14. Gold Leakage Audit

The inference API accepts no gold label, gold evidence, alignment, teacher class, eta, or oracle action. Feature construction remains the shared `build_selector_input`. A regression test changes the posterior gold label while keeping query, candidate text, and state fixed, and verifies identical model inputs and actions. Gold and posterior direction are used only by the read-only post-hoc analyzer after selection.

## 15. Backward Compatibility

Signed-v1, signed-v2, signed-v2.1, `src/selector_cross_encoder.py`, the teacher, shared losses, configurations, generator, retrieval, and split logic are untouched. V2.2 has its own method identity, training manifest schema, checkpoint metadata, selection metadata, tests, report, and artifact subtree under `signed_v22`.

## 16. Tests Run

All five new Python files passed bytecode compilation.

Focused v2.2 suite:

```text
.venv\Scripts\python.exe -m pytest -q tests/test_signed_selector_v22.py
25 passed in 1.31s
```

Required related regression suite:

```text
.venv\Scripts\python.exe -m pytest -q tests/test_signed_teacher_v1.py tests/test_signed_selector_v21.py tests/test_signed_selector_v22.py tests/test_signed_selector_v2.py tests/test_selector_loss_and_schema.py tests/test_method_failure_diagnostics.py tests/test_cbwdm_score.py
83 passed in 1.81s
```

`tests/test_signed_selector_v1.py` does not exist at this HEAD, so no artificial placeholder test was created. Existing v1 behavior is covered by signed-teacher, selector/schema, method-failure, CBWDM, and explicit v1-versus-v2.2 inference equivalence tests.

## 17. Known Limitations

- No server training was run.
- No five-query smoke selection was run.
- No 500-query selection was run.
- No generator evaluation was run.
- Seed 21 and seed 42 were not run.
- No preformal or formal experiment was run or registered.
- `lambda_top=0.25` was not tuned.
- `gamma_top=1.0` was not tuned.
- The authoritative signed-v1 teacher remains unchanged.
- Hard max gives auxiliary gradient only to the current admissible and harmful maxima.
- No step weighting or step-0 upweighting was added.
- No raw-sign alternative or raw gate was added.
- Relative top ordering alone does not calibrate the absolute zero STOP threshold.

## 18. Exact Proposed Server Commands

These commands were derived from the implemented CLIs' actual `--help` output and were not executed. After committing, replace `PENDING` with the actual new short SHA.

```bash
REPO=/root/rag-cbwdm
RUN=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13
PY=/root/miniconda3/envs/rag-cbwdm-baselines/bin/python
MODEL=/root/models/ms-marco-MiniLM-L-6-v2
V22="$RUN/artifacts/diagnostics/method_failure_audit/signed_v22"
TAG=head_PENDING_v22_seed13
cd "$REPO"
```

Fresh training, intentionally without `--resume`:

```bash
"$PY" scripts/diagnostics/train_signed_selector_v22.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-dir "$RUN" \
  --teacher "$RUN/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher/head6586_train_core/teacher.jsonl" \
  --posteriors "$RUN/artifacts/formal/fever2_train_core_posteriors.jsonl" \
  --retrieval "$RUN/artifacts/formal/fever2_train_core_bm25_top20.jsonl" \
  --output-dir "$V22/training/$TAG" \
  --model-name "$MODEL" \
  --epochs 3 \
  --lr 2e-5 \
  --batch-size 8 \
  --max-length 512 \
  --b-plus 0.01 \
  --b-minus 0.001 \
  --beta 0.25 \
  --gamma 1.0 \
  --lambda-top 0.25 \
  --gamma-top 1.0 \
  --neutral-sample-policy negative \
  --seed 13 \
  --device auto
```

Fresh five-query smoke selection, intentionally without `--resume`:

```bash
"$PY" scripts/diagnostics/select_signed_selector_v22.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-dir "$RUN" \
  --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" \
  --checkpoint-dir "$V22/training/$TAG/checkpoint" \
  --output "$V22/selections/${TAG}_smoke5/selection.jsonl" \
  --top-m 4 \
  --min-docs 0 \
  --score-threshold 0.0 \
  --validation-limit 5 \
  --device auto
```

Fresh 500-query selection, intentionally without `--resume`:

```bash
"$PY" scripts/diagnostics/select_signed_selector_v22.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-dir "$RUN" \
  --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" \
  --checkpoint-dir "$V22/training/$TAG/checkpoint" \
  --output "$V22/selections/${TAG}_full500/selection.jsonl" \
  --top-m 4 \
  --min-docs 0 \
  --score-threshold 0.0 \
  --validation-limit 500 \
  --device auto
```

Fresh 500-query generator evaluation:

```bash
"$PY" scripts/07_eval_rag_classification.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --split validation \
  --selection "$V22/selections/${TAG}_full500/selection.jsonl" \
  --output "$V22/evaluation/${TAG}_full500/predictions.jsonl" \
  --metrics-output "$V22/evaluation/${TAG}_full500/metrics.json" \
  --model-name /root/models/Qwen2.5-1.5B-Instruct \
  --method-name rag_cbwdm_signed_v22 \
  --limit 500
```

Read-only post-hoc analysis:

```bash
"$PY" scripts/diagnostics/analyze_signed_selector_v22.py \
  --config configs/fever2_server_pilot_5000_500.yaml \
  --run-dir "$RUN" \
  --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" \
  --selection "$V22/selections/${TAG}_full500/selection.jsonl" \
  --predictions "$V22/evaluation/${TAG}_full500/predictions.jsonl" \
  --metrics "$V22/evaluation/${TAG}_full500/metrics.json" \
  --oracle-selection "$RUN/artifacts/diagnostics/method_failure_audit/signed_gate/full500/selection.jsonl" \
  --output-dir "$V22/reports/${TAG}_full500" \
  --alignment-eps 0
```

Only a completed training or selection artifact with an identical fingerprint and valid checksums may be reused by repeating its identical command with `--resume`. The analyzer requires a new output directory.

## HANDOFF_TO_CHATGPT

STATUS: COMPLETED
GIT_HEAD: 10ebb95be518330ac418d7b6d480323ffaaddf6b
METHOD: rag_cbwdm_signed_v22
VARIANT: signed_selector_v22
MODEL_CLASS: CrossEncoderSelector
PRETRAINED_CLASSIFIER_PRESERVED: YES
NEW_MODEL_HEAD_ADDED: NO

BASE_LOSS: cbwdm_multitask_loss

TOP_AUXILIARY: softplus(gamma_top * (max harmful score - max admissible score))
ADMISSIBLE_SET: supervision_class in {positive, negative, neutral}
HARMFUL_SET: supervision_class == explicit_harmful_negative

LAMBDA_TOP: 0.25
GAMMA_TOP: 1.0

USES_HARD_MAX: YES
USES_PAIRWISE_MEAN: NO
USES_LOGSUMEXP: NO

INFERENCE_CHANGED: NO
SCORE_THRESHOLD: 0.0
TOP_M: 4
MIN_DOCS: 0

RAW_GATE_IMPLEMENTED: NO
HARD_GATE_IMPLEMENTED: NO
STEP_WEIGHTING_IMPLEMENTED: NO

V1_BEHAVIOR_CHANGED: NO
V2_BEHAVIOR_CHANGED: NO
V21_BEHAVIOR_CHANGED: NO

GOLD_USED_AT_INFERENCE: NO

TESTS: 25 focused passed; 83 required related regression tests passed; py_compile passed

FILES_ADDED: src/diagnostics/signed_selector_v22.py, scripts/diagnostics/train_signed_selector_v22.py, scripts/diagnostics/select_signed_selector_v22.py, scripts/diagnostics/analyze_signed_selector_v22.py, tests/test_signed_selector_v22.py, SIGNED_V22_IMPLEMENTATION_REPORT.md
FILES_MODIFIED: NONE

SERVER_TRAIN_COMMAND: "$PY" scripts/diagnostics/train_signed_selector_v22.py --config configs/fever2_server_pilot_5000_500.yaml --run-dir "$RUN" --teacher "$RUN/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher/head6586_train_core/teacher.jsonl" --posteriors "$RUN/artifacts/formal/fever2_train_core_posteriors.jsonl" --retrieval "$RUN/artifacts/formal/fever2_train_core_bm25_top20.jsonl" --output-dir "$V22/training/$TAG" --model-name "$MODEL" --epochs 3 --lr 2e-5 --batch-size 8 --max-length 512 --b-plus 0.01 --b-minus 0.001 --beta 0.25 --gamma 1.0 --lambda-top 0.25 --gamma-top 1.0 --neutral-sample-policy negative --seed 13 --device auto
SERVER_SMOKE_COMMAND: "$PY" scripts/diagnostics/select_signed_selector_v22.py --config configs/fever2_server_pilot_5000_500.yaml --run-dir "$RUN" --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" --checkpoint-dir "$V22/training/$TAG/checkpoint" --output "$V22/selections/${TAG}_smoke5/selection.jsonl" --top-m 4 --min-docs 0 --score-threshold 0.0 --validation-limit 5 --device auto
SERVER_SELECT500_COMMAND: "$PY" scripts/diagnostics/select_signed_selector_v22.py --config configs/fever2_server_pilot_5000_500.yaml --run-dir "$RUN" --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" --checkpoint-dir "$V22/training/$TAG/checkpoint" --output "$V22/selections/${TAG}_full500/selection.jsonl" --top-m 4 --min-docs 0 --score-threshold 0.0 --validation-limit 500 --device auto
SERVER_EVAL500_COMMAND: "$PY" scripts/07_eval_rag_classification.py --config configs/fever2_server_pilot_5000_500.yaml --split validation --selection "$V22/selections/${TAG}_full500/selection.jsonl" --output "$V22/evaluation/${TAG}_full500/predictions.jsonl" --metrics-output "$V22/evaluation/${TAG}_full500/metrics.json" --model-name /root/models/Qwen2.5-1.5B-Instruct --method-name rag_cbwdm_signed_v22 --limit 500
SERVER_ANALYZE_COMMAND: "$PY" scripts/diagnostics/analyze_signed_selector_v22.py --config configs/fever2_server_pilot_5000_500.yaml --run-dir "$RUN" --posteriors "$RUN/artifacts/formal/fever2_validation_posteriors.jsonl" --selection "$V22/selections/${TAG}_full500/selection.jsonl" --predictions "$V22/evaluation/${TAG}_full500/predictions.jsonl" --metrics "$V22/evaluation/${TAG}_full500/metrics.json" --oracle-selection "$RUN/artifacts/diagnostics/method_failure_audit/signed_gate/full500/selection.jsonl" --output-dir "$V22/reports/${TAG}_full500" --alignment-eps 0

REPORT_PATH: C:\Users\wenhao\Desktop\CBWDM\rag_cbwdm\SIGNED_V22_IMPLEMENTATION_REPORT.md
ALLOWLIST_UNCHANGED: YES
GIT_DIFF_STAT: task delta is 6 new files, 1933 insertions(+); no existing file modified by this task

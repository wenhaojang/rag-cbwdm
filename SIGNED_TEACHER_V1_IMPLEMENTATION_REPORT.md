# signed_teacher_v1 Implementation Report

## 1. Summary

This change adds an isolated repair-validation pipeline: a read-only signed-gate full500 residual audit, a train-core `signed_teacher_v1` with terminal-state supervision, fixed-contract cross-encoder training, deployable state-aware inference, and post-hoc validation diagnostics. It does not change production RAG-CBWDM behavior or run any server experiment locally.

## 2. Files changed

| File | Change | Purpose | Production impact |
|---|---|---|---|
| `src/diagnostics/signed_gate_error_audit.py` | added | CPU-only 106-zero-doc/59-error/gold-evidence audit | None |
| `src/diagnostics/signed_teacher_v1.py` | added | signed action, all-remaining supervision, terminal groups, teacher stats | None |
| `src/diagnostics/signed_selector_v1.py` | added | gold-free greedy action API and post-hoc diagnostics | None |
| `scripts/diagnostics/analyze_signed_gate_full500.py` | added | residual-audit CLI | None |
| `scripts/diagnostics/build_signed_teacher_v1.py` | added | isolated teacher artifact CLI | None |
| `scripts/diagnostics/train_signed_selector_v1.py` | added | frozen first-round training CLI | None |
| `scripts/diagnostics/select_signed_selector_v1.py` | added | deployable experimental inference CLI | None |
| `scripts/diagnostics/analyze_signed_selector_v1.py` | added | evidence/alignment/oracle-imitation post-hoc CLI | None |
| `tests/test_signed_teacher_v1.py` | added | 11 repair-contract regression tests | None |
| `SIGNED_TEACHER_V1_SERVER_RUNBOOK.md` | added | gated, copyable server workflow | None |
| `SIGNED_TEACHER_V1_IMPLEMENTATION_REPORT.md` | added | implementation record | None |

## 3. Production behavior audit

| Surface | Changed? |
|---|---:|
| `src/cbwdm_score.py` default | NO |
| Production teacher | NO |
| Production selector default/training/inference | NO |
| Production evaluator | NO |
| Formal artifact schema | NO |
| `RAG-CBWDM.tex` | NO |

All code is new under `src/diagnostics/` or `scripts/diagnostics/`. Evaluation is performed by the existing `scripts/07_eval_rag_classification.py` with its existing serialization, prompt, `LabelLogitScorer`, argmax and metrics.

## 4. Exact signed_teacher_v1 mathematical contract

The experiment retains production posterior construction and Euclidean local effects:

\[
x_{ij}=L_i(\eta_{ij}-\eta_{i0}),\qquad d_i=L_i(e_{y_i}-\eta_{i0}),\qquad L_i=I.
\]

It defines `alignment = x_ij^T d_i` and admits `j` iff `alignment > alignment_eps`, with `alignment_eps=0`. At each state it maximizes the unchanged production marginal

\[
\Theta_i(S\cup\{j\})-\Theta_i(S)
\]

over admissible remaining candidates. `top_m=4`; a semantic STOP occurs when no admissible candidate exists or the best marginal is strictly below `0.001`. `top_m_reached` is a budget STOP and does not create an extra terminal state. The builder asserts that selected indices and stop reason equal the existing `signed_gated_greedy` helper.

## 5. Exact training-label semantics

- `alignment <= 0`: `explicit_harmful_negative`, BCE target 0, ranking-negative. Its diagnostic unsigned Theta gain is preserved, while its effective training gain is forced to 0 so it cannot become positive.
- Admissible and `gain > 0.01`: positive, BCE target 1, ranking-positive.
- Admissible and `gain < 0.001`: negative, BCE target 0, ranking-negative.
- Admissible and `0.001 <= gain <= 0.01`: neutral. With the fixed `neutral_sample_policy=negative`, it enters BCE with target 0 and is excluded from ranking.

Training calls production `cbwdm_multitask_loss` on the effective gains. Ranking is valid only when a group has at least one strict positive and one strict negative; otherwise ranking loss is zero, but BCE still trains the group. A terminal all-negative group, including a one-candidate group, is therefore retained and trained rather than dropped.

At the exact floating-point boundary `gain == 0.001`, teacher stopping (`best_gain < stop_threshold`) selects the candidate, while the gain-label contract calls it neutral. This follows the requested/current strict inequalities and is recorded as an edge-case uncertainty rather than silently changing either rule.

## 6. STOP / zero-doc learning mechanism

There is no STOP token and no STOP head. Semantic terminal states are included as ordinary candidate groups. With `stop_threshold=b_minus=0.001`, semantic terminal candidates become BCE-negative under the fixed contract. At inference, raw negative logits plus `min_docs=0` and `score_threshold=0.0` allow step-0 STOP. `score_threshold=0.0` is the natural BCE logit boundary because `CrossEncoderSelector.score_texts` returns the raw scalar sequence-classification logit; stopping uses strict `best_score < 0.0`.

## 7. Inference gold-leakage audit

The action function `greedy_select_without_gold` accepts only query, candidate documents, a state-aware scoring callback, and stopping parameters. It has no `label`, `gold`, `d_i`, `alignment`, or posterior-direction argument. The callback builds production `build_selector_input(query, selected_docs, candidate)` and returns raw logits. The wrapper copies `label` only after action selection into the standard selection row so the production validation evaluator can score it; it is never passed to the selector or stopping rule.

Every selection row/manifest declares `uses_gold_at_inference=false`, `deployable_selector=true`, and `experimental=true`. Post-hoc alignment code refuses artifacts that do not carry this no-gold declaration.

**Conclusion: NO GOLD LEAKAGE in inference decisions.**

## 8. Residual-audit coverage

The server audit reads existing signed-gate trajectories, selections, predictions, metrics, validation retrieval/posteriors and the existing supplement without generator calls. It reports:

- Zero-doc counts/ratios, query-only confusion/distribution, exact trajectory-derived reason and positive-candidate histograms by ALL/SUPPORTS/REFUTES.
- All 59 SUPPORTS→REFUTES examples with query-only posterior, doc-count distribution, BM25 gold presence/rank, selected coverage/ranks, positive-candidate availability and per-gold-candidate alignment, gold-probability delta and singleton Theta.
- Mutually exclusive priority `F5 → F1 → F2 → F4 → F3` plus raw non-exclusive counts.
- Gold-candidate alignment over all 500 validation examples.
- Signed selection quality by label and existing production controls loaded from the supplement.

The local repository does not include `RAG_CBWDM_METHOD_FAILURE_AUDIT.md` or the server-generated supplement Markdown/JSON, so their server contents were not fabricated. Gate A consumes the real JSON directly.

## 9. Tests

Local bundled Python command:

```text
<python> -m compileall -q src/diagnostics scripts/diagnostics tests/test_method_failure_diagnostics.py tests/test_signed_teacher_v1.py
result: PASS
```

```text
<python> -m unittest tests.test_cbwdm_score tests.test_method_failure_diagnostics tests.test_signed_teacher_v1 -v
passed: 23
failed: 0
skipped: 0
environment errors: 0
```

```text
<python> -m unittest tests.test_selector_loss_and_schema -v
passed: 0
assertion failures: 0
environment dependency errors: 1 (`torch` missing before test import)
```

```text
<python> -m unittest discover -s tests -v
passed: 35
assertion failures: 0
environment dependency errors: 12 (`PyYAML`, `pytest`, or `torch` unavailable)
skipped: 0
```

No dependency was installed. Gate 0 reruns selector tests in the authoritative server environment.

## 10. Key server commands

Gate A residual audit:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/analyze_signed_gate_full500.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --signed-gate-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_gate/full500 --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_gate/full500_error_audit --supplement-json /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/RAG_CBWDM_METHOD_FAILURE_AUDIT_SERVER_SUPPLEMENT.json
```

Gate B full teacher:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/build_signed_teacher_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --split train_core --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher/full_train_core --top-m 4 --teacher-stop-threshold 0.001 --alignment-eps 0 --b-plus 0.01 --b-minus 0.001 --neutral-sample-policy negative --resume
```

Gate C 100-group training smoke:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/train_signed_selector_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --teacher /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher/full_train_core/teacher.jsonl --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_posteriors.jsonl --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_bm25_top20.jsonl --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/training/smoke100 --model-name /root/models/ms-marco-MiniLM-L-6-v2 --epochs 3 --lr 2e-5 --batch-size 8 --beta 0.25 --gamma 1.0 --teacher-temperature 0.1 --b-plus 0.01 --b-minus 0.001 --neutral-sample-policy negative --seed 13 --max-train-groups 100 --device auto --resume
```

Gate C 20-validation selection smoke:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/select_signed_selector_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_validation_posteriors.jsonl --checkpoint-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/training/smoke100/checkpoint --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/selections/smoke20/selection.jsonl --top-m 4 --min-docs 0 --score-threshold 0.0 --validation-limit 20 --device auto --resume
```

Gate D full training:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/train_signed_selector_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --teacher /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/teacher/full_train_core/teacher.jsonl --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_posteriors.jsonl --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_bm25_top20.jsonl --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/training/full --model-name /root/models/ms-marco-MiniLM-L-6-v2 --epochs 3 --lr 2e-5 --batch-size 8 --beta 0.25 --gamma 1.0 --teacher-temperature 0.1 --b-plus 0.01 --b-minus 0.001 --neutral-sample-policy negative --seed 13 --device auto --resume
```

Gate E full500 primary selection:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/select_signed_selector_v1.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_validation_posteriors.jsonl --checkpoint-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/training/full/checkpoint --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_teacher_v1/selections/primary_min0_threshold0/selection.jsonl --top-m 4 --min-docs 0 --score-threshold 0.0 --validation-limit 500 --device auto --resume
```

Evaluation and post-hoc commands immediately following each selection are in the runbook. Every model argument is a local absolute path.

## 11. Expected compute

- Residual audit: CPU JSONL/NumPy work, linear in 500×20 candidates plus selected states; no model calls.
- Teacher build: CPU NumPy, using existing 5000×20 single-doc posteriors; no Qwen calls. Each visited state computes remaining production Theta marginals.
- Training smoke: GPU recommended; exactly at most 100 state groups for 3 fixed epochs.
- Full training: GPU recommended; 3 fixed epochs over all emitted groups, each scoring every remaining candidate text under the state-aware cross-encoder.
- 500 validation selection: GPU recommended for selector logits, at most `500×(20+19+18+17)` candidate scores if no early stop.
- 500 validation evaluation: generator GPU work, exactly one production evaluation prompt per selected row, batched only as implemented by the production evaluator.

No wall-clock time is invented.

## 12. Interpretation framework

1. Deployable selector approaches signed oracle (for example, clearly enters the ~0.8 range): teacher sign repair is learnable; formal directional redesign and a future production-method rebuild become justified.
2. Oracle remains 0.882 while deployable selector remains near old ~0.59: teacher target is repaired, but the main residual problem moves to selector training, state representation or stop learning.
3. `min_docs=0` materially beats `min_docs=2`: zero-doc/stopping is an important secondary mechanism.
4. Signed selector still chooses many negative-alignment documents: the textual selector failed to learn the directional teacher; inspect supervision/loss before theory expansion.
5. Selector rarely chooses harmful evidence but accuracy stays low: then investigate true joint-posterior/set interaction.

## 13. Remaining uncertainties

- The 0.882 signed-gate oracle uses gold and is not deployable.
- Whether signed_teacher_v1 can be learned by the selector is unknown.
- Score-threshold calibration has not been searched; only the natural BCE logit boundary 0.0 is used.
- The joint-posterior surrogate gap has not been formally excluded in this repair round.
- Query-conditional BW geometry remains unimplemented.
- Final theory formulas and production method remain unchanged.
- `teacher_temperature=0.1` is recorded for parity but does not affect production `cbwdm_multitask_loss`; it only affects the unused listwise-distillation alternative.

## AUDIT CORRECTION

- Production selector scores are raw logits, not probabilities. Therefore threshold 0.0 is meaningful as the BCE boundary.
- Production `min_docs=0` already works because inference clamps it to a non-negative integer and checks threshold before the first selection.
- The old production group builder emits only saved selected-action `steps`, drops groups with fewer than two candidates, and has no terminal STOP supervision.
- Under `neutral_sample_policy=negative`, neutral enters BCE as target 0 but never enters the positive/negative ranking sets.
- `teacher_temperature` is irrelevant when `loss_type=cbwdm_multitask`.

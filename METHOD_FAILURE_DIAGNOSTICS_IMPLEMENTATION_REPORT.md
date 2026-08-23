# Method-Failure Diagnostics Implementation Report

## 1. Summary

This change adds a read-only server supplement, a signed-gated gold oracle, and a true joint-posterior gold oracle. All generated artifacts are constrained below `$RUN/artifacts/diagnostics/method_failure_audit/`. No production method, training, selection, evaluation, configuration, theory, or formal artifact was changed.

The local checkout does not contain the named server run, so no actual 500-validation result is claimed here. The supplement will report `UNAVAILABLE` rather than infer a missing statistic.

## 2. Files changed

| File | Change | Purpose | Production? |
|---|---|---|---:|
| `src/diagnostics/__init__.py` | added | diagnostic package boundary | No |
| `src/diagnostics/method_failure.py` | added | isolation, signed greedy, joint greedy, production-evaluator reuse, cache/stat helpers | No |
| `src/diagnostics/server_supplement.py` | added | read existing grid/teacher/training/selection/evaluation artifacts and publish supplement | No |
| `scripts/diagnostics/build_method_failure_server_supplement.py` | added | read-only supplement CLI | No |
| `scripts/diagnostics/run_signed_gate_oracle.py` | added | validation-only signed-gate CLI and generator evaluation | No |
| `scripts/diagnostics/run_joint_posterior_oracle.py` | added | validation-only real set-posterior CLI, cache, comparison and metrics | No |
| `tests/test_method_failure_diagnostics.py` | added | six diagnostic regression tests covering the seven requested behaviors (existing tests are separate) | No |
| `METHOD_FAILURE_DIAGNOSTICS_SERVER_RUNBOOK.md` | added | gated server commands | No |
| `METHOD_FAILURE_DIAGNOSTICS_IMPLEMENTATION_REPORT.md` | added | implementation and verification record | No |

Pre-existing tracked report deletions in the worktree were not created, restored, or edited by this work.

## 3. Production behavior audit

| Surface | Changed? |
|---|---:|
| `src/cbwdm_score.py` default behavior | NO |
| Production teacher default behavior | NO |
| Selector architecture/training/inference | NO |
| Production evaluation | NO |
| Production artifact schema | NO |
| `RAG-CBWDM.tex` | NO |
| Formal config/defaults | NO |

The diagnostics dynamically load `scripts/07_eval_rag_classification.py` and call its `recover_selected_docs`, `build_evidence_context`, and `argmax_label`; prompts use the same `build_fever_prompt`, scorer uses the same `LabelLogitScorer`, metrics use the same `ClassificationMetrics`, and selections use `rag_cbwdm_selection.v2` helpers. This avoids a second drifting evaluator implementation.

## 4. Supplement coverage

Directly available when the corresponding server artifacts exist:

- Actual RAG-CBWDM candidate IDs, parameters, paths, fingerprints, manifests, training config, and metrics from `calibration_candidates.json` and its referenced artifacts.
- Teacher candidate gains, original BM25 ranks, selected flag, stop reasons, recomputed signed alignment, trajectory length, and step-0 Spearman correlations, stratified by all/gold/step/gold×step.
- Effective supervision class counts and ranking-valid/skipped groups under the actual `b_plus`, `b_minus`, and neutral policy. Current loss semantics are: `gain > b_plus` positive; `gain < b_minus` negative; the closed interval is neutral; neutral enters BCE as negative when policy is `negative`; ranking requires at least one positive and one negative.
- Candidate prediction confusion, distributions, accuracy/macro-F1, and SUPPORTS→REFUTES selected-document diagnostics.
- BM25 and selector evidence coverage using the repository's existing `gold_evidence_keys` plus `meta.page_id/meta.sentence_id` contract. It reports any-hit and complete flattened-key-union coverage. True “one complete alternative FEVER evidence group” coverage is explicitly `UNAVAILABLE`, because the retrieval artifact flattens groups and the tool will not invent grouping semantics.
- InfoGain candidate contract, prediction, and evidence statistics when InfoGain records exist in the same aggregate.

Explicitly `UNAVAILABLE` when missing:

- Validation teacher/selector top-1 and full ranking agreement requires a validation gold trajectory and/or persisted full candidate selector scores. Existing train-core teacher plus final selected IDs cannot reconstruct this.
- Validation teacher evidence recall requires an isolated validation oracle selection.
- Any absent manifest, selection, prediction, metrics, posterior, or retrieval artifact is named together with the existing stage/diagnostic that can generate it.

The supplement never invokes a generator, rebuilds a teacher, trains a selector, or reruns evaluation.

## 5. Signed-gate implementation

The diagnostic imports production `build_local_effects`, so

\[
x_{ij}=L_i(\eta_{ij}-\eta_{i0}),\qquad d_i=L_i(e_{y_i}-\eta_{i0}).
\]

It admits exactly candidates satisfying

\[
x_{ij}^{\top}d_i>\texttt{alignment\_eps},
\]

with default `alignment_eps=0`. Within that fixed admissible pool it calls production `theta_for_indices` and `marginal_gain`; ridge, smoothing, posterior definition, candidate pool, stopping rule, evaluator, and generator are otherwise unchanged. Trajectories persist every candidate's posterior, shift, alignment sign/admissibility, and every greedy state/gain/Theta decision. The selection is explicitly `diagnostic_only`, non-deployable, and `uses_gold_at_test=true`.

## 6. Joint-posterior implementation

For every ordered state `S`, documents are passed to production evaluator `build_evidence_context`; the resulting context is passed to the common FEVER prompt builder and label-logit scorer:

\[
\eta_{i,S}=P_\phi(Y\mid q_i,Z_S).
\]

Selection maximizes the probability-difference utility

\[
\eta_{i,S\cup j}[y_i]-\eta_{i,S}[y_i]
\]

and accepts only a strictly positive best marginal. All remaining candidates at one state are scored in one batched call (subject to scorer sub-batching). Cache keys bind diagnostic fingerprint, query ID, ordered document IDs, and exact prompt hash, so document ordering is never canonicalized away.

Stage 03 `eta0` is reused because Stage 03 and production query-only evaluation both call `build_fever_prompt(..., evidence=None)` with the same labels/verbalizers. The diagnostic fingerprint records prompt and verbalizer hashes plus generator revision. It does not replace or rewrite Stage 03.

At every joint-created state, original CBWDM surrogate gains are computed on the exact same remaining candidate indices. `comparison.json` reports matched-state Spearman, top-1 agreement, and gain-sign agreement by all/gold/step/gold×step.

## 7. Tests

Local bundled interpreter: `C:\Users\wenhao\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.

```text
<python> -m compileall -q src/diagnostics scripts/diagnostics tests/test_method_failure_diagnostics.py
passed: compilation completed
failed: 0
skipped: 0
```

```text
<python> -m unittest tests.test_cbwdm_score tests.test_method_failure_diagnostics -v
passed: 12
failed: 0
skipped: 0
```

Coverage includes: original binary sign blindness, signed admissibility, harmful-candidate exclusion and beneficial greedy order, mock joint positive-marginal selection/stopping/top-m, artifact isolation, cache resume, and cache fingerprint rejection. Six existing production CBWDM score tests also passed unchanged.

A full discovery attempt was also run:

```text
<python> -m unittest discover -s tests -v
passed: 24
failed: 0 assertions
environment/import errors: 12
skipped: 0
```

All 12 errors are due to this bundled local runtime lacking repository dependencies (`PyYAML`, `pytest`, or `torch`); they are not failing assertions. Dependencies were not installed because this task forbids internet-dependent environment mutation. The authoritative server baselines environment is checked in Gate 0.

## 8. Server commands

Supplement:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/build_method_failure_server_supplement.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit
```

Signed-gate 10-query smoke:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/run_signed_gate_oracle.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/signed_gate/smoke10 --validation-limit 10 --alignment-eps 0 --generator-model /root/models/Qwen2.5-1.5B-Instruct --device auto --resume
```

Joint-posterior 10-query smoke:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python scripts/diagnostics/run_joint_posterior_oracle.py --config /root/rag-cbwdm/configs/fever2_server_pilot_5000_500.yaml --run-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13 --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/diagnostics/method_failure_audit/joint_posterior_oracle/smoke10 --validation-limit 10 --candidate-limit 20 --generator-model /root/models/Qwen2.5-1.5B-Instruct --device auto --resume
```

## 9. Expected runtime / GPU demand

- Supplement: CPU/lightweight JSONL traversal only; no model calls. Complexity is linear in saved teacher candidate-state rows plus prediction/selection rows.
- Signed gate: teacher construction is CPU NumPy; evaluation requires one generator posterior per query, batched. For `N` queries the generator workload is `N` prompts.
- Joint posterior: GPU is strongly recommended. With candidate count `C` and selected length `M`, the upper bound is `C+(C-1)+...+(C-M+1)` posterior prompts per query. At `C=20,M=4`, that is at most 74 prompts/query, organized into at most four state-level scorer calls and internally sub-batched. Early stopping and cache hits reduce this.

No wall-clock minutes are asserted without a server measurement.

## 10. Interpretation framework

1. If signed-gated oracle recovers substantially and joint-posterior oracle is also strong, original Theta sign blindness is likely the primary cause; the single-doc→set surrogate can still be imperfect without being first-order.
2. If signed-gated remains weak but joint-posterior is strong, the main problem is likely the single-document posterior-span surrogate versus true set-posterior interaction.
3. If both are weak, investigate candidate-pool coverage, generator/evidence contract, and FEVER adaptation; further selector tuning has limited value at that point.
4. If signed-gated approaches or exceeds InfoGain, redesign the formal directional/admissibility mechanism before rebuilding the production teacher.

These are future interpretation rules, not conclusions from experiments that have not run.

## 11. Remaining uncertainties

- The current 500-validation teacher selected negative-alignment ratio has not been measured locally.
- Current formal 500 teacher-oracle generator performance remains unknown if no existing server artifact supplies it.
- Whether signed gating restores performance is unknown.
- Whether the joint-posterior oracle is strong is unknown.
- The earliest layer responsible for the REFUTES bias still awaits the server supplement and gated diagnostics.
- Actual server candidate path portability must be confirmed by Gate 1; the resolver reads paths from the aggregate/manifests and does not invent missing artifacts.

## AUDIT CORRECTION

The requested filename `scripts/07_eval_fever_generation.py` does not exist at the fixed commit. The actual production evaluator is `scripts/07_eval_rag_classification.py`. Its evidence serializer, prompt builder, label scorer, argmax, and metrics are therefore the authoritative contract used by these diagnostics. This changes no scientific conclusion, but it corrects the implementation reference and prevents evaluator drift.

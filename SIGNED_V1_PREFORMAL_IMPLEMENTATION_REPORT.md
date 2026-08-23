# Signed-v1 Preformal Implementation Report

## 1. Outcome

`signed_teacher_v1` is now registered as the independent experimental formal method `rag_cbwdm_signed_v1`. The implementation adds a clean-split builder, frozen three-seed signed training and gold-free inference entry points, shared-artifact fairness checks, formal summary/paired statistics, and post-hoc-only signed diagnostics. No heavy experiment was run.

This work does not modify `RAG-CBWDM.tex`, the production `rag_cbwdm` mathematics, an existing formal/diagnostic artifact, or any production result.

## 2. Repository state recorded before work

- Working directory: `C:\Users\wenhao\Desktop\CBWDM\rag_cbwdm`
- Branch: `feature/fever-formal-readiness`
- HEAD: `9ddccbb4366ff08801f8744e9bd1b5af6281e05c`
- Recent commits: `9ddccbb`, `221394f`, `4b914d2`, `01c121a`, `83a5f06`
- Pre-existing worktree state: fourteen tracked report files were already deleted. They were not restored or otherwise modified by this implementation.

## 3. Current split audit

Observed implementation facts:

1. `formal_splits.validation_size=5000` defines the frozen full validation target before profile limits.
2. `profile_limits.validation=500` becomes `validation_limit=500`. `src/formal_splits.py::_apply_limit` is called only after the full normalized-claim-group partition has been constructed. Therefore the pilot 500 is a deterministic whole-group subset/limit of the formal validation partition, not a separately sampled validation set.
3. `profile_limits.train_core=5000` is applied through the same post-partition whole-group limiting mechanism. It is a limit of the full legal train_core.
4. Full validation can be reconstructed without changing seed, stable hash, stratification, conflict policy, held-out precedence, or normalized-claim grouping by rerunning the pure `build_splits` function with the manifest's source SHAs/partition contract and limits removed.
5. The unused rows are recoverable as full validation minus the exact normalized-claim groups present in the pilot validation artifact.
6. The formal split implementation already guarantees zero ID and normalized-claim overlap among full train_core, full validation, and official-dev held_out_test. The new builder additionally checks the produced preformal rows against the current limited train_core, pilot validation, and full held-out role and fails if any of six overlap counts is nonzero.

Not locally observable:

- `outputs/formal_splits/fever2_seed13/fever2_formal_splits.manifest.json` is absent in this workspace.
- The current server run manifest, train_core/validation artifact manifests, `calibration_candidates.json`, calibration manifest, and frozen parameters are also absent.
- Local `data/raw/fever/{train,dev}.jsonl` contain 2000-row fixtures, not the server FEVER sources.

Consequently, the server-observed `actual_validation_size`, actual pilot-limit allocation, source SHA values, current baseline winner fingerprints, and exact final preformal row count cannot honestly be reported from this machine. The protocol-planned count is 4500 (`5000 - 500`); Gate A treats that as an expectation and publishes the actual count from the completed manifest. A non-4500 result is not silently accepted as 4500.

## 4. Clean evaluation strategy

Chosen strategy: **A**.

`scripts/preformal/24_build_preformal_eval.py` validates the existing completed split manifest and every split artifact checksum. It reconstructs the full frozen partition from the exact manifest source paths, SHAs, seed, stable-hash/group policy, stratification, and validation target; removes all current pilot-validation normalized-claim groups; and publishes only `preformal_eval`.

Canonical server outputs:

- `$RUN/artifacts/preformal_signed_v1/splits/preformal_eval.jsonl`
- `$RUN/artifacts/preformal_signed_v1/splits/preformal_eval.manifest.json`
- `$RUN/artifacts/preformal_signed_v1/splits/preformal_split_audit.md`

The manifest records strategy, source split manifest SHA, excluded pilot-validation SHA, current train_core SHA, preformal file SHA, count, label counts, ID-set SHA, normalized-claim-set SHA, all six overlap checks, Git state, and creation time. Official dev is consulted only inside the frozen split/leakage audit. No held-out example is emitted or made eligible for modeling.

## 5. Exact frozen signed-v1 contract

- Canonical method: `rag_cbwdm_signed_v1`
- Status: experimental formal method; it coexists with production `rag_cbwdm`
- `alignment_ij = x_ij^T d_i`
- Admissible iff `alignment_ij > 0`
- Teacher action: maximize the current production Theta marginal among admissible candidates
- `top_m=4`
- `teacher_stop_threshold=0.001`
- `alignment_eps=0`
- `b_plus=0.01`, `b_minus=0.001`
- `neutral_sample_policy=negative`
- nonpositive alignment: `explicit_harmful_negative`, BCE target 0, ranking negative
- selector: `/root/models/ms-marco-MiniLM-L-6-v2`
- training: 3 epochs, `lr=2e-5`, batch size 8, `beta=0.25`, `gamma=1.0`, `loss_type=cbwdm_multitask`
- inference: `min_docs=0`, `score_threshold=0.0`, `top_m=4`
- learned seeds: 13, 21, 42
- preformal calibration eligibility: false

`src/preformal/registry.py` is the machine-readable registry. Formal teacher materialization calls `src.diagnostics.signed_teacher_v1.build_signed_teacher_row` directly. Diagnostic and formal inference call the same `src.diagnostics.signed_selector_v1.select_row_without_gold` helper. This is an additive refactor; old diagnostic artifact schemas remain readable.

## 6. Baseline frozen contract

The registry includes `no_evidence`, `naive_topm`, `bge`, `infogain_fever`, `rag_cbwdm`, `rag_cbwdm_signed_v1`, and optional diagnostic-only `signed_gate_oracle`.

- Deterministic methods run once: no-evidence, Naive top-4, and the current formal BGE configuration.
- InfoGain runs seeds 13/21/42 from the current pilot reference configuration.
- Signed-v1 runs seeds 13/21/42.
- Old RAG-CBWDM runs 13/21/42 only if its frozen/pilot-selected training contract supports retraining those seeds. A seed is never fabricated from a copied checkpoint.
- The new config marks InfoGain and old RAG reference parameters as `reference_config_from_pilot=true` and `formal_optimum_claimed=false` until the server calibration artifacts prove otherwise.

The exact selected InfoGain/old-RAG candidate parameters and fingerprints are a server-side blocker because their manifests are not present locally. The runbook requires auditing them before Gate D; no baseline parameter was guessed or changed.

## 7. Training scale and seed protocol

Signed teacher/training consumes only the existing 5000-row train_core retrieval/posterior pair. It does not invoke Qwen for train_core. Formal teacher materialization verifies the input checksums and materializes the same diagnostic trajectory with a complete formal manifest.

Training seed is part of the contract, training manifest, and checkpoint fingerprint. Resume requires the identical contract and checkpoint tree SHA. Seeds 13, 21, and 42 therefore produce distinct requested fingerprints; an identical split/model/seed resumes the same completed checkpoint rather than retraining over it.

## 8. Fairness contract

`src/preformal/fairness.py` and `scripts/preformal/28_audit_fairness.py` fail closed unless:

- retrieval query SHA equals the clean preformal split SHA;
- shared posterior input SHA equals the retrieval output SHA;
- every method selection consumes either that retrieval SHA or the shared posterior SHA;
- every method has exactly the same query ID set;
- every evaluation consumes the corresponding selection SHA;
- every evaluation uses `split=preformal_eval`;
- generator SHA, prompt hash, and verbalizer hash equal the shared posterior contract;
- every prediction ID set exactly equals the clean split ID set.

The generic retrieval/posterior/evaluation scripts received only additive `preformal_eval` role support and stronger config/model provenance. Existing roles and production behavior are unchanged.

## 9. Statistical comparison

`scripts/preformal/29_summarize_results.py` emits:

- `PREFORMAL_SIGNED_V1_RESULTS.json`
- `PREFORMAL_SIGNED_V1_RESULTS.csv`
- `PREFORMAL_SIGNED_V1_RESULTS.md`
- `PREFORMAL_PAIRED_COMPARISONS.json`
- `PREFORMAL_PAIRED_COMPARISONS.md`

The per-seed table contains accuracy, macro-F1, SUPPORTS/REFUTES recall and F1, average selected documents, average evidence characters, average original BM25 rank, gold-evidence any-hit, and complete flattened-union coverage. Learned methods receive mean/sample-SD summaries for accuracy, macro-F1, and average documents.

Paired analysis aligns an identical ID set, reports the four correctness cells, exact two-sided McNemar/binomial p-value, and deterministic paired-bootstrap 95% percentile CIs for accuracy and macro-F1 differences. Default bootstrap seed is 130421 with 10,000 samples.

The code reports results only; it has no path that updates calibration or frozen parameters.

## 10. Signed post-hoc diagnostics

`scripts/preformal/30_signed_posthoc.py` reports selected positive/zero/negative alignment ratios, zero-document ratio, document-count distribution, stop reasons, evidence coverage, and—only if the optional oracle selection is supplied—step-0 agreement and selected-set Jaccard. Output always declares `uses_gold_for_selection=false` and `posthoc_only=true`.

The oracle is deliberately outside the deployable Gates and may run only after Gate G is frozen.

## 11. Files changed

Added:

- `configs/fever2_server_preformal_signed_v1.yaml`
- `src/preformal/__init__.py`
- `src/preformal/registry.py`
- `src/preformal/splits.py`
- `src/preformal/fairness.py`
- `src/preformal/statistics.py`
- `scripts/preformal/24_build_preformal_eval.py`
- `scripts/preformal/25_materialize_signed_v1_teacher.py`
- `scripts/preformal/26_train_signed_v1.py`
- `scripts/preformal/27_select_signed_v1.py`
- `scripts/preformal/27a_select_no_evidence.py`
- `scripts/preformal/28_audit_fairness.py`
- `scripts/preformal/29_summarize_results.py`
- `scripts/preformal/30_signed_posthoc.py`
- `tests/test_preformal_signed_v1.py`
- `SIGNED_V1_PREFORMAL_IMPLEMENTATION_REPORT.md`
- `SIGNED_V1_PREFORMAL_EXPERIMENT_RUNBOOK.md`

Additively modified:

- `scripts/02_retrieve_bm25.py`
- `scripts/03_compute_label_posteriors.py`
- `scripts/07_eval_rag_classification.py`
- `scripts/diagnostics/select_signed_selector_v1.py`
- `src/diagnostics/signed_selector_v1.py`

## 12. Production impact

- old `rag_cbwdm` changed? **NO**
- `RAG-CBWDM.tex` changed? **NO**
- held_out_test consumed for modeling/evaluation? **NO**
- current pilot validation 500 modified? **NO**
- existing formal/diagnostic artifacts overwritten? **NO**
- existing signed-v1 mathematics changed? **NO**

## 13. Tests

Coverage includes the requested boundaries:

1. preformal/pilot ID overlap zero;
2. normalized-claim overlap zero;
3. preformal/held-out overlap zero;
4. preformal rejected as calibration input;
5. preformal rejected as frozen-parameter source;
6. formal teacher equals diagnostic trajectory;
7. formal inference shares diagnostic inference implementation;
8. inference API has no gold/alignment input;
9. seed changes requested fingerprint;
10. same split/model/seed fingerprint and resume contract are stable;
11. fairness resolves one canonical retrieval SHA for every method;
12. generator/prompt/verbalizer mismatches block fairness;
13. paired comparison requires exactly identical IDs;
14. held-out references in preformal training/calibration fail closed.

Latest local result before final handoff: `104 passed, 2 subtests passed` under pytest. Final compileall, pytest, unittest, and `git diff --check` results are recorded in the handoff response.

## 14. Estimated compute by stage

These are workload counts/capabilities, not invented wall-clock estimates.

- Gate A: CPU JSON parsing/hashing over official train/dev and deterministic partition reconstruction; no model.
- Gate B: BM25 top-20 retrieval for actual preformal row count; CPU/index I/O; no Qwen; existing compatible index only.
- Gate C: Qwen query-only plus 20 single-document posteriors per query, up to `21 × N` prompts; GPU recommended.
- Gate D: no train-core Qwen. Signed teacher is CPU/NumPy over existing 5000 posteriors; signed and InfoGain reranker training use MiniLM for three seeds; old RAG seeds are conditional on available frozen contract.
- Gate E: deterministic/BGE/learned selection. Signed state-aware worst case scores `20+19+18+17 = 74` candidate states per query and seed.
- Gate F: one generator evaluation prompt per method/seed/query; GPU recommended; shared generator/prompt/verbalizer.
- Gate G: CPU JSON/statistics; paired bootstrap is CPU-bound.
- Gate H: optional diagnostic oracle; run once only after deployable artifacts freeze.

## 15. Blockers

1. The authoritative server split manifest and current run/artifact manifests are absent locally. Gate A must report the actual full/pilot/preformal row counts and source SHAs before retrieval.
2. Current calibration candidates/manifest/frozen parameter files are absent locally. Exact InfoGain and old-RAG reference candidate fingerprints, parameters, and checkpoint paths must be audited on the server before Gate D.
3. The current server run manifest must supply the checksum-compatible Lucene index path/fingerprint. The runbook never rebuilds it automatically.
4. Old RAG-CBWDM seed21/42 stability is conditional. If only seed13 exists and the exact selected training contract cannot be safely replayed, report `old rag_cbwdm stability unavailable / pending`.

## 16. Scientific interpretation boundary

The 500-query pilot validation has been used for method development. `preformal_eval` is intended to be clean at this run's start and is a formal-grade internal benchmark, not the final paper held-out result. If signed-v1 or any baseline parameter is changed after inspecting preformal outcomes, `preformal_eval` becomes development data. Final conclusions require a separately frozen algorithm followed by one evaluation on untouched official-dev `held_out_test`.

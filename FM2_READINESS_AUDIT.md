# FM2 readiness audit

Audit baseline: branch `feature/fever-formal-readiness`, commit `3d3aa4fb94cb3496e45309c003618f0bf91aeaa6`. This records the read-only audit performed before the FM2 implementation. Unrelated pre-existing Markdown deletions in the working tree were not changed.

## Executive answer

At the audited HEAD, FM2 could not run directly. No `fm2`, `fool_me_twice`, `fool-me-twice`, or dataset-adapter implementation existed. Naive selection, BGE scoring/selection, label-logit scoring, metrics, the signed-v1 mathematical helper, and most InfoGain teacher/training logic were already schema-generic. The blockers were data preparation/candidate provenance, the FEVER-named prompt API and hashes, FEVER-oriented InfoGain naming, and signed preformal entrypoints that required `train_core`/`preformal_eval`.

The minimum safe implementation is one FM2 adapter plus small generalizations of the existing entrypoints. It must not fork the selector, posterior, evaluator, BGE, Naive, or signed-v1 algorithms.

## Source audit and real schema

The authoritative source is [google-research/fool-me-twice](https://github.com/google-research/fool-me-twice), pinned here to commit `d9db753e5acf91c0d9bf543db327ab655661eb94`. The three official JSONL files were inspected directly, not inferred from dataset descriptions.

Every inspected row has exactly these fields:

`category`, `correct_votes`, `gold_evidence`, `id`, `label`, `retrieved_evidence`, `text`, `total_likes`, `total_votes`, `wikipedia_page`.

Both evidence arrays contain objects with exactly `section_header` and `text`. Official labels are already `SUPPORTS` and `REFUTES`; the adapter must nevertheless validate and record the explicit mapping `{SUPPORTS: SUPPORTS, REFUTES: REFUTES}` and retain `original_label`.

Verified source facts:

| Split | Rows | Pages | SUPPORTS | REFUTES | Raw SHA-256 |
|---|---:|---:|---:|---:|---|
| train | 10,419 | 1,811 | 5,126 | 5,293 | `1f47e035650aa5734301afb36a94afa610af73ac908fd4c4009e281b9956a311` |
| dev | 1,169 | 209 | 596 | 573 | `eeb36a4757fb86412f1d7ad5197ffd6ffa837c1cf7fb3c0067d82d0b65257229` |
| test | 1,380 | 234 | 681 | 699 | `4a45aa8edd45ea8ff54eb66be06e8c6113873c22496b209925dd5b529ce00721` |

`wikipedia_page` intersections are zero for train/dev, train/test, and dev/test. The adapter must fail closed if that invariant changes.

## Existing module readiness

### Reusable unchanged

- `src/label_logits.py`: dataset-neutral label verbalizer scorer.
- `src/metrics.py`: already emits accuracy, macro-F1, per-class scores, confusion matrix, document/evidence averages, and selection distribution.
- `scripts/08_select_naive_topm.py`: consumes the shared retrieval schema; only server commands must pass `--min-docs 0` because a few FM2 pools have fewer than four items.
- `src/baselines/bge_reranker.py` and `scripts/12_select_bge_reranker.py`: model/scoring logic is dataset-neutral despite a FEVER-oriented CLI description/template version name.
- `src/diagnostics/signed_teacher_v1.py`: signed alignment, admissibility, Theta marginal greedy, and supervision are dataset-neutral and must remain the one implementation.
- `src/diagnostics/signed_selector_v1.py`: inference API has no gold/direction/alignment argument.
- `src/baselines/infogain_fever.py`: the actual teacher equation, grouping, loss, and pointwise input consume generic posterior/candidate fields.
- `src/selection_schema.py`, `src/baselines/common.py`, and evaluation metrics/manifest helpers.

### Generalize, do not copy

- `src/prompts.py`: retain `build_fever_prompt` and its exact v1 hash, add a dataset registry/wrapper and a distinct FM2 prompt version.
- `scripts/03_compute_label_posteriors.py` and `scripts/07_eval_rag_classification.py`: select the registered prompt by `config.dataset`; retain FEVER defaults for direct function callers.
- `scripts/preformal/25_materialize_signed_v1_teacher.py`, `26_train_signed_v1.py`, `27_select_signed_v1.py`, and `27a_select_no_evidence.py`: parameterize split role with FEVER-compatible defaults.
- InfoGain scripts: import through a generic adapter name and permit a dataset-specific output method name while keeping `infogain_fever` as the FEVER default.

### FEVER-specific and intentionally untouched

- FEVER raw conversion, FEVER Wikipedia corpus creation, Lucene index building, formal FEVER split publication/calibration/readiness, and `run_fever_cbwdm.py` remain FEVER-only.
- Existing FEVER configs, artifacts, prompt version, prompt bytes, signed-v1 contract, results, and checkpoint manifests must not be rewritten.

## Candidate-pool audit

### A. Official `retrieved_evidence`

Official packaging code maps the claim record's `evidence` list to `retrieved_evidence` and separately maps `gold` to `gold_evidence`. Therefore construction does not read the gold label or gold-evidence list. It is suitable as a shared gold-free selector input under the official closed-page task.

It is not an open-domain retrieval result: the pool is tied to the known `wikipedia_page` from which the game claim was authored. Results must be described as the `fm2_official_closed_page_v1` protocol, not as deployable open-web retrieval.

Pool distribution:

| Split | Min | Max | Mean | Rows `<4` | Rows `<20` |
|---|---:|---:|---:|---:|---:|
| train | 1 | 12 | 10.357 | 48 | 10,419 |
| dev | 2 | 12 | 10.308 | 3 | 1,169 |
| test | 2 | 12 | 10.383 | 2 | 1,380 |

Thus top4 is feasible as a maximum-budget convention, with fewer documents used when the pool itself is smaller. Top20 is impossible and must not be claimed. Exact gold-evidence text is present for all dev/test rows and 10,418/10,419 train rows. This is expected for the closed-page evidence pool and makes gold coverage high, but gold identity must remain in a separate diagnostic artifact.

### B. Independent Wikipedia BM25

Open-domain BM25 would require a corpus snapshot compatible with FM2 page titles and section text. The FEVER Lucene index cannot be reused without proving snapshot and document-ID compatibility; the audited repository contains no such proof. Building a new snapshot/index substantially expands scope and is not required for the first controlled cross-dataset run.

Decision: first round uses the official closed-page pool, checksum-frozen and shared by every method. A later open-domain protocol requires a separately versioned FM2 Wikipedia corpus/index and must not be compared as if it were the same first-stage pool.

## Leakage audit

- Teacher generation may read FM2 train labels only.
- Signed and InfoGain selector inference APIs must not receive gold/direction features. Selection output may carry the label solely for the common evaluator, as in the frozen FEVER schema.
- Gold evidence is written to `*_gold_diagnostics.jsonl`, never to `*_official_pool.jsonl`.
- Dev may calibrate the InfoGain filter threshold; reports must distinguish this from the fixed 0.5 cross-dataset reference.
- Test is not downloaded/prepared by default and may be scored only after model seeds, prompt/verbalizers, candidate SHA, document budget, and threshold policy are frozen.
- Page disjointness is enforced across every set of splits prepared together.

## Minimal implementation outcome

The implementation follows the audit: `src/datasets/fm2.py`, download/prepare entrypoints, two FM2 configs, generic prompt wrappers, generic InfoGain import/method naming, and split parameters on the existing signed/no-evidence entrypoints. No signed-v1 equation, frozen constant, FEVER config, FEVER result, or FEVER artifact is modified.

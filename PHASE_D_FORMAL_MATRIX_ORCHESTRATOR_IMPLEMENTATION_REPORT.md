# Phase D Formal Matrix Orchestrator Implementation Report

## 1. Summary

Phase D adds a thin formal-v2 matrix planner and sequential server executor around the existing workers. The planner expands Dataset x Generator x Method, and Seed only for learned methods, into a validated DAG. It reuses the Phase A experiment identity paths, Phase B atomic generator contract, and Phase C formal registry rather than reimplementing any scientific method.

The implementation includes a bounded FEVER development smoke config, machine-readable plan manifests, shell-safe Linux commands, deterministic semantic plan fingerprints, fail-closed held-out gating, focused matrix tests, and a minimal BGE worker extension for verifying a frozen local model SHA.

No model was run. No formal matrix was executed. No held-out split was opened.

## 2. Git Baseline

- Required and observed HEAD: `b8267d74a9b4b4bab63a8573108bba592200b436`
- Required and observed branch: `feature/fever-formal-readiness`
- Platform: Windows, repository `C:\Users\wenhao\Desktop\CBWDM\rag_cbwdm`
- The three allowlisted pre-existing dirty paths were not modified by Phase D. Their final SHA-256 values exactly match the values recorded at the hard gate.

## 3. Files Changed

Modified:

- `scripts/12_select_bge_reranker.py`

Added:

- `src/formal_matrix.py`
- `scripts/run_formal_matrix.py`
- `configs/formal/fever_qwen15_development_smoke.yaml`
- `tests/test_formal_matrix.py`
- `PHASE_D_FORMAL_MATRIX_ORCHESTRATOR_IMPLEMENTATION_REPORT.md`

The BGE worker change is backward-compatible. `--model-sha256` is optional; when supplied it requires a local artifact and verifies its recursive SHA before scoring. The value also enters the BGE score-cache contract and model metadata.

## 4. Orchestrator Architecture

`src/formal_matrix.py` is the pure planning/execution layer. It resolves the Phase C dataset and method registry, previews or validates Phase B generator manifests, constructs nodes with actual worker CLI arguments, validates the DAG, and emits a semantic plan fingerprint.

`scripts/run_formal_matrix.py` is the command entrypoint. It supports mutually exclusive `--dry-run` and `--execute`, optional generator/method/seed subsets, a separate server project root and artifact root, optional BGE/held-out freeze inputs through the matrix config and freeze manifest, and atomic plan/command-file writes.

The executor is intentionally simple: it validates the plan, traverses it topologically, validates external inputs, validates existing generator manifests, and launches each worker synchronously with failure propagation.

## 5. Matrix Semantics

The Phase C registry remains authoritative. The planned main-table methods are exactly:

1. `no_evidence`
2. `retrieval_topk`
3. `bge`
4. `infogain`
5. `rag_cbwdm_signed_v1`

Canonical Ours is `rag_cbwdm_signed_v1`. Deterministic methods have no seed dimension. Learned methods expand only over the requested legal formal seeds. Matched-main evaluation is generated once per generator/method/seed row; diagnostic and transfer experiments are not part of this plan.

## 6. DAG Model

Each node records:

- deterministic `node_id`
- stage, dataset, generator, method, seed, and split
- named inputs and outputs
- explicit dependencies
- exact argv command
- `formal_v2`
- reuse eligibility
- expected status
- execution policy

Validation fails on duplicate node IDs, unknown dependencies, cycles, and output-path collisions. Node IDs contain stable logical IDs, never absolute model paths.

## 7. Shared Stages

The dataset config and retrieval candidate pools are modeled as external shared dependencies. No Evidence selection, Retrieval Top-k selection, and BGE scoring/selection each have one generator-independent node per compatible dataset/split plan. Their paths are referenced by every generator-specific evaluation instead of being duplicated.

FEVER uses registry protocol `fever_bm25_v1` and display `BM25 Top-k`. FM2 uses `fm2_official_closed_page_v1` and display `Official-pool Top-k`; the planner does not label FM2 as BM25 and implements no FM2 BM25 stage.

## 8. Generator-Specific Stages

Each generator has its own atomic generator manifest, train/evaluation posteriors, InfoGain teacher/training/selection branch, signed-v1 teacher/training/selection branch, and all five evaluation branches. Learned nodes depend only on artifacts under the same stable `generator_id`. Evaluation always lives under and references the evaluation generator's manifest, including for shared deterministic selections.

## 9. Learned-Method Seed Expansion

Development smoke explicitly uses seed 13. Full development defaults to Phase C formal seeds 13, 21, and 42 when no subset is supplied. Both InfoGain and signed-v1 receive the same selected legal seed set. Seed values outside the registry set fail planning. Held-out plans must match the complete seed set recorded by the signed freeze.

## 10. Artifact Hierarchy

New outputs are rooted under `artifacts/formal_v2/<dataset_id>/`. Shared retrieval and deterministic selections live under `shared/`. Generator-conditioned posteriors, learned checkpoints/selections, and matched-main evaluation outputs live under stable generator IDs.

Phase A helpers construct dataset, generator, posterior, learned method/seed, and evaluation roots. Existing historical formal-v1 paths and artifacts are neither migrated nor modified.

## 11. Development / Held-Out Separation

Development is the default. A development plan must target a registry development role and has `formal_result_claim: false`. Merely putting a held-out profile in a config does not authorize it; explicit `--held-out` is required.

Before any held-out worker can launch, planning validates the Phase C freeze/sign-off contract and then matches the actual plan against its dataset, final split, formal registry, retrieval protocol, generator registry fingerprint, Git commit, config SHA, prompt hash, verbalizer hash, formal seeds, Top-k, evidence budget, and BGE model ID/revision/SHA. Failure is closed and occurs while building the plan.

## 12. Smoke Profile

`configs/formal/fever_qwen15_development_smoke.yaml` defines the bounded engineering smoke:

- FEVER `fever_binary_v2`
- Qwen2.5-1.5B-Instruct stable generator ID
- all five formal main-table methods
- `train_core` to `validation`
- learned seed 13 only
- existing `configs/fever2_server_smoke.yaml` limits: train 200, development 100

The tested plan contains 20 DAG nodes and five discoverable result rows. It is labeled development-only and cannot claim a formal final result.

## 13. Full Development Profile

`full_development` accepts an explicit legal learned-seed subset or defaults to 13/21/42 from the registry. It uses a registry development split and remains non-final. The same compact matrix config can express additional generators without creating a config per matrix cell.

## 14. Generator Contract Integration

A generator spec accepts a stable `generator_id` and either a config or an existing generator manifest. Config-backed specs use the Phase B manifest builder to resolve model/revision, tokenizer, dtype, device map, trust setting, prompt hash, and verbalizer hash without requiring the model artifact locally. Existing manifests are validated through the Phase B validator.

Stable IDs are separate from model paths. Known model/ID conflicts fail in the Phase A identity builder. A two-generator focused test proves distinct learned branches with shared compatible deterministic selections.

## 15. BGE Freeze Handling

The orchestrator never chooses base versus large. A matrix may directly supply `model_id`/`bge_model_id`, revision, SHA, and model path, or supply a local identity manifest that resolves those fields. Development may use an explicit existing model with a clear `development_only` marker and unresolved final revision/SHA; the plan records that unresolved decision.

Held-out planning requires an exact signed BGE model ID/revision/SHA match and no unresolved BGE fields. The worker's optional SHA guard prevents a named local path from silently resolving to different bytes.

## 16. Execution Plan Manifest

The plan includes both required schema versions, creation time, Git state, registry version/fingerprint, dataset/retrieval identities, generator identities, generator registry fingerprint, methods, seed policy, profile, split role, held-out/final-claim state, limits, retrieval inputs, BGE contract, nodes, dependency edges, artifact roots, explicit result index, unresolved freeze decisions, and the plan fingerprint.

The result index directly maps dataset, generator, method, seed, split, experiment type, node ID, and evaluation manifest. A future collector therefore needs no fuzzy filename matching.

The semantic fingerprint excludes timestamps, server-root relocation, generated commands, and output locations. It includes registry, dataset, generator/config identities, retrieval input references, methods, seeds, profile/splits/limits, BGE identity, Git commit, DAG semantics, result semantics, and unresolved freeze decisions. Tests prove determinism, server-root independence, and sensitivity to seed and retrieval-input changes.

## 17. Dry-Run Behavior

`--dry-run` builds and validates the complete DAG, atomically writes the JSON plan, optionally writes shell-safe Linux commands, and prints commands. It never calls the executor. The focused test injects an executor sentinel and proves it receives zero calls.

No repository experiment artifact was produced during Phase D; the CLI dry-run test writes only into pytest's temporary directory.

## 18. Resume / Reuse Validation

File existence alone is not considered proof of reusable formal-v2 identity. External generator manifests and existing config-backed generator manifests are checked by the Phase B validator. Worker nodes receive their existing formal-v2 identity flags, manifest inputs, and `--resume`; those workers validate their own provenance/cache contracts before reuse. The BGE cache now includes the optional frozen model SHA.

Mismatched artifacts are not deleted or overwritten automatically. A worker validation failure propagates and stops sequential execution.

## 19. Failure Semantics

Planning or execution fails closed for unknown datasets or methods, dataset/protocol mismatch, generator identity conflicts, illegal seeds, final splits in development mode, held-out without valid freeze/sign-off, freeze-to-plan mismatches, unresolved held-out BGE identity, missing external dependencies, invalid generator manifests, duplicate semantic nodes, output collisions, dependency cycles, and worker failures.

## 20. Tests Run

- `python -m py_compile` on all four changed Python files
- Focused: `tests/test_formal_matrix.py`
- Related: `tests/test_experiment_identity.py`, `tests/test_atomic_generator_binding.py`, `tests/test_formal_registry.py`, `tests/test_preformal_signed_v1.py`, `tests/test_fever_formal_protocol.py`, `tests/test_fever_baselines.py`, `tests/test_fm2_support.py`
- Full: `python -m pytest -q`
- Final whitespace audit: `git diff --check`

The repository-local Windows interpreter `.venv\Scripts\python.exe` was used because this host has no global `python` command.

## 21. Test Results

- Py compile: PASS
- Focused Phase D: PASS, 21 tests
- Related Phase A/B/C and protocol suites: PASS, 110 tests
- Full suite: PASS, 236 tests and 2 subtests
- `git diff --check`: PASS

All tests are source/unit tests. No live model calls or formal worker execution occurred.

## 22. Remaining P0

The final BGE model choice and immutable identity, production generator registry, final server config freeze, and genuine Phase C runtime/human held-out sign-off remain intentionally unresolved. Held-out planning/execution is blocked until those external release inputs exist and match the plan.

## 23. Remaining P1

Run the committed development smoke dry-run on the Linux server, inspect its plan and commands, provision/validate the referenced retrieval and model artifacts, then execute the development smoke. Add production generator specs only after their exact model/revision/tokenizer identities are chosen.

## 24. Risks / Limitations

Execution is sequential by design and was not run in Phase D. Retrieval inputs are external prerequisites; the plan records their references but cannot hash server-only files on this Windows planning host. Scientific reuse beyond generator manifests remains delegated to the existing worker manifest validators. Final publication aggregation is intentionally out of scope.

## HANDOFF_TO_CHATGPT

STATUS: COMPLETED
GIT_BASELINE: b8267d74a9b4b4bab63a8573108bba592200b436

FILES_CHANGED: scripts/12_select_bge_reranker.py
FILES_ADDED: src/formal_matrix.py; scripts/run_formal_matrix.py; configs/formal/fever_qwen15_development_smoke.yaml; tests/test_formal_matrix.py; PHASE_D_FORMAL_MATRIX_ORCHESTRATOR_IMPLEMENTATION_REPORT.md

ORCHESTRATOR_SCHEMA_VERSION: rag_cbwdm_formal_matrix_orchestrator.v1
PLAN_MANIFEST_SCHEMA_VERSION: rag_cbwdm_formal_matrix_plan.v1

DRY_RUN_STATUS: PASS; DAG and command plan generated in tests with zero executor calls
EXECUTION_MODE_STATUS: IMPLEMENTED_SEQUENTIAL_NOT_RUN

MAIN_TABLE_METHODS: no_evidence,retrieval_topk,bge,infogain,rag_cbwdm_signed_v1
CANONICAL_OURS: rag_cbwdm_signed_v1

FEVER_SMOKE_PROFILE_STATUS: PASS; 20 nodes, 5 result rows, development-only, seed 13
FM2_PLAN_STATUS: PASS; official-pool protocol, no FM2 BM25

GENERATOR_INDEPENDENT_REUSE_STATUS: PASS; No Evidence, Retrieval Top-k, and BGE selections shared
GENERATOR_CONDITIONED_BRANCHING_STATUS: PASS; posteriors, teachers, learned selectors, and evaluations separated by generator

FORMAL_SEED_EXPANSION_STATUS: PASS; smoke subset supported and full development expands 13/21/42

HELDOUT_GATE_STATUS: PASS; explicit opt-in plus signed freeze/sign-off and exact plan matching required
BGE_FREEZE_STATUS: DEVELOPMENT_EXPLICIT_SUPPORTED; HELDOUT_UNRESOLVED_BLOCKED

PLAN_FINGERPRINT_STATUS: PASS; deterministic, relocation-independent, semantic-change-sensitive
OUTPUT_COLLISION_GUARD_STATUS: PASS
RESUME_VALIDATION_STATUS: PASS; Phase B generator validation plus worker-owned formal provenance validation

PY_COMPILE: PASS
FOCUSED_TESTS: PASS; 21 passed
RELATED_TESTS: PASS; 110 passed
FULL_SUITE: PASS; 236 passed, 2 subtests passed

METHOD_MATH_CHANGED: NO
PROMPT_BYTES_CHANGED: NO
METRIC_DEFINITIONS_CHANGED: NO
HISTORICAL_ARTIFACTS_CHANGED: NO

REMAINING_P0: Final BGE/generator/config freeze and genuine runtime-human held-out sign-off
REMAINING_P1: Linux server development dry-run review, artifact provisioning, then bounded development smoke execution

RECOMMENDED_NEXT_STEP: Commit/review Phase D, then run scripts/run_formal_matrix.py --dry-run with the FEVER smoke config on the Linux server and inspect the emitted plan before any --execute use.

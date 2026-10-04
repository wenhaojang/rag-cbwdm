# FM2 Formal-v2 Config Readiness Report

## Scope

This change adds FM2 server configuration only. It does not change algorithm
source, formal-v2 schemas, held-out policy, or any existing historical dirty
file. The worker-facing dataset alias remains `fm2`; the stable formal dataset
and retrieval identities are both `fm2_official_closed_page_v1`.

## Added files

- `configs/fm2_qwen05_full_development.server.yaml`
- `configs/fm2_qwen15_full_development.server.yaml`
- `configs/fm2_qwen7_full_development.server.yaml`
- `configs/fm2_mistral7_full_development.server.yaml`
- `configs/fm2_qwen15_development_smoke.server.yaml`
- `configs/formal/fm2_qwen15_development_smoke.seed13.matrix.server.yaml`
- `tests/test_fm2_formal_configs.py`
- `FM2_FORMAL_V2_CONFIG_READINESS_REPORT.md`

The dedicated smoke server config is required by the existing planner design:
limits and shared method settings come from `dataset_config`, while the
generator branch uses its own generator config. This preserves unlimited
full-development configs and reuses the established FEVER smoke pattern
without adding a schema field. Smoke limits are `train_core=200` and
`validation=100`.

## Generator identities

| Generator ID | Family | Server model path | Posterior batch size |
|---|---|---|---:|
| `qwen2.5-0.5b-instruct` | `qwen2.5` | `/root/models/Qwen2.5-0.5B-Instruct` | 64 |
| `qwen2.5-1.5b-instruct` | `qwen2.5` | `/root/models/Qwen2.5-1.5B-Instruct` | 16 |
| `qwen2.5-7b-instruct` | `qwen2.5` | `/root/models/Qwen2.5-7B-Instruct` | 8 |
| `mistral-7b-instruct-v0.3` | `mistral` | `/root/models/Mistral-7B-Instruct-v0.3` | 8 |

All configs retain the existing FM2 SUPPORTS/REFUTES and A/B prompt/verbalizer
contract. Generator-manifest construction was exercised for all four configs.

## Frozen method settings

- Retrieval Top-k: official candidate-pool source order, `top_m=4`,
  `min_docs=0`, so short pools use every available candidate.
- BGE: `BAAI/bge-reranker-large`, local path
  `/root/models/bge-reranker-large`, `revision=null`, directory SHA-256
  `b01f9eac1483006e54a902eb0d272f8738e94a98f9f36e4a2b2e3b3eb8c335a2`,
  `top_m=4`, `min_docs=4`, local files only.
- InfoGain: positive/negative quantiles `0.75/0.25`, `beta=0.75`,
  `top_m=4`, `min_docs=2`, vectorized rank-loss implementation, `epochs=3`
  to match the FEVER formal-v2 training budget, seed 13.
- Ours: `rag_cbwdm_signed_v1`, `top_m=4`, `min_docs=0`,
  `score_threshold=0.0`, `runtime_implementation=block_v1`,
  `forward_batch_size=32`, seed 13.

## Smoke matrix and retrieval bindings

The only new matrix is the Qwen2.5-1.5B `development_smoke` matrix with the
five main-table methods and learned seed 13. It binds only `train_core` and
`validation` through the formal-v2 FM2 object contract.

- Prepare manifest:
  `/root/experiments/rag_cbwdm/formal_v2_external_inputs/fm2_official_closed_page_v1/prepared_v2/fm2_prepare.manifest.json`
  (reported SHA-256 `39c5d528dba72946ade897aed7364dd97bcbb3381ad1f76a90f48339f86a4be0`).
- Train pool:
  `/root/experiments/rag_cbwdm/formal_v2_external_inputs/fm2_official_closed_page_v1/prepared_v2/fm2_train_official_pool.jsonl`
  (10,419 rows; reported SHA-256
  `54a927c3eebc1645949cb0af21b84c74cbcb99c0a0746802dfac2f96968093e4`).
- Validation pool:
  `/root/experiments/rag_cbwdm/formal_v2_external_inputs/fm2_official_closed_page_v1/prepared_v2/fm2_dev_official_pool.jsonl`
  (1,169 rows; reported SHA-256
  `f371600059769a245db8696b97cc55a176e523a225518410076308ef68ff53c3`).

Static inspection found no FEVER/BM25/Pyserini command or protocol leakage and
no held-out/test binding. No full-development or held-out FM2 matrix was added.

## Validation

- Focused config tests: `7 passed in 13.77s`.
- FM2/formal related tests: `81 passed in 6.57s`.
- Full suite: `293 passed, 2 subtests passed in 27.34s`.
- Local real-artifact dry-run: not run, because the authoritative `/root/...`
  inputs are server-only and no substitute artifacts were fabricated.

Before the real server smoke, verify the checked-in Git commit, all three
reported FM2 artifact hashes and row counts, the BGE directory hash, local
model/tokenizer loadability and A/B single-token verbalizers. Then build the
matrix plan on the server so the authoritative FM2 manifest validator checks
both retrieval bindings before any worker executes.

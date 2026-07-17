# Persistent BM25 and FEVER limit fix report

## Outcome

The production server profiles now select a reusable Pyserini/Lucene on-disk BM25
backend. Index construction is an independent `index` stage
(`build_bm25_index` alias). Retrieval opens the completed index, restores
`doc_id/title/text` from Lucene stored raw JSON, and atomically publishes JSONL plus
a completed manifest. The in-memory `rank_bm25` implementation remains available
only as `memory_rank_bm25` with a 50,000-document default safety cap.

The previous OOM was caused by retaining the complete sentence corpus, a token list
for every document, and rank_bm25 posting/statistics Python objects at the same
time. Materializing token dictionaries or adding swap would preserve that scaling
failure. The new main path streams a temporary JSONL collection into Lucene and
does not construct a corpus-wide Python token object graph during search.

## Linux Pyserini 2.3.0 completion-validation follow-up

A real Linux integration with Python 3.12, Java 21, Pyserini 2.3.0, and Lucene
10.4 successfully built and probed a 200,000-document index. Direct
`LuceneSearcher` open/search/raw-document recovery also passed. The index stage
nevertheless returned 1 after publishing `index_manifest.json` with
`completed=true`.

The failing call path was:

```text
scripts/02a_build_bm25_index.py:main
  -> build_index
  -> atomic_write_json(completed=true)
  -> validate_index
  -> index_inventory
  -> ValueError("Index has no non-empty Lucene files")
```

The root cause was the validator condition
`any(item["size_bytes"] <= 0 for item in inventory)`. It rejected the whole index
when any inventory entry was empty, including the legal `write.lock`, while the
build path had only required a non-empty inventory. The error text therefore
incorrectly said there were no non-empty files even though Lucene data and
`segments_1` were present.

The patch makes `index_inventory()` the single scanner used by build and resume,
and makes `validate_lucene_inventory()` the single structural rule:

- a non-empty `segments_N` file must exist;
- at least one other non-empty data file must exist;
- `write.lock` may be inventoried and may be zero bytes, but never counts as data;
- Lucene codec extensions are not allow-listed, so Lucene 10.4 names such as
  `_0_Lucene104_0.doc`, `.pos`, `.tim`, `.tip`, and `_0_Lucene90_0.dvd` work;
- completed manifest, self-consistent fingerprint/contract, positive and matching
  document count, exact canonical inventory, `probe_passed=true`, and successful
  `LuceneSearcher` open are all required.

During build, structural validation and the live probe now finish first. The
post-probe canonical inventory is captured, and only then is `completed=true`
atomically published. There is no fallible second validator after publication.
During `--resume`, the completed manifest is validated with the same inventory
rules and the indexer subprocess is not invoked.

## Changed files

- `src/retrieval/{base,memory_bm25,pyserini_bm25}.py` and compatibility exports
- `scripts/02a_build_bm25_index.py`, `scripts/02_retrieve_bm25.py`
- `scripts/00_prepare_fever.py`, `scripts/01_prepare_fever_corpus.py`
- `scripts/run_fever_cbwdm.py`, `scripts/check_environment.py`
- three `configs/fever2_server_*.yaml` profiles
- `requirements-retrieval.txt`
- `tests/test_disk_bm25_and_limits.py`
- `DISK_BM25_SERVER_RUNBOOK.md` and this report

Pre-existing deletions in the worktree, including `SERVER_RUNBOOK.md`, were not
restored or overwritten. The new runbook satisfies the task's “update or add”
documentation requirement without discarding those user changes.

## Data flow and shared artifacts

```text
raw FEVER
  -> run-specific prepared train/dev claims + prepare manifests
  -> shared sentence corpus keyed by corpus config/limit
  -> shared Lucene index keyed by corpus/retrieval config
  -> run-specific atomic train/dev retrieval
  -> posterior -> teacher -> cross-encoder -> selection -> evaluation
```

Prepared claims are now written inside each run's artifacts, so runs with different
limits cannot overwrite `outputs/processed/fever2_{split}.jsonl`. Corpus and index
are under `<output-root>/_shared`; exact corpus content is verified by SHA-256 in
the index manifest before reuse.

## CLI, config, and limit semantics

- `--limit N` on prepare now stops after N valid mapped rows are emitted.
- `--raw-limit N` independently caps raw rows scanned; the first reached condition
  is recorded as `stop_reason`.
- runner adds `--train-limit`, `--dev-limit`, and `--corpus-limit`.
- resolution order is explicit split CLI, profile limit, generic `--limit`, then
  `None`.
- resolved train/dev/corpus/raw limits are included in the run fingerprint and run
  manifest.
- corpus limit counts sentence documents rather than wiki pages. Limited and full
  corpora have distinct shared keys. Formal profile rejects a non-null corpus
  limit.
- FEVER-2 drop manifests record `NOT ENOUGH INFO`; FEVER-3 retains it through its
  existing label map.

Server retrieval config is:

```yaml
retrieval:
  backend: pyserini_lucene
  top_n: 10  # or 20
  bm25: {k1: 0.9, b: 0.4}
  index: {path: null, analyzer: english, store_raw: true}
```

The old implementation used library defaults. The new explicit k1=0.9/b=0.4 are
Pyserini's standard BM25 settings and are recorded as compatibility-defining
parameters. Lucene and rank_bm25 scores must not be compared for floating-point
equality: analyzers, IDF details, length normalization, and tie-breaking differ.
Only ranking/relevance behavior should be compared on a server fixture.

## Manifest and completion contracts

Index manifests include schema/completion state, backend and installed version,
corpus path/hash/size, document schema, contents rule, analyzer/language, BM25
parameters, timestamps, Git HEAD, config hash, document count, non-empty file
inventory, and a successful probe query. Resume recomputes the expected contract
and refuses corpus, backend-version, analyzer, or BM25 mismatches. Only explicit
overwrite removes and rebuilds the target index.

Prepare/corpus/retrieval write a `.partial`, flush and fsync it, then use
`os.replace`. Retrieval final and manifest include input/index fingerprints,
top-N/limit, row counts, candidate-count statistics, output SHA-256, and
timestamps. Fine-grained retrieval resume is intentionally not implemented:
after failure the diagnostic partial remains, final is absent, and an explicit
rerun starts retrieval from the first query. Runner completion checks require
completed manifests and matching output checksums, not merely existing JSONL.

## Dependencies

Pinned server dependency: `pyserini==2.3.0`. The upstream Pyserini documentation
for this generation specifies Python 3.12 and Java 21. Install with:

```bash
conda create -n rag-cbwdm python=3.12 -y
conda activate rag-cbwdm
sudo apt-get install -y openjdk-21-jre-headless
python -m pip install -r requirements.txt
python -m pip install -r requirements-retrieval.txt
python -m pip check
java -version
python scripts/check_environment.py --config configs/fever2_server_smoke.yaml
```

Pyserini's documented JSONL `JsonCollection`, `--storeRaw`, Lucene indexer, and
`LuceneSearcher` APIs are used. No optional backend is imported by any production
script's `--help` path.

## Verification performed locally

- `python -m compileall -q src scripts tests`: PASS.
- original focused `unittest` for this change: 4/4 PASS.
  - post-filter FEVER-2 limit and raw limit
  - resolved limit priority/fingerprint
  - index completion/reuse/mismatch/overwrite using a mocked Lucene process
  - retrieval failure partial and successful atomic final/manifest
- completion-validator regression suite after the real Linux finding: 5/5 PASS.
  It includes Lucene 10.4 underscore-prefixed files, non-empty `segments_1`, a
  zero-byte `write.lock`, and completed/fingerprint-matching resume with an
  assertion that the indexing subprocess call count does not increase.
- runner dry-run with train=10, dev=10, corpus=200000 and local model overrides:
  PASS; all required stages and shared index paths were printed.
- `git diff --check`: PASS.
- existing dependency-light CBWDM tests: 6 PASS.
- full `python -m unittest discover`: 6 PASS, 2 import errors because this local
  environment has no PyTorch.
- `python -m pytest -q`: not run because this local environment has no pytest.
- `bash -n scripts/run_fever_cbwdm.sh`: not run because this Windows environment
  has no bash.

No model or full FEVER download, heavy run, push, commit, branch change, or tag
change was performed.

## Server validation commands

Follow `DISK_BM25_SERVER_RUNBOOK.md`. The exact new 100-row full-corpus retrieval
smoke is:

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_pilot.yaml \
  --run-name fever2_full_corpus_smoke_v02_seed13 \
  --stages prepare,corpus,index,retrieve \
  --train-limit 100 --dev-limit 100 \
  --seed 13 --output-root "$EXP_ROOT" --cache-root "$HF_HOME"
```

Build/reuse the formal full index:

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_formal.yaml \
  --run-name fever2_index_build_v02 \
  --stages corpus,index \
  --output-root "$EXP_ROOT" --cache-root "$HF_HOME"
```

Exact retest for the failed 200,000-document smoke run (use the same run name,
output root, cache root, and model overrides as the original run; do not retain
`--overwrite-stage index`):

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_smoke.yaml \
  --run-name fever2_micro_smoke_v02_seed13 \
  --stages index,retrieve \
  --corpus-limit 200000 \
  --resume \
  --output-root "$EXP_ROOT" \
  --cache-root "$HF_HOME"
```

Expected evidence is an `[index] completed ...` line without a Pyserini indexing
subprocess run, index-stage exit code 0, retrieval-stage start, and unchanged
`index_manifest.json` fingerprint/inventory.

## Remaining P0/P1

P0:

- Re-run the existing real Pyserini 2.3.0/Java 21 200,000-document index with
  `--resume` and verify it is reused, then complete retrieval. The Lucene build,
  probe, direct search, and raw-document recovery are already confirmed PASS; the
  completion/resume path is the remaining server check.
- Run the complete pytest/unittest/bash syntax suite in the server environment
  with project dependencies installed.
- Record peak RSS, wall time, CPU, corpus bytes, temporary collection bytes, and
  final Lucene bytes for the full index.

P1:

- Compare tiny-fixture top-k relevance/order between Lucene English analysis and
  the legacy regex/rank_bm25 backend; document analyzer-driven differences without
  asserting score equality.
- If repeated retrieval jobs need mid-file recovery, add a validated query-ID
  checkpoint protocol. Current restart-from-head behavior is safe but not
  fine-grained.

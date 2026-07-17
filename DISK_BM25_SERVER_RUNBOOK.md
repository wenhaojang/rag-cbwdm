# Persistent BM25 server runbook

## Environment

The server retrieval profile uses Pyserini 2.3.0, Python 3.12, and Java 21. Pyserini's
official documentation identifies Python 3.12 and Java 21 as its supported build
environment. Install the CUDA-specific PyTorch build first so that Pyserini cannot
silently replace it.

```bash
conda create -n rag-cbwdm python=3.12 -y
conda activate rag-cbwdm
sudo apt-get update
sudo apt-get install -y openjdk-21-jre-headless
java -version
python -m pip install -r requirements.txt
python -m pip install -r requirements-retrieval.txt
python -m pip check
```

Preflight does not download FEVER or models:

```bash
python scripts/check_environment.py \
  --config configs/fever2_server_smoke.yaml \
  --output-root "$EXP_ROOT" \
  --cache-root "$HF_HOME"
```

The result must show `retrieval.ready=true`, Pyserini 2.3.0, and Java 21.

## Dry-run and limited micro-smoke

`--limit` is a compatibility fallback. Prefer the split-specific switches. Limits
count valid prepared samples after FEVER-2 `NOT ENOUGH INFO` filtering.
`--raw-limit` caps scanned source rows. `--corpus-limit` caps emitted sentence
documents and is only for micro/debug runs.

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_smoke.yaml \
  --run-name fever2_micro_smoke_v02_seed13 \
  --stages prepare,corpus,index,retrieve \
  --train-limit 10 \
  --dev-limit 10 \
  --corpus-limit 200000 \
  --seed 13 \
  --output-root "$EXP_ROOT" \
  --cache-root "$HF_HOME" \
  --generator-model "$GENERATOR_MODEL" \
  --selector-model "$SELECTOR_MODEL" \
  --dry-run
```

Remove `--dry-run` to execute. Inspect:

```bash
jq . "$EXP_ROOT/fever2_micro_smoke_v02_seed13/run_manifest.json"
find "$EXP_ROOT/_shared" -name '*manifest.json' -print
jq . "$EXP_ROOT/_shared"/indexes/*/index_manifest.json
jq . "$EXP_ROOT/fever2_micro_smoke_v02_seed13/artifacts/fever2_train_bm25_top10.manifest.json"
```

Only proceed when the index manifest has `completed=true`, non-empty inventory,
matching corpus hash/count, and `probe_passed=true`; retrieval manifests must have
matching row counts and output SHA-256.

## 100-row smoke on the full corpus

The smoke YAML defaults to a 200,000-document debug corpus. For a full corpus,
use a profile whose `profile_limits.corpus` is `null` and do not pass
`--corpus-limit`. For example:

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_pilot.yaml \
  --run-name fever2_full_corpus_smoke_v02_seed13 \
  --stages prepare,corpus,index,retrieve \
  --train-limit 100 \
  --dev-limit 100 \
  --output-root "$EXP_ROOT" \
  --cache-root "$HF_HOME"
```

Then execute the remaining stages with the same run name and `--resume`:

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_pilot.yaml \
  --run-name fever2_full_corpus_smoke_v02_seed13 \
  --stages posterior,teacher,train_cross_encoder,select_cross_encoder,eval,no_evidence \
  --train-limit 100 \
  --dev-limit 100 \
  --resume \
  --output-root "$EXP_ROOT" \
  --cache-root "$HF_HOME"
```

## Full index build, reuse, and overwrite

The runner stores shared data under:

```text
<output-root>/_shared/corpora/<configuration-key>/
<output-root>/_shared/indexes/<configuration-key>/
```

An exact corpus SHA-256, size, document count, Pyserini version, analyzer, and BM25
parameters are recorded in the index manifest. A compatible `--resume` reuses it.
Any mismatch fails rather than silently using stale data.

To build only the shared artifacts:

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_formal.yaml \
  --run-name fever2_index_build_v02 \
  --stages corpus,index \
  --output-root "$EXP_ROOT" \
  --cache-root "$HF_HOME"
```

Reuse from another run by using the same output root/config and `--stages index`
or the normal full chain. Explicit rebuild:

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_formal.yaml \
  --run-name fever2_index_rebuild_v02 \
  --stages corpus,index \
  --overwrite-stage corpus \
  --overwrite-stage index \
  --output-root "$EXP_ROOT" \
  --cache-root "$HF_HOME"
```

## Resource monitoring

The indexer streams JSONL into a temporary collection beside the index; allow free
space for the sentence corpus, temporary collection, Lucene index, and safety
margin. Measure rather than assuming a fixed multiplier:

```bash
watch -n 5 'free -h; df -h "$EXP_ROOT"; du -sh "$EXP_ROOT/_shared"'
pidstat -r -u -d -p "$(pgrep -n -f pyserini.index.lucene)" 5
/usr/bin/time -v python scripts/02a_build_bm25_index.py --help
```

`rank_bm25` is never selected by server smoke/pilot/formal configs. The
`memory_rank_bm25` backend is retained for tiny fixtures and refuses to exceed its
configured safety threshold.

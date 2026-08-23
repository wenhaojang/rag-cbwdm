# Signed-v1 Preformal Experiment Runbook

## Fixed locations and rules

- Repository: `/root/rag-cbwdm`
- Current pilot run: `/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13`
- Preformal root: `/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1`
- Config: `/root/rag-cbwdm/configs/fever2_server_preformal_signed_v1.yaml`
- Retrieval environment: `/root/miniconda3/envs/rag-cbwdm-retrieval/bin/python`
- Baseline/Qwen environment: `/root/miniconda3/envs/rag-cbwdm-baselines/bin/python`

Never run `retrieve_test`, `posterior_test`, or any evaluation over `held_out_test`. Do not use preformal outputs in calibration or formal config freezing. Do not rebuild the index unless Gate 0 proves the existing index contract incompatible and the rebuild is separately reviewed.

Every command below is one line and copyable. Commands state Qwen and resume behavior in the note immediately above them.

## Gate 0 — tests and code/artifact audit

Environment: baseline. Qwen: no. Resume: not applicable.

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python -m compileall -q src scripts tests
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python -m pytest -q
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python -m unittest discover
```

```bash
cd /root/rag-cbwdm && git diff --check
```

Environment: system Git/Python. Qwen: no. Resume: read-only.

```bash
cd /root/rag-cbwdm && pwd && git branch --show-current && git rev-parse HEAD && git status --short && git log -5 --oneline
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python -c "import json; p='/root/rag-cbwdm/outputs/formal_splits/fever2_seed13/fever2_formal_splits.manifest.json'; m=json.load(open(p)); print(json.dumps({'fingerprint':m['fingerprint'],'contract':m['contract'],'full_split_rows_before_limits':m['full_split_rows_before_limits'],'published':m['splits'],'overlap_checks':m['overlap_checks']},indent=2,sort_keys=True))"
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python -c "import json; p='/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/run_manifest.json'; m=json.load(open(p)); print(json.dumps({'git':m.get('git'),'paths':m.get('paths'),'stages':m.get('stages')},indent=2,sort_keys=True))"
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python -c "import json; p='/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/calibration_candidates.json'; m=json.load(open(p)); print(json.dumps({'status':m.get('status'),'request_contract':m.get('request_contract'),'candidate_count':len(m.get('candidates',[])),'completed':sum(x.get('status')=='completed' for x in m.get('candidates',[]))},indent=2,sort_keys=True))"
```

Review conditions: completed split/checksums, zero overlap, 5000/500 actual pilot limits, compatible train retrieval/posterior SHAs, current calibration status and selected reference candidates, and an existing compatible index path from the run manifest.

## Gate A — build and audit clean preformal_eval

Environment: baseline. Qwen: no. Resume: yes; exact contract/checksum only.

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/preformal/24_build_preformal_eval.py --split-manifest /root/rag-cbwdm/outputs/formal_splits/fever2_seed13/fever2_formal_splits.manifest.json --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/splits --resume
```

Environment: baseline. Qwen: no. Resume: read-only.

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python -c "import json; p='/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/splits/preformal_eval.manifest.json'; m=json.load(open(p)); print(json.dumps(m,indent=2,sort_keys=True)); assert m['source_strategy']=='A' and not any(m['overlap_checks'].values()) and m['held_out_test_consumed_for_modeling'] is False"
```

**STOP FOR REVIEW.** Confirm actual row count, label counts, source/pilot/train SHAs, ID-set SHA, normalized-claim SHA, and six zero overlap checks. Expected protocol count is 4500, but use the manifest's actual count.

## Gate B — preformal retrieval

Environment: retrieval. Qwen: no. Resume: the retrieval script is atomic but not resumable; if completed output exists, audit and reuse it. Do not pass `--overwrite` without review.

```bash
cd /root/rag-cbwdm && INDEX_PATH=$(/root/miniconda3/envs/rag-cbwdm-baselines/bin/python -c "import json; print(json.load(open('/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/run_manifest.json'))['paths']['index_path'])") && /root/miniconda3/envs/rag-cbwdm-retrieval/bin/python /root/rag-cbwdm/scripts/02_retrieve_bm25.py --config /root/rag-cbwdm/configs/fever2_server_preformal_signed_v1.yaml --split preformal_eval --queries /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/splits/preformal_eval.jsonl --index "$INDEX_PATH" --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/retrieval/preformal_eval_bm25_top20.jsonl --top-n 20
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python -c "import json; p='/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/retrieval/preformal_eval_bm25_top20.manifest.json'; m=json.load(open(p)); s=json.load(open('/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/splits/preformal_eval.manifest.json')); print(json.dumps(m,indent=2,sort_keys=True)); assert m['completed'] and m['query_input_sha256']==s['preformal_eval_sha256'] and m['num_output_rows']==s['row_count'] and m['candidate_count_statistics']['max']<=20"
```

**STOP FOR REVIEW.** Confirm index fingerprint matches the current formal run and no index was rebuilt.

## Gate C — shared preformal posteriors

Environment: baseline. Qwen: **yes**, `/root/models/Qwen2.5-1.5B-Instruct`. Resume: yes, row-level partial resume.

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/03_compute_label_posteriors.py --config /root/rag-cbwdm/configs/fever2_server_preformal_signed_v1.yaml --split preformal_eval --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/retrieval/preformal_eval_bm25_top20.jsonl --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/posteriors/preformal_eval_posteriors.jsonl --model-name /root/models/Qwen2.5-1.5B-Instruct --max-candidates 20 --resume
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python -c "import json; p='/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/posteriors/preformal_eval_posteriors.manifest.json'; m=json.load(open(p)); r=json.load(open('/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/retrieval/preformal_eval_bm25_top20.manifest.json')); print(json.dumps(m,indent=2,sort_keys=True)); assert m['status']=='completed' and m['provenance']['input_sha256']==r['output_sha256'] and m['completed_rows']==r['num_output_rows']"
```

**STOP FOR REVIEW.** Freeze posterior SHA, generator SHA, prompt hash, verbalizer hash, labels, truncation, and completed row count. This one posterior artifact is shared by InfoGain, old RAG-CBWDM, and signed RAG-CBWDM.

## Gate D — train missing learned-method seeds

First audit current `calibration_candidates.json`, `calibration.manifest.json`, and `frozen_parameters.yaml`. Record the selected InfoGain/old-RAG parameter JSON, candidate/training/checkpoint fingerprints, and whether the grid is full or partial. If only the pilot reference exists, mark `reference_config_from_pilot=true` and `formal_optimum_claimed=false`. Do not run a new grid.

Environment: baseline. Qwen: no; reuses existing train_core posterior. Resume: yes.

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/preformal/25_materialize_signed_v1_teacher.py --config /root/rag-cbwdm/configs/fever2_server_preformal_signed_v1.yaml --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_posteriors.jsonl --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_bm25_top20.jsonl --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/training/rag_cbwdm_signed_v1/teacher --resume
```

Environment: baseline. Qwen: no. Resume: yes, exact seed/checkpoint contract.

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/preformal/26_train_signed_v1.py --config /root/rag-cbwdm/configs/fever2_server_preformal_signed_v1.yaml --teacher /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/training/rag_cbwdm_signed_v1/teacher/teacher.jsonl --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_posteriors.jsonl --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_bm25_top20.jsonl --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/training/rag_cbwdm_signed_v1/seed13 --model-name /root/models/ms-marco-MiniLM-L-6-v2 --seed 13 --device auto --resume
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/preformal/26_train_signed_v1.py --config /root/rag-cbwdm/configs/fever2_server_preformal_signed_v1.yaml --teacher /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/training/rag_cbwdm_signed_v1/teacher/teacher.jsonl --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_posteriors.jsonl --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_bm25_top20.jsonl --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/training/rag_cbwdm_signed_v1/seed21 --model-name /root/models/ms-marco-MiniLM-L-6-v2 --seed 21 --device auto --resume
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/preformal/26_train_signed_v1.py --config /root/rag-cbwdm/configs/fever2_server_preformal_signed_v1.yaml --teacher /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/training/rag_cbwdm_signed_v1/teacher/teacher.jsonl --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_posteriors.jsonl --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/formal/fever2_train_core_bm25_top20.jsonl --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/training/rag_cbwdm_signed_v1/seed42 --model-name /root/models/ms-marco-MiniLM-L-6-v2 --seed 42 --device auto --resume
```

InfoGain seed13/21/42 and old-RAG seed commands must use the exact server-audited selected candidate parameters and training teacher. They are intentionally not fabricated here because those artifacts are absent from the implementation workspace. If the old-RAG contract cannot be replayed, record `old rag_cbwdm stability unavailable / pending` and keep only its genuine existing seed13 checkpoint.

**STOP FOR REVIEW.** Compare input SHAs and verify distinct seed/checkpoint fingerprints; do not inspect preformal results because none exist yet.

## Gate E — deployable selections

Run no-evidence/Naive/BGE once, InfoGain for three genuine checkpoints, old RAG for available genuine checkpoints, and signed-v1 for all three seeds. Every selection must consume the Gate B retrieval or Gate C posterior SHA.

Environment: baseline. Qwen: no. Resume: yes.

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/preformal/27a_select_no_evidence.py --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/retrieval/preformal_eval_bm25_top20.jsonl --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/no_evidence/seed13/selection.jsonl --resume
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/08_select_naive_topm.py --config /root/rag-cbwdm/configs/fever2_server_preformal_signed_v1.yaml --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/retrieval/preformal_eval_bm25_top20.jsonl --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/naive_topm/seed13/selection.jsonl --top-m 4 --min-docs 4 --method-name naive_topm --resume
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/12_select_bge_reranker.py --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/retrieval/preformal_eval_bm25_top20.jsonl --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/bge/seed13/selection.jsonl --score-cache /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/bge/seed13/scores.jsonl --model-name-or-path /root/models/bge-reranker-large --device auto --dtype auto --batch-size 8 --max-length 512 --top-m 4 --min-docs 4 --local-files-only --resume
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/preformal/27_select_signed_v1.py --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/posteriors/preformal_eval_posteriors.jsonl --checkpoint-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/training/rag_cbwdm_signed_v1/seed13/checkpoint --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/rag_cbwdm_signed_v1/seed13/selection.jsonl --seed 13 --device auto --resume
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/preformal/27_select_signed_v1.py --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/posteriors/preformal_eval_posteriors.jsonl --checkpoint-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/training/rag_cbwdm_signed_v1/seed21/checkpoint --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/rag_cbwdm_signed_v1/seed21/selection.jsonl --seed 21 --device auto --resume
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/preformal/27_select_signed_v1.py --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/posteriors/preformal_eval_posteriors.jsonl --checkpoint-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/training/rag_cbwdm_signed_v1/seed42/checkpoint --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/rag_cbwdm_signed_v1/seed42/selection.jsonl --seed 42 --device auto --resume
```

Use existing `08_select_naive_topm.py`, `12_select_bge_reranker.py`, `12c_select_infogain_reranker.py`, and `11_select_with_cross_encoder.py` for the other methods with the frozen parameters audited at Gate D. Do not change thresholds/min-docs/top-m here.

**STOP FOR REVIEW.** Check exact ID sets, method metadata, `uses_gold_at_test=false`, selection SHAs, and document budgets.

## Gate F — generator evaluations

Environment: baseline. Qwen: **yes**, `/root/models/Qwen2.5-1.5B-Instruct`. Resume: yes, completed exact-contract evaluation; individual evaluation is not row-partial.

The following is the signed seed13 pattern; repeat only the seed directory/method seed for 21 and 42, and use the same command contract for every deployable baseline selection.

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/07_eval_rag_classification.py --config /root/rag-cbwdm/configs/fever2_server_preformal_signed_v1.yaml --split preformal_eval --selection /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/no_evidence/seed13/selection.jsonl --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/evaluations/no_evidence/seed13/predictions.jsonl --metrics-output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/evaluations/no_evidence/seed13/metrics.json --model-name /root/models/Qwen2.5-1.5B-Instruct --method-name no_evidence --no-evidence --resume
```

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/07_eval_rag_classification.py --config /root/rag-cbwdm/configs/fever2_server_preformal_signed_v1.yaml --split preformal_eval --selection /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/rag_cbwdm_signed_v1/seed13/selection.jsonl --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/evaluations/rag_cbwdm_signed_v1/seed13/predictions.jsonl --metrics-output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/evaluations/rag_cbwdm_signed_v1/seed13/metrics.json --model-name /root/models/Qwen2.5-1.5B-Instruct --method-name rag_cbwdm_signed_v1 --resume
```

**STOP FOR REVIEW.** Before reading scores, verify all evaluation manifests share generator SHA, prompt hash, verbalizer hash, context/truncation, split, and exact ID set.

## Gate G — fairness, paired statistics, and summary

Environment: baseline. Qwen: no. Resume: summary is cheap and atomic; use a new output directory if contracts change.

Pass one `--selection-manifest METHOD:SEED=PATH` and one `--evaluation-manifest METHOD:SEED=PATH` for every completed deployable run. The command below shows the required signed entries; append the frozen baseline entries before execution.

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/preformal/28_audit_fairness.py --split-manifest /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/splits/preformal_eval.manifest.json --retrieval-manifest /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/retrieval/preformal_eval_bm25_top20.manifest.json --posterior-manifest /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/posteriors/preformal_eval_posteriors.manifest.json --selection-manifest rag_cbwdm_signed_v1:13=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/rag_cbwdm_signed_v1/seed13/selection.manifest.json --selection-manifest rag_cbwdm_signed_v1:21=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/rag_cbwdm_signed_v1/seed21/selection.manifest.json --selection-manifest rag_cbwdm_signed_v1:42=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/rag_cbwdm_signed_v1/seed42/selection.manifest.json --evaluation-manifest rag_cbwdm_signed_v1:13=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/evaluations/rag_cbwdm_signed_v1/seed13/metrics.manifest.json --evaluation-manifest rag_cbwdm_signed_v1:21=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/evaluations/rag_cbwdm_signed_v1/seed21/metrics.manifest.json --evaluation-manifest rag_cbwdm_signed_v1:42=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/evaluations/rag_cbwdm_signed_v1/seed42/metrics.manifest.json --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/fairness/preformal_fairness.json
```

Append matching `--evaluation-manifest METHOD:SEED=...` and `--selection METHOD:SEED=...` arguments for all deployable methods to the summary command.

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/preformal/29_summarize_results.py --fairness-audit /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/fairness/preformal_fairness.json --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/retrieval/preformal_eval_bm25_top20.jsonl --evaluation-manifest rag_cbwdm_signed_v1:13=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/evaluations/rag_cbwdm_signed_v1/seed13/metrics.manifest.json --evaluation-manifest rag_cbwdm_signed_v1:21=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/evaluations/rag_cbwdm_signed_v1/seed21/metrics.manifest.json --evaluation-manifest rag_cbwdm_signed_v1:42=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/evaluations/rag_cbwdm_signed_v1/seed42/metrics.manifest.json --selection rag_cbwdm_signed_v1:13=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/rag_cbwdm_signed_v1/seed13/selection.jsonl --selection rag_cbwdm_signed_v1:21=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/rag_cbwdm_signed_v1/seed21/selection.jsonl --selection rag_cbwdm_signed_v1:42=/root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/rag_cbwdm_signed_v1/seed42/selection.jsonl --bootstrap-seed 130421 --bootstrap-samples 10000 --output-dir /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/summary
```

Interpret only after the complete baseline arguments are present and fairness status is `comparable`:

- STRONG GO: signed three-seed mean accuracy/macro-F1 meets or exceeds InfoGain and paired CIs do not stably show signed disadvantage.
- GO/BORDERLINE: gap to InfoGain is at most 1 percentage point, evidence cost is materially lower, and seeds are stable.
- OPTIMIZE BEFORE THEORY FREEZE: stable loss greater than 2 points, all three seeds lose, or paired analysis shows a clear stable disadvantage.
- 1–2 point gray zone: manually review failure categories; do not tune on preformal_eval.

## Gate H — optional signed oracle diagnostic

Only after all Gate G deployable artifacts and summaries are frozen. Environment: baseline. Qwen: may be required by the existing oracle runner. Resume: yes. Run once, seed13 metadata only. Never feed its output into parameters or deployable selection.

After the optional oracle selection exists, post-hoc analysis is:

```bash
cd /root/rag-cbwdm && /root/miniconda3/envs/rag-cbwdm-baselines/bin/python /root/rag-cbwdm/scripts/preformal/30_signed_posthoc.py --config /root/rag-cbwdm/configs/fever2_server_preformal_signed_v1.yaml --selection /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/rag_cbwdm_signed_v1/seed13/selection.jsonl --posteriors /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/posteriors/preformal_eval_posteriors.jsonl --retrieval /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/retrieval/preformal_eval_bm25_top20.jsonl --oracle-selection /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/selections/signed_gate_oracle/selection.jsonl --output /root/experiments/rag_cbwdm/fever2_formal_pilot_5000_500_seed13/artifacts/preformal_signed_v1/statistics/signed_seed13_posthoc.json
```

Final boundary: this run never authorizes official-dev held_out_test. If preformal findings cause any algorithm/parameter change, record preformal_eval as development data, refreeze the algorithm, and reserve untouched held_out_test for the later final evaluation.

# FEVER Budget Sweep：Qwen0.5B 与 Mistral7B 增量部署手册

## 1. 范围与状态

本手册只覆盖 `fever_binary_v2 / validation / 5000 / seed=13`，新增 generator 为：

- `qwen2.5-0.5b-instruct`
- `mistral-7b-instruct-v0.3`

方法固定为 `retrieval_topk`、`bge`、`infogain_fever`、`rag_cbwdm_signed_v1`、`kcbwdm_linear_gate_v2`，预算固定为 k=2/3/4。逻辑矩阵恰好 30 点；k=4 的 10 点只允许通过 provenance gate alias 到正式主实验。GPU job 数由 selection audit 的实际 evidence-vector 结果决定，范围为 0–20；约 18 仅是 Mistral signed/KCBWDM 等平台期成立时的预期，不是执行条件。

本轮本地实现没有运行服务器命令、训练、teacher/posterior 构造或 generator evaluation。

## 2. 本地改动文件

- `src/fever_budget_sweep.py`：将 generator 集合、30点矩阵、16个 selection task、10个 k4 alias 与排序改为由受控配置导出；加入上一轮共享 RAG/BGE selection 的严格只读复用；加强跨 generator alias、输出根隔离、有限数值和 summary 完整性检查；提供60点合并审计函数。
- `scripts/diagnostics/run_fever_budget_sweep.py`：保留原六阶段 CLI；允许配置指定 summary 文件名。
- `scripts/diagnostics/merge_fever_budget_summaries.py`：只读校验两份30点 CSV 和48组主表 k4 anchor，成功后原子写出60点 CSV/manifest。
- `configs/formal/fever_budget_sweep_extra_05b_mistral_caps234.seed13.yaml`：新增两 generator 的独立正式配置。
- `tests/test_fever_budget_sweep.py`：覆盖新旧配置、身份隔离、动态 job、alias、summary 和60点合并。

`plots/` 不属于本次部署内容，不应上传或改动。

## 3. 新旧配置差异

| 项目 | 原配置 | 新配置 |
|---|---|---|
| 配置 | `fever_budget_sweep_document_caps_234.seed13.yaml` | `fever_budget_sweep_extra_05b_mistral_caps234.seed13.yaml` |
| Generator | Qwen1.5B、Qwen7B | Qwen0.5B、Mistral7B |
| 输出根 | `formal_v2_fever_budget_sweep_document_caps_234` | `formal_v2_fever_budget_sweep_extra_05b_mistral_caps234` |
| Summary | `fever_budget_accuracy_tokens.csv` | `fever_budget_accuracy_tokens_extra.csv` |
| RAG/BGE k2/k3 | 从冻结 retrieval/cache 物化 | 优先严格复用上一轮4份共享 selection；两文件均缺失时由同一冻结 retrieval/cache 重建 |

旧配置的计划指纹仍为 `db6a27554db3aca0085fe6ac944a9a5ccda80e972ef7427596d6b896c12b93bf`。新配置不会写入旧输出根。

## 4. Generator 与正式来源映射

| Generator | Generator config | 普通 generator manifest | KCBWDM generator manifest |
|---|---|---|---|
| Qwen2.5-0.5B | `/root/experiments/rag_cbwdm/formal_v2_deployment/configs/fever_qwen05_full_development.server.yaml` | `formal_v2_full_development/fever_binary_v2/qwen2.5-0.5b-instruct/generator/manifest.json` | `formal_v2_fever_kcbwdm_linear_gate_v2_full_development/fever_binary_v2/qwen2.5-0.5b-instruct/generator/manifest.json` |
| Mistral-7B-v0.3 | `/root/experiments/rag_cbwdm/formal_v2_deployment/configs/fever_mistral7_full_development.server.yaml` | `formal_v2_full_development/fever_binary_v2/mistral-7b-instruct-v0.3/generator/manifest.json` | `formal_v2_fever_kcbwdm_linear_gate_v2_full_development/fever_binary_v2/mistral-7b-instruct-v0.3/generator/manifest.json` |

InfoGain 使用 `formal_v2_infogain_top4_filter02_final/fever_binary_v2/{generator_id}/...` 中的 corrected selection/evaluation；其 checkpoint 与 training manifest 仍由该 selection manifest 的 formal binding 指向 `formal_v2_full_development`。signed CBWDM 使用 `formal_v2_full_development`；KCBWDM v2A 使用 `formal_v2_fever_kcbwdm_linear_gate_v2_full_development`。不要根据目录名人工替换 provenance。

## 5. Preflight 前必须存在的服务器来源

至少核实：

1. 冻结 FEVER BM25 Top20 retrieval、manifest 与 SHA `a71fffda269f9112c6c243cee3384df072455a604eb801c838b69e32993786eb`。
2. 两个 generator 的 server config、模型/tokenizer 本地路径与 generator manifest。
3. 五种方法的10份原生 k4 selection、predictions、metrics、evaluation manifest。
4. InfoGain 两个 generator 的完整 `candidate_score_audit`、checkpoint 与 training binding。
5. signed/KCBWDM 的 k4 causal trace；默认 replay 要求每步 candidate scores、winner、remaining count 与停止步合法。
6. `main_48_v3/main_results_48.csv`、`token_audit.json`、`analysis_manifest.json`，且 audit/manifest 均为已完成的 `mode=all` 48组分析。
7. 上一轮 RAG/BGE k2/k3 selection 与 manifest。若 selection 和 manifest 均不存在，`prefer` 策略会安全回退到冻结 retrieval/BGE cache 重建；若只缺一个、SHA漂移或内容不等价则 fail closed。

## 6. 公共变量

以下命令均在服务器执行：

```bash
cd /root/rag-cbwdm
PY=/root/miniconda3/envs/rag-cbwdm-mistral/bin/python
CFG=configs/formal/fever_budget_sweep_extra_05b_mistral_caps234.seed13.yaml
OUT=/root/experiments/rag_cbwdm/formal_v2_fever_budget_sweep_extra_05b_mistral_caps234
OLD=/root/experiments/rag_cbwdm/formal_v2_fever_budget_sweep_document_caps_234
TOK=/root/experiments/rag_cbwdm/formal_v2_analysis/evidence_token_comparison_v1/main_48_v3
```

## 7. Phase 0：只读 preflight

```bash
"$PY" scripts/diagnostics/run_fever_budget_sweep.py \
  --config "$CFG" \
  --preflight
```

期望：`status=passed`、`native_evaluation_count=10`、`native_token_record_count=10`。若上一轮4份共享 selection 全部通过复用 gate，还应出现 `external_budget_selection_reuse_count=4`。此阶段不写新实验目录、不启动 GPU。

如果 signed/KCBWDM 的历史 k4 没有足够的逐步 score trace，可只读检查同 checkpoint 推理回退条件：

```bash
"$PY" scripts/diagnostics/run_fever_budget_sweep.py \
  --config "$CFG" \
  --preflight \
  --state-aware-inference
```

该选项不训练模型，也不运行 generator；正式 materialize 必须使用相同模式。

## 8. Phase 0b：dry-run

```bash
"$PY" scripts/diagnostics/run_fever_budget_sweep.py \
  --config "$CFG" \
  --dry-run
```

期望 `points=30`、`new_selections=16`。其中4个共享 RAG/BGE task、12个 generator-conditioned task；10个 k4 点均显示 `native_k4_alias_pending_provenance`。

## 9. Phase 1：物化16份 selection

默认 causal-prefix replay：

```bash
"$PY" scripts/diagnostics/run_fever_budget_sweep.py \
  --config "$CFG" \
  --materialize-selections \
  --resume
```

只有 preflight 明确表明 causal trace 不足、且 checkpoint/training/posterior identity 全部通过时，才使用同 checkpoint 回退：

```bash
"$PY" scripts/diagnostics/run_fever_budget_sweep.py \
  --config "$CFG" \
  --materialize-selections \
  --state-aware-inference \
  --resume
```

输出：`$OUT/plan/plan.json` 与 `$OUT/plan/materialization.manifest.json`。RAG/BGE 外部复用的输出 manifest 会记录旧 selection/manifest 的 path、SHA 和 contract fingerprint；新根中发布的是独立 selection，不会改写旧目录。

## 10. Phase 2：selection audit 与动态 GPU job 冻结

```bash
"$PY" scripts/diagnostics/run_fever_budget_sweep.py \
  --config "$CFG" \
  --audit-selections
```

输出：

- `$OUT/audit/selection_audit.json`
- `$OUT/audit/evaluation_jobs.json`
- `$OUT/audit/evaluation_commands.sh`

只读查看计数：

```bash
"$PY" -c 'import json; p=json.load(open("/root/experiments/rag_cbwdm/formal_v2_fever_budget_sweep_extra_05b_mistral_caps234/audit/selection_audit.json", encoding="utf-8")); print({k:p[k] for k in ("logical_point_count","new_selection_count","native_k4_alias_count","evidence_vector_alias_count","evaluation_job_count")})'
```

必须看到 `logical_point_count=30`、`new_selection_count=16`、`native_k4_alias_count=10`。`evaluation_job_count` 必须以该文件为准，不能手改成18。Mistral signed/KCBWDM 的 k3/k4 只有在5000行 ordered doc IDs、rendered evidence、num_docs 和 generator-input fingerprint 全同后才会成为 `evidence_vector_alias`。

## 11. Phase 3：GPU evaluation——人工审批点

在人工检查 `selection_audit.json`、`evaluation_jobs.json` 和 job 数后才能执行。启动 tmux：

```bash
mkdir -p "$OUT/logs"
tmux new-session -d -s fever-budget-extra -c /root/rag-cbwdm
tmux send-keys -t fever-budget-extra "/root/miniconda3/envs/rag-cbwdm-mistral/bin/python scripts/diagnostics/run_fever_budget_sweep.py --config configs/formal/fever_budget_sweep_extra_05b_mistral_caps234.seed13.yaml --run-evaluations --job-plan /root/experiments/rag_cbwdm/formal_v2_fever_budget_sweep_extra_05b_mistral_caps234/audit/evaluation_jobs.json 2>&1 | tee -a /root/experiments/rag_cbwdm/formal_v2_fever_budget_sweep_extra_05b_mistral_caps234/logs/evaluations.log" C-m
```

状态与日志：

```bash
tmux list-sessions
tmux attach -t fever-budget-extra
tail -n 100 -f "$OUT/logs/evaluations.log"
```

安全断开 tmux 使用 `Ctrl-b d`。中断后使用完全相同的 `--run-evaluations --job-plan ...` 命令恢复；冻结 job 中的 evaluator 命令自带 `--resume`，执行前会重新检查 audit fingerprint 以及 selection、manifest、generator config/manifest SHA。

## 12. Phase 4：汇总与30点验收

```bash
"$PY" scripts/diagnostics/run_fever_budget_sweep.py \
  --config "$CFG" \
  --summarize
```

默认输出：

- `$OUT/summaries/fever_budget_accuracy_tokens_extra.csv`
- `$OUT/summaries/fever_budget_accuracy_tokens_extra.manifest.json`

验收：

```bash
"$PY" -c 'import csv; p="/root/experiments/rag_cbwdm/formal_v2_fever_budget_sweep_extra_05b_mistral_caps234/summaries/fever_budget_accuracy_tokens_extra.csv"; r=list(csv.DictReader(open(p, encoding="utf-8"))); print(len(r), len({(x["generator_id"],x["method_id"],x["budget_k"]) for x in r}), {x["analysis_status"] for x in r})'
```

期望输出为 `30 30 {'completed'}`。summarize 会拒绝非有限指标、`avg_num_docs > k`、跨 generator evaluation path 和错误 token basis。

## 13. 合并为四 generator 的60点 CSV

先只读验证：

```bash
"$PY" scripts/diagnostics/merge_fever_budget_summaries.py \
  --original-csv "$OLD/summaries/fever_budget_accuracy_tokens.csv" \
  --extra-csv "$OUT/summaries/fever_budget_accuracy_tokens_extra.csv" \
  --main-results "$TOK/main_results_48.csv" \
  --validate-only
```

通过后写新文件：

```bash
"$PY" scripts/diagnostics/merge_fever_budget_summaries.py \
  --original-csv "$OLD/summaries/fever_budget_accuracy_tokens.csv" \
  --extra-csv "$OUT/summaries/fever_budget_accuracy_tokens_extra.csv" \
  --main-results "$TOK/main_results_48.csv" \
  --output-csv "$OUT/summaries/fever_budget_accuracy_tokens_all4.csv" \
  --output-manifest "$OUT/summaries/fever_budget_accuracy_tokens_all4.manifest.json"
```

工具要求两份输入各30个 completed 唯一点、generator 集合互补、20个 k4 点的 Accuracy/Macro-F1/Avg Docs/Avg Tokens/selection/evaluation 来源与主表完全一致，并禁止跨 generator evaluation path。输入 CSV 不会被修改。

## 14. 备份与本地下载建议

在60点验收后创建只包含新根的归档：

```bash
tar -C /root/experiments/rag_cbwdm -czf \
  /root/experiments/rag_cbwdm/formal_v2_fever_budget_sweep_extra_05b_mistral_caps234.tar.gz \
  formal_v2_fever_budget_sweep_extra_05b_mistral_caps234
sha256sum /root/experiments/rag_cbwdm/formal_v2_fever_budget_sweep_extra_05b_mistral_caps234.tar.gz
```

随后从本地使用组织批准的 `scp`/备份通道下载归档和 SHA 文件。不要把归档写进旧 Budget Sweep、formal-v2 或 token-analysis 根目录。

## 15. 常见失败与禁止绕过

- **旧共享 selection 缺失**：两文件都缺失时允许由冻结 retrieval/BGE cache 重建；只缺一份、SHA/contract/content 不同则停止，不能复制文件名相似的产物冒充。
- **InfoGain candidate score audit 缺失**：停止；不能截断 k4 filtered selection，也不能重新定义近似协议。
- **causal trace 不足**：只有 provenance 完整时才使用 `--state-aware-inference`；不能直接 `selected_docs[:k]`。
- **generator config/model/tokenizer/prompt/verbalizer 漂移**：停止并调查；禁止跨 generator alias。
- **k4 主表或 token analysis 不一致**：停止；不能放宽 SHA、binding 或数值比较。
- **job 数不是18**：检查逐点 evidence vector；以 audit 动态结果为准，不能编辑 job JSON/命令文件。
- **输出目录已存在**：使用 `--resume`；不要覆盖旧目录、删除 manifest 或执行 `git clean`/`git reset --hard`。
- **GPU 阶段失败**：保留日志和 partial state，修复明确原因后重跑冻结 job plan；禁止重新训练或重新下载模型作为临时绕过。

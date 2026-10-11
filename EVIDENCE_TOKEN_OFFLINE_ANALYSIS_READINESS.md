# formal-v2 Evidence Token 离线分析就绪报告

日期：2026-10-11

基线分支：`feature/fever-formal-readiness`

基线提交：`8a42315`

## 1. 结论

已完成 formal-v2 主实验 Evidence Token 的离线统计与 48 组统一汇总工具。实现只读取已完成的 matched-main validation 产物，只加载对应 generator 的 tokenizer，不加载 generator 模型、不运行 selector、不训练模型，也不修改任何正式实验产物。

本地 Windows 环境没有服务器 `/root/experiments/rag_cbwdm` 数据，因此本阶段只完成代码、fixture 回归测试和运行方案；没有宣称真实服务器 dry-run 或 token 统计已经通过，也没有启动任何 CPU/GPU 正式任务。

## 2. 审计结论

本次实现以仓库现有正式 evaluator 为唯一计算口径：

- `scripts/07_eval_rag_classification.py`：直接复用 `recover_selected_docs()` 与 `build_evidence_context()`；严格复现 `no_evidence`、`max_docs` 和 selected-doc 顺序。
- `src/label_logits.py`：离线工具只复现 `AutoTokenizer.from_pretrained(..., trust_remote_code=..., revision=tokenizer_revision or model_revision)`，不调用 `AutoModelForCausalLM`。
- `src/metrics.py`：沿用 `generator_tokenizer_rendered_evidence_block_before_prompt_truncation` 作为 token 统计口径。
- `src/artifact_binding.py`：复用正式 selection、generator 和 evaluation binding 校验，未建立宽松的旁路规则。
- `src/formal_matrix.py` 及 FEVER/FM2 formal-v2 配置：确认 shared selection、learned selection、matched-main evaluation 的正式路径结构及 deterministic/seed13 规则。
- `scripts/infogain_final_protocol.py` 及 corrected InfoGain 配置：确认最终 InfoGain 来源必须是 `formal_v2_infogain_top4_filter02_final`，不能回退到旧版无阈值产物。

实际 token 定义为：

```text
len(tokenizer.encode(rendered_evidence_block, add_special_tokens=False))
```

统计发生在 prompt 模板拼接及 prompt truncation 之前。No Evidence 先按 evaluator 规则恢复为空证据，再验证 token 数为 0。

## 3. 新增文件

- `configs/formal/evidence_token_analysis_48.development.server.yaml`
  - 冻结两个数据集、四个 generator、六个方法、seed 策略、tokenizer identity、expected rows 和三个正式 source root。
  - loader 将配置展开为严格唯一的 48 个 source record，并拒绝非预期方法、seed、root、tokenizer 或 split。
- `scripts/analyze_formal_evidence_tokens.py`
  - 提供 `--dry-run`、`--validate-infogain`、`--all`、`--output-dir`、`--resume`。
  - 每组校验 source provenance、文件 SHA、行身份、实际 evidence 与既有 metrics 后再统计 token。
- `tests/test_evidence_token_offline_analysis.py`
  - 新增 15 个离线分析专项回归测试。
- `EVIDENCE_TOKEN_OFFLINE_ANALYSIS_READINESS.md`
  - 本报告。

未修改 generator、selector、训练或正式 evaluation 逻辑。

## 4. 48 组 source mapping

固定维度为：

| 维度 | 值 |
|---|---|
| Dataset | `fever_binary_v2`（5000）；`fm2_official_closed_page_v1`（1169） |
| Generator | `qwen2.5-0.5b-instruct`；`qwen2.5-1.5b-instruct`；`qwen2.5-7b-instruct`；`mistral-7b-instruct-v0.3` |
| Split / experiment | `validation` / `matched_main` |

每个 dataset × generator 均使用以下六组映射：

| 方法 | Source root | Selection 路径（相对 dataset root） | Evaluation 路径（相对 dataset root） | Seed |
|---|---|---|---|---|
| `no_evidence` | main | `shared/no_evidence/validation/selection.jsonl` | `{generator}/evaluation/no_evidence/matched_main/deterministic/validation` | null |
| `retrieval_topk` | main | `shared/retrieval_topk/validation/selection.jsonl` | `{generator}/evaluation/retrieval_topk/matched_main/deterministic/validation` | null |
| `bge` | main | `shared/bge/validation/selection.jsonl` | `{generator}/evaluation/bge/matched_main/deterministic/validation` | null |
| `infogain_fever` | corrected InfoGain | `{generator}/infogain/seed13/selection/validation.jsonl` | `{generator}/evaluation/infogain/matched_main/seed13/validation` | 13 |
| `rag_cbwdm_signed_v1` | main | `{generator}/rag_cbwdm_signed_v1/seed13/selection/validation.jsonl` | `{generator}/evaluation/rag_cbwdm_signed_v1/matched_main/seed13/validation` | 13 |
| `kcbwdm_linear_gate_v2` | dataset-specific KCBWDM v2A | `{generator}/kcbwdm_linear_gate_v2/seed13/selection/validation.jsonl` | `{generator}/evaluation/kcbwdm_linear_gate_v2/matched_main/seed13/validation` | 13 |

Root 固定为：

| 名称 | 路径 |
|---|---|
| main | `/root/experiments/rag_cbwdm/formal_v2_full_development` |
| corrected InfoGain | `/root/experiments/rag_cbwdm/formal_v2_infogain_top4_filter02_final` |
| FEVER KCBWDM v2A | `/root/experiments/rag_cbwdm/formal_v2_fever_kcbwdm_linear_gate_v2_full_development` |
| FM2 KCBWDM v2A | `/root/experiments/rag_cbwdm/formal_v2_kcbwdm_linear_gate_v2_full_development` |

所有组的 generator manifest 都绑定原始 `formal_v2_full_development/{dataset}/{generator}/generator/manifest.json`。工具不会仅凭上表路径信任文件：evaluation manifest 中记录的 selection path/SHA、evaluation binding、generator identity 和 predictions/metrics 路径/SHA 必须与实际文件逐项一致。

## 5. 逐组 fail-fast 校验

在 token 编码前，每组必须同时通过：

1. generator manifest schema/status、dataset/generator identity、config SHA、模型及 tokenizer artifact identity；
2. tokenizer path/revision 与冻结计划及 generator config 一致；
3. selection manifest、formal-v2 provenance、dataset/method/seed、selection SHA 和 expected rows；
4. evaluation manifest schema/status/completed、matched-main identity、binding、contract fingerprint；
5. `split=validation`、`limit=null`、`no_evidence`、`max_docs`、selection path/SHA；
6. predictions、metrics 的记录路径和实际 SHA；
7. selection/predictions 行数、唯一 ID、顺序、query、gold label、method；
8. evaluator 实际使用的 `selected_doc_ids` 和 `num_docs`；
9. 离线重算的 `avg_num_docs`、`avg_evidence_chars`、文档数 histogram 与 `metrics.json`；
10. corrected InfoGain 八组从各自正式 `metrics.json` 动态读取 `avg_evidence_tokens` 与 token basis，并以 `1e-9` 容差交叉验证。代码中未写死用户提供的八个数值。

任一失败都会终止正式汇总，只保留带错误信息的 `token_audit.json` / `token_audit.md`；不会输出伪完整的 48 组主表。

## 6. 输出与恢复语义

默认独立输出目录：

```text
/root/experiments/rag_cbwdm/formal_v2_analysis/evidence_token_comparison_v1
```

正式成功后生成：

| 文件 | 内容 |
|---|---|
| `main_results_48.csv` | 48 组完整原始指标、token、文档分布、source path/SHA、tokenizer identity；Accuracy/F1 保持 0–1 |
| `main_results_48.md` | 便于人工审阅的主表 |
| `main_results_48.tex` | 论文表格行，Accuracy/F1 转百分比并转义 LaTeX 特殊字符 |
| `accuracy_token_plot_data.csv` | Accuracy/F1–token 作图数据及 KCBWDM 对 RAG/BGE/InfoGain 的派生 reduction ratio |
| `token_audit.json` | 机器可读的逐组审计结果 |
| `token_audit.md` | 人工可读的审计摘要 |
| `analysis_manifest.json` | plan SHA/fingerprint、输出 SHA、完成状态、Git 状态和完成时间 |

逐组成功结果保存在 `group_audits/`。默认拒绝覆盖已有分析输出；只有显式 `--resume` 才允许恢复，且 source fingerprint（包含组身份及所有关键输入 SHA）不一致时立即拒绝复用。输出目录若位于任一正式实验 root 内也会被拒绝。

## 7. 服务器运行命令

先进入部署后的仓库并设置固定命令变量：

```bash
cd /root/rag-cbwdm
PY=/root/miniconda3/envs/rag-cbwdm-mistral/bin/python
PLAN=configs/formal/evidence_token_analysis_48.development.server.yaml
FINAL_OUT=/root/experiments/rag_cbwdm/formal_v2_analysis/evidence_token_comparison_v1
INFOGAIN_OUT="$FINAL_OUT/infogain_validation"
```

### 7.1 48 组只读 source dry-run

```bash
"$PY" scripts/analyze_formal_evidence_tokens.py \
  --plan "$PLAN" \
  --dry-run
```

验收：显示 48 个组逐组 `PASS`，最后显示 `dry-run=PASS groups=48 tokenizer_loaded=false`。失败时不得继续正式统计。

### 7.2 corrected InfoGain 八组交叉验证

```bash
"$PY" scripts/analyze_formal_evidence_tokens.py \
  --plan "$PLAN" \
  --validate-infogain \
  --output-dir "$INFOGAIN_OUT"
```

中断后从相同输入恢复：

```bash
"$PY" scripts/analyze_formal_evidence_tokens.py \
  --plan "$PLAN" \
  --validate-infogain \
  --output-dir "$INFOGAIN_OUT" \
  --resume
```

验收：8 组全部 `PASS`，`token_audit.json` 的 `status` 为 `PASS`、`num_groups` 为 8，且每组重算 token 与正式 metrics 动态值一致。

### 7.3 48 组正式离线统计

仅在前两步通过后执行：

```bash
"$PY" scripts/analyze_formal_evidence_tokens.py \
  --plan "$PLAN" \
  --all \
  --output-dir "$FINAL_OUT"
```

中断恢复：

```bash
"$PY" scripts/analyze_formal_evidence_tokens.py \
  --plan "$PLAN" \
  --all \
  --output-dir "$FINAL_OUT" \
  --resume
```

建议完成后核对：

```bash
"$PY" - <<'PY'
import csv, json
from pathlib import Path

root = Path('/root/experiments/rag_cbwdm/formal_v2_analysis/evidence_token_comparison_v1')
rows = list(csv.DictReader((root / 'main_results_48.csv').open(encoding='utf-8')))
audit = json.loads((root / 'token_audit.json').read_text(encoding='utf-8'))
manifest = json.loads((root / 'analysis_manifest.json').read_text(encoding='utf-8'))
assert len(rows) == 48
assert len({(r['dataset_id'], r['generator_id'], r['method_id']) for r in rows}) == 48
assert audit['status'] == 'PASS' and audit['num_groups'] == 48
assert manifest['completed'] is True and manifest['num_groups'] == 48
print('evidence-token-analysis: PASS, rows=48')
PY
```

## 8. 本地测试结果

| 检查 | 结果 |
|---|---|
| 新增专项测试 | `15 passed` |
| formal/InfoGain/FEVER/FM2 跨模块专项回归 | `207 passed` |
| 全仓 pytest | `498 passed, 2 subtests passed` |
| `python -m compileall -q src scripts tests` | PASS |
| 全部 9 个 `.sh` 的 `bash -n` | PASS |
| `git diff --check`（本次新增文件） | PASS |

全仓 pytest 首次运行时，Windows 沙箱禁止仓库外系统临时目录写入，导致 51 个 `PermissionError`；将 `TEMP/TMP` 指向仓库内隔离目录后完整重跑，得到上述全通过结果。该问题不涉及测试断言或实现逻辑。

专项覆盖包括：No Evidence 零 token、标题/顺序/分隔符、`max_docs`、tokenizer identity 且只加载 tokenizer、prediction selected-doc mismatch、字符数 mismatch、动态 InfoGain token mismatch、train/validation 混淆、旧 InfoGain root、缺失 manifest、SHA 错误、重复 ID、48 组唯一性、输出隔离、resume identity mismatch，以及 CSV/LaTeX 输出。

## 9. 仍需服务器确认的风险

- 本地不存在 `/root` 正式产物，必须在服务器实际运行 `--dry-run` 才能确认所有路径、权限、manifest 和 SHA 当前仍一致。
- generator identity 校验会读取并哈希服务器模型/tokenizer artifact；这不加载权重，但需要这些目录可读，首次检查可能有明显磁盘 I/O。
- `--validate-infogain` 和 `--all` 依赖服务器环境已有兼容的 `transformers` 及四个本地 tokenizer；仍不需要 GPU。
- 正式结果只有在 corrected InfoGain 八组动态 token 交叉验证和全部 48 组 docs/chars/provenance 校验均通过后才可接受。
- `infogain_validation` 使用独立子目录，不与最终 48 组输出或任何正式实验产物相互覆盖。

当前工作区未 commit、未 push；等待确认后再进行 GitHub 同步和服务器只读验证。

# formal-v2 Evidence Token 离线分析就绪报告

日期：2026-10-11

基线分支：`feature/fever-formal-readiness`

当前 histogram 语义修复基线提交：`cf92693`

## 1. 结论

已完成 formal-v2 主实验 Evidence Token 的离线统计与 48 组统一汇总工具。实现只读取已完成的 matched-main validation 产物，只加载对应 generator 的 tokenizer，不加载 generator 模型、不运行 selector、不训练模型，也不修改任何正式实验产物。

本地 Windows 环境没有服务器 `/root/experiments/rag_cbwdm` 数据，因此本地仍只执行代码和 fixture 回归测试。根据服务器实测回报，48/48 source dry-run 和 8/8 corrected InfoGain token 交叉验证已经通过；正式 48 组 histogram 独立对账为 40 组完全一致、8 组仅含零频 bin 格式差异、0 组非零频数差异、0 组错误。

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
  - 冻结两个数据集、四个 generator、六个方法、seed 策略、tokenizer identity、expected rows、正式 source root 和逐方法 generator-manifest root。
  - loader 将配置展开为严格唯一的 48 个 source record，并拒绝非预期方法、seed、root、tokenizer 或 split。
- `scripts/analyze_formal_evidence_tokens.py`
  - 提供 `--dry-run`、`--validate-infogain`、`--all`、`--output-dir`、`--resume`。
  - 每组校验 source provenance、文件 SHA、行身份、实际 evidence 与既有 metrics 后再统计 token。
- `tests/test_evidence_token_offline_analysis.py`
  - 新增 15 个离线分析专项回归测试。
- `EVIDENCE_TOKEN_OFFLINE_ANALYSIS_READINESS.md`
  - 本报告。

未修改 generator、selector、训练或正式 evaluation 逻辑。

### 3.1 KCBWDM generator manifest 来源修复

服务器在提交 `77cdd7f` 上的首次 48 组 dry-run 暴露出：原分析计划把所有方法的 generator manifest 都强制映射到 `formal_v2_full_development`，但 KCBWDM v2A 的正式 evaluation binding 指向各自 KCBWDM 实验 root。该错误导致首个 FEVER KCBWDM 组在完整 artifact-binding 比较时失败。

修复后，计划中的 `generator_manifest_root` 成为独立冻结字段：五种非 KCBWDM 方法固定为 `main`，KCBWDM 固定为 `kcbwdm_v2a`。loader 同时用代码内正式协议白名单核对该字段，不能通过篡改 YAML 将任意 evaluation 记录路径变成可信来源。

generator provenance 仍由 `validate_generator_manifest()` 和 `validate_generator_identity_against_config()` 完整验证，包括 manifest SHA、dataset/generator identity、model/tokenizer artifact identity、config SHA 与 tokenizer 配置。evaluation manifest 与 contract 中的 artifact binding 仍和 `build_evaluation_binding()` 的完整结果做字典级严格相等比较，未跳过或放宽 KCBWDM 的 path/SHA 字段。

generator cache 的键是解析后的真实 manifest 绝对路径。因此同一 dataset/generator 在 main root 与 KCBWDM root 的两个 manifest 会形成不同缓存项，不会跨 root 复用。

### 3.2 稀疏 num_docs histogram 修复

服务器在提交 `820849d` 上运行正式 `--all` 时发现，既有 FEVER No Evidence `metrics.json` 记录 `{"0": 5000}`，而离线脚本曾强制补入 `"1"` 至 `"4"` 的零计数 bin，导致严格比较失败。提交 `cf92693` 改为稀疏重算后，No Evidence 等 40 组通过，但 corrected InfoGain 的八组正式 metrics 保留了零频 `"0"`、`"1"` bin，因未经规范化的字典比较在第 4/48 组失败。

服务器对全部 48 组 `metrics.json` 与 `predictions.jsonl` 的独立对账结论为：`EXACT=40`、`ZERO_BIN_ONLY=8`、`NONZERO_DIFFERENCE=0`、`ERROR=0`。八个 `ZERO_BIN_ONLY` 全部是覆盖 FEVER、FM2 和四个 generator 的 `infogain_fever`；其 row ID、`avg_num_docs` 和 predictions 重算结果均一致。

最终统一规则如下：`audit_evidence_rows()` 继续只记录实际出现的 bin；比较器要求两侧 histogram 均为字典，键必须是规范的 `"0"` 至 `"4"`，值必须是非 bool 的非负整数，并分别要求频数总和等于 expected rows。只有通过这些格式和总数校验后才移除零频 bin，并对剩余非零 bin 及频数执行严格相等比较。任何非零频数差异仍立即失败，即使 `avg_num_docs` 恰好相同也不例外。

CSV 的 `num_docs_0` 至 `num_docs_4` 是稳定的展示列，仍以 `histogram.get(key, 0)` 生成，因此稀疏 histogram 不会破坏表结构。`num_docs_histogram_basis=observed_num_docs_values_only` 继续描述离线重算结果；新增 `num_docs_histogram_comparison=zero_bins_elided_nonzero_counts_exact_v1` 记录比较语义。逐组缓存、token audit 和 analysis manifest 都携带该标识，旧比较语义的 group cache 会在 `--resume` 时被拒绝，不能静默绕过新校验。

## 4. 48 组 source mapping

固定维度为：

| 维度 | 值 |
|---|---|
| Dataset | `fever_binary_v2`（5000）；`fm2_official_closed_page_v1`（1169） |
| Generator | `qwen2.5-0.5b-instruct`；`qwen2.5-1.5b-instruct`；`qwen2.5-7b-instruct`；`mistral-7b-instruct-v0.3` |
| Split / experiment | `validation` / `matched_main` |

每个 dataset × generator 均使用以下六组映射：

| 方法 | Selection/evaluation root | Generator manifest root | Selection 路径（相对 dataset root） | Evaluation 路径（相对 dataset root） | Seed |
|---|---|---|---|---|---|
| `no_evidence` | main | main | `shared/no_evidence/validation/selection.jsonl` | `{generator}/evaluation/no_evidence/matched_main/deterministic/validation` | null |
| `retrieval_topk` | main | main | `shared/retrieval_topk/validation/selection.jsonl` | `{generator}/evaluation/retrieval_topk/matched_main/deterministic/validation` | null |
| `bge` | main | main | `shared/bge/validation/selection.jsonl` | `{generator}/evaluation/bge/matched_main/deterministic/validation` | null |
| `infogain_fever` | corrected InfoGain | main | `{generator}/infogain/seed13/selection/validation.jsonl` | `{generator}/evaluation/infogain/matched_main/seed13/validation` | 13 |
| `rag_cbwdm_signed_v1` | main | main | `{generator}/rag_cbwdm_signed_v1/seed13/selection/validation.jsonl` | `{generator}/evaluation/rag_cbwdm_signed_v1/matched_main/seed13/validation` | 13 |
| `kcbwdm_linear_gate_v2` | dataset-specific KCBWDM v2A | dataset-specific KCBWDM v2A | `{generator}/kcbwdm_linear_gate_v2/seed13/selection/validation.jsonl` | `{generator}/evaluation/kcbwdm_linear_gate_v2/matched_main/seed13/validation` | 13 |

Root 固定为：

| 名称 | 路径 |
|---|---|
| main | `/root/experiments/rag_cbwdm/formal_v2_full_development` |
| corrected InfoGain | `/root/experiments/rag_cbwdm/formal_v2_infogain_top4_filter02_final` |
| FEVER KCBWDM v2A | `/root/experiments/rag_cbwdm/formal_v2_fever_kcbwdm_linear_gate_v2_full_development` |
| FM2 KCBWDM v2A | `/root/experiments/rag_cbwdm/formal_v2_kcbwdm_linear_gate_v2_full_development` |

No Evidence、RAG、BGE、corrected InfoGain 和 CBWDM 的 generator manifest 绑定 `formal_v2_full_development/{dataset}/{generator}/generator/manifest.json`。FEVER KCBWDM 绑定 `formal_v2_fever_kcbwdm_linear_gate_v2_full_development/.../generator/manifest.json`，FM2 KCBWDM 绑定 `formal_v2_kcbwdm_linear_gate_v2_full_development/.../generator/manifest.json`。工具不会仅凭 evaluation manifest 的记录信任任意路径：记录的 selection path/SHA、完整 evaluation binding、generator identity 和 predictions/metrics 路径/SHA 必须与冻结计划及实际文件逐项一致。

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

本次来源映射修复部署后，重新执行的精确命令为：

```bash
cd /root/rag-cbwdm
/root/miniconda3/envs/rag-cbwdm-mistral/bin/python \
  scripts/analyze_formal_evidence_tokens.py \
  --plan configs/formal/evidence_token_analysis_48.development.server.yaml \
  --dry-run
```

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

当前服务器在 `cf92693` 上的 `--all` 已通过前三组并在第 4 组 corrected InfoGain 失败，因此旧输出目录内可能同时存在失败审计和三个旧比较语义的 group cache。新代码会按设计拒绝这些缓存；不得用旧目录强行绕过。保留旧失败目录，并使用全新的独立输出目录重新执行：

```bash
cd /root/rag-cbwdm
PY=/root/miniconda3/envs/rag-cbwdm-mistral/bin/python
PLAN=configs/formal/evidence_token_analysis_48.development.server.yaml
FINAL_OUT_HIST_V1=/root/experiments/rag_cbwdm/formal_v2_analysis/evidence_token_comparison_histogram_exact_v1
"$PY" scripts/analyze_formal_evidence_tokens.py \
  --plan "$PLAN" \
  --all \
  --output-dir "$FINAL_OUT_HIST_V1"
```

若仅中断了这次新目录运行，才对同一个 `FINAL_OUT_HIST_V1` 增加 `--resume`。正式验收不能仅依据进程或 tmux 退出状态。

建议完成后核对：

```bash
"$PY" - <<'PY'
import csv, json
from pathlib import Path

root = Path('/root/experiments/rag_cbwdm/formal_v2_analysis/evidence_token_comparison_histogram_exact_v1')
rows = list(csv.DictReader((root / 'main_results_48.csv').open(encoding='utf-8')))
audit = json.loads((root / 'token_audit.json').read_text(encoding='utf-8'))
manifest = json.loads((root / 'analysis_manifest.json').read_text(encoding='utf-8'))
assert len(rows) == 48
assert len({(r['dataset_id'], r['generator_id'], r['method_id']) for r in rows}) == 48
assert audit['status'] == 'PASS' and audit['num_groups'] == 48
assert audit['num_docs_histogram_basis'] == 'observed_num_docs_values_only'
assert audit['num_docs_histogram_comparison'] == 'zero_bins_elided_nonzero_counts_exact_v1'
no_evidence = next(
    group for group in audit['groups']
    if group['dataset_id'] == 'fever_binary_v2'
    and group['generator_id'] == 'qwen2.5-0.5b-instruct'
    and group['method_id'] == 'no_evidence'
)
assert no_evidence['num_docs_histogram'] == {'0': 5000}
assert manifest['completed'] is True and manifest['num_groups'] == 48
print('evidence-token-analysis: PASS, rows=48')
PY
```

## 8. 本地测试结果

| 检查 | 结果 |
|---|---|
| Evidence Token 专项测试 | `35 passed` |
| formal/InfoGain/FEVER/FM2 跨模块专项回归 | `227 passed` |
| 全仓 pytest | `518 passed, 2 subtests passed` |
| `python -m compileall -q src scripts tests` | PASS |
| 全部 9 个 `.sh` 的 `bash -n` | PASS |
| `git diff --check`（本次修改） | PASS |

全仓 pytest 首次运行时，Windows 沙箱禁止仓库外系统临时目录写入，导致 51 个 `PermissionError`；将 `TEMP/TMP` 指向仓库内隔离目录后完整重跑，得到上述全通过结果。该问题不涉及测试断言或实现逻辑。

专项覆盖包括：No Evidence 零 token、标题/顺序/分隔符、`max_docs`、tokenizer identity 且只加载 tokenizer、prediction selected-doc mismatch、字符数 mismatch、动态 InfoGain token mismatch、train/validation 混淆、旧 InfoGain root、缺失 manifest、SHA 错误、重复 ID、48 组唯一性、输出隔离、resume identity mismatch，以及 CSV/LaTeX 输出。来源映射测试覆盖 FEVER/FM2 KCBWDM 独立 manifest 路径、其他五种方法的 main-root 映射、冻结配置拒绝篡改、main/KCBWDM 缓存隔离、evaluation 记录路径偏离和 generator-manifest SHA mismatch。Histogram 测试覆盖 FEVER No Evidence `{"0": 5000}`、corrected InfoGain 的 dense-zero 与 sparse 表示等价、平均值相同但非零频数不同仍拒绝、非法字典/键、负数/浮点/bool 计数、总频数错误、旧比较语义缓存拒绝，以及稀疏 histogram 到固定 CSV 列的安全映射；corrected InfoGain 动态 token 校验保持通过。

## 9. 仍需服务器确认的风险

- 服务器已完成 48/48 source dry-run；部署新提交后仍建议先重跑只读 dry-run，确认部署路径、权限、manifest 和 SHA 未发生变化。
- generator identity 校验会读取并哈希服务器模型/tokenizer artifact；这不加载权重，但需要这些目录可读，首次检查可能有明显磁盘 I/O。
- `--validate-infogain` 和 `--all` 依赖服务器环境已有兼容的 `transformers` 及四个本地 tokenizer；仍不需要 GPU。
- 正式结果只有在 corrected InfoGain 八组动态 token 交叉验证和全部 48 组 docs/chars/provenance 校验均通过后才可接受。
- `infogain_validation` 使用独立子目录，不与最终 48 组输出或任何正式实验产物相互覆盖。

当前工作区未 commit、未 push；等待确认后再进行 GitHub 同步和服务器只读验证。

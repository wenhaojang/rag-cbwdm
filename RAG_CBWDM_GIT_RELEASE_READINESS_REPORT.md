# RAG-CBWDM Git Release Readiness Report

## 1. 结论

```text
READY TO PUSH AND DEPLOY FOR SERVER SMOKE
```

RAG-CBWDM 主流程已进入本地 server-ready commit，并从该 commit 的 annotated tag
完成独立 clean-clone 验收。本轮没有 push、没有创建或修改远程仓库，也没有运行真实
Qwen 或下载大型模型。

## 2. Git snapshot

| 项目 | 结果 |
|---|---|
| branch | `rag-cbwdm-server-ready` |
| commit | `3fc7f7f4f4d66751f71fa97a80d27f9385f0ec3a` |
| parent commit | `4848ba8fa6a08922b563c5939b39b69a0246d724` |
| commit message | `Implement server-ready RAG-CBWDM pipeline` |
| annotated tag | `rag-cbwdm-server-smoke-v0.1` |
| tag target | `3fc7f7f4f4d66751f71fa97a80d27f9385f0ec3a` |
| remote | `origin = https://github.com/wenhaojang/rag-cbwdm.git` |
| pushed in this task | **No** |
| tracked dirty state after commit | clean |
| local untracked after report | 论文原稿、两份 Codex task、本文报告 |

报告是在已验证 commit/tag 之后按任务书生成，因此不属于上述 tag。它不影响 server
pipeline 或 clean-clone 结论。

## 3. 文件分类

### 3.1 初始 modified/untracked 分类表

“机器路径/敏感”一列中的“否”表示未发现用户绝对路径、token、密码、私钥或 SSH
凭据；通用示例路径不算敏感信息。

| 路径 | 初始类型 | 建议/实际动作 | 理由 | 机器路径/敏感 | 进入 commit |
|---|---|---|---|---|---|
| `.gitignore` | modified | 更新并提交 | 补齐 cache、partial、命令历史等规则 | 否 | 是 |
| `.gitattributes` | untracked | 新增并提交 | 固定 Linux 文本文件 LF | 否 | 是 |
| `requirements.txt` | modified | 提交 | 主流程依赖与 pytest | 否 | 是 |
| `configs/fever2_minimal.yaml` | modified | 提交 | 现有兼容配置 | 否 | 是 |
| `configs/fever3_minimal.yaml` | modified | 提交 | 现有兼容配置 | 否 | 是 |
| `configs/fever2_server_smoke.yaml` | untracked | 提交 | server smoke 配置 | 否 | 是 |
| `configs/fever2_server_pilot.yaml` | untracked | 提交 | server pilot 配置 | 否 | 是 |
| `configs/fever2_server_formal.yaml` | untracked | 提交 | formal 配置模板 | 否 | 是 |
| `scripts/03_compute_label_posteriors.py` | untracked | 提交 | batched/resumable posterior | 否 | 是 |
| `scripts/04_build_cbwdm_teacher.py` | untracked | 提交 | v2 teacher | 否 | 是 |
| `scripts/05_train_selector.py` | untracked | 提交 | feature-MLP 兼容训练 | 否 | 是 |
| `scripts/06_select_with_selector.py` | untracked | 提交 | feature-MLP inference | 否 | 是 |
| `scripts/07_eval_rag_classification.py` | untracked | 提交 | 统一 evaluator | 否 | 是 |
| `scripts/08_select_naive_topm.py` | untracked | 提交 | sanity baseline | 否 | 是 |
| `scripts/09_select_cbwdm_oracle_from_teacher.py` | untracked | 提交 | oracle diagnostic | 否 | 是 |
| `scripts/10_train_cross_encoder_selector.py` | untracked | 提交 | state-aware selector 训练 | 否 | 是 |
| `scripts/11_select_with_cross_encoder.py` | untracked | 提交 | greedy selector inference | 否 | 是 |
| `scripts/check_environment.py` | untracked | 提交 | server 预检 | 否 | 是 |
| `scripts/download_fever_hf.py` | untracked | 提交 | FEVER 下载辅助 | 仅通用 `D:/hf_cache` 示例 | 是 |
| `scripts/run_fever_cbwdm.py` | untracked | 提交 | manifest-aware orchestrator | 否 | 是 |
| `scripts/run_fever_cbwdm.sh` | untracked | 提交 | Linux 统一入口 | 否 | 是 |
| `src/cbwdm_score.py` | untracked | 提交 | CBWDM 数值核心 | 否 | 是 |
| `src/label_logits.py` | untracked | 提交 | batched label posterior | 否 | 是 |
| `src/metrics.py` | untracked | 提交 | metrics v2 | 否 | 是 |
| `src/prompts.py` | untracked | 提交 | prompt 及 hash | 否 | 是 |
| `src/run_manifest.py` | untracked | 提交 | provenance/manifest | 否 | 是 |
| `src/selection_schema.py` | untracked | 提交 | selection schema | 否 | 是 |
| `src/selector_cross_encoder.py` | untracked | 提交 | manuscript selector loss/model | 否 | 是 |
| `src/selector_dataset.py` | untracked | 提交 | feature selector dataset | 否 | 是 |
| `src/selector_model.py` | untracked | 提交 | feature selector model | 否 | 是 |
| `tests/__init__.py` | 本轮新增 | 提交 | 默认 unittest discovery | 否 | 是 |
| `tests/test_cbwdm_score.py` | untracked | 提交 | 数值/teacher tests | 否 | 是 |
| `tests/test_label_logits_and_manifest.py` | untracked | 提交 | batch/resume tests | 否 | 是 |
| `tests/test_selector_loss_and_schema.py` | untracked | 提交 | selector/schema tests | 否 | 是 |
| `tests/fixtures/posteriors/fever2_toy_posteriors.jsonl` | untracked | 提交 | 2-row public fixture | 否 | 是 |
| `RAG_CBWDM_THEORY_CODE_ALIGNMENT.md` | untracked | 提交 | 理论—代码对齐依据 | 否 | 是 |
| `RAG_CBWDM_SERVER_READINESS_REPORT.md` | untracked | 清理路径后提交 | server readiness 记录 | 初始含本机路径，已替换 | 是 |
| `SERVER_RUNBOOK.md` | untracked | 提交 | 部署操作手册 | 仅通用 server 路径 | 是 |
| `PROJECT_STATUS_REPORT.md` | untracked | 提交 | 历史项目审计 | 含通用 Windows/Unix 示例 | 是 |
| `INFOGAIN_RAG_INTEGRATION_AUDIT.md` | untracked | 清理路径后提交 | 方法关联审计 | 初始含本机路径，已替换 | 是 |
| `RAG-CBWDM.tex` | untracked | 保留本地，不提交 | 原稿含既有 trailing whitespace；禁止为格式检查改写论文 | 未发现凭据 | 否 |
| `RAG_CBWDM_SERVER_READINESS_CODEX_TASK.md` | untracked | 保留本地 | 任务说明，不是运行依赖 | 否 | 否 |
| `RAG_CBWDM_GIT_SNAPSHOT_AND_CLEAN_CLONE_CODEX_TASK.md` | untracked | 保留本地 | 当前任务说明 | 否 | 否 |
| `运行过的指令/` | untracked | 精确 ignore，不删除 | 14 个历史任务/命令文件，192,404 bytes，含本机路径 | 有机器路径，无凭据 | 否 |

### 3.2 Committed production/config/tests/docs

- Production：Stage 03–11、environment check、FEVER download helper、Python/Bash runner、
  CBWDM/posterior/selector/schema/manifest/metrics 模块。
- Config：FEVER-2/3 minimal 和 smoke/pilot/formal server configs。
- Tests：3 个测试模块、`tests/__init__.py`、2-row posterior fixture；原有 BM25/FEVER
  fixtures 由 parent commit 提供。
- Docs：theory-code alignment、server readiness、runbook、project status、InfoGain audit。
- Git policy：`.gitattributes`、强化后的 `.gitignore`。

### 3.3 Ignored local assets

已确认 ignore：

```text
__pycache__/
*.pyc
.venv/
data/
outputs/
checkpoints/
cache/
.cache/
wandb/
.pytest_cache/
*.log
*.faiss
*.pt
*.bin
*.safetensors
*.jsonl.partial
.DS_Store
运行过的指令/
```

没有删除任何上述本地资产。

## 4. 测试结果

### 4.1 原工作区

| 检查 | 结果 |
|---|---|
| `pip install -r requirements.txt` | PASS；首次沙箱网络受限，授权联网后仅补装 pytest 及小依赖 |
| `pytest -q` | PASS，12 passed |
| `python -m unittest discover` | PASS，12 tests |
| `py_compile` | PASS，29 Python files |
| `bash -n scripts/run_fever_cbwdm.sh` | PASS |
| environment check | PASS for CPU checks；CUDA unavailable 是当前 CPU 环境事实 |
| runner dry-run | PASS，全部 11 stages |
| cached whitespace check | PASS |
| staged sensitive scan | PASS；仅代码变量名误报 |

PowerShell 首次直接启动非-login Git Bash 时，Git Unix 工具 PATH 不完整，`dirname`
不可见；改用真实 login Git Bash 后同一 runner 命令 PASS。Linux 脚本逻辑未修改。

### 4.2 Clean clone

Clean clone：

```text
<TEMP>/rag_cbwdm_clean_verify_20260717_1854
```

| 检查 | 结果 |
|---|---|
| checkout | annotated tag `rag-cbwdm-server-smoke-v0.1` |
| HEAD | `3fc7f7f4f4d66751f71fa97a80d27f9385f0ec3a` |
| initial `git status --short` | empty |
| required pipeline files | PASS |
| forbidden data/output/cache/venv/weights | absent before tests |
| `pytest -q` | PASS，12 passed |
| `python -m unittest discover` | PASS，12 tests |
| `py_compile` | PASS，29 Python files |
| Bash syntax | PASS |
| environment check | PASS for CPU checks |
| full runner dry-run | PASS，全部 11 stages |
| final non-ignored Git status | clean |
| generated ignored state | only `.pytest_cache/`, `__pycache__/`, `outputs/`, `cache/` |

没有执行真实 Qwen、GPU 或 formal experiment。

## 5. LF / CRLF

Committed `.gitattributes`：

```gitattributes
* text=auto

*.sh    text eol=lf
*.py    text eol=lf
*.yaml  text eol=lf
*.yml   text eol=lf
*.json  text eol=lf
*.jsonl text eol=lf
*.md    text eol=lf
*.tex   text eol=lf
```

检查结果：

```text
scripts/run_fever_cbwdm.sh: text: set
scripts/run_fever_cbwdm.sh: eol: lf
committed shell CRLF pairs: 0
committed shell LF bytes: 6
```

因此不存在 `/bin/bash^M` 风险，也没有修改 Git 全局 `core.autocrlf`。

## 6. 敏感信息检查

- staged 内容未发现 Hugging Face token、密码、私钥或用户 home 绝对路径。
- 两份待提交报告中的本机 workspace 路径已替换为 `<WORKSPACE>`。
- `/path/to/big_disk`、`D:/hf_cache` 是文档/CLI 的通用示例，不是用户凭据。
- `tokenizer.pad_token`、`answer_token` 是代码变量，不是 secrets。
- `运行过的指令/` 含本机绝对路径，因此保持本地并被精确 ignore；未发现凭据。
- 没有提交 `.venv`、数据、输出、cache、checkpoint、模型权重或 FAISS index。

## 7. 用户下一步动作（本轮未执行）

远程 `origin` 已存在。建议用户先确认：

```bash
git remote -v
```

然后由用户手动发布：

```bash
git push -u origin rag-cbwdm-server-ready
git push origin rag-cbwdm-server-smoke-v0.1
```

如果需要将本文报告也纳入 Git，应先单独审阅并创建后续 documentation commit；不要移动
或覆盖已经 clean-clone 验证的 tag。

## 8. 服务器部署入口

以 committed [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md) 为准。服务器 clone 后首轮：

```bash
python scripts/check_environment.py \
  --config configs/fever2_server_smoke.yaml \
  --output-root /mnt/experiments/rag_cbwdm \
  --cache-root /mnt/fast-cache/huggingface

bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_smoke.yaml \
  --run-name fever2_smoke_seed13 \
  --stages prepare,corpus,retrieve,posterior,teacher,train,select,eval \
  --limit 100 \
  --seed 13 \
  --output-root /mnt/experiments/rag_cbwdm \
  --cache-root /mnt/fast-cache/huggingface \
  --posterior-batch-size 2 \
  --selector-batch-size 4
```

当前结论只覆盖 server smoke 部署准备，不代表 BW local metric、formal 超参数或 GPU
吞吐已经验证。

# RAG-CBWDM Master Server Runbook

**版本日期：** 2026-08-24  
**适用范围：** 从一台全新的 MatPool/Ubuntu GPU 服务器恢复 RAG-CBWDM，运行 FEVER2 与 FM2、`rag_cbwdm_signed_v1` 及核心 baselines，并在释放服务器前完整归档。  
**唯一部署提交变量：** `FINAL_DEPLOY_COMMIT=<FM2 + signed-v1 + 当前全部需要代码最终 push 后的 clean commit>`  
**重要：** 上述值当前故意是 placeholder。`4b914d...`、`3d3aa4...` 等只属于历史 provenance，不是未来部署目标。

---

## 0. 本文复现什么

本文是服务器操作的唯一主流程。冲突按以下顺序裁决：

1. 当前仓库代码、真实 `--help`、config 与 `environment/server/*.yml`；
2. `RAG_CBWDM_SERVER_REALITY_ALIGNMENT_NOTES_2026-08-24.md` 中已在服务器验证的事实；
3. 最新 signed-v1/FM2 implementation report 与 experiment runbook；
4. 2026-08-20 新服务器 preflight/two-env 文档；
5. 更旧文档与 commit 只作历史出处。

当前方法状态：

- FEVER2 原有已冻结 pipeline 不因本文改变；正式 split 为 `train_core`（训练）、`validation`（开发/校准）、`preformal_eval`（干净内部比较，禁止再校准）、`held_out_test`（冻结后一次性最终评估）。
- `rag_cbwdm_signed_v1` 是独立 experimental formal method，不替换旧 `rag_cbwdm`。
- FM2 第一轮是 `fm2_official_closed_page_v1`：所有方法共享官方 `retrieved_evidence` 候选池；它是 construction-gold-free，但依赖已知 `wikipedia_page`，不是开放域检索协议。
- deployable 核心方法为 `no_evidence`、`naive_topm`、BGE、项目内 InfoGain classification adaptation、`rag_cbwdm_signed_v1`。旧 `rag_cbwdm` 仅在精确旧训练合约可恢复时作 repair/control。
- `signed_gate_oracle` 使用 gold，只能诊断，永远不得混入 deployable 排名。
- 禁止 inference gold leakage、用 held-out/test 调参、根据 clean preformal 结果静默改冻结参数、把 InfoGain adaptation 冒充论文 exact reproduction。

本文命令会下载/计算数据；执行者应逐 Stage 过门。本次编写本文时没有下载数据/模型、建索引、算 posterior、训练或运行 GPU evaluation。

---

## 1. 发布提交与 Git 硬门

### Stage 1 — Clone 精确 release

**Purpose:** 只运行最终已 push 的、干净的部署提交。  
**Environment:** base shell。  
**GPU:** 否。  
**Inputs:** GitHub 仓库与发布负责人给出的完整 40 位 SHA。  
**Outputs:** `/root/rag-cbwdm`。  
**First run:** 如下。  
**Resume:** 网络中断后重新 `fetch`；不要 rebase/merge。  
**Completion check:** HEAD 等于变量且 `git status --porcelain` 为空。  
**Expected invariant:** 不把历史开发 HEAD 当作 release。  
**Common failure:** placeholder 未替换、GitHub HTTP/2 失败、工作树非空。  
**Next gate:** 目录/环境变量。

```bash
export FINAL_DEPLOY_COMMIT='<FM2 + signed-v1 + current needed code final pushed clean commit>'
[[ "$FINAL_DEPLOY_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || { echo 'BLOCKED: set the final pushed 40-char commit SHA'; exit 2; }
git -c http.version=HTTP/1.1 ls-remote https://github.com/wenhaojang/rag-cbwdm.git HEAD
git -c http.version=HTTP/1.1 clone https://github.com/wenhaojang/rag-cbwdm.git /root/rag-cbwdm
cd /root/rag-cbwdm && git -c http.version=HTTP/1.1 fetch --all --tags --prune && git checkout --detach "$FINAL_DEPLOY_COMMIT"
cd /root/rag-cbwdm && test "$(git rev-parse HEAD)" = "$FINAL_DEPLOY_COMMIT" && test -z "$(git status --porcelain)" && echo 'GIT RELEASE PASS'
```

每次开始实验会话都记录：

```bash
cd /root/rag-cbwdm && pwd && git branch --show-current && git rev-parse HEAD && git status --short && git log -5 --oneline
```

正式服务器上应处于 detached HEAD 或明确 release branch；无论哪种，最终硬门都是 SHA 相等和 clean worktree。

---

## 2. 统一目录、Python 与 tmux

```bash
export REPO=/root/rag-cbwdm
export EXP_ROOT=/root/experiments/rag_cbwdm
export HF_HOME=/root/huggingface
export MODEL_ROOT=/root/models
export MINICONDA_ROOT=/root/miniconda3
export RETRIEVAL_ENV=rag-cbwdm-retrieval
export BASELINE_ENV=rag-cbwdm-baselines
export RETR_PY="$MINICONDA_ROOT/envs/$RETRIEVAL_ENV/bin/python"
export BASE_PY="$MINICONDA_ROOT/envs/$BASELINE_ENV/bin/python"
mkdir -p "$EXP_ROOT" "$HF_HOME" "$MODEL_ROOT" /root/rag-cbwdm-recovery-logs
```

所有长任务都放进 tmux；一个实验 run 使用一个明确 session：

```bash
tmux new -s rag-cbwdm
```

SSH 断线后：

```bash
tmux ls
tmux attach -t rag-cbwdm
```

不要因为 SSH 断线就启动第二个相同进程。先在 tmux 内检查原进程和 log。

---

## 3. 全新实例 preflight 与网络

### Stage 2 — Hardware / OS / storage gate

**Purpose:** 在下载或创建环境前确认实例可承担当前工作负载。  
**Environment:** base shell。  
**GPU:** 仅查询。  
**Inputs:** Ubuntu 22.04+、NVIDIA driver。  
**Outputs:** 保存到 recovery log 的只读盘点。  
**First run/Resume:** 同一组只读命令。  
**Completion check:** GPU 可见，最大显存至少 24000 MiB，RAM 至少 64 GiB，目标盘空闲至少 500 GiB。  
**Expected invariant:** 驱动可用；无需系统 CUDA toolkit 才能使用 pip CUDA Torch。  
**Common failure:** 买到小显存实例、`/root` 实际落在小系统盘。  
**Next gate:** 网络。

```bash
cat /etc/os-release
uname -a
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
free -h
df -hT /root
lsblk -o NAME,SIZE,FSTYPE,TYPE,MOUNTPOINTS,MODEL
```

若低于上述资源，停止或显式重新规划；不要把 OOM 当成方法失败。

### Stage 3 — GitHub / HF / Conda / PyPI network gate

**Purpose:** 在大下载前区分网络问题与环境问题。  
**Environment:** base shell。  
**GPU:** 否。  
**Outputs:** HTTP header 与 `ls-remote` 输出。  
**Completion check:** GitHub 可列远端；HF config endpoint、两个 TUNA Conda endpoint、TUNA PyPI endpoint 返回可用 HTTP 状态。  
**Common failure:** GitHub HTTP/2 reset、MatPool Aliyun channel 404、PyTorch special index 被误用作普通 PyPI。  
**Next gate:** Conda 配置。

```bash
git -c http.version=HTTP/1.1 ls-remote https://github.com/wenhaojang/rag-cbwdm.git HEAD
curl -I -L --connect-timeout 15 --max-time 30 https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/config.json
curl -I --connect-timeout 15 --max-time 30 https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/repodata.json
curl -I --connect-timeout 15 --max-time 30 https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/repodata.json
curl -I --connect-timeout 15 --max-time 30 https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/tqdm/
curl -I --connect-timeout 15 --max-time 30 https://download.pytorch.org/whl/cu124/torch/
```

`curl -I` 只测试网络，不构成 revision/content 验证。

---

## 4. MatPool `.condarc` 与 Miniconda

### Stage 4 — 移走 Aliyun 污染并建立项目 Conda 配置

**Purpose:** 避免 MatPool 自动生成的 Aliyun channel 造成 404 或混源。  
**Environment:** base shell。  
**GPU:** 否。  
**First run:** 备份后移走 `/root/.condarc`；建立项目配置。  
**Resume:** 每次重新登录和每次 `conda create` 前重新检查。  
**Completion check:** `--show-sources` 不含 `mirrors.aliyun.com`。  
**Expected invariant:** TUNA main + conda-forge；不加入本项目不需要且曾 404 的 `pkgs/r`。  
**Common failure:** MatPool 登录钩子重新生成 `/root/.condarc`。  
**Next gate:** 双环境恢复。

```bash
if [ -f /root/.condarc ]; then mv /root/.condarc "/root/.condarc.matpool-$(date +%Y%m%d-%H%M%S)"; fi
```

```bash
cat > /root/rag-cbwdm-condarc <<'EOF_CONDARC'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF_CONDARC
```

```bash
CONDARC=/root/rag-cbwdm-condarc "$MINICONDA_ROOT/bin/conda" config --show-sources
CONDARC=/root/rag-cbwdm-condarc "$MINICONDA_ROOT/bin/conda" config --show channels default_channels custom_channels
```

如果尚无 Miniconda，先从官方发布页取得当前 installer 的 SHA-256，再执行；不得跳过 hash：

```bash
export MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
export MINICONDA_SHA256='<verified installer sha256>'
[[ "$MINICONDA_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || { echo 'BLOCKED: verified 64-char Miniconda SHA required'; exit 2; }
curl -fL --retry 5 "$MINICONDA_URL" -o /tmp/miniconda-installer.sh && echo "$MINICONDA_SHA256  /tmp/miniconda-installer.sh" | sha256sum -c - && bash /tmp/miniconda-installer.sh -b -p "$MINICONDA_ROOT"
```

`scripts/20_bootstrap_new_server.sh` 当前会把 `openjdk-21-jre-headless` 放进 apt 列表，并且全量 pip 可能继承 PyTorch extra index。主流程因此采用下节的 split-install；不要在未检查 Conda retrieval Java 前为 Pyserini安装另一个系统 JDK。

---

## 5. 从当前 YAML 恢复双环境

当前权威版本摘要：

| 环境 | Python | Torch | Transformers | Java | Pyserini |
|---|---:|---:|---:|---:|---:|
| retrieval | 3.12.13 | 2.9.1+cpu | 5.14.1 | OpenJDK/Javac 21.0.11 | 2.3.0 |
| baselines | 3.12.13 | 2.6.0+cu124 | 4.57.6 | 不使用 | 必须不可导入 |

不要用 `*.from-history.yml.template`。以这两个文件为准：

```text
$REPO/environment/server/rag-cbwdm-retrieval.yml
$REPO/environment/server/rag-cbwdm-baselines.yml
```

先从 YAML 自动生成临时 conda-only 与 pip 文件：

```bash
cd "$REPO" && "$MINICONDA_ROOT/bin/python" - <<'PY'
from pathlib import Path
specs = [
    (Path('environment/server/rag-cbwdm-retrieval.yml'), Path('/root/retrieval-conda-only.yml'), Path('/root/retrieval-pip-requirements.txt')),
    (Path('environment/server/rag-cbwdm-baselines.yml'), Path('/root/baseline-conda-only.yml'), Path('/root/baseline-pip-requirements.txt')),
]
for src, conda_out, pip_out in specs:
    lines = src.read_text(encoding='utf-8-sig').splitlines()
    conda_lines = []
    for line in lines:
        if line.startswith('prefix:'):
            continue
        if line.strip() == '- pip:':
            break
        conda_lines.append(line)
    in_pip = False
    pip_lines = []
    for line in lines:
        if line.strip() == '- pip:':
            in_pip = True
            continue
        if in_pip and line.startswith('      - '):
            pip_lines.append(line[8:])
    conda_out.write_text('\n'.join(conda_lines) + '\n', encoding='utf-8')
    pip_out.write_text('\n'.join(pip_lines) + '\n', encoding='utf-8')
    print(src, 'pip_count=', len(pip_lines))
PY
```

预期逻辑记录数：retrieval 138，baseline 67。若不同，先重新读取当前 YAML；不要继续复制旧数字。

### Stage 5 — retrieval Conda 环境

**Purpose:** 建立只负责 corpus/index/BM25 的 CPU/Java 环境。  
**Environment:** Conda create + `$RETR_PY`。  
**GPU:** 否。  
**Inputs:** 当前 retrieval YAML。  
**Outputs:** `$MINICONDA_ROOT/envs/$RETRIEVAL_ENV`。  
**First run:** conda-only → CPU Torch → 其余 pip。  
**Resume:** pip 自身可重试；若已有同名环境，先审计，不要盲删。  
**Completion check:** pip check、版本、下一节 Java 硬门。  
**Common failure:** `.condarc` 再出现、普通包被送到 PyTorch index。  
**Next gate:** Java/Javac/Pyserini。

```bash
CONDARC=/root/rag-cbwdm-condarc "$MINICONDA_ROOT/bin/conda" env create -n "$RETRIEVAL_ENV" -f /root/retrieval-conda-only.yml
export RETR_PY="$MINICONDA_ROOT/envs/$RETRIEVAL_ENV/bin/python"
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null "$RETR_PY" -m pip install --index-url https://download.pytorch.org/whl/cpu --timeout 120 --retries 5 --no-deps 'torch==2.9.1+cpu'
grep -v '^torch==2\.9\.1+cpu$' /root/retrieval-pip-requirements.txt > /root/retrieval-pip-requirements-no-torch.txt
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null "$RETR_PY" -m pip install --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --timeout 120 --retries 5 -r /root/retrieval-pip-requirements-no-torch.txt 2>&1 | tee -a /root/rag-cbwdm-recovery-logs/retrieval_pip_install.log
"$RETR_PY" -m pip check
```

### Stage 6 — Java 21 + Javac 21 + real Pyserini hard gate

**Purpose:** 修正系统 Java 11/缺失 javac 与 Conda Java 21 的已知错位。  
**Environment:** retrieval。  
**GPU:** 否。  
**Inputs:** retrieval env。  
**Outputs:** PASS log。  
**First run/每个新 shell:** 必须重新 export。  
**Resume:** 同一只读 gate。  
**Completion check:** `java 21`、`javac 21`、真实 `LuceneSearcher` import 全部 PASS。  
**Expected invariant:** Pyserini/Lucene 不依赖 `/usr/bin/java`。  
**Common failure:** 只看 `java -version`；系统 shell 的 Java 11 掩盖 Conda Java；`import pyserini` 成功但 Lucene API 失败。  
**Next gate:** baseline 环境。

```bash
export JAVA_HOME="$MINICONDA_ROOT/envs/$RETRIEVAL_ENV"
export PATH="$JAVA_HOME/bin:$PATH"
test "$(command -v java)" = "$JAVA_HOME/bin/java" && test "$(command -v javac)" = "$JAVA_HOME/bin/javac"
java -version
javac -version
"$RETR_PY" -c "import importlib.metadata as m; assert m.version('pyserini') == '2.3.0'; from pyserini.search.lucene import LuceneSearcher; print('pyserini_java_import=PASS')"
```

服务器已验证的 base shell 可能仍为 `/usr/bin/java` → Java 11 且无 javac；这不构成失败。失败条件是上述 Conda contract 不能提供 Java/Javac 21 或真实 import 失败。

### Stage 7 — baseline CUDA 环境

**Purpose:** 建立 posterior、selector、BGE、InfoGain、Qwen evaluation 环境。  
**Environment:** `$BASE_PY`。  
**GPU:** 是，验收时必须 CUDA 可用。  
**Inputs:** 当前 baseline YAML。  
**Outputs:** `$MINICONDA_ROOT/envs/$BASELINE_ENV`。  
**First run:** conda-only → cu124 三件套 → 其余 pip。  
**Resume:** pip 可重试；已有环境先审计。  
**Completion check:** exact core versions、CUDA true、pip check、Pyserini isolation。  
**Common failure:** 忘记安装 `nvidia-*` requirements 就测试 Torch；装成 CPU Torch。  
**Next gate:** freeze audit。

```bash
test ! -f /root/.condarc || { echo 'BLOCKED: MatPool .condarc returned'; exit 2; }
CONDARC=/root/rag-cbwdm-condarc "$MINICONDA_ROOT/bin/conda" env create -n "$BASELINE_ENV" -f /root/baseline-conda-only.yml
export BASE_PY="$MINICONDA_ROOT/envs/$BASELINE_ENV/bin/python"
grep -vE '^(torch|torchaudio|torchvision)==.*$' /root/baseline-pip-requirements.txt > /root/baseline-pip-requirements-no-torch.txt
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null "$BASE_PY" -m pip install --index-url https://download.pytorch.org/whl/cu124 --timeout 120 --retries 5 --no-deps 'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124' 'torchaudio==2.6.0+cu124'
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null "$BASE_PY" -m pip install --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --timeout 120 --retries 5 -r /root/baseline-pip-requirements-no-torch.txt 2>&1 | tee -a /root/rag-cbwdm-recovery-logs/baseline_pip_install.log
"$BASE_PY" -m pip check
```

```bash
"$BASE_PY" - <<'PY'
import torch, torchvision, torchaudio, transformers
print(torch.__version__, torch.version.cuda, torchvision.__version__, torchaudio.__version__, transformers.__version__)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.__version__ == '2.6.0+cu124'
assert torchvision.__version__ == '0.21.0+cu124'
assert torchaudio.__version__ == '2.6.0+cu124'
assert transformers.__version__ == '4.57.6'
assert torch.cuda.is_available()
PY
if "$BASE_PY" -c 'import pyserini' >/dev/null 2>&1; then echo 'BLOCKED: baseline imports Pyserini'; exit 2; else echo 'BASELINE PYSERINI ISOLATION PASS'; fi
```

### Stage 8 — pip-freeze audit

**Purpose:** 比较恢复环境与真实服务器 freeze。  
**Environment:** 两个环境。  
**Completion check:** 规范化大小写后的 package/version diff 为空；任何差异均记录并在实验前裁决。  
**Expected invariant:** freeze 是审计文件，不是盲目的 CUDA 安装 requirements。

```bash
diff -u <(sed '/^[[:space:]]*$/d' "$REPO/environment/server/rag-cbwdm-retrieval-pip-freeze.txt" | tr '[:upper:]' '[:lower:]' | sort) <("$RETR_PY" -m pip freeze | sed '/^[[:space:]]*$/d' | tr '[:upper:]' '[:lower:]' | sort)
diff -u <(sed '/^[[:space:]]*$/d' "$REPO/environment/server/rag-cbwdm-baselines-pip-freeze.txt" | tr '[:upper:]' '[:lower:]' | sort) <("$BASE_PY" -m pip freeze | sed '/^[[:space:]]*$/d' | tr '[:upper:]' '[:lower:]' | sort)
```

当前 reproducibility gap：`scripts/20_bootstrap_new_server.sh`、`scripts/19_verify_resumed_server.sh`、继而 `scripts/23_verify_rebuilt_smoke.sh` 尚未同时检查 javac 21 与真实 `LuceneSearcher` import；所以即使它们 PASS，也不能替代 Stage 6。

---

## 6. 模型资产

当前 `scripts/21_download_project_assets.sh` 管理五个目录：

```text
$MODEL_ROOT/Qwen2.5-1.5B-Instruct
$MODEL_ROOT/Qwen2.5-7B-Instruct
$MODEL_ROOT/ms-marco-MiniLM-L-6-v2
$MODEL_ROOT/bge-reranker-base
$MODEL_ROOT/bge-reranker-large
```

FEVER2/FM2 当前主流程实际需要 Qwen 1.5B、MiniLM-L6、BGE-large；7B 与 BGE-base 仍属于当前项目资产清单，不要只因主流程未调用就伪造其 revision。

### Stage 9 — 下载/验证 pinned models

**Purpose:** 取得本地、可校验的模型快照。  
**Environment:** baseline。  
**GPU:** 下载否；actual-load gate 是。  
**Inputs:** 五个 immutable HF revision，优先来自已归档 `model_inventory.json`/asset manifest。  
**Outputs:** 五个本地模型目录与 `data/raw/fever/project_assets.manifest.json`。  
**First run:** 设置五个 revision 后运行 assets script。  
**Resume:** 脚本下载默认可恢复；相同 revision/manifest 可复用。  
**Completion check:** check-only、manifest file SHA、actual local weight load。  
**Common failure:** 用 `--allow-unpinned` 后仍声称 byte-identical reproduction。  
**Next gate:** 数据。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export QWEN15_REVISION='<immutable HF commit>' QWEN7_REVISION='<immutable HF commit>' MINILM_REVISION='<immutable HF commit>' BGE_BASE_REVISION='<immutable HF commit>' BGE_LARGE_REVISION='<immutable HF commit>'
for v in QWEN15_REVISION QWEN7_REVISION MINILM_REVISION BGE_BASE_REVISION BGE_LARGE_REVISION; do eval "x=\${$v}"; case "$x" in ''|\<*) echo "BLOCKED: $v"; exit 2;; esac; done
cd "$REPO" && bash scripts/21_download_project_assets.sh --models-only --resume
cd "$REPO" && bash scripts/21_download_project_assets.sh --models-only --check-only
```

若确实无法恢复 immutable revision，`--allow-unpinned` 只允许建立“functional recovery”，manifest 必须保留 `revision_mode=unpinned`；不得把它与 pinned 结果混为一谈。

实际加载三个主模型权重：

```bash
"$BASE_PY" - <<'PY'
import os, torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
root = os.environ['MODEL_ROOT']
q = f'{root}/Qwen2.5-1.5B-Instruct'
AutoTokenizer.from_pretrained(q, local_files_only=True)
m = AutoModelForCausalLM.from_pretrained(q, local_files_only=True, torch_dtype='auto', device_map='auto'); del m
for name in ('ms-marco-MiniLM-L-6-v2', 'bge-reranker-large'):
    p = f'{root}/{name}'
    AutoTokenizer.from_pretrained(p, local_files_only=True)
    m = AutoModelForSequenceClassification.from_pretrained(p, local_files_only=True); del m
if torch.cuda.is_available(): torch.cuda.empty_cache()
print('LOCAL MODEL LOAD PASS')
PY
```

---

## 7. FEVER2 数据、split、corpus 与 index

### Stage 10 — FEVER2 原始数据

**Purpose:** 下载 official train/dev/wiki pages 并固定 revision/content SHA。  
**Environment:** baseline。  
**GPU:** 否。  
**Inputs:** immutable `FEVER_DATASET_REVISION`。  
**Outputs:** `$REPO/data/raw/fever/{train.jsonl,dev.jsonl,wiki-pages/}` 与 asset manifest。  
**First run:** data-only。  
**Resume:** assets script 默认可恢复。  
**Completion check:** check-only 验证 manifest 中每个 size/SHA。  
**Common failure:** 未知 revision 时无意使用 `main`。  
**Next gate:** formal splits/corpus/index。

```bash
export FEVER_DATASET_REVISION='<immutable dataset revision>'
case "$FEVER_DATASET_REVISION" in ''|\<*) echo 'BLOCKED: FEVER_DATASET_REVISION'; exit 2;; esac
cd "$REPO" && bash scripts/21_download_project_assets.sh --data-only --resume
cd "$REPO" && bash scripts/21_download_project_assets.sh --data-only --check-only
```

FEVER2 formal split 的当前 config：`configs/fever2_server_pilot_5000_500.yaml`。它先按 seed 13、normalized-claim grouping 构造完整 partition，再对 pilot 应用 `train_core=5000`、`validation=500` 限制。`held_out_test` 来自 official dev，未获授权前禁止 retrieval/posterior/evaluation。

### Stage 11 — FEVER corpus/index build or reuse

**Purpose:** 建立 content-addressed sentence corpus 与 Pyserini Lucene v2 index。  
**Environment:** retrieval，且先执行 Stage 6 的 JAVA_HOME/PATH。  
**GPU:** 否。  
**Inputs:** wiki pages、config。  
**Outputs:** `$EXP_ROOT/_shared/corpora/.../fever_corpus_sentence.jsonl`、`$EXP_ROOT/_shared/indexes/<fingerprint>/index_manifest.json`。  
**First run:** 由 Stage 13 的 runner/scripts 22 构建。  
**Resume:** corpus/index 的 `--resume` 只验证并复用已完成 artifact；不是 row-level resume。  
**Completion check:** `validate_index()` 重新打开 searcher、核对 inventory、文档数、probe 与 stored raw metadata。  
**Expected invariant:** BM25 `k1=0.9,b=0.4`、English analyzer、storeRaw/Positions/Docvectors；run manifest 的 index fingerprint 一致。  
**Common failure:** incomplete 非空 index 目录没有 completed manifest；不得把它当成 resumable index。用新 fingerprint/path，或在人工确认无有效内容后显式 `--overwrite`。  
**Next gate:** smoke rebuild。

复用检查（`INDEX_PATH` 必须从 run manifest 读取，不猜目录）：

```bash
export JAVA_HOME="$MINICONDA_ROOT/envs/$RETRIEVAL_ENV"; export PATH="$JAVA_HOME/bin:$PATH"
export FEVER_RUN="$EXP_ROOT/fever2_formal_pilot_5000_500_seed13"
export INDEX_PATH="$("$BASE_PY" -c "import json; print(json.load(open('$FEVER_RUN/run_manifest.json'))['paths']['index_path'])")"
PYTHONPATH="$REPO" "$RETR_PY" -c "import sys; from src.retrieval.pyserini_bm25 import validate_index; m=validate_index(sys.argv[1]); print(m['fingerprint'],m['num_documents'])" "$INDEX_PATH"
```

绝不默认让 FM2 复用这个 index；见第 17 节。

---

## 8. FM2 数据与 candidate pool

### Stage 12 — 下载/准备 FM2 train/dev

**Purpose:** 把官方 FM2 JSONL 严格适配为 query、共享候选池、独立 gold diagnostic。  
**Environment:** baseline。  
**GPU:** 否。  
**Inputs:** Google Research `fool-me-twice` commit `d9db753e5acf91c0d9bf543db327ab655661eb94`。  
**Outputs:** `data/raw/fm2/{train,dev}.jsonl`、`data/processed/fm2/fm2_{train,dev}.jsonl`、`*_official_pool.jsonl`、`*_gold_diagnostics.jsonl`、`fm2_prepare.manifest.json`。  
**First run:** 不带 overwrite；默认只取 train/dev。  
**Resume:** downloader 会按 SHA 复用已完成文件；prepare 没有 resume，已有任一输出会 fail closed。  
**Completion check:** raw SHA、10419/1169 rows、1811/209 pages、页面零交集、gold 独立。  
**Expected invariant:** label identity mapping `SUPPORTS→SUPPORTS, REFUTES→REFUTES`；未知标签 fail closed。  
**Common failure:** selector 误读 `*_gold_diagnostics.jsonl`；用 `--allow-noncanonical-source` 却声称官方复现。  
**Next gate:** code/smoke gate。

```bash
cd "$REPO" && "$BASE_PY" scripts/download_fm2.py --output-dir data/raw/fm2 --splits train dev
cd "$REPO" && "$BASE_PY" scripts/prepare_fm2.py --raw-dir data/raw/fm2 --output-dir data/processed/fm2 --splits train dev
sha256sum "$REPO/data/raw/fm2/train.jsonl" "$REPO/data/raw/fm2/dev.jsonl"
```

预期 raw SHA：

```text
train 1f47e035650aa5734301afb36a94afa610af73ac908fd4c4009e281b9956a311
dev   eeb36a4757fb86412f1d7ad5197ffd6ffa837c1cf7fb3c0067d82d0b65257229
```

```bash
"$BASE_PY" - <<'PY'
import json
p='data/processed/fm2/fm2_prepare.manifest.json'
m=json.load(open(p))
assert m['status']=='completed'
assert m['label_mapping']=={'SUPPORTS':'SUPPORTS','REFUTES':'REFUTES'}
assert m['splits']['train']['num_rows']==10419 and m['splits']['dev']['num_rows']==1169
assert m['splits']['train']['num_pages']==1811 and m['splits']['dev']['num_pages']==209
assert m['page_overlaps']['dev/train']['count']==0
assert m['candidate_pool_contract']['gold_evidence_stored_separately'] is True
print('FM2 PREPARE PASS')
PY
```

候选池来自官方 `retrieved_evidence`，保持源顺序；full data 每行 1–12 个候选，全部少于 20。公平预算是“最多 4 篇”，候选少于 4 时使用全部可用候选。第一轮不建 FM2 Lucene index，也不复用 FEVER index；开放域扩展必须另行固定兼容的 FM2 Wikipedia snapshot/index。

---

## 9. 低成本代码 gate

### Stage 13 — Tests 与 CLI audit

**Purpose:** 在重计算前确认代码/release 可运行。  
**Environment:** baseline；shell syntax 用 bash。  
**GPU:** 否。  
**Outputs:** 保存测试 log。  
**First run/Resume:** 可重复。  
**Completion check:** compileall、pytest、unittest、diff check、四个 shell `bash -n` 全 PASS。  
**Common failure:** 在错误 Python 环境运行；release worktree 不干净。  
**Next gate:** FEVER smoke rebuild。

```bash
cd "$REPO" && "$BASE_PY" -m compileall -q src scripts tests
cd "$REPO" && "$BASE_PY" -m pytest -q
cd "$REPO" && "$BASE_PY" -m unittest discover
cd "$REPO" && git diff --check
cd "$REPO" && bash -n scripts/20_bootstrap_new_server.sh scripts/21_download_project_assets.sh scripts/22_rebuild_to_current_smoke.sh scripts/23_verify_rebuilt_smoke.sh scripts/run_fever_cbwdm.sh
```

关键 CLI 复核：

```bash
cd "$REPO" && "$BASE_PY" scripts/03_compute_label_posteriors.py --help && "$BASE_PY" scripts/07_eval_rag_classification.py --help && "$BASE_PY" scripts/preformal/25_materialize_signed_v1_teacher.py --help && "$BASE_PY" scripts/preformal/26_train_signed_v1.py --help && "$BASE_PY" scripts/preformal/27_select_signed_v1.py --help
cd "$REPO" && "$RETR_PY" scripts/02_retrieve_bm25.py --help && "$RETR_PY" scripts/02a_build_bm25_index.py --help
```

---

## 10. FEVER2 smoke rebuild 与 verify

当前 `scripts/22_rebuild_to_current_smoke.sh` 有一个 fresh-run gap：它对每个 runner 调用都加顶层 `--resume`；03 在新的 posterior manifest 不存在时会拒绝 `--resume`。以下两段式路径是当前代码的安全绕行，不修改 artifact 合约。

### Stage 14 — Retrieval side smoke rebuild

**Purpose:** 构造 split、corpus、index、train_core/validation BM25。  
**Environment:** retrieval + Stage 6 Java contract。  
**GPU:** 否。  
**Inputs:** FEVER raw data、pilot config、release SHA。  
**Outputs:** split manifest、shared corpus/index、两份 top20 retrieval、run manifest。  
**First run:** scripts 22 停在 `retrieve_validation`。  
**Resume:** 重跑相同命令；每个 artifact 按自身合约复用。  
**Completion check:** 两份 retrieval manifest completed，5000/500 rows，candidate max ≤20，index fingerprint 一致。  
**Expected invariant:** held-out 文件不存在。  
**Common failure:** Java export 未做；incomplete index 非原子恢复。  
**Next gate:** posterior 首跑。

```bash
export FEVER_CONFIG="$REPO/configs/fever2_server_pilot_5000_500.yaml"
export FEVER_RUN_NAME=fever2_formal_pilot_5000_500_seed13
export FEVER_RUN="$EXP_ROOT/$FEVER_RUN_NAME"
export EXPECTED_GIT_HEAD="$FINAL_DEPLOY_COMMIT"
export JAVA_HOME="$MINICONDA_ROOT/envs/$RETRIEVAL_ENV"; export PATH="$JAVA_HOME/bin:$PATH"
cd "$REPO" && bash scripts/22_rebuild_to_current_smoke.sh --stop-after retrieve_validation
```

```bash
"$BASE_PY" - <<'PY'
import json, os
r=os.environ['FEVER_RUN']
for name,n,split in [('fever2_train_core_bm25_top20',5000,'train_core'),('fever2_validation_bm25_top20',500,'validation')]:
    p=f'{r}/artifacts/formal/{name}.manifest.json'; m=json.load(open(p))
    assert m['completed'] and m['num_output_rows']==n and m['split']==split
    assert m['candidate_count_statistics']['max']<=20
print('FEVER RETRIEVAL PASS')
PY
```

### Stage 15 — Posterior first run, then resume smoke wrapper

**Purpose:** 首次正确创建两个 posterior manifest，再让 scripts 22 完成 smoke calibration 与 verify。  
**Environment:** baseline。  
**GPU:** 是。  
**Inputs:** 两份 retrieval、Qwen 1.5B。  
**Outputs:** train_core/validation posterior 与 manifests；InfoGain/old-RAG smoke candidate artifacts；rebuild report。  
**First run:** 两个 03 命令绝对不带 `--resume`。  
**Resume:** 只有 manifest 已由第一次调用写出后，才以完全相同命令加 `--resume`。  
**Completion check:** posterior status completed、5000/500 rows、input/prompt/verbalizer/model SHA 固定；InfoGain 与 old-RAG 各至少两个 completed smoke evaluations；当前-schema hard verify PASS。  
**Expected invariant:** held-out 未使用。  
**Common failure:** 首次直接加 resume；batch size 改变导致 fingerprint mismatch。  
**Next gate:** FEVER preformal。

FIRST RUN：

```bash
cd "$REPO" && "$BASE_PY" scripts/03_compute_label_posteriors.py --config "$FEVER_CONFIG" --split train_core --retrieval "$FEVER_RUN/artifacts/formal/fever2_train_core_bm25_top20.jsonl" --output "$FEVER_RUN/artifacts/formal/fever2_train_core_posteriors.jsonl" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --batch-size 16
cd "$REPO" && "$BASE_PY" scripts/03_compute_label_posteriors.py --config "$FEVER_CONFIG" --split validation --retrieval "$FEVER_RUN/artifacts/formal/fever2_validation_bm25_top20.jsonl" --output "$FEVER_RUN/artifacts/formal/fever2_validation_posteriors.jsonl" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --batch-size 16
```

INTERRUPTED RESUME（manifest 已存在后）：

```bash
cd "$REPO" && "$BASE_PY" scripts/03_compute_label_posteriors.py --config "$FEVER_CONFIG" --split train_core --retrieval "$FEVER_RUN/artifacts/formal/fever2_train_core_bm25_top20.jsonl" --output "$FEVER_RUN/artifacts/formal/fever2_train_core_posteriors.jsonl" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --batch-size 16 --resume
cd "$REPO" && "$BASE_PY" scripts/03_compute_label_posteriors.py --config "$FEVER_CONFIG" --split validation --retrieval "$FEVER_RUN/artifacts/formal/fever2_validation_bm25_top20.jsonl" --output "$FEVER_RUN/artifacts/formal/fever2_validation_posteriors.jsonl" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --batch-size 16 --resume
```

posterior 完成后让 wrapper 验证/复用并先跑 InfoGain smoke；停在该 stage 可避免 wrapper 尾部调用当前有 schema gap 的 verifier：

```bash
cd "$REPO" && bash scripts/22_rebuild_to_current_smoke.sh --stop-after infogain_smoke
```

再用 scripts 22 内同一真实 runner contract 跑 old-RAG smoke：

```bash
cd "$REPO" && "$BASE_PY" scripts/run_fever_cbwdm.py --config "$FEVER_CONFIG" --run-name "$FEVER_RUN_NAME" --stages run_calibration_grid --output-root "$EXP_ROOT" --cache-root "$HF_HOME" --resume --methods rag_cbwdm --candidate-limit 2 --max-training-candidates 1 --generator-model "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --selector-model "$MODEL_ROOT/ms-marco-MiniLM-L-6-v2" --selector-device cuda --infogain-device cuda --skip-completed --continue-on-error
```

当前-schema hard verify：

```bash
export INDEX_PATH="$("$BASE_PY" -c "import json; print(json.load(open('$FEVER_RUN/run_manifest.json'))['paths']['index_path'])")"
PYTHONPATH="$REPO" "$RETR_PY" -c "import sys; from src.retrieval.pyserini_bm25 import validate_index; m=validate_index(sys.argv[1]); assert m['completed'] and m['probe_passed']; print('INDEX PASS',m['fingerprint'],m['num_documents'])" "$INDEX_PATH"
"$BASE_PY" - <<'PY'
import hashlib,json,os,pathlib
run=pathlib.Path(os.environ['FEVER_RUN']); repo=pathlib.Path(os.environ['REPO'])
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
split=repo/'outputs/formal_splits/fever2_seed13/fever2_formal_splits.manifest.json'
from src.formal_splits import validate_split_manifest
validate_split_manifest(split)
rm=json.load(open(run/'run_manifest.json')); corpus=pathlib.Path(rm['paths']['corpus']); cm=json.load(open(rm['paths']['corpus_manifest']))
assert cm['completed'] is True and cm['output_sha256']==sha(corpus)
for role,n in [('train_core',5000),('validation',500)]:
    r=pathlib.Path(rm['paths']['formal_retrieval'][role]); m=json.load(open(r.with_suffix('.manifest.json')))
    assert m['completed'] and m['num_output_rows']==n and m['output_sha256']==sha(r)
    p=pathlib.Path(rm['paths']['formal_posteriors'][role]); pm=json.load(open(p.with_suffix('.manifest.json')))
    assert pm['status']=='completed' and pm['completed_rows']==n and pm['output_sha256']==sha(p)
    assert pm['provenance']['input_sha256']==sha(r)
for p in (pathlib.Path(rm['paths']['formal_retrieval']['held_out_test']),pathlib.Path(rm['paths']['formal_posteriors']['held_out_test'])):
    assert not p.exists() and not p.with_suffix('.manifest.json').exists(), p
c=json.load(open(run/'artifacts/formal/calibration_candidates.json')).get('candidates',[])
for method in ('infogain_fever','rag_cbwdm'):
    ok=[x for x in c if x.get('method')==method and x.get('status')=='completed' and isinstance(x.get('metrics'),dict) and x['metrics'].get('accuracy') is not None and x['metrics'].get('macro_f1') is not None]
    assert len(ok)>=2,(method,len(ok))
print('CURRENT-SCHEMA FEVER SMOKE PASS')
PY
```

仍要运行并保存官方 verifier 输出：

```bash
cd "$REPO" && bash scripts/23_verify_rebuilt_smoke.sh || true
```

当前代码的 corpus builder 写 `completed=true` 但不写 `status`，而 `scripts/19_verify_resumed_server.sh`（由 scripts 23 调用）还要求 corpus `status=completed`。因此 fresh rebuild 上 scripts 23 可能仅因该字段 BLOCKED。只有当输出的唯一剩余 blocker 确实是这个已知 schema mismatch、上面的 current-schema hard verify PASS、Stage 6 PASS 时，才可记录为 verifier automation gap；任何其他 blocker 都必须先解决。

---

## 11. FEVER2 normal / formal / pilot / clean preformal 边界

### 11.1 工作流角色

| 角色 | 数据 | 允许用途 | 禁止用途 |
|---|---|---|---|
| legacy normal/dev | `train`,`dev` | 代码/API smoke、历史兼容 | 新 formal claim |
| formal pilot | `train_core=5000`,`validation=500` | training、calibration、方法开发 | 最终 test claim |
| clean preformal | 从冻结 full validation 去掉已用 pilot groups，预计约 4500，实际数以 manifest 为准 | 冻结方法比较、paired statistics | 调参、更新 frozen parameters |
| held-out | official-dev `held_out_test` | 算法/参数完全冻结后的一次最终评估 | 训练、校准、seed/threshold 选择 |
| oracle | gold-dependent | deployable 结果冻结后的 post-hoc 诊断 | deployable comparison |

legacy runner 可用 stage 名以当前 `run_fever_cbwdm.py` 为准；`pilot` alias 展开为 split、corpus、index、两份 retrieval 与两份 posterior。由于双环境和 Stage 15 的首次 resume 陷阱，服务器重建优先使用 scripts 22 的两段式路径，不要跨环境用一个 Python 进程跑完整 `pilot` alias。

正式 freeze 的 CLI 是：

```bash
export FEVER_INFOGAIN_CHECKPOINT='<selected completed InfoGain checkpoint path>' FEVER_OLD_RAG_CHECKPOINT='<selected completed old rag_cbwdm checkpoint path>'
export QWEN15_REVISION='<immutable HF commit>' BGE_LARGE_REVISION='<immutable HF commit>' MINILM_REVISION='<immutable HF commit>' INFOGAIN_CHECKPOINT_REVISION='<selected candidate/checkpoint fingerprint>' OLD_RAG_CHECKPOINT_REVISION='<selected candidate/checkpoint fingerprint>'
for v in FEVER_INFOGAIN_CHECKPOINT FEVER_OLD_RAG_CHECKPOINT QWEN15_REVISION BGE_LARGE_REVISION MINILM_REVISION INFOGAIN_CHECKPOINT_REVISION OLD_RAG_CHECKPOINT_REVISION; do eval "x=\${$v}"; case "$x" in ''|\<*) echo "BLOCKED: $v must come from completed calibration/model manifests"; exit 2;; esac; done
cd "$REPO" && "$BASE_PY" scripts/16_freeze_fever_formal_config.py --base-config "$FEVER_CONFIG" --split-manifest "$REPO/outputs/formal_splits/fever2_seed13/fever2_formal_splits.manifest.json" --calibration-manifest "$FEVER_RUN/artifacts/formal/calibration/calibration.manifest.json" --corpus "$("$BASE_PY" -c "import json; print(json.load(open('$FEVER_RUN/run_manifest.json'))['paths']['corpus'])")" --index "$("$BASE_PY" -c "import json; print(json.load(open('$FEVER_RUN/run_manifest.json'))['paths']['index_path'])")" --output-dir "$FEVER_RUN/artifacts/formal/frozen" --model "generator=$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --model "tokenizer=$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --model "bge=$MODEL_ROOT/bge-reranker-large" --model "infogain=$FEVER_INFOGAIN_CHECKPOINT" --model "rag_cbwdm=$FEVER_OLD_RAG_CHECKPOINT" --revision "generator=$QWEN15_REVISION" --revision "tokenizer=$QWEN15_REVISION" --revision "bge=$BGE_LARGE_REVISION" --revision "infogain=$INFOGAIN_CHECKPOINT_REVISION" --revision "rag_cbwdm=$OLD_RAG_CHECKPOINT_REVISION"
```

只有 calibration grid 完成、`15_calibrate_fever_methods.py` 生成 completed calibration manifest 后才可执行。不要用 clean preformal metrics 回写该目录。

冻结输出名含 fingerprint，不能猜。执行后读取脚本打印的 `config=`/`manifest=`，再定义：

```bash
export FEVER_FROZEN_CONFIG='<printed fever2_formal_frozen_*.yaml path>' FEVER_FROZEN_MANIFEST='<printed fever2_formal_frozen_*.manifest.json path>'
test -f "$FEVER_FROZEN_CONFIG" && test -f "$FEVER_FROZEN_MANIFEST"
```

### 11.2 FEVER held-out/test gate

当前 clean preformal config 明确 `held_out_test: forbidden`；不得拿它跑 held-out。只有上面的 generated frozen config/manifest 已完成、所有 deployable preformal artifacts/summary 已归档、任何算法修改已重新冻结后才开 gate。

```bash
export SPLIT_MANIFEST="$REPO/outputs/formal_splits/fever2_seed13/fever2_formal_splits.manifest.json"
export PREFORMAL="$FEVER_RUN/artifacts/preformal_signed_v1"
```

先在同一 release 上真实运行并记录 required tests；JSON 只会在四个命令全部成功后写出：

```bash
mkdir -p "$FEVER_RUN/artifacts/formal"
cd "$REPO" && set -o pipefail && { "$BASE_PY" -m compileall -q src scripts tests && "$BASE_PY" -m pytest -q && "$BASE_PY" -m unittest discover && git diff --check; } 2>&1 | tee "$FEVER_RUN/artifacts/formal/tests_status.log" && "$BASE_PY" - <<'PY'
import json, os, pathlib, subprocess
out=pathlib.Path(os.environ['FEVER_RUN'])/'artifacts/formal/tests_status.json'
out.write_text(json.dumps({'status':'passed','git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'commands':['python -m compileall -q src scripts tests','python -m pytest -q','python -m unittest discover','git diff --check'],'log':str(out.with_name('tests_status.log'))},indent=2,sort_keys=True)+'\n')
print(out)
PY
```

当前 `scripts/17_check_fever_formal_readiness.py` 只接受旧 production formal suite：`no_evidence,naive_topm,bge,infogain_fever,rag_cbwdm,cbwdm_oracle`；它要求旧 baseline fairness/summary schema。`preformal_fairness.json` 与 `PREFORMAL_SIGNED_V1_RESULTS.json` 不是兼容替代，且 readiness canonical methods 当前不含 `rag_cbwdm_signed_v1`。因此**不得**把 signed preformal 文件硬塞给 scripts 17 并把预期 BLOCKED 当成 ready。

这是当前 FEVER signed-v1 final-held-out 的明确 integration blocker。先用真实 CLI 查看要求：

```bash
cd "$REPO" && "$BASE_PY" scripts/17_check_fever_formal_readiness.py --help
```

只有后续代码/协议评审完成 signed-v1 final-readiness integration，或项目负责人明确批准旧 production formal suite 的独立 final run，并提供其真实 completed fairness/summary/diagnostics paths，才能运行 scripts 17。`tests_status.json` 也必须来自上面的真实命令，不得手工写 PASS。readiness 未给出 `status=ready,p0_passed=true` 前，本节后续命令全部 BLOCKED。

ready 后先在 retrieval env 运行 frozen held-out retrieval（第一次不带 resume）：

```bash
export FEVER_FINAL_RUN_NAME=fever2_formal_frozen_final_seed13 FEVER_FINAL_RUN="$EXP_ROOT/fever2_formal_frozen_final_seed13"
export JAVA_HOME="$MINICONDA_ROOT/envs/$RETRIEVAL_ENV"; export PATH="$JAVA_HOME/bin:$PATH"
cd "$REPO" && "$RETR_PY" scripts/run_fever_cbwdm.py --config "$FEVER_FROZEN_CONFIG" --frozen-manifest "$FEVER_FROZEN_MANIFEST" --run-name "$FEVER_FINAL_RUN_NAME" --stages retrieve_test --output-root "$EXP_ROOT" --cache-root "$HF_HOME"
```

然后在 baseline env 对实际 runner path 做 posterior FIRST RUN，仍然不带 resume：

```bash
export FEVER_TEST_RETRIEVAL="$FEVER_FINAL_RUN/artifacts/formal/fever2_held_out_test_bm25_top20.jsonl" FEVER_TEST_POSTERIOR="$FEVER_FINAL_RUN/artifacts/formal/fever2_held_out_test_posteriors.jsonl"
cd "$REPO" && "$BASE_PY" scripts/03_compute_label_posteriors.py --config "$FEVER_FROZEN_CONFIG" --split held_out_test --retrieval "$FEVER_TEST_RETRIEVAL" --output "$FEVER_TEST_POSTERIOR" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --batch-size 16
cd "$REPO" && "$BASE_PY" scripts/run_fever_cbwdm.py --config "$FEVER_FROZEN_CONFIG" --frozen-manifest "$FEVER_FROZEN_MANIFEST" --run-name "$FEVER_FINAL_RUN_NAME" --stages posterior_test --output-root "$EXP_ROOT" --cache-root "$HF_HOME" --resume --generator-model "$MODEL_ROOT/Qwen2.5-1.5B-Instruct"
```

最后只用已冻结的 signed/InfoGain/BGE/Naive/no-evidence/conditional old-RAG checkpoints 进行 gold-free selection 和统一 evaluation；split 均为 `held_out_test`。不得建 held-out teacher、训练、校准或挑 seed。当前仓库没有一个把 signed-v1 三 seed与全部 baseline 一次性跑完并做 paired summary 的 FEVER final wrapper；因此 final comparison 在 freeze artifact 路径全部解析前保持 **BLOCKED FOR OPERATOR FREEZE REVIEW**，不能从本文猜旧-RAG/InfoGain checkpoint path。可复用的 signed 命令是当前真实 CLI：

```bash
for SEED in 13 21 42; do mkdir -p "$FEVER_FINAL_RUN/artifacts/selections/rag_cbwdm_signed_v1/seed${SEED}"; "$BASE_PY" "$REPO/scripts/preformal/27_select_signed_v1.py" --posteriors "$FEVER_TEST_POSTERIOR" --checkpoint-dir "$PREFORMAL/training/rag_cbwdm_signed_v1/seed${SEED}/checkpoint" --output "$FEVER_FINAL_RUN/artifacts/selections/rag_cbwdm_signed_v1/seed${SEED}/selection.jsonl" --seed "$SEED" --split held_out_test --device auto --batch-size 32 || exit 1; done
```

任何 held-out artifact 出现后，所有算法/参数变化都视为 test exposure incident，必须保留原结果与书面原因。

---

## 12. FEVER2 clean preformal shared artifacts

统一变量：

```bash
export PREFORMAL_CONFIG="$REPO/configs/fever2_server_preformal_signed_v1.yaml"
export SPLIT_MANIFEST="$REPO/outputs/formal_splits/fever2_seed13/fever2_formal_splits.manifest.json"
export PREFORMAL="$FEVER_RUN/artifacts/preformal_signed_v1"
export INDEX_PATH="$("$BASE_PY" -c "import json; print(json.load(open('$FEVER_RUN/run_manifest.json'))['paths']['index_path'])")"
mkdir -p "$PREFORMAL"/{splits,retrieval,posteriors,training,selections,evaluations,fairness,summary,statistics}
```

### Stage 16 — 构造 clean `preformal_eval`

**Purpose:** 按冻结 strategy A 重建 full validation，排除已使用 pilot validation group。  
**Environment:** baseline。  
**GPU:** 否。  
**Inputs:** 完成且 checksum-valid 的 formal split manifest 与原始 source files。  
**Outputs:** `$PREFORMAL/splits/preformal_eval.jsonl`、`.manifest.json`、audit Markdown。  
**First run:** 不带 resume。  
**Resume:** completed exact-contract artifact 才带 `--resume` 复用。  
**Completion check:** source_strategy A、六类 ID/normalized-claim overlap 全零、`held_out_test_consumed_for_modeling=false`。  
**Expected invariant:** 预计 4500 只是协议参考，实际 count 以 manifest 为准。  
**Common failure:** 本地 fixture 或 changed source SHA；绝不硬写 4500 通过。  
**Next gate:** BM25。

```bash
cd "$REPO" && "$BASE_PY" scripts/preformal/24_build_preformal_eval.py --split-manifest "$SPLIT_MANIFEST" --output-dir "$PREFORMAL/splits"
```

```bash
"$BASE_PY" - <<'PY'
import json, os
p=os.environ['PREFORMAL']+'/splits/preformal_eval.manifest.json'; m=json.load(open(p))
assert m['source_strategy']=='A' and not any(m['overlap_checks'].values())
assert m['held_out_test_consumed_for_modeling'] is False
print('rows=',m['row_count'],'sha=',m['preformal_eval_sha256'])
PY
```

### Stage 17 — clean preformal BM25

**Purpose:** 使用已经通过 checksum/fingerprint 验证的 FEVER index 检索 top20。  
**Environment:** retrieval + Java gate。  
**GPU:** 否。  
**Inputs:** clean split、现有 index。  
**Outputs:** `$PREFORMAL/retrieval/preformal_eval_bm25_top20.jsonl` 与 manifest。  
**First run:** 02 没有 `--resume` 参数；直接运行。  
**Resume:** retrieval 是 atomic、非 resumable。若 completed artifact 合约匹配就只读复用；若中断只留下 `.partial`，同命令会从头写 partial。不得无审计地 `--overwrite` completed output。  
**Completion check:** completed、query input SHA 等于 split SHA、rows 相等、candidate max ≤20、index fingerprint 等于 formal run。  
**Expected invariant:** 不重建 index。  
**Common failure:** 从旧文档给 02 加 `--resume`（当前 CLI 不存在）。  
**Next gate:** posterior。

```bash
export JAVA_HOME="$MINICONDA_ROOT/envs/$RETRIEVAL_ENV"; export PATH="$JAVA_HOME/bin:$PATH"
cd "$REPO" && "$RETR_PY" scripts/02_retrieve_bm25.py --config "$PREFORMAL_CONFIG" --split preformal_eval --queries "$PREFORMAL/splits/preformal_eval.jsonl" --index "$INDEX_PATH" --output "$PREFORMAL/retrieval/preformal_eval_bm25_top20.jsonl" --top-n 20
```

### Stage 18 — shared preformal posteriors

**Purpose:** 用统一 Qwen/prompt/verbalizer 计算 query-only + 单文档 posteriors，供 InfoGain、old-RAG、signed-v1 共享。  
**Environment:** baseline。  
**GPU:** 是。  
**Inputs:** preformal top20 retrieval。  
**Outputs:** `$PREFORMAL/posteriors/preformal_eval_posteriors.jsonl` 与 manifest。  
**First run:** 不带 `--resume`。  
**Resume:** manifest 已存在后，同一 batch/model/config/input 命令加 `--resume`，row-level 续算。  
**Completion check:** status completed；completed_rows 等于 retrieval rows；input SHA、generator SHA、prompt hash、verbalizer hash 固定。  
**Expected invariant:** 所有需 posterior 的方法共享此文件。  
**Common failure:** 为 OOM 改 batch size 后尝试续旧 manifest；batch size 在 fingerprint 内，必须新路径重算。  
**Next gate:** learned-method training。

FIRST RUN：

```bash
cd "$REPO" && "$BASE_PY" scripts/03_compute_label_posteriors.py --config "$PREFORMAL_CONFIG" --split preformal_eval --retrieval "$PREFORMAL/retrieval/preformal_eval_bm25_top20.jsonl" --output "$PREFORMAL/posteriors/preformal_eval_posteriors.jsonl" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --max-candidates 20 --batch-size 16
```

INTERRUPTED RESUME：

```bash
cd "$REPO" && "$BASE_PY" scripts/03_compute_label_posteriors.py --config "$PREFORMAL_CONFIG" --split preformal_eval --retrieval "$PREFORMAL/retrieval/preformal_eval_bm25_top20.jsonl" --output "$PREFORMAL/posteriors/preformal_eval_posteriors.jsonl" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --max-candidates 20 --batch-size 16 --resume
```

在进入下节前检查服务器上的 `calibration_candidates.json`、`calibration/calibration.manifest.json` 与 `frozen_parameters.yaml`，记录 selected candidate/training/checkpoint fingerprint。若 calibration grid 只是 partial/reference，必须标记 `reference_config_from_pilot=true`、`formal_optimum_claimed=false`，不能补造最优性。

---

## 13. signed-v1 固定科学合约

当前代码 `src/preformal/registry.py` 与 signed scripts 冻结以下值，本文不得改动：

```text
alignment = x_ij^T d_i
admissible iff alignment > 0
teacher action = admissible 内最大 current production Theta marginal
top_m = 4
teacher_stop_threshold = 0.001
alignment_eps = 0
b_plus = 0.01
b_minus = 0.001
neutral_sample_policy = negative
nonpositive alignment = explicit harmful negative

selector = /root/models/ms-marco-MiniLM-L-6-v2
epochs = 3
lr = 2e-5
batch_size = 8
beta = 0.25
gamma = 1
loss_type = cbwdm_multitask
teacher_temperature = 0.1
seeds = 13,21,42

inference: min_docs=0, score_threshold=0.0, top_m=4
```

### Stage 19 — signed-v1 teacher materialization

**Purpose:** 只用 train_core gold label 与已有 posterior 构造 signed teacher；不调用 Qwen。  
**Environment:** baseline。  
**GPU:** 否/CPU。  
**Inputs:** FEVER train_core retrieval+posterior。  
**Outputs:** `$PREFORMAL/training/rag_cbwdm_signed_v1/teacher/{teacher.jsonl,statistics.json,manifest.json}`。  
**First run:** 不带 resume。  
**Resume:** 仅 completed exact fingerprint/checksum 复用；非 row-level。  
**Completion check:** manifest completed、method 正确、training_only、evaluation/calibration ineligible、teacher/statistics SHA 对上。  
**Expected invariant:** 不接触 preformal labels 作为 teacher。  
**Common failure:** 把 validation/preformal/test posterior 当训练 teacher。  
**Next gate:** 三 seed 训练。

```bash
cd "$REPO" && "$BASE_PY" scripts/preformal/25_materialize_signed_v1_teacher.py --config "$PREFORMAL_CONFIG" --posteriors "$FEVER_RUN/artifacts/formal/fever2_train_core_posteriors.jsonl" --retrieval "$FEVER_RUN/artifacts/formal/fever2_train_core_bm25_top20.jsonl" --output-dir "$PREFORMAL/training/rag_cbwdm_signed_v1/teacher" --training-split train_core
```

### Stage 20 — signed-v1 seeds 13/21/42 training

**Purpose:** 用同一 teacher、同一 MiniLM base、三个固定 seed 训练独立 checkpoint。  
**Environment:** baseline。  
**GPU:** 是。  
**Inputs:** Stage 19 teacher 与 train_core shared artifacts。  
**Outputs:** 每 seed 的 `checkpoint/`、`training_history.json`、`training_config.json`、`training_manifest.json`。  
**First run:** 不带 resume。  
**Resume:** 当前 `--resume` 只复用已完成、checksum-compatible checkpoint，不是 epoch/batch-level continuation。中断留下歧义状态时使用新的、明确标记的 output dir；不要删除未审计内容。  
**Completion check:** 三个 manifest completed；seed 分别 13/21/42；checkpoint SHA/fingerprint 有效且不同。  
**Expected invariant:** batch 8 等冻结参数不可为 OOM 调小；需要更大 GPU。  
**Common failure:** 复制 seed13 checkpoint 冒充其他 seed。  
**Next gate:** signed selection。

```bash
for SEED in 13 21 42; do cd "$REPO" && "$BASE_PY" scripts/preformal/26_train_signed_v1.py --config "$PREFORMAL_CONFIG" --teacher "$PREFORMAL/training/rag_cbwdm_signed_v1/teacher/teacher.jsonl" --posteriors "$FEVER_RUN/artifacts/formal/fever2_train_core_posteriors.jsonl" --retrieval "$FEVER_RUN/artifacts/formal/fever2_train_core_bm25_top20.jsonl" --output-dir "$PREFORMAL/training/rag_cbwdm_signed_v1/seed${SEED}" --model-name "$MODEL_ROOT/ms-marco-MiniLM-L-6-v2" --seed "$SEED" --device auto --training-split train_core || exit 1; done
```

### Stage 21 — signed-v1 gold-free selection

**Purpose:** 在 clean preformal 上运行 deployable state-aware selection。  
**Environment:** baseline。  
**GPU:** 是。  
**Inputs:** shared preformal posterior + 每 seed checkpoint。  
**Outputs:** `$PREFORMAL/selections/rag_cbwdm_signed_v1/seed*/selection.jsonl` 与 manifests。  
**First run:** 不带 resume。  
**Resume:** completed exact artifact 才带 resume 复用；selection 是 atomic、非 row-level。  
**Completion check:** 三 seed 相同 ID set；`uses_gold_at_inference=false`；max docs ≤4；允许 0 docs。  
**Expected invariant:** inference API 不接收 label/gold/alignment。  
**Common failure:** 用 diagnostic oracle selection 代替 deployable selection。  
**Next gate:** baselines。

```bash
for SEED in 13 21 42; do mkdir -p "$PREFORMAL/selections/rag_cbwdm_signed_v1/seed${SEED}"; cd "$REPO" && "$BASE_PY" scripts/preformal/27_select_signed_v1.py --posteriors "$PREFORMAL/posteriors/preformal_eval_posteriors.jsonl" --checkpoint-dir "$PREFORMAL/training/rag_cbwdm_signed_v1/seed${SEED}/checkpoint" --output "$PREFORMAL/selections/rag_cbwdm_signed_v1/seed${SEED}/selection.jsonl" --seed "$SEED" --split preformal_eval --device auto --batch-size 32 || exit 1; done
```

---

## 14. FEVER2 core baselines

### Stage 22 — no_evidence / naive_topm / BGE

**Purpose:** 生成三个 deterministic/pretrained deployable controls。  
**Environment:** baseline。  
**GPU:** no-evidence/naive 否；BGE 是。  
**Inputs:** shared retrieval。  
**Outputs:** 各自 selection 与 manifest；BGE score cache。  
**First run:** 不带 resume。  
**Resume:** selection/BGE 只复用 completed exact artifact；BGE scoring 非 row-level，cache 最终写入是 atomic。  
**Completion check:** 相同 ID set；Naive/BGE top4/min_docs4；无 gold inference。  
**Expected invariant:** deterministic methods 只跑 seed13 metadata。  
**Common failure:** 把 FM2 的 min_docs0 复制给 FEVER；FEVER clean protocol 是 4。  
**Next gate:** InfoGain。

```bash
mkdir -p "$PREFORMAL/selections/no_evidence/seed13" "$PREFORMAL/selections/naive_topm/seed13" "$PREFORMAL/selections/bge/seed13"
cd "$REPO" && "$BASE_PY" scripts/preformal/27a_select_no_evidence.py --retrieval "$PREFORMAL/retrieval/preformal_eval_bm25_top20.jsonl" --output "$PREFORMAL/selections/no_evidence/seed13/selection.jsonl" --split preformal_eval
cd "$REPO" && "$BASE_PY" scripts/08_select_naive_topm.py --config "$PREFORMAL_CONFIG" --retrieval "$PREFORMAL/retrieval/preformal_eval_bm25_top20.jsonl" --output "$PREFORMAL/selections/naive_topm/seed13/selection.jsonl" --top-m 4 --min-docs 4 --method-name naive_topm
cd "$REPO" && "$BASE_PY" scripts/12_select_bge_reranker.py --retrieval "$PREFORMAL/retrieval/preformal_eval_bm25_top20.jsonl" --output "$PREFORMAL/selections/bge/seed13/selection.jsonl" --score-cache "$PREFORMAL/selections/bge/seed13/scores.jsonl" --model-name-or-path "$MODEL_ROOT/bge-reranker-large" --device auto --dtype auto --batch-size 8 --max-length 512 --top-m 4 --min-docs 4 --local-files-only
```

### Stage 23 — InfoGain classification adaptation

**Purpose:** 运行项目内 probability-difference classification adapter；不是 InfoGain 论文的 answer-sequence DIG exact reproduction。  
**Environment:** baseline。  
**GPU:** teacher 否，训练/selection 是。  
**Inputs:** train_core posteriors、MiniLM、clean retrieval。  
**Outputs:** teacher、三 seed checkpoints、三 selections。  
**First run:** teacher/训练/selection 均不带 resume。  
**Resume:** teacher/训练只复用 completed exact contract；训练无 mid-epoch resume；selection atomic completed reuse。  
**Completion check:** teacher 只来自 train_core；三 seed；filter policy 已记录；preformal 未参与阈值选择。  
**Expected invariant:** 服务器现实记录的当前 reference 是 beta .75、train quantiles .8/.2、epochs3、lr2e-5、filter .5、top4/min2；它标记为 pilot reference，`formal_optimum_claimed=false`，除非 calibration artifact 证明相反。  
**Common failure:** 新 calibration winner 与 reference 不同却静默覆盖同一目录。  
**Next gate:** old-RAG conditional control / evaluation。

先审计实际 pilot artifact；如其 selected parameter/fingerprint 与下列 reference 不同，停止并建立新命名 protocol/output，不得静默替换：

```bash
"$BASE_PY" -c "import json; p='$FEVER_RUN/artifacts/formal/calibration_candidates.json'; m=json.load(open(p)); print(json.dumps([x for x in m.get('candidates',[]) if x.get('method')=='infogain_fever' and x.get('status')=='completed'],indent=2,sort_keys=True))"
```

当前 reference 命令：

```bash
mkdir -p "$PREFORMAL/training/infogain_fever" "$PREFORMAL/selections/infogain_fever"
cd "$REPO" && "$BASE_PY" scripts/12a_build_infogain_teacher.py --posteriors "$FEVER_RUN/artifacts/formal/fever2_train_core_posteriors.jsonl" --output "$PREFORMAL/training/infogain_fever/teacher.jsonl" --purpose training --threshold-mode train_quantile --positive-quantile 0.8 --negative-quantile 0.2
for SEED in 13 21 42; do cd "$REPO" && "$BASE_PY" scripts/12b_train_infogain_reranker.py --teacher "$PREFORMAL/training/infogain_fever/teacher.jsonl" --output-dir "$PREFORMAL/training/infogain_fever/seed${SEED}" --model-name-or-path "$MODEL_ROOT/ms-marco-MiniLM-L-6-v2" --device auto --max-length 512 --epochs 3 --lr 2e-5 --beta 0.75 --seed "$SEED" || exit 1; done
for SEED in 13 21 42; do mkdir -p "$PREFORMAL/selections/infogain_fever/seed${SEED}"; cd "$REPO" && "$BASE_PY" scripts/12c_select_infogain_reranker.py --retrieval "$PREFORMAL/retrieval/preformal_eval_bm25_top20.jsonl" --checkpoint-dir "$PREFORMAL/training/infogain_fever/seed${SEED}/checkpoint" --output "$PREFORMAL/selections/infogain_fever/seed${SEED}/selection.jsonl" --device auto --batch-size 32 --top-m 4 --min-docs 2 --filter-threshold 0.5 --method-name infogain_fever || exit 1; done
```

### 14.1 old `rag_cbwdm` repair/control

它不是当前 signed-v1 主方法。只有 calibration manifest、teacher、训练参数、checkpoint fingerprint 能精确恢复时才运行。`scripts/10_train_cross_encoder_selector.py` 当前没有 `--resume`/`--overwrite`，训练不是 resumable；不要在已有目录上重跑。若只能验证真实 seed13 checkpoint，就只报告 seed13；seed21/42 标记：

```text
old rag_cbwdm stability unavailable / pending
```

不得复制 seed13 checkpoint、不得根据 preformal 分数反推旧阈值。可用 CLI 为 `04_build_cbwdm_teacher.py` → `10_train_cross_encoder_selector.py` → `11_select_with_cross_encoder.py`；参数必须逐项来自上述 calibration/frozen artifact，而不是本文猜测。这个条件性 blocker 不阻止 signed-v1 与其他核心 baseline 完成。

---

## 15. Unified Qwen evaluation

### Stage 24 — 所有 deployable selections 的统一评估

**Purpose:** 用同一个本地 Qwen 1.5B、同一 FEVER prompt/verbalizer、同一 exact ID set 评估。  
**Environment:** baseline。  
**GPU:** 是。  
**Inputs:** Stage 21–23 selections。  
**Outputs:** 每 method/seed 的 predictions、metrics、`metrics.manifest.json`。  
**First run:** 不带 resume。  
**Resume:** 当前 evaluation 是整次 atomic publish，非 row-level；`--resume` 仅复用 completed exact-contract output。中断只留下 `.partial` 时用相同 first-run 命令从头评估。  
**Completion check:** 每 manifest status completed；generator SHA、prompt hash、verbalizer hash、split、ID set 全同。  
**Expected invariant:** `no_evidence` 必须显式 `--no-evidence --max-docs 0`；其他最多 4 docs。  
**Common failure:** 首次机械加 resume、method/path 对错。  
**Next gate:** fairness/statistics。

先定义可复制的 first-run 函数：

```bash
fever_eval() { METHOD="$1"; SEED="$2"; SELECTION="$3"; shift 3; mkdir -p "$PREFORMAL/evaluations/$METHOD/seed$SEED"; "$BASE_PY" "$REPO/scripts/07_eval_rag_classification.py" --config "$PREFORMAL_CONFIG" --split preformal_eval --selection "$SELECTION" --output "$PREFORMAL/evaluations/$METHOD/seed$SEED/predictions.jsonl" --metrics-output "$PREFORMAL/evaluations/$METHOD/seed$SEED/metrics.json" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --method-name "$METHOD" "$@"; }
```

```bash
fever_eval no_evidence 13 "$PREFORMAL/selections/no_evidence/seed13/selection.jsonl" --no-evidence --max-docs 0
fever_eval naive_topm 13 "$PREFORMAL/selections/naive_topm/seed13/selection.jsonl" --max-docs 4
fever_eval bge 13 "$PREFORMAL/selections/bge/seed13/selection.jsonl" --max-docs 4
for SEED in 13 21 42; do fever_eval infogain_fever "$SEED" "$PREFORMAL/selections/infogain_fever/seed${SEED}/selection.jsonl" --max-docs 4 || exit 1; done
for SEED in 13 21 42; do fever_eval rag_cbwdm_signed_v1 "$SEED" "$PREFORMAL/selections/rag_cbwdm_signed_v1/seed${SEED}/selection.jsonl" --max-docs 4 || exit 1; done
```

已完成 exact artifact 的复用命令是在相同 `07_eval...` 命令末尾加 `--resume`。不要用 `--overwrite` 修复 contract mismatch；用新输出目录。

---

## 16. FEVER fairness、paired statistics 与 signed post-hoc

### Stage 25 — Fairness audit

**Purpose:** fail closed 地验证 shared retrieval/posterior、selection/evaluation SHA、prompt/model 与 ID set。  
**Environment:** baseline。  
**GPU:** 否。  
**Inputs:** 所有 completed manifests。  
**Outputs:** `$PREFORMAL/fairness/preformal_fairness.json`。  
**First run/Resume:** cheap atomic summary；contract 改变时新路径。  
**Completion check:** status `comparable`；所有 deployable 方法都列入。  
**Expected invariant:** 不能只审计 signed 而漏 baseline。  
**Common failure:** 任一 seed manifest 缺失。  
**Next gate:** paired summary。

```bash
SEL_ARGS=(--selection-manifest "no_evidence:13=$PREFORMAL/selections/no_evidence/seed13/selection.manifest.json" --selection-manifest "naive_topm:13=$PREFORMAL/selections/naive_topm/seed13/selection.manifest.json" --selection-manifest "bge:13=$PREFORMAL/selections/bge/seed13/selection.manifest.json")
EVAL_ARGS=(--evaluation-manifest "no_evidence:13=$PREFORMAL/evaluations/no_evidence/seed13/metrics.manifest.json" --evaluation-manifest "naive_topm:13=$PREFORMAL/evaluations/naive_topm/seed13/metrics.manifest.json" --evaluation-manifest "bge:13=$PREFORMAL/evaluations/bge/seed13/metrics.manifest.json")
for METHOD in infogain_fever rag_cbwdm_signed_v1; do for SEED in 13 21 42; do SEL_ARGS+=(--selection-manifest "$METHOD:$SEED=$PREFORMAL/selections/$METHOD/seed$SEED/selection.manifest.json"); EVAL_ARGS+=(--evaluation-manifest "$METHOD:$SEED=$PREFORMAL/evaluations/$METHOD/seed$SEED/metrics.manifest.json"); done; done
cd "$REPO" && "$BASE_PY" scripts/preformal/28_audit_fairness.py --split-manifest "$PREFORMAL/splits/preformal_eval.manifest.json" --retrieval-manifest "$PREFORMAL/retrieval/preformal_eval_bm25_top20.manifest.json" --posterior-manifest "$PREFORMAL/posteriors/preformal_eval_posteriors.manifest.json" "${SEL_ARGS[@]}" "${EVAL_ARGS[@]}" --output "$PREFORMAL/fairness/preformal_fairness.json"
```

若 old `rag_cbwdm` 有真实完成的 seed，再把它们加入两个数组；不能列空或伪造路径。

### Stage 26 — Results / paired statistics

**Purpose:** 生成 per-seed、mean/SD、McNemar、paired bootstrap CI。  
**Environment:** baseline。  
**GPU:** 否。  
**Inputs:** comparable fairness audit、所有 metrics/selections、shared retrieval。  
**Outputs:** `PREFORMAL_SIGNED_V1_RESULTS.*` 与 `PREFORMAL_PAIRED_COMPARISONS.*`。  
**Completion check:** bootstrap seed 130421、samples 10000；完整方法集合；identical IDs。  
**Expected invariant:** 只报告结果，不更新 calibration/frozen config。  
**Next gate:** optional post-hoc。

```bash
SUMMARY_EVAL_ARGS=(--evaluation-manifest "no_evidence:13=$PREFORMAL/evaluations/no_evidence/seed13/metrics.manifest.json" --evaluation-manifest "naive_topm:13=$PREFORMAL/evaluations/naive_topm/seed13/metrics.manifest.json" --evaluation-manifest "bge:13=$PREFORMAL/evaluations/bge/seed13/metrics.manifest.json")
SUMMARY_SEL_ARGS=(--selection "no_evidence:13=$PREFORMAL/selections/no_evidence/seed13/selection.jsonl" --selection "naive_topm:13=$PREFORMAL/selections/naive_topm/seed13/selection.jsonl" --selection "bge:13=$PREFORMAL/selections/bge/seed13/selection.jsonl")
for METHOD in infogain_fever rag_cbwdm_signed_v1; do for SEED in 13 21 42; do SUMMARY_EVAL_ARGS+=(--evaluation-manifest "$METHOD:$SEED=$PREFORMAL/evaluations/$METHOD/seed$SEED/metrics.manifest.json"); SUMMARY_SEL_ARGS+=(--selection "$METHOD:$SEED=$PREFORMAL/selections/$METHOD/seed$SEED/selection.jsonl"); done; done
cd "$REPO" && "$BASE_PY" scripts/preformal/29_summarize_results.py --fairness-audit "$PREFORMAL/fairness/preformal_fairness.json" --retrieval "$PREFORMAL/retrieval/preformal_eval_bm25_top20.jsonl" "${SUMMARY_EVAL_ARGS[@]}" "${SUMMARY_SEL_ARGS[@]}" --bootstrap-seed 130421 --bootstrap-samples 10000 --output-dir "$PREFORMAL/summary"
```

### Stage 27 — signed post-hoc / optional oracle

**Purpose:** 在 deployable artifacts 与 summary 已冻结后看 alignment、zero-doc、coverage；oracle 只用于差距诊断。  
**Environment:** baseline。  
**GPU:** post-hoc 否；oracle runner 可能调用 Qwen。  
**Inputs:** signed selection/shared posterior/retrieval；optional oracle selection。  
**Outputs:** statistics JSON。  
**Completion check:** `uses_gold_for_selection=false`、`posthoc_only=true`；oracle 未进入 fairness comparison。  
**Expected invariant:** oracle 结果不得驱动 selector 参数。  
**Next gate:** 解释/冻结，不是 held-out 自动授权。

无 oracle 的 deployable post-hoc：

```bash
cd "$REPO" && "$BASE_PY" scripts/preformal/30_signed_posthoc.py --config "$PREFORMAL_CONFIG" --selection "$PREFORMAL/selections/rag_cbwdm_signed_v1/seed13/selection.jsonl" --posteriors "$PREFORMAL/posteriors/preformal_eval_posteriors.jsonl" --retrieval "$PREFORMAL/retrieval/preformal_eval_bm25_top20.jsonl" --output "$PREFORMAL/statistics/signed_seed13_posthoc.json"
```

如果 preformal 结果导致任何算法或参数改变，则本次 `preformal_eval` 从此成为 development data；重新冻结后必须保留另一个未触碰 held-out 才能作最终结论。

---

## 17. FM2 train/dev workflow

FM2 与 FEVER 共享 generator、prompt registry interface、signed implementation 与 baseline scripts，但数据、prompt version/hash、candidate pool、posterior、teacher、checkpoint、calibration 和结果全部 dataset-specific。

可直接 transfer 的只有：

- Qwen generator weights/tokenizer；
- BGE pretrained reranker；
- MiniLM base encoder；
- deterministic no-evidence/Naive 逻辑。

primary FM2 comparison 必须重训：

- signed-v1 selector heads/checkpoints，train split，seeds 13/21/42；
- InfoGain adapter heads/checkpoints，train split，seeds 13/21/42。

FEVER-trained signed/InfoGain checkpoint 只能作为清楚标记的 zero-shot transfer control，不能代替 primary FM2 方法。FEVER posterior、index、pool、split manifest、threshold 与 result 均不可直接 transfer。

统一变量：

```bash
export FM2_CONFIG="$REPO/configs/fm2_server_preformal_signed_v1.yaml"
export FM2_RUN="$EXP_ROOT/fm2_preformal_v1"
export FM2_DATA="$REPO/data/processed/fm2"
mkdir -p "$FM2_RUN/artifacts"/{posteriors,training,selections,evaluations,protocol,logs}
```

### Stage 28 — FM2 smoke / pilot / full 数据入口

**Purpose:** 先测 API/吞吐，再进入 full train/dev。  
**Environment:** baseline；official-pool 第一轮不使用 retrieval env/Java/Pyserini。  
**GPU:** prepare 否，后续是。  
**Inputs:** Stage 12 canonical train/dev。  
**Outputs:** 独立 smoke、pilot、full run roots。  
**First run:** smoke limit8；pilot limit500 只测吞吐；full 不带 limit。  
**Resume:** prepare 无 resume；每个规模必须新 output dir。  
**Completion check:** smoke 8/8，pilot 500/500，full 10419/1169；pool max≤12；gold diagnostic 分离。  
**Expected invariant:** pilot prefix metrics 不作 formal claim。  
**Common failure:** 在 canonical `data/processed/fm2` 上用 `--overwrite --limit`，破坏 full pool。  
**Next gate:** posterior。

Smoke：

```bash
export FM2_CONFIG="$REPO/configs/fm2_server_smoke.yaml" FM2_RUN="$EXP_ROOT/fm2_smoke_v1" FM2_DATA="$EXP_ROOT/fm2_smoke_v1/artifacts/data"
mkdir -p "$FM2_DATA" "$FM2_RUN/artifacts"/{posteriors,training,selections,evaluations,protocol,logs}
cd "$REPO" && "$BASE_PY" scripts/prepare_fm2.py --raw-dir data/raw/fm2 --output-dir "$FM2_DATA" --splits train dev --limit 8
```

Pilot（吞吐/API，不用于调参结论）：

```bash
export FM2_CONFIG="$REPO/configs/fm2_server_preformal_signed_v1.yaml" FM2_RUN="$EXP_ROOT/fm2_pilot500_v1" FM2_DATA="$EXP_ROOT/fm2_pilot500_v1/artifacts/data"
mkdir -p "$FM2_DATA" "$FM2_RUN/artifacts"/{posteriors,training,selections,evaluations,protocol,logs}
cd "$REPO" && "$BASE_PY" scripts/prepare_fm2.py --raw-dir data/raw/fm2 --output-dir "$FM2_DATA" --splits train dev --limit 500
```

Full：

```bash
export FM2_CONFIG="$REPO/configs/fm2_server_preformal_signed_v1.yaml" FM2_RUN="$EXP_ROOT/fm2_preformal_v1" FM2_DATA="$REPO/data/processed/fm2"
mkdir -p "$FM2_RUN/artifacts"/{posteriors,training,selections,evaluations,protocol,logs}
```

以下 Stage 29–35 对当前 `FM2_RUN`/`FM2_DATA` 执行；smoke → pilot → full 必须分别完整过门。

### Stage 29 — FM2 train/dev shared posteriors

**Purpose:** FM2 prompt v1 下生成 shared train/dev posteriors。  
**Environment:** baseline。  
**GPU:** 是。  
**Inputs:** `fm2_{train,dev}_official_pool.jsonl`。  
**Outputs:** `$FM2_RUN/artifacts/posteriors/{train,dev}.jsonl` 与 manifests。  
**First run:** 两个命令不带 resume。  
**Resume:** manifest 已存在后相同命令加 resume，row-level。  
**Completion check:** dataset `fm2`、split 正确、prompt version `fm2_classification_v1`、input SHA/rows 对上。  
**Expected invariant:** train/dev 使用同一 Qwen revision、labels `[SUPPORTS,REFUTES]`、A/B verbalizers。  
**Common failure:** 从旧 FM2 runbook 复制首跑 `--resume`。  
**Next gate:** teachers/training。

FIRST RUN：

```bash
cd "$REPO" && "$BASE_PY" scripts/03_compute_label_posteriors.py --config "$FM2_CONFIG" --split train --retrieval "$FM2_DATA/fm2_train_official_pool.jsonl" --output "$FM2_RUN/artifacts/posteriors/train.jsonl" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --batch-size 16
cd "$REPO" && "$BASE_PY" scripts/03_compute_label_posteriors.py --config "$FM2_CONFIG" --split dev --retrieval "$FM2_DATA/fm2_dev_official_pool.jsonl" --output "$FM2_RUN/artifacts/posteriors/dev.jsonl" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --batch-size 16
```

INTERRUPTED RESUME：

```bash
cd "$REPO" && "$BASE_PY" scripts/03_compute_label_posteriors.py --config "$FM2_CONFIG" --split train --retrieval "$FM2_DATA/fm2_train_official_pool.jsonl" --output "$FM2_RUN/artifacts/posteriors/train.jsonl" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --batch-size 16 --resume
cd "$REPO" && "$BASE_PY" scripts/03_compute_label_posteriors.py --config "$FM2_CONFIG" --split dev --retrieval "$FM2_DATA/fm2_dev_official_pool.jsonl" --output "$FM2_RUN/artifacts/posteriors/dev.jsonl" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --batch-size 16 --resume
```

### Stage 30 — FM2 InfoGain teacher/training/selection

**Purpose:** 在 FM2 train 重训 classification adaptation，并在 dev gold-free selection。  
**Environment:** baseline。  
**GPU:** teacher 否；训练/selection 是。  
**Inputs:** train posterior、dev official pool、MiniLM。  
**Outputs:** teacher、三 seed checkpoint/selection。  
**First run:** 不带 resume。  
**Resume:** completed exact reuse；训练不支持 mid-epoch continuation。  
**Completion check:** beta .75、train quantiles .8/.2、epochs3、lr2e-5、seeds13/21/42；dev top4/min2/filter.5。  
**Expected invariant:** 0.5 是 cross-dataset reference，不是 FM2 optimum。  
**Common failure:** 用 dev 造 training teacher；将 adaptation accuracy 与论文 EM 当同指标。  
**Next gate:** signed-v1。

```bash
mkdir -p "$FM2_RUN/artifacts/training/infogain" "$FM2_RUN/artifacts/selections/infogain_adapter"
cd "$REPO" && "$BASE_PY" scripts/12a_build_infogain_teacher.py --posteriors "$FM2_RUN/artifacts/posteriors/train.jsonl" --output "$FM2_RUN/artifacts/training/infogain/teacher.jsonl" --purpose training --threshold-mode train_quantile --positive-quantile 0.8 --negative-quantile 0.2
for SEED in 13 21 42; do cd "$REPO" && "$BASE_PY" scripts/12b_train_infogain_reranker.py --teacher "$FM2_RUN/artifacts/training/infogain/teacher.jsonl" --output-dir "$FM2_RUN/artifacts/training/infogain/seed${SEED}" --model-name-or-path "$MODEL_ROOT/ms-marco-MiniLM-L-6-v2" --device auto --max-length 512 --epochs 3 --lr 2e-5 --beta 0.75 --seed "$SEED" || exit 1; done
for SEED in 13 21 42; do mkdir -p "$FM2_RUN/artifacts/selections/infogain_adapter/seed${SEED}"; cd "$REPO" && "$BASE_PY" scripts/12c_select_infogain_reranker.py --retrieval "$FM2_DATA/fm2_dev_official_pool.jsonl" --checkpoint-dir "$FM2_RUN/artifacts/training/infogain/seed${SEED}/checkpoint" --output "$FM2_RUN/artifacts/selections/infogain_adapter/seed${SEED}/selection.jsonl" --device auto --batch-size 16 --top-m 4 --min-docs 2 --filter-threshold 0.5 --method-name infogain_adapter || exit 1; done
```

### Stage 31 — FM2 signed-v1 teacher/training/selection

**Purpose:** 复用同一 signed 算法实现，但用 FM2 train 重新 materialize/retrain。  
**Environment:** baseline。  
**GPU:** teacher 否；训练/selection 是。  
**Inputs:** FM2 train retrieval/posterior、dev posterior。  
**Outputs:** signed teacher、三 seed checkpoints、三 dev selections。  
**First run:** 不带 resume。  
**Resume:** completed exact reuse；训练非 mid-epoch。  
**Completion check:** frozen signed contract；teacher split=train；selection split=dev、no gold、0–4 docs。  
**Expected invariant:** FEVER learned head 不作为 primary 初始化/结果。  
**Common failure:** training posterior 被 selection script 拒绝是正确防泄漏行为。  
**Next gate:** deterministic/BGE baselines。

```bash
cd "$REPO" && "$BASE_PY" scripts/preformal/25_materialize_signed_v1_teacher.py --config "$FM2_CONFIG" --posteriors "$FM2_RUN/artifacts/posteriors/train.jsonl" --retrieval "$FM2_DATA/fm2_train_official_pool.jsonl" --output-dir "$FM2_RUN/artifacts/training/signed_teacher" --training-split train
for SEED in 13 21 42; do cd "$REPO" && "$BASE_PY" scripts/preformal/26_train_signed_v1.py --config "$FM2_CONFIG" --teacher "$FM2_RUN/artifacts/training/signed_teacher/teacher.jsonl" --posteriors "$FM2_RUN/artifacts/posteriors/train.jsonl" --retrieval "$FM2_DATA/fm2_train_official_pool.jsonl" --output-dir "$FM2_RUN/artifacts/training/signed_seed${SEED}" --model-name "$MODEL_ROOT/ms-marco-MiniLM-L-6-v2" --seed "$SEED" --device auto --training-split train || exit 1; done
for SEED in 13 21 42; do mkdir -p "$FM2_RUN/artifacts/selections/rag_cbwdm_signed_v1/seed${SEED}"; cd "$REPO" && "$BASE_PY" scripts/preformal/27_select_signed_v1.py --posteriors "$FM2_RUN/artifacts/posteriors/dev.jsonl" --checkpoint-dir "$FM2_RUN/artifacts/training/signed_seed${SEED}/checkpoint" --output "$FM2_RUN/artifacts/selections/rag_cbwdm_signed_v1/seed${SEED}/selection.jsonl" --seed "$SEED" --split dev --device auto --batch-size 32 || exit 1; done
```

### Stage 32 — FM2 no_evidence / Naive / BGE

**Purpose:** 在完全相同 official pool 上跑其余核心 baselines。  
**Environment:** baseline。  
**GPU:** BGE 是。  
**Inputs:** FM2 dev official pool。  
**Outputs:** selection/manifests 与 BGE score cache。  
**First run:** 不带 resume。  
**Resume:** completed exact reuse；BGE scoring 非 row-level。  
**Completion check:** no-evidence 0 docs；Naive/BGE max4、min_docs0，以容纳少于4候选的行。  
**Expected invariant:** 不向任何 selector 传 gold diagnostic。  
**Common failure:** 错用 FEVER min_docs4，导致 FM2 少候选行失败。  
**Next gate:** Qwen dev evaluation。

```bash
mkdir -p "$FM2_RUN/artifacts/selections/no_evidence/seed13" "$FM2_RUN/artifacts/selections/naive_topm/seed13" "$FM2_RUN/artifacts/selections/bge/seed13"
cd "$REPO" && "$BASE_PY" scripts/preformal/27a_select_no_evidence.py --retrieval "$FM2_DATA/fm2_dev_official_pool.jsonl" --output "$FM2_RUN/artifacts/selections/no_evidence/seed13/selection.jsonl" --split dev
cd "$REPO" && "$BASE_PY" scripts/08_select_naive_topm.py --config "$FM2_CONFIG" --retrieval "$FM2_DATA/fm2_dev_official_pool.jsonl" --output "$FM2_RUN/artifacts/selections/naive_topm/seed13/selection.jsonl" --top-m 4 --min-docs 0 --method-name naive_topm
cd "$REPO" && "$BASE_PY" scripts/12_select_bge_reranker.py --retrieval "$FM2_DATA/fm2_dev_official_pool.jsonl" --output "$FM2_RUN/artifacts/selections/bge/seed13/selection.jsonl" --score-cache "$FM2_RUN/artifacts/selections/bge/seed13/scores.jsonl" --model-name-or-path "$MODEL_ROOT/bge-reranker-large" --device auto --dtype auto --batch-size 8 --max-length 512 --top-m 4 --min-docs 0 --local-files-only
```

### Stage 33 — FM2 unified dev evaluation

**Purpose:** 所有方法使用同一 FM2 prompt/Qwen/ID set。  
**Environment:** baseline。  
**GPU:** 是。  
**Inputs:** Stage 30–32 selections。  
**Outputs:** 每 method/seed predictions/metrics/manifest。  
**First run:** 不带 resume。  
**Resume:** completed exact reuse；非 row-level。  
**Completion check:** accuracy、macro-F1、per-class、confusion、avg docs/chars、prediction distribution 均存在；IDs 相同；max docs≤4。  
**Expected invariant:** dev 是唯一 calibration split；未触碰 test。  
**Next gate:** FM2 audit/calibration freeze。

```bash
fm2_eval() { METHOD="$1"; SEED="$2"; SELECTION="$3"; shift 3; mkdir -p "$FM2_RUN/artifacts/evaluations/$METHOD/seed$SEED"; "$BASE_PY" "$REPO/scripts/07_eval_rag_classification.py" --config "$FM2_CONFIG" --split dev --selection "$SELECTION" --output "$FM2_RUN/artifacts/evaluations/$METHOD/seed$SEED/predictions.jsonl" --metrics-output "$FM2_RUN/artifacts/evaluations/$METHOD/seed$SEED/metrics.json" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --method-name "$METHOD" --max-docs 4 "$@"; }
fm2_eval no_evidence 13 "$FM2_RUN/artifacts/selections/no_evidence/seed13/selection.jsonl" --no-evidence --max-docs 0
fm2_eval naive_topm 13 "$FM2_RUN/artifacts/selections/naive_topm/seed13/selection.jsonl"
fm2_eval bge 13 "$FM2_RUN/artifacts/selections/bge/seed13/selection.jsonl"
for SEED in 13 21 42; do fm2_eval infogain_adapter "$SEED" "$FM2_RUN/artifacts/selections/infogain_adapter/seed${SEED}/selection.jsonl" || exit 1; done
for SEED in 13 21 42; do fm2_eval rag_cbwdm_signed_v1 "$SEED" "$FM2_RUN/artifacts/selections/rag_cbwdm_signed_v1/seed${SEED}/selection.jsonl" || exit 1; done
```

### Stage 34 — FM2 fairness/provenance audit

当前没有 FM2 专用的 `28_audit_fairness.py` 等价入口；FEVER preformal fairness CLI 需要 FEVER split/retrieval/posterior manifest schema，不能冒充 FM2 audit。这是当前 **BLOCKED/TODO automation gap**，但可用下列只读 hard audit 过第一轮 gate：

```bash
"$BASE_PY" - <<'PY'
import glob, hashlib, json, os
run=os.environ['FM2_RUN']; data=os.environ['FM2_DATA']
pool=f'{data}/fm2_dev_official_pool.jsonl'
def rows(p):
    return [json.loads(x) for x in open(p,encoding='utf-8') if x.strip()]
pool_rows=rows(pool); expected=[str(x['id']) for x in pool_rows]
assert all('gold_evidence' not in x and 'gold_evidence_keys' not in x for x in pool_rows)
selection_paths=sorted(glob.glob(f'{run}/artifacts/selections/**/selection.jsonl',recursive=True))
assert selection_paths
for p in selection_paths:
    r=rows(p); assert [str(x['id']) for x in r]==expected, p
    assert max((len(x.get('selected_docs',[])) for x in r),default=0)<=4, p
    text='\n'.join(json.dumps(x,sort_keys=True) for x in r)
    assert 'gold_evidence' not in text and 'gold_evidence_keys' not in text, p
metrics_manifests=sorted(glob.glob(f'{run}/artifacts/evaluations/**/metrics.manifest.json',recursive=True))
assert metrics_manifests
contracts=[json.load(open(p))['contract'] for p in metrics_manifests]
for key in ('generator_sha256','prompt_hash','verbalizer_hash','split'):
    assert len({json.dumps(x.get(key),sort_keys=True) for x in contracts})==1, key
assert contracts[0]['split']=='dev'
print('FM2 FAIRNESS HARD AUDIT PASS',len(selection_paths),len(metrics_manifests))
PY
```

完整 freeze record 还必须人工记录：release SHA、raw/source commit/SHA、prepare manifest SHA、pool SHA、prompt/verbalizer/generator SHA、BGE revision、每个 checkpoint SHA、seeds、InfoGain threshold policy、每个 selection/evaluation SHA、上述 audit output。自动化 gap 未补前不得宣称与 FEVER fairness script 同级自动验证。

### Stage 35 — FM2 dev calibration policy

- Primary/reference 结果固定保留 InfoGain `filter_threshold=0.5`，标记 `cross_dataset_reference_not_fm2_optimum`。
- 如确需 FM2 calibration，只能在 official dev 建立**另一个明确命名**的 threshold grid/output，选择一次后写入 freeze record；不得覆盖或改名冒充 0.5 reference。
- signed-v1 的 `min_docs=0,score_threshold=0,top_m=4` 不参与这轮 calibration。
- Qwen prompt、document budget、seeds、训练超参均不从 dev 随意修改。
- test 不能用于 seed、threshold、model、prompt、budget 选择；learned methods 在 test 报三 seed 汇总，而不是挑最好 seed。

只有 Stage 29–34 full train/dev 全部完成且选择已冻结，才能写授权 record：

```bash
"$BASE_PY" - <<'PY'
import hashlib, json, os, pathlib, subprocess
run=pathlib.Path(os.environ['FM2_RUN']); repo=pathlib.Path(os.environ['REPO'])
data=pathlib.Path(os.environ['FM2_DATA'])
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
required=[data/'fm2_prepare.manifest.json',run/'artifacts/posteriors/train.manifest.json',run/'artifacts/posteriors/dev.manifest.json']
required += [run/f'artifacts/training/signed_seed{s}/training_manifest.json' for s in (13,21,42)]
required += [run/f'artifacts/training/infogain/seed{s}/training_manifest.json' for s in (13,21,42)]
selection_manifests=sorted((run/'artifacts/selections').glob('**/selection.manifest.json'))
evaluation_manifests=sorted((run/'artifacts/evaluations').glob('**/metrics.manifest.json'))
assert len(selection_manifests)==9 and len(evaluation_manifests)==9,(len(selection_manifests),len(evaluation_manifests))
required += selection_manifests + evaluation_manifests
assert all(p.is_file() for p in required), [str(p) for p in required if not p.is_file()]
payload={'schema_version':'rag_cbwdm_fm2_freeze_record.v1','status':'frozen_before_test','git_head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'reference_infogain_filter_threshold':0.5,'signed_contract':'src/preformal/registry.py','seeds':[13,21,42],'required_manifests':{str(p):sha(p) for p in required},'test_consumed':False}
out=run/'artifacts/protocol/fm2_freeze_record.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(out,sha(out))
PY
```

该 record 是操作层 gate；它不取代对所有 selection/evaluation SHA 的外部归档。

---

## 18. FM2 final held-out/test gate

### Stage 36 — 解锁 official test

**Purpose:** 只在 full train/dev freeze 后一次性准备、选择、评估 FM2 official test。  
**Environment:** baseline；无 FM2 Lucene。  
**GPU:** posterior/selection/evaluation 是。  
**Inputs:** completed freeze record、已训练 checkpoints。  
**Outputs:** canonical test pool、test posterior、所有 frozen method test results。  
**First run:** 显式设置人工授权变量；test download/prepare 均不带 overwrite。  
**Resume:** posterior 仅 manifest 存在后 row-level resume；其余按 completed exact reuse。  
**Completion check:** test 1380 rows/234 pages；train/dev/test page overlaps 0；所有 IDs/pool/prompt SHA 一致；三 seed均完成。  
**Expected invariant:** test 后不训练、不重校准、不选 seed、不改 threshold/prompt/budget。  
**Common failure:** 只因脚本 CLI 允许 `--split test` 就认为科学 gate 已开。  
**Next gate:** 最终 summary/archive。

```bash
export FM2_TEST_AUTHORIZED=NO
test "$FM2_TEST_AUTHORIZED" = YES || { echo 'BLOCKED: obtain signed-off FM2 freeze authorization'; exit 2; }
test -f "$FM2_RUN/artifacts/protocol/fm2_freeze_record.json"
cd "$REPO" && "$BASE_PY" scripts/download_fm2.py --output-dir data/raw/fm2 --splits test
cd "$REPO" && "$BASE_PY" scripts/prepare_fm2.py --raw-dir data/raw/fm2 --output-dir data/processed/fm2 --splits test
```

预期 test raw SHA：

```text
4a45aa8edd45ea8ff54eb66be06e8c6113873c22496b209925dd5b529ce00721
```

test prepare 实际 manifest 名为 `data/processed/fm2/fm2_prepare_test.manifest.json`；它要求 canonical train/dev raw files 在场，并重新检查三组 page intersection，不替换 train/dev prepare manifest。

FIRST test posterior：

```bash
mkdir -p "$FM2_RUN/artifacts/test"/{posteriors,selections,evaluations}
cd "$REPO" && "$BASE_PY" scripts/03_compute_label_posteriors.py --config "$FM2_CONFIG" --split test --retrieval "$REPO/data/processed/fm2/fm2_test_official_pool.jsonl" --output "$FM2_RUN/artifacts/test/posteriors/test.jsonl" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --batch-size 16
```

使用 Stage 30–33 的**同一 frozen checkpoint/参数**生成完整 test selections：

```bash
export FM2_TEST_POOL="$REPO/data/processed/fm2/fm2_test_official_pool.jsonl" FM2_TEST_ROOT="$FM2_RUN/artifacts/test"
mkdir -p "$FM2_TEST_ROOT/selections/no_evidence/seed13" "$FM2_TEST_ROOT/selections/naive_topm/seed13" "$FM2_TEST_ROOT/selections/bge/seed13"
cd "$REPO" && "$BASE_PY" scripts/preformal/27a_select_no_evidence.py --retrieval "$FM2_TEST_POOL" --output "$FM2_TEST_ROOT/selections/no_evidence/seed13/selection.jsonl" --split test
cd "$REPO" && "$BASE_PY" scripts/08_select_naive_topm.py --config "$FM2_CONFIG" --retrieval "$FM2_TEST_POOL" --output "$FM2_TEST_ROOT/selections/naive_topm/seed13/selection.jsonl" --top-m 4 --min-docs 0 --method-name naive_topm
cd "$REPO" && "$BASE_PY" scripts/12_select_bge_reranker.py --retrieval "$FM2_TEST_POOL" --output "$FM2_TEST_ROOT/selections/bge/seed13/selection.jsonl" --score-cache "$FM2_TEST_ROOT/selections/bge/seed13/scores.jsonl" --model-name-or-path "$MODEL_ROOT/bge-reranker-large" --device auto --dtype auto --batch-size 8 --max-length 512 --top-m 4 --min-docs 0 --local-files-only
for SEED in 13 21 42; do mkdir -p "$FM2_TEST_ROOT/selections/infogain_adapter/seed${SEED}"; "$BASE_PY" "$REPO/scripts/12c_select_infogain_reranker.py" --retrieval "$FM2_TEST_POOL" --checkpoint-dir "$FM2_RUN/artifacts/training/infogain/seed${SEED}/checkpoint" --output "$FM2_TEST_ROOT/selections/infogain_adapter/seed${SEED}/selection.jsonl" --device auto --batch-size 16 --top-m 4 --min-docs 2 --filter-threshold 0.5 --method-name infogain_adapter || exit 1; done
for SEED in 13 21 42; do mkdir -p "$FM2_TEST_ROOT/selections/rag_cbwdm_signed_v1/seed${SEED}"; "$BASE_PY" "$REPO/scripts/preformal/27_select_signed_v1.py" --posteriors "$FM2_TEST_ROOT/posteriors/test.jsonl" --checkpoint-dir "$FM2_RUN/artifacts/training/signed_seed${SEED}/checkpoint" --output "$FM2_TEST_ROOT/selections/rag_cbwdm_signed_v1/seed${SEED}/selection.jsonl" --seed "$SEED" --split test --device auto --batch-size 32 || exit 1; done
```

统一 test evaluation：

```bash
fm2_test_eval() { METHOD="$1"; SEED="$2"; SELECTION="$3"; shift 3; mkdir -p "$FM2_TEST_ROOT/evaluations/$METHOD/seed$SEED"; "$BASE_PY" "$REPO/scripts/07_eval_rag_classification.py" --config "$FM2_CONFIG" --split test --selection "$SELECTION" --output "$FM2_TEST_ROOT/evaluations/$METHOD/seed$SEED/predictions.jsonl" --metrics-output "$FM2_TEST_ROOT/evaluations/$METHOD/seed$SEED/metrics.json" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --method-name "$METHOD" --max-docs 4 "$@"; }
fm2_test_eval no_evidence 13 "$FM2_TEST_ROOT/selections/no_evidence/seed13/selection.jsonl" --no-evidence --max-docs 0
fm2_test_eval naive_topm 13 "$FM2_TEST_ROOT/selections/naive_topm/seed13/selection.jsonl"
fm2_test_eval bge 13 "$FM2_TEST_ROOT/selections/bge/seed13/selection.jsonl"
for SEED in 13 21 42; do fm2_test_eval infogain_adapter "$SEED" "$FM2_TEST_ROOT/selections/infogain_adapter/seed${SEED}/selection.jsonl" || exit 1; done
for SEED in 13 21 42; do fm2_test_eval rag_cbwdm_signed_v1 "$SEED" "$FM2_TEST_ROOT/selections/rag_cbwdm_signed_v1/seed${SEED}/selection.jsonl" || exit 1; done
```

test hard audit 并核对 freeze record 中的训练 manifests 未改变：

```bash
"$BASE_PY" - <<'PY'
import glob,hashlib,json,os
root=os.environ['FM2_TEST_ROOT']; pool=os.environ['FM2_TEST_POOL']; run=os.environ['FM2_RUN']
def rows(p): return [json.loads(x) for x in open(p,encoding='utf-8') if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
expected=[str(x['id']) for x in rows(pool)]; assert len(expected)==1380 and len(expected)==len(set(expected))
selections=sorted(glob.glob(f'{root}/selections/**/selection.jsonl',recursive=True)); assert len(selections)==9
for p in selections:
    r=rows(p); assert [str(x['id']) for x in r]==expected,p
    assert max((len(x.get('selected_docs',[])) for x in r),default=0)<=4,p
manifests=sorted(glob.glob(f'{root}/evaluations/**/metrics.manifest.json',recursive=True)); assert len(manifests)==9
contracts=[json.load(open(p))['contract'] for p in manifests]
for key in ('generator_sha256','prompt_hash','verbalizer_hash','split'):
    assert len({json.dumps(x.get(key),sort_keys=True) for x in contracts})==1,key
assert contracts[0]['split']=='test'
freeze=json.load(open(f'{run}/artifacts/protocol/fm2_freeze_record.json'))
for p,digest in freeze['required_manifests'].items(): assert sha(p)==digest,p
print('FM2 TEST HARD AUDIT PASS')
PY
```

保留 pre-test freeze record 不变，另写 test exposure record：

```bash
"$BASE_PY" - <<'PY'
import glob,hashlib,json,os,pathlib,subprocess
run=pathlib.Path(os.environ['FM2_RUN']); root=pathlib.Path(os.environ['FM2_TEST_ROOT'])
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
freeze=run/'artifacts/protocol/fm2_freeze_record.json'
manifests=sorted(root.glob('evaluations/**/metrics.manifest.json')); assert len(manifests)==9
payload={'schema_version':'rag_cbwdm_fm2_test_exposure.v1','status':'completed','git_head':subprocess.check_output(['git','-C',os.environ['REPO'],'rev-parse','HEAD'],text=True).strip(),'freeze_record_sha256':sha(freeze),'test_prepare_manifest_sha256':sha(pathlib.Path(os.environ['REPO'])/'data/processed/fm2/fm2_prepare_test.manifest.json'),'evaluation_manifests':{str(p):sha(p) for p in manifests},'post_test_tuning_allowed':False}
out=run/'artifacts/protocol/fm2_test_exposure_record.json'; out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); print(out,sha(out))
PY
```

完整方法数是 9（3 deterministic/pretrained + InfoGain 3 seed + signed 3 seed）。任何 test 暴露后的重跑都写 incident reason 并保留旧 artifacts；不得训练、调参或选择最佳 seed。当前仓库尚无 FM2 专用 paired-summary CLI；不能把 FEVER preformal summary script 的 schema 假设套到 FM2。先报告各 seed/aggregate metrics 与 hard-audit provenance；若论文需要 FM2 paired CIs，应在单独评审的 FM2 summary implementation 完成后补算，并且不得反向调参。

---

## 19. Resume / reuse / overwrite matrix

“Resume”不统一等于断点续算。以当前代码为准：

| Stage | First-run command/flag | Interrupted/resume command | Row-level | Atomic publish | Completed artifact | Manifest | 何时允许 overwrite |
|---|---|---|---:|---:|---|---|---|
| FEVER corpus | `01_prepare...`；runner 可在空路径传 `--resume` | `--resume` 只验证/reuse completed | 否 | JSONL 是 | corpus JSONL | `.manifest.json` | source/contract 改变且旧 artifact 已归档；优先新 path |
| Lucene index | `02a_build...`; 空 dir 可带 resume | `--resume` 只验证 completed index | 否 | **否**，直接写 final dir，manifest 最后发布 | index directory | `index_manifest.json` | incomplete/nonempty dir 经人工确认；优先新 fingerprint dir |
| BM25 retrieval | 当前 CLI **无 resume** | completed 合约匹配则只读复用；中断后同命令从头 | 否 | 是，`.partial→output` | retrieval JSONL | `.manifest.json` | 仅显式重算且旧结果归档；`--overwrite` |
| Posterior | **不带 `--resume`** | manifest 已存在后相同命令 `--resume` | **是** | 最终 promote 是 | posterior JSONL；运行中 `.partial` | `.manifest.json`，running/failed/completed | contract 改变用新 path；`--overwrite` 会移除 output/partial/manifest，须先归档 |
| signed teacher | 不带 resume | `--resume` 只复用完整 SHA-compatible teacher | 否 | teacher JSONL 是；整个三文件集合非单事务 | `teacher.jsonl`,`statistics.json` | `manifest.json` | CLI 无 overwrite；歧义状态用新 output dir |
| signed training | 不带 resume | `--resume` 只复用 completed checkpoint | 否/无 mid-epoch | 否 | `checkpoint/`、history/config | `training_manifest.json` | CLI 无 overwrite；中断/改 contract 用新 dir |
| signed selection | 不带 resume | completed exact artifact 加 `--resume` | 否 | 是 | `selection.jsonl` | `selection.manifest.json` | formal CLI 无 overwrite；新 path |
| InfoGain teacher | 不带 resume | completed exact `--resume` | 否 | 是 | teacher JSONL | `.manifest.json` | 只在旧结果归档后；优先新 path |
| InfoGain training | 不带 resume | completed exact `--resume`；无 mid-epoch | 否 | 否 | `checkpoint/` | `training_manifest.json` | `--overwrite` 仅经人工确认；正式 run 优先新 dir |
| InfoGain selection | 不带 resume | completed exact `--resume` | 否 | 是 | selection JSONL | `.manifest.json` | contract 改变新 path；必要时显式 overwrite |
| BGE scoring | 不带 resume | `--resume` 只复用 completed valid cache；中断从头 score | 否 | 是 | score cache JSONL | cache `.manifest.json` | mismatch 时新 cache；归档后才 overwrite |
| BGE selection | 不带 resume | completed exact `--resume` | 否 | 是 | selection JSONL | selection `.manifest.json` | 新 path 优先 |
| Qwen evaluation | 不带 resume | completed exact `--resume`；中断从头 eval | 否 | predictions 是；metrics/manifest 随后发布 | predictions + metrics | `metrics.manifest.json` | 新 output dir；归档后才 overwrite |
| FM2 download | 不带 overwrite | 已有文件按 expected SHA 自动 reuse；下载本身不续 partial | 否 | 是 | raw JSONL | 无独立 manifest；hash 在代码/prepare manifest | 只有已知需重取且旧文件已隔离 |
| FM2 prepare | 不带 overwrite | 无 resume；中断产生任一 output 时 fail closed | 否 | 单文件 atomic，整组非事务 | query/pool/gold files | `fm2_prepare*.manifest.json` | 审计/归档全部 partial outputs 后，或新 output dir |

绝对规则：

1. changed seed/threshold/limit/model/prompt/split/candidate SHA → 新 output path；
2. 不把 `--overwrite` 当通用恢复键；
3. 不删除 FEVER2/FM2 正式 artifact 来“试一次”；
4. 首次 posterior manifest 不存在时，`--resume` 必然失败，这是预期保护。

---

## 20. Recovery 与 troubleshooting

### 20.1 SSH 断线 / 不确定进程是否仍在

先恢复 tmux，不要先重启命令：

```bash
tmux ls
tmux attach -t rag-cbwdm
pgrep -af 'run_fever_cbwdm|compute_label_posteriors|train_signed|train_infogain|eval_rag|select_bge'
nvidia-smi
```

若原进程仍在，继续观察。若已退出，读取最后的 manifest：posterior 的 `status=failed/running` 且 fingerprint 相同才走 row-level resume；其他 expensive stages 按 Resume Matrix 判断是 completed reuse 还是从头/新目录。

### 20.2 OOM

1. 保存完整 traceback、`nvidia-smi`、命令行与 manifest fingerprint。
2. 确认没有另一个重复进程占显存。
3. posterior 的 `--batch-size` 在 fingerprint 中；调小 batch 必须使用新 output path，从头运行，不能 resume 旧 manifest。
4. BGE scoring/InfoGain selection/signed selection 的 inference batch 可调小；中断 scoring 会从头写 cache，selection 仍必须保持相同 top_m/min_docs/threshold。
5. signed-v1 training `batch_size=8` 是冻结科学合约，不因 OOM 改成别的值；换更大 GPU。
6. InfoGain training超参同样按已冻结 protocol，不在看过 dev/preformal/test 后因 OOM 改科学参数。
7. evaluation 当前 CLI 没有 batch-size 参数；不要通过改源码临时改变 production evaluator。

### 20.3 Partial artifact

```bash
find "$EXP_ROOT" -type f \( -name '*.partial' -o -name '*manifest*.json' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort
```

- posterior：保留 `.partial` + running/failed manifest；相同 contract 加 `--resume`。
- retrieval/BGE/evaluation：partial 不代表 row-level resume；同一 first-run command 从头写 partial，completed final artifact 则先校验/复用。
- index：无 completed `index_manifest.json` 的非空 final directory 是 incomplete，不能 `--resume` 冒充完成。优先新 fingerprint/path；只有确认无有效数据且已有外部备份时才显式 overwrite。
- signed/InfoGain training：没有 mid-epoch checkpoint 合约。保留故障目录，新的尝试用新 run/output dir。

### 20.4 Conda / mirror

若 log 出现 `mirrors.aliyun.com`：

```bash
[ -f /root/.condarc ] && cat /root/.condarc || true
CONDARC=/root/rag-cbwdm-condarc "$MINICONDA_ROOT/bin/conda" config --show-sources
```

停止 create/install，备份移走新生成的 `.condarc`，再重试。PyTorch wheel 只从官方 cpu/cu124 index 以 `--no-deps` 安装；其余 packages 走 TUNA PyPI。

### 20.5 Java/Pyserini

症状包括 `UnsupportedClassVersionError`、找不到 javac、`LuceneSearcher` import/open 失败。不要先 apt install JDK；重新执行：

```bash
export JAVA_HOME="$MINICONDA_ROOT/envs/$RETRIEVAL_ENV"; export PATH="$JAVA_HOME/bin:$PATH"
command -v java; command -v javac; java -version; javac -version
"$RETR_PY" -c "from pyserini.search.lucene import LuceneSearcher; print('pyserini_java_import=PASS')"
```

### 20.6 Contract/checksum mismatch

不要用 overwrite 压过 mismatch。比较 config SHA、input SHA、model/revision/SHA、prompt/verbalizer hash、split、limit、seed、threshold、index fingerprint。若请求本来就不同，创建新命名 output；若本应相同，先调查 bit rot/误路径并保留旧 artifact。

---

## 21. GPU / RAM / disk / log monitoring

### 21.1 计算量规划（不是 pass threshold）

- FEVER clean preformal posterior 最多约 `21 × N` 个 prompts；signed state-aware selection 最坏每 query/seed 打分 `20+19+18+17=74` 个 state-candidate pairs。
- FM2 full train+dev posterior 约 126k 个短 prompts，BGE 约 120k pairs，训练为 signed 三 seed + InfoGain 三 seed各 3 epochs，dev generator evaluation 约 5.8k 次。
- 历史硬件无关规划区间为 FM2 full train/dev 约 12–30 modern-GPU hours；它不是 SLA。必须用 smoke8 与 pilot500 实测 throughput/peak VRAM，再预约 full 窗口。
- held-out 的任何耗时预估都不能成为提前打开 test 的理由。

在另一个 tmux pane 运行：

```bash
watch -n 2 nvidia-smi
```

```bash
nvidia-smi dmon -s pucvmet
```

周期盘点：

```bash
free -h
df -hT /root "$EXP_ROOT"
du -sh "$EXP_ROOT" "$HF_HOME" "$MODEL_ROOT" "$REPO/data" 2>/dev/null
ps -eo pid,ppid,etime,%cpu,%mem,rss,cmd --sort=-rss | head -30
```

长命令 log 模式：

```bash
set -o pipefail
test -f "$FM2_RUN/artifacts/posteriors/dev.manifest.json" && "$BASE_PY" "$REPO/scripts/03_compute_label_posteriors.py" --config "$FM2_CONFIG" --split dev --retrieval "$FM2_DATA/fm2_dev_official_pool.jsonl" --output "$FM2_RUN/artifacts/posteriors/dev.jsonl" --model-name "$MODEL_ROOT/Qwen2.5-1.5B-Instruct" --batch-size 16 --resume 2>&1 | tee -a "$FM2_RUN/artifacts/logs/posterior_dev_resume.log"
```

上行是实际 FM2 posterior resume 示例，manifest 不存在时会被前置 `test` 阻止。给每次尝试唯一 log，不要把 secret/token 打进 log。

磁盘告警时先停止启动新 stage，盘点 checkpoint/cache/artifacts；不要在运行中的目录执行清理。HF cache、模型目录、shared index 与正式 posterior 都可能很大。

---

## 22. Artifact provenance 与 checksum

### Stage 37 — 每个 gate 后固定 provenance

**Purpose:** 结果身份由内容与合约决定，而不是历史 accuracy。  
**Environment:** base/baseline。  
**GPU:** 否。  
**Inputs:** completed run。  
**Outputs:** per-run checksum inventory、Git/env snapshot。  
**First run:** 每个主要 gate 生成一个新 timestamped inventory。  
**Resume:** 不覆盖旧 inventory。  
**Completion check:** inventory 自身 SHA 已记录，路径可从外部 archive 读取。  
**Expected invariant:** commit、dirty status、data/model/index/prompt/checkpoint/artifact SHA 可追溯。  
**Next gate:** archive。

每份核心 manifest 至少核对：

- Git commit/dirty state；
- config path + SHA；
- dataset/source revision + raw SHA；
- split role、row count、ID set；
- retrieval pool/index fingerprint/input SHA；
- generator/model/checkpoint revision + SHA；
- prompt version/hash、verbalizer hash、max context；
- method parameters、seed；
- output SHA、status/completed time；
- gold/deployable/diagnostic flags。

为一个 run 建 checksum inventory：

```bash
export RUN_TO_FREEZE="$FM2_RUN"
export FREEZE_TS="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_TO_FREEZE/artifacts/provenance"
find "$RUN_TO_FREEZE" -type f ! -path "$RUN_TO_FREEZE/artifacts/provenance/*" -print0 | sort -z | xargs -0 sha256sum > "$RUN_TO_FREEZE/artifacts/provenance/files-$FREEZE_TS.sha256"
sha256sum "$RUN_TO_FREEZE/artifacts/provenance/files-$FREEZE_TS.sha256"
```

FEVER run 使用 `RUN_TO_FREEZE="$FEVER_RUN"` 再做一次；不要让 checksum inventory 自己递归进入自己。

环境/release 快照：

```bash
cd "$REPO" && git rev-parse HEAD && git status --short
"$RETR_PY" -m pip freeze > "/root/rag-cbwdm-recovery-logs/retrieval-pip-freeze-$FREEZE_TS.txt"
"$BASE_PY" -m pip freeze > "/root/rag-cbwdm-recovery-logs/baseline-pip-freeze-$FREEZE_TS.txt"
nvidia-smi > "/root/rag-cbwdm-recovery-logs/nvidia-smi-$FREEZE_TS.txt"
```

---

## 23. 释放服务器前必须保存/同步

### Stage 38 — Capture + external archive

**Purpose:** 服务器删除后仍可验证和恢复全部实验。  
**Environment:** base shell。  
**GPU:** 否。  
**Inputs:** completed/failed runs、shared assets、logs。  
**Outputs:** timestamped server-state snapshot 与持久外部 archive。  
**First run:** capture 后 rsync。  
**Resume:** rsync `--partial` 可重试；snapshot 目录必须是新 timestamp。  
**Completion check:** archive 端文件数量/大小/checksum；至少随机 rehash 核心 manifests/checkpoints/posteriors。  
**Expected invariant:** archive 不依赖临时系统盘。  
**Common failure:** 只保存 metrics，不保存 checkpoint/manifests/commands；只保存 repo 不保存 unpushed release；capture script 误被当成 FM2 全覆盖。  
**Next gate:** 才可释放实例。

先捕获当前 FEVER server state：

```bash
cd "$REPO" && export RUN_NAME=fever2_formal_pilot_5000_500_seed13 && export SNAPSHOT_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)" && bash scripts/18_capture_server_resume_state.sh
```

当前 `scripts/18` 主要盘点 FEVER pilot/shared index，并只检查 Java/package-level Pyserini；它没有完整纳入 FM2/preformal，也没有 javac/LuceneSearcher hard gate。因此还必须同步以下内容：

1. `$EXP_ROOT/fever2_formal_pilot_5000_500_seed13/`，包括 formal、preformal、logs、commands、checkpoints、manifests、summary；
2. `$EXP_ROOT/fm2_smoke_v1/`、pilot/full/test（若存在）的完整 run roots；
3. `$EXP_ROOT/_shared/` 的 corpus/index；
4. `$REPO/data/raw/{fever,fm2}` 与 `$REPO/data/processed/fm2`，或至少可重下载的 immutable revisions + 所有 SHA/manifests；最终论文复现优先同步数据本体；
5. `$MODEL_ROOT`，或至少五个 pinned revision + asset manifest + file SHA；若上游可用性不确定则同步模型本体；
6. 两个 conda export/pip freeze、Java/Javac/Pyserini real import log、CUDA/Torch log；
7. release commit/remote/status、本 master runbook、所有实际命令与 stderr/stdout logs；
8. calibration/freeze records、fairness、paired statistics、test authorization/incident records。

外部目标必须是持久挂载/对象存储同步目录，不是实例临时盘：

```bash
export ARCHIVE_ROOT='<persistent mounted archive path>'
case "$ARCHIVE_ROOT" in ''|\<*) echo 'BLOCKED: persistent ARCHIVE_ROOT required'; exit 2;; esac
export ARCHIVE_RUN="$ARCHIVE_ROOT/rag-cbwdm-$FINAL_DEPLOY_COMMIT-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$ARCHIVE_RUN"
rsync -aH --partial "$EXP_ROOT/" "$ARCHIVE_RUN/experiments/"
rsync -aH --partial "$REPO/data/" "$ARCHIVE_RUN/repo-data/"
rsync -aH --partial "$MODEL_ROOT/" "$ARCHIVE_RUN/models/"
rsync -aH --partial /root/rag-cbwdm-recovery-logs/ "$ARCHIVE_RUN/recovery-logs/"
rsync -aH --partial "$REPO/environment/server/" "$ARCHIVE_RUN/environment-server/"
cp "$REPO/RAG_CBWDM_MASTER_SERVER_RUNBOOK_2026-08-24.md" "$ARCHIVE_RUN/"
git -C "$REPO" rev-parse HEAD > "$ARCHIVE_RUN/git-head.txt"
git -C "$REPO" status --short > "$ARCHIVE_RUN/git-status.txt"
```

归档验收：

```bash
du -sh "$ARCHIVE_RUN" "$EXP_ROOT" "$REPO/data" "$MODEL_ROOT"
find "$ARCHIVE_RUN" -type f | wc -l
find "$ARCHIVE_RUN/experiments" -type f \( -name '*manifest*.json' -o -name '*.sha256' -o -name 'metrics.json' \) | sort | head -100
```

`scripts/18` 会刷新 repo 下小型 `environment/server/captured/` metadata，可能使 release worktree 变 dirty；将其作为 archive metadata 审计，不要未经 credential/size review 就 commit/push。

实例释放硬门：

```bash
test -s "$ARCHIVE_RUN/git-head.txt" && test -f "$ARCHIVE_RUN/RAG_CBWDM_MASTER_SERVER_RUNBOOK_2026-08-24.md" && test "$(find "$ARCHIVE_RUN/experiments" -type f | wc -l)" -gt 0 && echo 'ARCHIVE BASIC GATE PASS'
```

只有外部存储侧再次确认可读，才停止/释放服务器。

---

## 24. 从新机到最终结果的最短已验证路径

这是索引，不替代各 Stage 的 gate：

1. 填真实 `FINAL_DEPLOY_COMMIT`，clone/fetch/checkout；确认 HEAD 相等、worktree clean。
2. `nvidia-smi/free/df` 与 GitHub/HF/TUNA/PyTorch network gate。
3. 备份移走 MatPool `/root/.condarc`；建立 project condarc。
4. 按当前 YAML split-install retrieval/baseline；pip-freeze diff。
5. 每个 retrieval shell export Conda `JAVA_HOME/PATH`；Java21 + Javac21 + real `LuceneSearcher` import。
6. 用 pinned revisions 下载五个模型；check-only + actual local load。
7. 下载/验证 FEVER train/dev/wiki；下载/prepare FM2 train/dev；冻结 SHA/manifest。
8. compileall + pytest + unittest + diff check + bash -n。
9. FEVER scripts22 先 `--stop-after retrieve_validation`。
10. baseline env 中两个 posterior FIRST RUN 不带 resume；之后重跑 scripts22 与 scripts23。
11. 构造 clean preformal split → 复用 index retrieval → shared posterior（首跑无 resume）。
12. FEVER signed teacher → signed seeds13/21/42 → signed selections；no-evidence/Naive/BGE/InfoGain reference；old-RAG 只在精确合约可恢复时运行。
13. 统一 Qwen evaluation → complete fairness audit → paired summary → optional post-hoc；不从 preformal 静默调参。
14. FM2 smoke8 → pilot500 → full train/dev posterior；FM2-specific signed/InfoGain 各三 seed；no-evidence/Naive/BGE。
15. FM2 unified dev evaluation → hard fairness/provenance audit → dev-only threshold policy freeze。
16. 只有 freeze record 与人工授权后才下载/prepare/score FM2 test；不再训练/调参。
17. 每 run 建 checksum inventory；运行 capture；完整 rsync 到外部持久存储并在 archive 端验收。

---

## Appendix A. 历史提交，只作 provenance

- `4b914d2...`：早期 reproducible new-server workflow；不是最终 FM2/signed-v1 release。
- `3d3aa4fb94cb3496e45309c003618f0bf91aeaa6`：signed-v1 preformal pipeline 的历史实现基线；本 master 编写时 FM2 代码仍位于 dirty worktree，因此不能成为未来 deployment target。
- `9ddccbb...`、`221394f...`：signed teacher / failure diagnostics 历史节点。

未来所有部署只使用发布负责人填入并已 push 的 `FINAL_DEPLOY_COMMIT`。

## Appendix B. FEVER2 historical clean-preformal sanity references

这些是历史结果，不是环境恢复 hard threshold；未来 commit/data/model revision 变化时以 artifact SHA/manifest 为准：

| 方法 | Seed | Accuracy（约） |
|---|---:|---:|
| no_evidence | 13 | 0.7191 |
| naive top4 | 13 | 0.7316 |
| BGE | 13 | 0.7909 |
| InfoGain adaptation | 13 | 0.7864 |
| InfoGain adaptation | 21 | 0.7829 |
| signed-v1 | 13 | 0.7860 |
| signed-v1 | 21 | 0.8013 |
| signed-v1 | 42 | 0.7898 |

不要因新结果未落在这些数附近就覆盖 artifacts 或调参；先做 provenance/fairness 调查。

## Appendix C. 本文读取的 source runbooks/reports

完整审计源位于 `_codex_context/server_runbook_2026-08-24/`：

1. `RAG_CBWDM_SERVER_REALITY_ALIGNMENT_NOTES_2026-08-24.md`
2. `RAG_CBWDM_NEW_SERVER_PREFLIGHT_FAST_PATH_2026-08-20.md`
3. `RAG_CBWDM_TWO_ENV_REBUILD_RUNBOOK_2026-08-20.md`
4. `SIGNED_TEACHER_V1_SERVER_RUNBOOK.md`
5. `SIGNED_TEACHER_V1_IMPLEMENTATION_REPORT.md`
6. `SIGNED_V1_PREFORMAL_EXPERIMENT_RUNBOOK.md`
7. `SIGNED_V1_PREFORMAL_IMPLEMENTATION_REPORT.md`
8. `FM2_SERVER_RUNBOOK.md`
9. `FM2_IMPLEMENTATION_REPORT.md`

## Appendix D. Superseded/迁移差异

1. 旧 fixed commits 被 release placeholder + SHA/clean gate 取代。
2. 系统 `/usr/bin/java` 不再作为 Pyserini Java；Conda retrieval Java/Javac21 + real Lucene import 是 P0 gate。
3. `scripts/20/19/23` 的 Java/package-only check 不再被当作完整 acceptance；建议后续补 javac 与 `LuceneSearcher`。
4. 全量 bootstrap pip-extra-index 路径被当前 YAML split-install 路径取代，以规避普通依赖在 PyTorch index 上长时间卡住。
5. `scripts/22` fresh posterior 的 `--resume` 错位由 Stage 14–15 两段式路径绕开；建议后续修改 wrapper，仅在 manifest 已存在时透传 resume。
6. current corpus manifest 与 scripts19/23 的 `status` 要求存在 schema gap；本文 current-schema hard verify 不伪造字段，建议后续修正 verifier/builder 合约。
7. 当前 02 retrieval CLI 没有 resume；旧文档任何 retrieval `--resume` 命令均失效。
8. posterior 明确区分首次无 resume 与 manifest 存在后的 row-level resume。
9. FM2 第一轮不建 index、不复用 FEVER index，候选预算不再称 top20；少于4候选时用 available pool。
10. FEVER checkpoints 不再作为 primary FM2 learned methods；signed/InfoGain 必须 FM2 train 重训。
11. FM2 专用自动 fairness/paired summary 与 capture coverage 尚未实现；本文以 hard audit + 显式 blocker 诚实补位，不把 FEVER CLI 错用到 FM2。

## Appendix E. 发布前 master 自审清单

- [ ] 没有把历史 SHA 写成未来固定部署目标；
- [ ] 没有任何 Pyserini 主流程依赖系统 Java；
- [ ] 明确检查 javac 21；
- [ ] 明确真实 import `LuceneSearcher`；
- [ ] retrieval/baseline 环境无混用；
- [ ] 所有命令参数来自当前 `--help`/源码；
- [ ] posterior first run 无 `--resume`；
- [ ] FEVER2/FM2 pool/index/split 明确区分；
- [ ] training/dev/preformal/held-out/oracle 明确区分；
- [ ] InfoGain 明确标为 classification adaptation；
- [ ] no-evidence/Naive/BGE 均存在；
- [ ] release 前 archive 包含 artifacts/checkpoints/manifests/logs；
- [ ] tmux/SSH/OOM/partial recovery 已覆盖；
- [ ] 每个大 Stage 有 completion check；
- [ ] 第 24 节构成从新机到最终结果的连续主路径。

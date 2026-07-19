# New server quick start

本页只提供实际操作入口。完整的灾备协议、审计边界和逐阶段命令继续保留在
`NEW_SERVER_FROM_ZERO_RUNBOOK.md` 与
`RAG_CBWDM_SERVER_REPRODUCTION_AND_RESUME_RUNBOOK.md`。

## 前置准备

新服务器建议使用 Ubuntu 22.04+、24 GB 以上 NVIDIA GPU、64 GB 以上内存和
至少 500 GiB 可用持久化磁盘。先取得最终恢复 commit 和官方 Miniconda
installer SHA-256：

```bash
sudo apt-get update
sudo apt-get install -y git ca-certificates
git clone https://github.com/wenhaojang/rag-cbwdm.git /root/rag-cbwdm
cd /root/rag-cbwdm
git checkout --detach '<final recovery commit>'
test -z "$(git status --porcelain)"

export EXPECTED_GIT_HEAD="$(git rev-parse HEAD)"
export MINICONDA_SHA256='<verified official Miniconda installer SHA-256>'
mkdir -p /root/rag-cbwdm-recovery-logs
set -o pipefail
```

不要在命令或日志中写入密码、Token、SSH key或其他凭据。

## 1. 配置系统与两个 conda 环境

```bash
bash scripts/20_bootstrap_new_server.sh 2>&1 \
  | tee -a /root/rag-cbwdm-recovery-logs/20_bootstrap.log
```

该命令检查 Ubuntu、GPU、driver、RAM和磁盘，安装系统依赖与 Miniconda，
再创建：

- `rag-cbwdm-retrieval`：Java 21、Pyserini 2.3.0、corpus/index/retrieval；
- `rag-cbwdm-baselines`：CUDA Torch、Transformers、posterior、teacher、
  training、selection和evaluation；该环境不得依赖 Pyserini。

脚本按 YAML 顶层 `name:` 自动选中真实 Linux 文件：

```text
environment/server/rag-cbwdm-retrieval.yml
environment/server/rag-cbwdm-baselines.yml
```

旧服务器的顶层 `prefix:` 会在临时副本中删除。两份
`*-pip-freeze.txt` 只用于安装后的版本审计；两份 `*.template` 永远不会被
用于部署。可用以下命令只做检查：

```bash
bash scripts/20_bootstrap_new_server.sh --check-only --skip-apt
```

## 2. 下载五个模型与 FEVER 数据

当前环境文件没有记录不可变 Hugging Face revision。先从旧服务器审计记录
补齐下列公开资产的 commit revision；禁止猜测或使用移动的 `main`：

```bash
export QWEN15_REVISION='<Qwen/Qwen2.5-1.5B-Instruct commit>'
export QWEN7_REVISION='<Qwen/Qwen2.5-7B-Instruct commit>'
export MINILM_REVISION='<cross-encoder/ms-marco-MiniLM-L-6-v2 commit>'
export BGE_BASE_REVISION='<BAAI/bge-reranker-base commit>'
export BGE_LARGE_REVISION='<BAAI/bge-reranker-large commit>'
export FEVER_DATASET_REVISION='<Hugging Face fever dataset commit>'
```

默认先尝试严格固定模式；缺少任何 revision 时该命令会明确拒绝下载：

```bash
bash scripts/21_download_project_assets.sh 2>&1 \
  | tee -a /root/rag-cbwdm-recovery-logs/21_assets.log
```

严格模式的核心命令是：

```bash
bash scripts/21_download_project_assets.sh
```

如果 revision 确实无法恢复，用户可以显式选择功能恢复模式：

```bash
bash scripts/21_download_project_assets.sh --allow-unpinned
```

等价的环境变量写法为
`ALLOW_UNPINNED_ASSETS=1 bash scripts/21_download_project_assets.sh`。该模式
只使用仓库中已经审计的五个真实模型 ID，并解析下载时实际使用的 Hub commit；
FEVER 记录物化文件的 dataset fingerprint。Asset manifest 会明确写入
`revision_mode=unpinned` 和功能恢复范围。它允许重新运行实验，但不保证与旧
模型文件、数据快照或 artifact 逐字节一致。

这是通常最受网络速度影响的步骤；脚本会写 asset manifest，并验证模型
config、tokenizer、权重以及 FEVER 文件的大小和 SHA-256。中断后直接重复
同一命令即可续传。无网络验收：

```bash
bash scripts/21_download_project_assets.sh --check-only
```

## 3. 从头重跑到 5000/500 pilot smoke

```bash
bash scripts/22_rebuild_to_current_smoke.sh 2>&1 \
  | tee -a /root/rag-cbwdm-recovery-logs/22_rebuild.log
```

该命令依次恢复 split、full corpus、Lucene v2 index、5000/500 retrieval、
5000/500 posterior，以及 InfoGain 和 RAG-CBWDM 各一组
`1 training + 2 selection/evaluation` smoke。Lucene index、posterior和GPU
训练通常是计算最耗时的阶段。

所有子阶段都使用现有 manifest/`--resume` 语义。中断或部分失败后直接重复
同一命令；已通过 fingerprint、状态和 SHA 校验的 completed 子阶段不会重算。
可先查看计划或仅验收：

```bash
bash scripts/22_rebuild_to_current_smoke.sh --dry-run
bash scripts/22_rebuild_to_current_smoke.sh --verify-only
```

正常完成时最终验收输出 `PASS`。也可单独运行：

```bash
bash scripts/23_verify_rebuilt_smoke.sh
```

该 `PASS` 同时证明 InfoGain 和 RAG-CBWDM 均至少有两个
`status=completed`、且 `accuracy`/`macro_f1` 非空的 smoke evaluation。
日志位于 `/root/rag-cbwdm-recovery-logs/`；实时查看可使用
`tail -f /root/rag-cbwdm-recovery-logs/22_rebuild.log`。

## Smoke 之后

完整 calibration grid 不再传 `--candidate-limit` 或
`--max-training-candidates`。在 `rag-cbwdm-baselines` 环境中依次执行复杂
runbook 的 10.1（full InfoGain）、10.2（full RAG-CBWDM）、10.3（合并
aggregate）和 10.4（`calibrate_methods`）；命令继续使用 `--resume
--skip-completed`。

在 `formal_readiness.json` 明确满足 `status=ready` 且 `blockers=[]` 之前，
禁止运行 `retrieve_test`、`posterior_test`、任何 `held_out_test` teacher、
held-out gold evaluation或用 held-out 数据做参数选择。

# RAG-CBWDM Server Runbook

## 1. 上传、环境与缓存

推荐在 Linux GPU 服务器 clone 仓库；若上传工作目录，必须连同 `configs/`、`scripts/`、
`src/`、`tests/fixtures/` 一起保留，避免上传本地 `.venv` 和旧 checkpoint。

```bash
git clone <YOUR_REPOSITORY_URL> rag_cbwdm
cd rag_cbwdm
conda create -n rag-cbwdm python=3.10 -y
conda activate rag-cbwdm
```

先根据服务器 CUDA 版本从 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)
安装匹配的 PyTorch，不在本仓库固定 CUDA wheel。例如 CUDA 12.4：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Hugging Face 公共 Qwen2.5 和默认 cross-encoder 通常不要求登录；若服务器或所选 revision
要求认证，再执行 `huggingface-cli login`。统一缓存：

```bash
export HF_HOME=/mnt/fast-cache/huggingface
mkdir -p "$HF_HOME"
```

## 2. FEVER 数据

仓库主流程预期：

```text
data/raw/fever/train.jsonl
data/raw/fever/dev.jsonl
data/raw/fever/shared_task_test.jsonl
data/raw/fever/wiki-pages/*.jsonl
```

若使用 Hugging Face 下载辅助脚本，先查看参数再执行；不要在 formal 运行中临时更换数据版本：

```bash
python scripts/download_fever_hf.py --help
python scripts/00_prepare_fever.py --help
```

## 3. 环境预检

环境检查不下载模型：

```bash
python scripts/check_environment.py \
  --config configs/fever2_server_smoke.yaml \
  --output-root /mnt/experiments/rag_cbwdm \
  --cache-root "$HF_HOME"
nvidia-smi
df -h /mnt/experiments "$HF_HOME"
```

应确认 PyTorch `cuda_available=true`、GPU 名称/显存正确、输出和缓存目录可写、
`trust_remote_code=false`。建议在运行前将 model/revision 固定为经验证值；`null`
表示仍使用 Hugging Face 默认分支，不是强复现状态。

## 4. Dry run

runner 可从任意当前目录调用；bash wrapper 自行定位项目根目录。

```bash
bash /path/to/rag_cbwdm/scripts/run_fever_cbwdm.sh \
  --config /path/to/rag_cbwdm/configs/fever2_server_smoke.yaml \
  --run-name fever2_smoke \
  --stages prepare,corpus,retrieve,posterior,teacher,train,select,eval \
  --limit 100 \
  --output-root /mnt/experiments/rag_cbwdm \
  --cache-root "$HF_HOME" \
  --dry-run
```

别名 `train`/`select` 分别解析为 `train_cross_encoder`/`select_cross_encoder`。

## 5. Smoke

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_smoke.yaml \
  --run-name fever2_smoke_seed13 \
  --stages prepare,corpus,retrieve,posterior,teacher,train,select,eval,no_evidence,naive_topm,oracle_diagnostic \
  --limit 100 \
  --seed 13 \
  --output-root /mnt/experiments/rag_cbwdm \
  --cache-root "$HF_HOME" \
  --posterior-batch-size 2 \
  --selector-batch-size 4
```

`oracle_diagnostic` 使用 gold-label teacher，只用于 sanity check，输出明确
`diagnostic_only=true, deployable=false`。

## 6. Pilot

先根据 smoke 显存逐步增加 posterior batch；pilot 阈值仍是初始值：

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_pilot.yaml \
  --run-name fever2_pilot_seed13 \
  --stages prepare,corpus,retrieve,posterior,teacher,train_cross_encoder,select_cross_encoder,eval \
  --limit 5000 \
  --seed 13 \
  --output-root /mnt/experiments/rag_cbwdm \
  --cache-root "$HF_HOME" \
  --posterior-batch-size 8 \
  --selector-batch-size 16
```

只使用 train/dev 比较 `b_plus`、`b_minus`、`gamma`、`beta`、selector stop threshold；
禁止查看 test 后调参。

## 7. Formal

formal 前必须完成：固定数据 hash、generator/selector revision、pilot 选出的 loss/stop
参数以及可承受的 batch size。然后对三个 seed 分别运行，不能共用 run name：

```bash
for seed in 13 21 42; do
  bash scripts/run_fever_cbwdm.sh \
    --config configs/fever2_server_formal.yaml \
    --run-name "fever2_formal_seed${seed}" \
    --stages prepare,corpus,retrieve,posterior,teacher,train_cross_encoder,select_cross_encoder,eval \
    --seed "$seed" \
    --output-root /mnt/experiments/rag_cbwdm \
    --cache-root "$HF_HOME"
done
```

若 posterior 与 selector 都固定且只研究 selector seed，可以复用经 hash 验证的只读
posterior artifact，但不要手工复制 manifest 后伪装成同一次 run。

## 8. Resume 与定向重跑

每个 run 保存 `run_manifest.json`、`logs/`、`commands/`、`artifacts/`。恢复：

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_smoke.yaml \
  --run-name fever2_smoke_seed13 \
  --stages posterior,teacher,train,select,eval \
  --resume \
  --limit 100 \
  --output-root /mnt/experiments/rag_cbwdm \
  --cache-root "$HF_HOME"
```

posterior 使用 `*.jsonl.partial` 与 `*.manifest.json`。输入/config/model/prompt/verbalizer
fingerprint 不同会拒绝 resume；只有显式 `--overwrite-stage posterior` 才重算。runner
不会自动删除旧目录。其他阶段定向重跑示例：

```bash
bash scripts/run_fever_cbwdm.sh \
  --config configs/fever2_server_smoke.yaml \
  --run-name fever2_smoke_seed13 \
  --stages train,select,eval \
  --resume \
  --overwrite-stage train_cross_encoder \
  --output-root /mnt/experiments/rag_cbwdm \
  --cache-root "$HF_HOME"
```

## 9. 日志、监控与 OOM

```bash
tail -f /mnt/experiments/rag_cbwdm/fever2_smoke_seed13/logs/posterior.log
watch -n 2 nvidia-smi
du -sh /mnt/experiments/rag_cbwdm/fever2_smoke_seed13 "$HF_HOME"
jq '.stages' /mnt/experiments/rag_cbwdm/fever2_smoke_seed13/run_manifest.json
```

posterior OOM：先降低 `--posterior-batch-size`，然后 `--resume`；partial 会保留。
selector OOM：降低 `--selector-batch-size`（推理候选 batch）或 config 中训练
`batch_size/max_length`。不要删除 partial 或 checkpoint 来“修复”OOM。

## 10. 结果下载与关机前保留

至少下载并校验：

```text
<run>/run_manifest.json
<run>/commands/
<run>/logs/
<run>/artifacts/*manifest.json
<run>/artifacts/*teacher.jsonl
<run>/artifacts/cross_encoder/checkpoint/
<run>/artifacts/*selection.jsonl
<run>/artifacts/*predictions.jsonl
<run>/artifacts/*metrics.json
最终使用的 YAML config
```

示例：

```bash
tar -C /mnt/experiments/rag_cbwdm -czf fever2_smoke_seed13.tar.gz fever2_smoke_seed13
sha256sum fever2_smoke_seed13.tar.gz
```

确认压缩包已下载并能解包、posterior partial/manifest 已保留、模型 revision 和数据 hash
已记录后，才能终止实例。

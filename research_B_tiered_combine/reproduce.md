# 复现入口（初始化中）
任务规格：configs/task_brief.txt。当前证据与未完成项：RESULTS.md。

本地回传：
```bash
cd research_B_tiered_combine
bash code/pull_results.sh
```
SSH 交互输入凭证；脚本不存储密码。远端实验目录默认 /root/autodl-tmp/research_B_tiered_combine，可用 EXPERIMENT_HOST / EXPERIMENT_PORT / EXPERIMENT_ROOT 覆盖。

已安装 vLLM 源码快照：runs/environment_20261016/vllm_moe_sources.tar.gz；解压副本在同目录 source/。这只证明源代码逻辑，不能替代 kernel 实际执行证据。

## 已有实现检查

在远端工作目录执行（均不占用 GPU）：
```bash
OMP_NUM_THREADS=2 /root/miniconda3/bin/python code/check_codec.py --output runs/UNIQUE/reference.json
/root/miniconda3/bin/python code/compile_codec.py --output runs/UNIQUE/compile.json
```
先创建唯一的输出目录，保留失败日志。GPU 空闲且无其他整组任务后，`bash code/run_codec_gpu.sh UNIQUE` 持共享锁执行数值检查和微基准；驱动遇到 GPU 利用率非零直接退出，不终止其他进程。脚本需随代码部署。每组完成后从本地执行 pull_results.sh。

动态 NVFP4 tensor scale 包含归约开销，微基准将动态 quant 与 given-scale pack 分列；只用前者评价可部署的动态 codec 成本。GB/s 是 BF16 解码载荷除以延时，不是实测显存带宽。

## vLLM 质量 hook 集成 smoke（已通过）

部署 code/ 后，用共享锁驱动：
```bash
bash code/run_quality_smoke.sh unique_smoke_run /absolute/path/to/olmoe
```
此入口目前明确检查 16 层 OLMoE。使用现存 Instruct 权重时，仅能证明该 checkpoint 的集成行为；不替代正式四模型质量结果。按各模式单独的 BF16 reference 比较，不能把 grouped 累加的变化全归因于量化。脚本覆盖两个短提示的 prompt logprobs 和完整分布 KL；仍不是 WikiText corpus 质量评价。命名 worker 扩展由 quality_worker.py 提供。

## 当前验证组与环境设置

`code/run_stage0_validation.sh UNIQUE MODEL` 为单个有界候锁/验证组：候锁600秒，获锁后 smoke600秒、codec图计时300秒。启动后必须跟踪原 driver，不重复提交。质量进程设置 CUDA13 `nvidia/cu13/lib` 动态库路径并关闭可选 FlashInfer sampler；所有质量臂共用设置。

数据输入原始文件在 runs/inputs_wikitext2_r01/；整理结果在 runs/quality_articles_r02/。固定版本与各文件 SHA256 见原始 download.json。读取 Parquet 的 pyarrow20.0.0 只安装到远端课题 .deps（不是共享环境）。

## 文章级质量采集

`evaluate_quality.py` 通过命名 worker RPC 运行，不需要开启 pickle RPC。`run_quality_corpus.sh` 持共享锁，最多等600秒、执行3600秒，保存源代码快照和原始参数。示例：
```bash
bash code/run_quality_corpus.sh UNIQUE --model "$PWD/models/llmjp" \
  --articles runs/quality_articles_r02/validation.jsonl --layers 16 --top-k 16 \
  --hidden 512 --context 512 --purpose calibration --gpu-memory-utilization 0.2
```
默认评价全部文章全文；`--start-doc` 仅用于按源文章索引恢复，必须用新的run目录。`--limit-docs`、`--max-article-tokens` 的任何限制都会写入manifest。每块1-token重叠且上下文重置；不能将该PPL称为无限上下文PPL。每臂原始logprob/KL写 metrics.jsonl，token记录写 windows.jsonl，完整文章汇总写 documents.json。只有完整文章才能纳入跨中断运行的合并结果。

当前校准：r01完成前4篇后外部中断；r02从索引4接续。测试集尚未运行。code/codec.py 的最新两级scale归约仍待GPU检查；已通过的图计时版本位于 runs/stage0_validation_r02/code.tar.gz。

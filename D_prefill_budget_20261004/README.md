# D：按近期完成耗时调整 prefill 预算

已完成单卡真实混合 serving 对照和论文。仅改变每步 aggregate prefill token 上限，保留原生 FCFS 顺序、准入/分配逻辑、decode 服务、模型精度和算子。没有改共享安装包。D 已于 2026-10-04 16:31 结束 GPU 实验并释放共享锁，后续会话已接手。

## 主要结果

- 共 57 次正式测量，完成 4,636 个请求、生成 1,974,016 个 token。所有请求完成且逐请求工作量校验通过，暖机不计入结果。
- 固定 2048 相对固定 512，开发负载吞吐提高 6.60%，但每请求最大生成间隔的 P95 从 19.10 ms 增至 28.69 ms。固定分块的吞吐—停顿权衡真实存在。
- 冻结的长文档确认集上，反馈规则相对开发阶段选定的固定 2048，吞吐和联合 goodput 均低 0.89%。背景短请求复用，因此不是完全独立的输入集合。
- 高压探索中，反馈相对所测固定配置中平均联合 goodput 最好的固定 384，goodput 高 4.28%，但吞吐低 8.15%、TTFT P95 高 20.06%、完成延迟 P95 高 9.52%。这是局部权衡，不是完整服务结果全面优胜。
- 正常 KV 容量下最高在途 160 个请求，峰值 KV 占用 88.31%，没有抢占。没有人为缩小缓存来制造收益。
- 开发负载中反馈 95.45% 的混合步骤落在 1024 档；固定 1024 和匹配均值的固定 1072 都已加入。高压也加入固定 256、匹配约 302-token 平均工作量的固定 304，以及固定 384/512/768/3840。
- 简单 decode-count 规则未充分调参；高压负载用于固定预算选择，尚无独立高压确认集。近期耗时相对强 count-only 控制器的独立价值仍未成立。因此没有扩展第二模型。

联合 SLO 在观察结果前固定：TTFT ≤ 4 s、每请求最大生成间隔 ≤ 100 ms、从计划到达到完成 ≤ 20 s，三者同时满足。goodput 分母包括整个 trace 的等待和排空时间。所有策略处理相同请求和固定长度输出；不评测自然 EOS 或任务质量。host 输出时间不包含网络传输。

## 文件入口

| 文件 | 内容 |
|---|---|
| `output/pdf/prefill_budget_study.pdf` | 完整论文，包含方法、对照表、步骤散点图、负结果、近邻和局限 |
| `paper.tex`, `results.tex`, `mechanism.tex` | 论文及图表源文件 |
| `prefill_policy.py`, `scheduler.patch` | 唯一调度干预和策略 |
| `run_normal.py` | 真实 vLLM 混合请求驱动、暖机、逐步观测、串行锁 |
| `analyze.py` | 从原始记录重算全部服务指标和工作量校验 |
| `build_report.py`, `campaign_summary.json` | 从 8 个正式阶段构建论文数值和汇总 |
| `reproduction/` | 可在同环境重跑的干净源码、冻结输入及实际执行命令 |
| `*/NN_policy/raw.json` | 每次运行的请求、输出 token 时间、步骤工作量与耗时 |
| `*/protocol.json`, `*/summary.json`, `*/summary.csv` | 实际协议和每次完整运行的统计结果 |
| `fixed-normal-r01/environment.json`, `engine_args.json`, `patch.json` | 软件、硬件、KV、引擎和补丁证据 |
| `remote_original_sha256.json`, `readback_verification.json` | 108 个远端原件的下载完整性校验，全部通过 |
| `model_verification.json`, `release_and_model_hashes.txt` | 模型分片核验及释放 GPU 后状态 |
| `normal-r01-controller.log` | 完整执行日志，包括暖机编译观测 |

`fixed-r01/` 是未执行请求的首次启动失败，不能作为测量结果。根目录曾有尚未被控制器读取的准备命令；复现请使用 `reproduction/normal-command-*.json`，这些文件以实际执行的 `protocol.command` 为准。例如高压主组实际执行 8 臂。

## 冻结实验与策略

运行环境为 RTX PRO 6000 Blackwell Server Edition（97,887 MiB）、驱动 595.71.05、vLLM 0.26.0、Torch 2.11.0+cu130、CUDA 13.0。模型为完整驻留 BF16 OLMoE-1B-7B-0924-Instruct，revision `7f1c97f440f06ce36705e4f2b843edb5925f4498`。运行时自动分配 KV 71.81 GiB，36,765 × 16-token blocks，其中一个保留。最大上下文 4096、序列槽 192、每步总 token 上限 4096、显存目标 90%。禁用 prefix caching、异步调度和 speculative decoding。

反馈规则初始为 1024，档位为 256/512/1024/2048。最近完成的混合 `engine.step()` 超过 24 ms 降一档，低于 16.8 ms 升一档，否则保持；没有活跃 decode 时用 2048。只用已完成的 host 执行包络，不添加设备同步。decode-count 规则为活跃 decode 少于 8 时用 2048，否则 512。所有固定配置在整个 trace 内保持同一常数，不使用分段事后 oracle。

冻结输入包含低压 12 请求、开发/长文档确认各 36 请求、高压 160 请求。长短请求分波到达；高压总 prompt 471,895 token、输出 77,824 token。`workload_*.json` 已包含完整 token IDs，无需下载数据才能复现调度输入。`prepare_inputs.py` 记录开发/确认输入构造；低压和高压以冻结 JSON 为准。

正式阶段依次为 `fixed-normal-r01`（4 次）、`tuning-normal-r01`（8）、`holdout-normal-r01`（11）、`high-normal-r01`（8）、`low-normal-r01`（6）、`extra-controls-dev`（6）、`extra-controls-high`（12）、`extra-controls-high384`（2）。控制器中的 `warm_*` 阶段仅用于暖机，额外固定档位均在正式计时前跑完整 trace。

## 重算本地结果

无需 GPU，使用 Python 标准库：

```sh
cd D_prefill_budget_20261004
python3 build_report.py
```

这会对 8 个正式阶段运行 `analyze.py --strict --exclude-warmup`，重建 summary 和论文数值段。单阶段可运行：

```sh
python3 analyze.py high-normal-r01 --strict --exclude-warmup
```

生成论文需要 LaTeX、PGFPlots 和 Poppler：

```sh
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
mkdir -p output/pdf
cp paper.pdf output/pdf/prefill_budget_study.pdf
```

## 在原环境重跑

将 `reproduction/` 复制到服务器一个新的空实验目录，保留原环境和模型目录 `/root/autodl-tmp/moe-research-20261002/model`。不要在已有输出目录上执行；驱动刻意拒绝覆盖。GPU UUID、模型路径和共享锁路径位于驱动中，其他机器上应显式修改并记录环境差异。无需再次创建基础设施或下载另一模型。

在复制后的目录运行：

```sh
/root/miniconda3/bin/python -u run_normal.py \
  --output fixed-normal-r01 \
  --workload workload_dev.json \
  --policies fixed512,fixed2048,fixed2048,fixed512 \
  --command-loop
```

驱动先非阻塞获取 `/root/autodl-tmp/moe-research-gpu.lock` 并检查无其他 GPU 进程，然后初始化同一个引擎，暖机所有首版档位，顺序执行冻结命令，最终 `normal-command-10.json` 退出释放 GPU。锁忙时返回 75 且不初始化 GPU；可显式提供 `--wait-lock 1800` 等候。不要同时启动其他整卡计时任务。`runtime_bootstrap.py` 只对该进程修复此环境 Torch 升级遗留的三个模块导入，不改安装文件。复现包不含密码和权重。

## 必须保留的结果边界

补充固定 512 的一次 108.97 ms 输出间隔导致 67 个 gap-SLO 失败，goodput 降低；记录完整保留，原因未确定。日志仅在初始 512 暖机中出现一次被监测到的 Triton JIT 告警，正式阶段没有该告警，这不能排除其他抖动。固定 3840 高压 goodput 为零是 160 个请求全部超过 20 s 完成限制，并非 gap 失败。

高压主组的 goodput 增益来自早期长请求跨过联合阈值，后续请求仍受损；不能靠一个 goodput 数值宣称全面改善。不同 batch shape 的贪心输出部分不同，但输出数量相同；没有精度变化或质量优势的证据。每臂 2–3 次重复支持本机复现性，不构成跨工作负载统计保证。

近邻核验覆盖 Sarathi-Serve、官方调度实现、QoServe 和最接近的 DLFP。动态分块、上下文感知与耗时反馈均不作为新颖性主张；准确来源见论文参考文献。

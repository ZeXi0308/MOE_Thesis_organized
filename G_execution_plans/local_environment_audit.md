# G 本地环境与复用材料审计

2026-10-08；只读现有 B/D/E/F 顶层文档、入口脚本及少量明确相关证据。未 SSH，未取得 GPU 锁，未初始化 CUDA。本文件的硬件/环境数据均为历史记录；G 必须以当前远端实查为准。未记录任何密码。

## 最有决定性的已知事实

- D 的 BF16 OLMoE/vLLM 已在正常显存目标 0.9 下使用 graph，原生运行时在 KV 分配前估计 CUDA graph 内存。43 个 PIECEWISE + 27 个 FULL graph 的总池实占日志为 **0.17 GiB**、估计 **0.14 GiB**，不是单 graph 峰值相加。capture 约 6 秒。来源：`../D_prefill_budget_20261004/normal-r01-controller.log:59–72`。
- D 高压固定 2048 的已完成轨迹峰值 **32031 / 36764 可用 KV blocks**，无抢占；每块 2 MiB，峰值尚有约 **9.24 GiB** 余量。因此该旧配置下，把全部 graph 池都回收的乐观上界仍不足以耗尽该轨迹的容量余量。它是 G 的门槛 C 强烈负面先验，**不等于当前其他合法 kernel/graph 组合已被测量**。来源：`../F_output_isolation/trace_audit.md`，以及 D 原始环境和 engine 参数。
- D 的已用实际 MoE 是 `TRITON Unquantized / TritonExperts / MoEPrepareAndFinalizeNoDPEPModular`。日志的“potential backends”包含 FlashInfer TRTLLM/CUTLASS，不是它们已通过当前设备兼容性与正确性验证的证据。D 当时缺 `E=64,N=1024,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.json`，使用默认配置。来源：D 控制器日志 38、48–49、57 行。

## 当前入口的历史记录与共享锁

最新本地 `../E_restore_choice_20261004/checkpoint.json`（18:49 更新）记录：

| 项目 | 值与限制 |
|---|---|
| SSH 入口 | `root@connect.westd.seetacloud.com:53005`；与本次用户提供入口一致 |
| 最近记录 GPU UUID | `GPU-51b8e4bb-27b8-4b82-5254-7317aae7298c`；不可沿用 D 的旧 UUID |
| 公共锁 | `/root/autodl-tmp/moe-research-gpu.lock` |
| E 代码 | `/root/autodl-tmp/moe-e-restore-choice-20261004` |
| D 代码 | `/root/autodl-tmp/moe-d-prefill-20261004`，见 D 控制器日志的 command 路径 |
| 共享模型 | `/root/autodl-tmp/moe-research-20261002/model` |
| Python | `/root/miniconda3/bin/python`；历史 site-packages 为 Python 3.12 |
| 最近锁状态 | 18:33 历史记录 E runner/waiter 消失、锁释放；**不代表目前空闲** |

已存在的合法协议是整组实验持一个排他 `fcntl.flock`，锁前不初始化 CUDA；默认非阻塞，忙则退出；锁后重新查 `nvidia-smi --query-compute-apps` 和 GPU UUID，foreign GPU process 或查询失败则停止。E 还容许前任上下文最多 60 秒自然排空。不得终止他人进程。来源：`../D_prefill_budget_20261004/run_normal.py:127–144`、`../E_restore_choice_20261004/run_group.py:58–109`。E 的旧 RUN.txt 仍含旧入口/旧 UUID，已被 checkpoint supersede，不能当当前配置。

E 最近记录系统盘 `/tmp` 曾只剩约 2.0 GiB，数据盘也接近满；应先只读 `df` 再选 G 输出位置，不直接复制大档案或改他人目录。来源：`../E_restore_choice_20261004/RESULTS.md:176`。

## 模型、运行时与可复用启动修复

历史 D 已验证模型为 `allenai/OLMoE-1B-7B-0924-Instruct`，revision `7f1c97f440f06ce36705e4f2b843edb5925f4498`，三权重分片 SHA 完整核验，BF16 完整驻留。模型 16 层、hidden 2048、16 KV heads/attention heads、上下文 4096；按当前 native KV 布局，每 token 131072 字节，每 16-token block 2 MiB。来源：`../D_prefill_budget_20261004/model_verification.json`、`../E_restore_choice_20261004/pro6000_runtime_audit.txt`、`../E_restore_choice_20261004/multinews320_design.txt`。

历史软件：vLLM 0.26.0、PyTorch 2.11.0+cu130、CUDA 13.0；RTX PRO 6000 Blackwell Server Edition 97887 MiB、driver 595.71.05。D 的旧 GPU UUID 是 `GPU-bf3fc5ab-804d-b9d6-759b-4390899f15b9`，E 的更早旧卡为 `GPU-94203fc3-1021-3a9c-a367-cff792479616`。来源：`../D_prefill_budget_20261004/fixed-normal-r01/environment.json`。当前架构、版本、可用 backend 仍需实查。

共享 Torch 安装有升级遗留文件。最小进程内补丁 `../D_prefill_budget_20261004/runtime_bootstrap.py` 按 distribution RECORD 过滤旧 inductor modules（当时是 flex_attention.py、flex_decoding.py、mm_scaled_grouped.py），不改共享安装。更彻底的现成 `../E_restore_choice_20261004/prepare_env.py` 建私有 RECORD-only 软链 `package_view`，E 用 `PYTHONPATH=.../package_view`。两者择一核对，不自行“修复”共享 site-packages。

E 启动环境还固定 `VLLM_ENABLE_V1_MULTIPROCESSING=0`、`VLLM_USE_FLASHINFER_SAMPLER=0`、离线 HF/Transformers、`TOKENIZERS_PARALLELISM=false`、`WISP_PLUGIN_DISABLE=1`，并设置 cu13/torch 动态库路径。见 `run_group.py:77–83`。D 从旧进程内修复启动即可；不要把两套启动参数盲目叠加。

## 可复用的冻结请求与 SLO

以下 D 文件已包含完整输入 token IDs，无需下载数据。表中 token 数为本次 CPU 直接读取 JSON 的计数。

| 输入 | 请求数 | 到达范围 (s) | prompt tokens | 固定输出 tokens | 适用性 |
|---|---:|---:|---:|---:|---|
| `../D_prefill_budget_20261004/workload_low.json` | 12 | 0–5.01 | 30811 | 2816 | 可预定为低压力 |
| `../D_prefill_budget_20261004/workload_dev.json` | 36 | 0–5.28 | 92625 | 8448 | 开发/中压力，不必增加第三负载 |
| `../D_prefill_budget_20261004/workload_high.json` | 160 | 0–5.41 | 471895 | 77824 | 旧 trace 最高 KV 约 87–88%，未形成抢占压力 |

D 在观察结果前固定联合 SLO：TTFT ≤ 4 秒、每请求最大生成间隔 ≤ 100 ms、计划到达到完成 ≤ 20 秒，三者同时满足；goodput 分母覆盖全轨迹等待与排空。`ignore_eos=True`、`min_tokens=max_tokens`，用于固定输出工作量；不是自然 EOS 任务质量评估。时间是 host `engine.step()` 完成，不含 HTTP 网络。来源：`../D_prefill_budget_20261004/README.md`、`run_normal.py:73–116,164–166`。

G 若沿用 D SLO，必须事前声明是复用的实验 SLO，而非生产 SLA。高压已有许多请求超过完成阈值；完整报告吞吐、延迟和失败原因，不能因 G 结果再调阈值。

D 可复用的是请求驱动、输出记录与分析口径；`prefill_policy.install()` 是 D 独立研究干预，G 不应未经说明照搬。旧脚本配置为 maxseq=192、batch token=4096、maxlen=4096、FCFS、完整序列容量保留、chunked prefill、禁 prefix caching/async scheduling/speculation、auto GPU 0.9、graph 开启。G 各执行组合必须保持同一调度/准入/恢复配置，仅改变合法执行计划。

本轮 G **只冻结 D low/high 两条输入**，不采用 E320，也不因已知容量余量而更换/扩展 workload 寻找收益。high 是合理的既有服务到达压力，尚不是已证明能暴露 graph→KV 容量效应的压力场景；允许这一门槛失败。

CPU 提取脚本 `extract_historical_feasibility.py` 已从 D 日志、environment、low/high protocol 和首个固定 2048 raw 自动生成 `evidence/historical_feasibility.json`，保存来源 SHA-256、原 GPU UUID、graph 池实测/估计、总/峰值 KV、请求长度/到达与 SLO。实际 low 峰值为 **914 / 36764**，high 为 **32031 / 36764**；均 0 次抢占。只复制原始冻结输入 `inputs/workload_low.json` 与 `inputs/workload_high.json`，复制内容与 D 协议 SHA 一致。

复现提取（无需 GPU）：`python3 G_execution_plans/extract_historical_feasibility.py`。SLO 固定沿用 TTFT 4 秒 / 最大 gap 100 ms / 到达到完成 20 秒。JSON 清楚区分 D 历史原卡与 E 最近记录的当前入口 UUID；本提取器不验证远端状态。

## 后续最小实查

1. 当前 SSH 只读确认 GPU、锁 owner、模型、安装路径、磁盘，确认现有 backend 的精度/架构条件。
2. 如 GPU 可用，锁内新进程运行极小同 tensor/同 route 两配置正确性与时延探针；无实际候选就停止。
3. 仅当前两门槛有正信号时，测整个 startup 计划集合的实际 graph/workspace/allocator/peak/KV blocks。不要把 D 的 0.17 GiB 或 E/D 跨引擎 KV 差直接归为计划代价。

本次未复制/改变既有结果，未启动新的负载、模型或 GPU 实验。

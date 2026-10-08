# F CPU 回放：既有真实输出轨迹审计

审计日期：2026-10-08。只读检查既有 B / D / E / C 成果，未使用 GPU、未启动实验、未修改其他方向。按主任务决定，仅冻结以下两条 D 轨迹；不继续扩大场景。

结论：两条轨迹足以保留**真实引擎已达到的逐批产出时序、批次突发、token IDs、请求到达与实验输出长度分布**。不能据此宣称真实在线 API 的端到端收益或自然 EOS 长度分布。回放不缩放时间、不增加并发、不插值 token 时刻。

| 轨迹 | 中压 | 高压 |
|---|---:|---:|
| 精确路径 | `D_prefill_budget_20261004/fixed-normal-r01/02_fixed2048/raw.json` | `D_prefill_budget_20261004/high-normal-r01/00_fixed2048/raw.json` |
| SHA-256 | `4428401279ebf741f8c73d9feb738008e61ca8ac2d3d0b1aff255e5738ed8bcc` | `0d788741fd813648b26670deed16e919949ac28cf5ab235a92abf9f2336cc176` |
| 请求 / 完整输出 tokens | 36 / 8,448 | 160 / 77,824 |
| episode 时长（秒） | 7.645201996 | 34.814416314 |
| 首 / 末批输出时间（秒） | 0.023902349 / 7.645162208 | 0.026591551 / 34.814278179 |
| 产出批次数 | 627 | 781 |
| 每批 token 数 min / p50 / p95 / max | 2 / 12 / 20 / 25 | 1 / 113 / 160 / 160 |
| 相邻产出批间隔 p50 / p95 / max（毫秒） | 12.167 / 24.782 / 28.578 | 53.313 / 60.668 / 72.628 |
| tokens / episode 秒 | 1,105.007 | 2,235.396 |
| 每请求输出长度（tokens: 请求数） | 160: 24；384: 12 | 384: 32；512: 128 |
| 最后请求到达（秒） | 5.28 | 5.41 |
| 峰值 KV 占用 / 可用 blocks | 4,069 / 36,764 | 32,031 / 36,764 |
| 抢占次数 | 0 | 0 |

分位数使用排序后的最近索引 `round((n-1)*q)`，以上速率仅用于描述；执行器必须使用原始逐批时间，不用平均速率替换。

## 结构与验证

两个 `raw.json` 顶层均为 `policy, elapsed_s, requests, steps`。每个 `requests` 项保留 `request_id, arrival_s, prompt_token_ids, max_tokens, output_token_ids, token_times_s, completion_s, finished, finish_reason`。每个 `steps` 项保留 `start_s, end_s, requests, preempted, kv_used_blocks, kv_total_blocks` 等。

重建输出批次：对每个请求逐一配对 `token_times_s[i]` 与 `output_token_ids[i]`；把完全相同的时间值归入同一批。时间由 episode 开始处的 `time.perf_counter()` 原点定义；不要为每个请求减去首 token 时间。请求到达应保留 `arrival_s`。同批请求输出顺序没有独立记录：列表请求顺序可作为确定性的重放顺序，但不能冒充真实 `engine.step()` 返回列表顺序。

对两条完整轨迹做过下列断言，全部通过：

- 每个请求 `len(output_token_ids) == len(token_times_s) == max_tokens`；全部完成且 `finish_reason == 'length'`。
- 每请求 token 时间单调、`completion_s == token_times_s[-1]`。
- 所有 token 时间精确等于某一 `steps[].end_s`；每请求每批恰好一个 token；所有 step 都有输出。
- 所有 step 的 `preempted` 为空。两组目录的 `status.json` 都为 `COMPLETE`。

## 来源、运行时和边界

采集实现：`D_prefill_budget_20261004/run_normal.py`。当前文件 SHA-256 为 `9ec455c78ed9ccc76236fccf38b84dcc3dff53540b6df608979c902bd75d3171`，与两组 `protocol.json` 的采集源哈希一致。时间记录发生于同步 `engine.step()` 返回之后：包含引擎、scheduler 和既有 output handling，不是 GPU kernel 完成时间；未额外 `cuda.synchronize()`。`detokenize=False`、累计输出、`stream_interval=1`、`async_scheduling=False`、`enable_multiprocessing=False`。

两条是 D 的 `fixed2048` arm：`prefill_policy.py` 在原生 FCFS 上增加固定 aggregate prompt token cap=2048；总 `max_num_batched_tokens=4096`。它们是实际 GPU 产生的轨迹，但不是未经修改的默认 scheduler / API frontend。采集器还记录每步 scheduler 信息和累积 token 列表，其 CPU 观测成本已进入后续产出间隔；不能把该轨迹当成零观测开销的性能上界。

`fixed-normal-r01/environment.json` 和 `engine_args.json` 记录：vLLM **0.26.0**、Torch **2.11.0+cu130**、CUDA **13.0**；NVIDIA RTX PRO 6000 Blackwell Server Edition，UUID `GPU-bf3fc5ab-804d-b9d6-759b-4390899f15b9`，97887 MiB，driver 595.71.05。模型路径 `/root/autodl-tmp/moe-research-20261002/model`；仅据此文件不认定模型名称。高压是同一 `run_normal.py` command loop 的后续 phase，其 `protocol.json.command`、相同源哈希和采集代码提供运行链证据；高压目录没有独立 `environment.json`。这些是**历史运行证据**，不代表当前远端 GPU/环境状态。

固定 `min_tokens=max_tokens`、`ignore_eos=True` 是原始实验合同。当前输出长度完整而真实可达，但不代表自然 EOS 或生产分布。轨迹不含 logprobs 概率、排名、备选 token，也不含 HTTP/SSE 前端、序列化、网络或客户端收包时间。CPU 回放若构造 logprobs，必须披露这是固定真实 token/time 上的 API 成本输入构造；不应把它写成真实重响应生产 trace。若请求 top-k logprobs 会改变 GPU 产出速度，其影响仍待真实闭环确认。

## GPU 锁与其他检索结果

公共 GPU 锁：`/root/autodl-tmp/moe-research-gpu.lock`。`D_prefill_budget_20261004/run_normal.py` 用非阻塞 `fcntl.flock(..., LOCK_EX|LOCK_NB)`，在取得锁及检查 GPU 占用后才初始化 CUDA。`B_recovery_order_20261004/output_event_compact/plan-20261008-r01.json` 也记录同一锁路径。本次不检查/获取远端锁，不推断现在空闲。

B 的 `pkg/request_measurement.py` 及 `output_event_compact` 保留更直接的 `engine_call_index, received_s, new_token_ids` 输出事件，但当前冻结 D 的正常容量轨迹，避免把恢复/小容量机制干预混入 F。E 的 `execution/oct07_multinews_r01/00_same_engine/00_host/raw.json` 确实含变长 EOS 输出 token/time，但有恢复实验干预，未选为第三种负载。C 的最小查找未发现比选定 D 更合适的已展开逐批原始文件；没有全量解包档案。

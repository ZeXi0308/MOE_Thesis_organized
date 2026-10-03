# 恢复服务窗口：近邻与强基线核对

2026-09-14；`SOURCE_VERIFIED / EXISTING_RESULTS_REUSED`，新增 GPU 运行 0。独立 worktree `/private/tmp/moe-window-neighbors-20260914`，HEAD `de64dae5ea4fa3c92ec605e9847e817d8e68f3ac`。共享工作区存在更晚未提交结果；本报告读取实际 addendum，不把准备期 UNRUN 当当前状态。没有修改共享代码、台账或原始实验。

结论：**最小有效服务保护、等待提权、恢复成本和其它请求损害都已有直接近邻。当前最窄待检验差异是：在实际队列与增量 KV 约束下，恢复资格是否对应可兑现的输出窗口，且将 KV 保留与执行预算分开后是否仍有强简单规则未覆盖的完整请求收益。**这是实验问题，尚非新颖性结论。

## 原始来源与动作级对照

先读已有[组件报告](/private/tmp/moe-a-recovery-components-20260914/refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_components_r01/REPORT.md)，再重读四项原文；LTR、Andes、TokenFlow 固定官方源码本轮重新下载到 `sources/`，来源、字节及散列见 [retrieval.json](sources/retrieval.json)。GitHub 网页提取失败后通过只读 HTTPS 取得源码；未执行这些外部脚本。

| 工作 | 状态/信号 | 动作与服务边界 | 成本、资源与其它请求 | 目标/运行域 | 本轮必须如何比较 |
|---|---|---|---|---|---|
| LTR | 预测完成排名；未执行迭代计数 | 200 次等待提权，10 次实际调度量子；实际调度便清 idle，重算也消耗量子 | 官方排名路径有 CPU SWAP、块预留、低优先级 victim；预算不足停止选取 | 请求平均延迟、吞吐及每请求 max(TTFT,maxITL) | 现有 200/10 和 packing 是组件对照。输出等待修正单独消融，不能称首创公平机制。 |
| Andes | 消费缓冲、QoE gain、上下文、batch latency | 排序/装箱决定保留、恢复和 victim；开销 refiner 取消净 QoE 不划算的切换 | 论文 profile swap/recompute，比较被恢复者收益与全部 ongoing 请求损害 | 有消费速率目标的流式 QoE；持续/突发到达 | 同预算、同恢复能力的成本感知保留/切换规则是强基线；不得只用 target 恢复费描述其动作。 |
| UniBoost/MemGuard | attained decode 输出量、到达和在线分布统计 | `Q(w)=k·2^floor(log2(max(w,k)/k))`；dispatch 后至下一门槛 `2Q(w)` 不驱逐；默认 k=256 | 算法含下一 chunk/step KV 检查和 priority/swap-cost victim；主文明确 useful-service 摊销 | TTFT/TTLT 尾部；未知 decode 长度、动态 batching | 几何服务保护必须作为已知规则。第一门槛为 512；不能把 recompute positions 计作 attained decode 服务。 |
| TokenFlow | 虚拟消费缓冲、输出速率、等待、GPU/host 命中、写入/加载积压 | 工作集选择、预抢占/恢复；主动 write-through、分块搬运与执行重叠 | 论文明确 eviction queue + eviction + load queue + load，并与重算比较；实现将 host 命中加载与未缓存部分重算共同计费 | 突发和持续到达的 streaming QoS/有效吞吐 | 必须保留其主机 KV 能力；共同 recompute 移植仅比较组件。排队/缓存位置进入模型也不是新颖性。 |

LTR 来源：[NeurIPS 原文 §4.3](https://papers.neurips.cc/paper_files/paper/2024/file/6c8985579293e0209bdaa4f21bb1d237-Paper-Conference.pdf)、[官方 scheduler 固定版](https://github.com/hao-ai-lab/vllm-ltr/blob/13bbf6ff3dab661791d41362551b089e5f77c91c/vllm/core/scheduler.py#L969)、[200/10 配置](https://github.com/hao-ai-lab/vllm-ltr/blob/13bbf6ff3dab661791d41362551b089e5f77c91c/benchmarks/fair-lmsys-70B.sh#L30)。源码 1360 附近按 `scheduled_seq_groups` 更新 idle/runs，1400 附近直接指定 SWAP。排名、CPU SWAP 和本仓库 FCFS/recompute 不等价。

Andes 来源：[原文 §4.2–4.3](https://arxiv.org/html/2404.16283v2)、[官方 scheduler](https://github.com/AmberLJC/vllm-0.6.1/blob/3cc5c8458bd2a2744b8392ab6aa311c645105655/vllm/core/andes_scheduler.py#L87)、[官方 solver](https://github.com/AmberLJC/vllm-0.6.1/blob/3cc5c8458bd2a2744b8392ab6aa311c645105655/vllm/core/andes_utils/knapsack_solver.py#L24)。该版 scheduler 默认 RECOMPUTE、每 10 次调用检查压力；solver 使用固定 unit overhead/latency 和 0.9 利用率阈值，其参数化简化实现不能冒称论文完整 refiner。其消费端缓冲与本仓库引擎返回 ITL 也不是同一边界。

UniBoost 来源：[原文 §3.3/附录 A.1](https://arxiv.org/html/2606.18431v1)、[作者页](https://yl3469.github.io/uniboost-icml26/)。本轮作者页 Code 链接仍为 `href="#"`，完整官方实现未取得。主文明确 decode-token 门槛；附录对统一 service/prefill 的记号有更宽表述，故移植时必须写清绑定到哪次 residency、EOS 提前释放以及无 eligible victim 的回退，不能补造未展示源码。论文自己说明动态 batch 速率违反 M/G/1 假设；理论结论不能改写为本系统最大 ITL 保证。

TokenFlow 来源：[原文 §4.2.3–5](https://arxiv.org/html/2510.02758v1)、[官方项目](https://github.com/SJTU-RTEAS/TokenFlow)、[固定策略源码](https://github.com/SJTU-RTEAS/TokenFlow/blob/2f2ec646a0ab495df9262d961c5ad30743bf49b4/python/sglang/srt/managers/schedule_policy.py#L282)。`_estimate_terms` 实际读取 host/GPU hit、backlog/speed、buffer 与 wait tick；使用加权 proxy 而非逐字实现原文效用式。相加的策略 score 不等于可相加的暴露墙钟时间，不能拿它替代本仓库互斥计时。

## 已完成 native offload：复用，不重复

实际安装证据为 vLLM 0.26.0、PyTorch 2.11.0+cu130。四格原生 FCFS 均 32 请求/32768 输出完成；KV 为 6656 usable blocks，APC off，`kv_offloading_size=16`、backend=native。connector 元数据确认为 OffloadingConnector，额外配置只有 `cpu_bytes_to_use=17179869184`。安装 `native-offload-base.py:510–512` 默认 `offload_prompt_only=True`；安装 scheduler `_get_max_offload_tokens` 再按 prompt 长度截断。**本组是默认 prompt offload，不是全历史 decode KV 保存。**[实际结果与配置](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_native_offload_baseline_r01/REPORT.md)

| 已有实测 | 直接含义 | 尚不能推导 |
|---|---|---|
| 两个 on 各 store 12 GiB、load 2.25 GiB；重算 21426→2994 位置，少 18432 | 外部 KV 恢复真实替代部分重算；与加载字节对应 | lookup 重复命中总数不是独立复用 token；低复制字节不是服务收益 |
| on/off wall +23.55%/+5.17%；平均完成 +23.83%/+6.73%；maxITL +27.79%/−2.10% | 当前默认 backend 没有净效率收益，全部慢格保留 | 不能以第二 on 替换第一 on，或扩写为 offload 家族 NO-GO |
| 同逻辑共同前 1026 步，额外 engine 时间 3.258/1.594 秒 | 外部恢复改变调度之前，持续保存/查询/布局等已有执行税 | 不能据逻辑签名相同声称物理布局相同 |
| 有限 profiling 四格和 decode trace 两格全部完成 | connector 可见项约 1.1–1.2 秒；固定 32-call trace GPU 活动并集约相同，主机区间不同 | 不能把 residual 全叫 GPU 时间/硬件空闲；不能直接减去 profiler span 宣称净收益 |

来源：[前缀定位](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_native_offload_baseline_r01/LOCALIZATION_ADDENDUM.md)、[有限成本四格](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_native_offload_cost_r01/REPORT.md)、[decode trace](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_offload_decode_trace_r01/REPORT.md)。16 GiB 是配置容量，实际 host 驻留/峰值与全部元数据仍未完整计费。`offload_prompt_only=False` 仅是安装源码存在的配置能力，full-decode 与 rotation 兼容仍未实测；不解除 connector guard 就宣称已兼容。

## 恢复窗口残差必须基于更新后的结果

prefix/fit 四格已完成，不能重跑：prefix 的短服务段 11→0，却把零输出重抢占 2→6、全局 maxITL 约 4.8–4.9→6.6–6.7 秒。[实际结果](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_ltr_packing_r01/RESULTS_ADDENDUM.md)

更晚的首次输出义务四格也已完成：每个 on 的 27 次义务全以新输出释放，零输出中断归零；但 4 次恢复仅产生 1 token 就再抢占，分别重算 3273/3382/3498/3614 个位置。重算 74982→96955，27/32 请求 maxITL 变差，wall/平均完成两重复变号。[实际结果](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_restore_completion_r01/RESULTS_ADDENDUM.md)

因此缺口已经从“有没有首输出”推进到“首输出后的服务量是否足以兑现恢复的机会成本，以及为其保留 KV 是否不必要地挤掉其它可共批工作”。原有 200/10、简单 first-output guard 均未充分解决；这仍只是在本仓库接入下的现象，**完整 Andes/MemGuard/TokenFlow 还未被实测排除**。

最低比较顺序为：同一恢复底座上的最强既有 fit/native/most/least → 只按新输出重置等待 → 只保护首新输出 → 固定小服务量或 MemGuard 几何保护组件 → 仅在以上仍有残差时比较窗口可行性模型。一次实验只选其中有新增辨别力的少量臂，不同时实现全部。未实际支持逐请求选择恢复方式时固定 backend，各 backend 分开验证。

当前已存在 [KV 保留与执行预算分离准备合同](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_restore_token_reservation_r01/REPORT.md)。其最小干预是给仍可执行的 resident pending=1 工作留本步 token 预算，验证首输出保护是否必须阻断它们；这属于 decode 优先/chunking 的接入修正，不能重新命名为论文贡献。本会话不重复实现或另占 GPU。

## 本轮裁决

| 字段 | 值 |
|---|---|
| Verdict | 主问题 OPEN；保护窗口本身无独立新颖性证据 |
| Evidence type | 原文/固定官方源码核对；已有 NATIVE_SERVING 原件复用 |
| What was measured | 本轮无新增 GPU；恢复生命周期、native prompt-offload 实测状态与配置由现存原件确认 |
| What was not measured | 完整近邻的同预算原生对照、全历史 offload、实际 host 峰值、第二 workload/模型的窗口收益 |
| Strongest baseline | 当前域的 fit_scan 与原生简单策略；MemGuard/Andes/TokenFlow 的重合规则必须保留组件对照；native offload 能力不可忽略 |
| Oracle/headroom status | 没有全请求 Oracle；复制微基准和有限动作模型都不是全局上界 |
| Claim ceiling | 当前适配器存在“首输出完成性不等于足够服务”的请求级现象；未证明近邻算法共同缺陷 |
| Failure category | 部分接入规则的服务摊销/损害转移；默认 offload 执行税待定位 |
| Resurrection condition | 本问题无需复活；模型只有产生简单规则之外的执行前决策增量，且真实改善权衡，才进入方法验证 |
| One next smallest experiment | 复用已有 KV/执行预算分离合同；判别当前税是否来自可避免的共批阻断。若强 fit 仍覆盖收益，保留简单实现；若仍反复一 token 恢复，再比较窗口长度。 |

直接回答：已有证据支持继续检验“昂贵恢复应得到足够服务”，但尚不支持新的保护机制优于现有规则；下一次应关闭当前具体的共批执行缺口，不能以新增 host KV 或恢复计数修正冒充算法收益。

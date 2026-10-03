# Recovery quantum：结果与复现入口

截至 2026-10-02，本轮预选运行全部完成，A组已无运行中 GPU 作业。原生实现通过动作核对，但 **adaptive 相对 fixed4 的收益方向随运行改变，尚无稳定优势**；两个健康 KV512 域均未触发恢复量子。当前结论保留该混合/未支持结果，不自动追加调参或运行。下列组分别保留，不把恢复事件当作独立重复，也不把不同输出长度视为等工作量加速。本文档是当前索引；冻结包内的早期状态说明、原计划及失败记录作为历史证据保留。

## 已有结果

| 运行与状态 | 核心观测 | 结果文件 |
|---|---|---|
| `strong-session-r01`，COMPLETE；ordinary → queue_fund → native | 每臂128/128完成。fund 相对 ordinary 的请求最大输出间隔 P95 降59.46%，实际输出率降1.52%，平均 flow 增1.85%；存在个体损失和输出差异。 | [原始判定](output/STRONG_NEWHOST_RESULT_R01_20261002.json)、[完整请求解释](output/STRONG_NEWHOST_INTERPRETATION_R01_20261002.json)、[成本](output/STRONG_NEWHOST_RUNTIME_COSTS_R01_20261002.json) |
| `lease-session-r01`，COMPLETE；q1 → fixed4 → adaptive | 每臂128/128完成。adaptive 相对 fixed4：P95 间隔增14.44%，输出率降1.00%，平均 flow 增0.35%。 | [对 q1](output/LEASE_SERVICE_VS_Q1_R01_20261002.json)、[对 fixed4](output/LEASE_SERVICE_VS_FIXED4_R01_20261002.json)、[成本](output/LEASE_RUNTIME_COSTS_R01_20261002.json) |
| `lease-session-r02`，COMPLETE；adaptive → fixed4 → q1 | 反序重复，每臂128/128完成。adaptive 相对 fixed4：P95 间隔降10.80%，输出率增0.74%，平均 flow 增1.62%。与 R01 分开解释。 | [对 q1](output/LEASE_SERVICE_VS_Q1_R02_20261002.json)、[对 fixed4](output/LEASE_SERVICE_VS_FIXED4_R02_20261002.json)、[成本](output/LEASE_RUNTIME_COSTS_R02_20261002.json) |
| `session-dev16-low-r02`，COMPLETE；Instruct / KV4096，q1、fixed4、adaptive | 每臂16/16自然停止，严格正确7/16，零抢占、零恢复动作；aggregate observer 未启用。 | [质量与服务](output/healthy_dev16_low_r02_quality_service.json)、[成本](output/DEV16_LOW_RUNTIME_COSTS_R02_20261002.json) |
| `session-dev16-kv512-comparison-cost-r02`，COMPLETE；native、ordinary、q1、fixed4、adaptive | 每臂16/16自然停止，严格正确8/16，1次原生抢占、零恢复动作。五臂都启用同一 aggregate observer。容量压力实际存在，但没有触发1秒恢复条件。 | [质量与服务](output/HEALTHY_DEV16_KV512_COST_R02_QUALITY_SERVICE.json)、[成本](output/HEALTHY_DEV16_KV512_RUNTIME_COSTS_R02_20261002.json) |
| `session-holdout16-steady-kv512-comparison-cost-r02`，COMPLETE；预选稳态16题、KV512、同一成本五臂 | 按 native / ordinary / q1 / fixed4 / adaptive 顺序：严格正确10/8/8/9/8，每臂16/16自然停止；原生抢占2/2/2/2/1，全部零恢复动作。最大输出间隔均低于0.575秒。 | [质量与服务](output/HEALTHY_HOLDOUT16_STEADY_KV512_COST_R02_QUALITY_SERVICE.json)、[成本](output/HEALTHY_HOLDOUT16_STEADY_KV512_RUNTIME_COSTS_R02_20261002.json)、[冻结计划](HEALTHY_HOLDOUT16_STEADY_KV512_COMPARISON_COST_PLAN_R02_20261002.json) |

所有 article lease 动作核对通过：[R01 adaptive](output/LEASE_ACTIONS_ADAPTIVE_R01_20261002.json)、[R01 fixed4](output/LEASE_ACTIONS_FIXED4_R01_20261002.json)、[R01 q1](output/LEASE_ACTIONS_Q1_R01_20261002.json)；[R02 adaptive](output/LEASE_ACTIONS_ADAPTIVE_R02_20261002.json)、[R02 fixed4](output/LEASE_ACTIONS_FIXED4_R02_20261002.json)、[R02 q1](output/LEASE_ACTIONS_Q1_R02_20261002.json)。保护期间没有目标实际被抢占，已接纳 peer 和被迫腾让的 victim 均随后输出并完成。adaptive 已知成本时的 desired q 在 R01 的103/103次、R02 的75/75次都为3；前沿裁剪后的实际 q 可为1或2。此处没有展示出广泛自适应性，动作合法也不等于收益成立。

健康任务使用独立的 OLMoE Instruct 模型、自然 EOS 和严格数值答案评分；article 使用 base 模型且大多触及输出上限，不能合并质量或绝对时间。健康任务的分数/输出变化是不同执行轨迹的观测；零量子动作不能解释为量子带来的质量或速度收益，也不证明质量等价。heldout 与 A 开发题号分离，但邻近研究使用过其中部分题目，不声称全局未见或严格盲测。KV768 旧准备计划、低压 native/ordinary 两臂补充计划均未执行；不自动追加运行。

## 实现及证据边界

| 冻结候选 | manifest 内文件数 | 使用范围 |
|---|---:|---|
| `../candidate_native_oldest_strong_r01` | 26 | strong 的 native / ordinary / queue_fund |
| `candidate_native_recovery_lease_r01` | 27 | article R01/R02，同资源路径的 q1 / fixed4 / adaptive |
| `candidate_native_recovery_lease_healthy_r01` | 27 | 自然结束 Instruct 适配，已完成低压三臂 |
| `candidate_native_recovery_lease_healthy_cost_r02` | 28 | healthy 五臂成本测量；仅新增 aggregate observer、runner 接线及 manifest verifier，策略和输入载荷字节不变 |

每包的 `manifest.json`、`verify_package.py` 及运行计划定义身份；healthy 两包另有 `PROVENANCE.json`。成本包 manifest SHA256 为 `1cccd7f0865fd7104a98a2a42e9a1c0b841a44733b9bb9ac84773c30f7f34f34`；[成本集成检查](output/HEALTHY_COST_INTEGRATION_R02_20261002.json)通过。控制器沿用历史 cell 子目录名，实际包身份以 receipt、manifest、归档 config/source hash 为准。

机制在恢复 funding 前及 commit 时共同选择 q 与完整 KV 增长需求，为目标保留运行槽位和执行 token；仅以剩余资源接纳其他等待请求。q 上限16，目标 overhead fraction 0.5，peer 年龄前沿为 max(1秒, target commit age)，未知成本用 Q1。成本估计只用历史 protection-start（早于原生接纳）至首输出间隔减去一个历史纯 decode 间隔，不把重叠 DMA 相加。`recovery_lease.py` 的 CPU 模型在 Q1 物理资金不足时返回 q=0 / `Q1_PHYSICALLY_UNFUNDED`；进展结论以实际 funding、原生 KV/传输完成和执行条件成立为前提。peer 已逾期时保留 Q1 不能证明 peer 年龄上界，更不是墙钟 SLO 保证。

本地原型有26项包内检查和6项纯模型检查；它们核对边界、资源保留和生命周期，不能代替 GPU 服务结果。[既有恢复驻留分析](recovery_residencies.json)是历史逐事件描述；[prior_actions.md](prior_actions.md)记录先行工作比较，fixed4 也不是完整 UniBoost/MemGuard 复现或已调优最强量子基线。

成本 observer 在暖机 drain 和 connector reset 后安装，按完成元数据的观察时刻分别累计 capture / drain。dev16 KV512 五臂均为 COMPLETE、零缺失 acknowledgement；每臂 load 1,239,416,832 B / 15 records，store 402,653,184 B / 104 records。原生 copy-time 之和允许重叠，不能当暴露延迟；存储池/host cache 占用不是传输量。strong、article lease 及旧低压三臂未启用此 observer，其缺失传输成本为 unavailable，不补成零。`lease_decision_cpu_s` 实际是 chooser 内的 host elapsed timer，不是完整控制器 CPU 成本；零动作时计时为零也不能证明框架零开销。初始化、图捕获、暖机、测量和 drain 分列保留，不重复相加。

heldout 五臂 observer 也均为 COMPLETE、零缺失 acknowledgement；每臂 load 16–17 records、store 96–101 records，具体字节与时长见成本报告。ordinary 和 q1 各有1条2 MiB store 的完成元数据在 drain 阶段观察到，不能只取 capture 当总传输成本。五臂 chooser 计时均为0，对应零恢复选择；该结果只界定本轮机制未激活的域。

## 复现与重新分析

**禁止直接重跑已有 session 路径，也不覆盖原计划、冻结包或结果。** 新实验须复制对应计划到新文件，分配全新 `session_dir`，保持拟复现实验的包/输入哈希、资源和策略配置，检查当前共享 GPU/锁状态后由主会话安排整组执行。下表是控制器与历史计划的映射，不是待自动执行队列。

| 范围 | 控制器 | 历史计划 |
|---|---|---|
| strong | `run_strong_newhost.py` | `STRONG_NEWHOST_PLAN_R01_20261002.json` |
| article R01 | `run_lease_newhost.py` | `LEASE_NEWHOST_PLAN_R01_20261002.json` |
| article R02 | `run_lease_repeat_newhost.py` | `LEASE_NEWHOST_REVERSE_PLAN_R02_20261002.json` |
| healthy 低压 | `run_healthy_newhost.py` | `HEALTHY_DEV16_LOW_PLAN_R02_20261002.json` |
| healthy dev16 成本五臂 | `run_healthy_cost_comparison_newhost.py` | `HEALTHY_DEV16_KV512_COMPARISON_COST_PLAN_R02_20261002.json` |
| healthy heldout 成本五臂 | `run_healthy_cost_comparison_newhost.py` | `HEALTHY_HOLDOUT16_STEADY_KV512_COMPARISON_COST_PLAN_R02_20261002.json` |

远端目录为 `/root/autodl-tmp/moe-a-20261002/{strong,lease,healthy}`，共同锁为 `/root/autodl-tmp/moe-research-gpu.lock`。控制器整组持锁，逐臂使用独立初始空运行缓存，验证私有固定模型、源码/包、GPU身份、host预算与超时，并归档失败。运行环境必须含已安装库路径：

```sh
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib
/root/miniconda3/bin/python /absolute/path/to/selected_controller.py --plan /absolute/path/to/NEW_PLAN.json
```

成本 wrapper 复用未修改的 `run_healthy_comparison_newhost.py` 和 `run_healthy_newhost.py`，强制每臂 `A_AGGREGATE_OFFLOAD_COSTS=1` 并写入真实 wrapper SHA / manifest receipt。普通候选缺省不安装 aggregate observer。增量成本包 `healthy_cost_candidate_plans_r02.tar.gz` 平铺解包到已部署的 `healthy/`；SHA256 为 `4764383584cd951d2fdb5fd171288d82da5d16c7efe882de26dd74f81dec737f`。只在未部署的新目标中核验并解包；既有文件冲突应停止，不覆盖。

以下命令只分析已下载的 archive；输出须用尚不存在的新文件名。健康质量分析需要该固定 Instruct tokenizer snapshot，`--inputs` 缺省取首格已归档的 selected inputs。

```sh
python3 ../analyze_native_oldest_strong_triplet_r01.py --session strong-session-r01 --output /absolute/path/to/NEW_STRONG.json
python3 analyze_service_session.py --session lease-session-r02 --reference fixed4 --expected-requests 128 --output /absolute/path/to/NEW_SERVICE.json
python3 analyze_lease_actions.py --archive lease-session-r02/cell-00-adaptive/archive --output /absolute/path/to/NEW_ACTIONS.json
python3 analyze_runtime_costs.py --session lease-session-r02 --output /absolute/path/to/NEW_COSTS.json
python3 analyze_healthy_session.py --session session-dev16-kv512-comparison-cost-r02 --tokenizer-dir /absolute/path/to/PINNED_INSTRUCT_TOKENIZER --expected-arms native ordinary q1 fixed4 adaptive --reference native --output /absolute/path/to/NEW_QUALITY_SERVICE.json
```

论文入口为 [paper/draft.tex](paper/draft.tex) 和 [当前 PDF](output/pdf/draft.pdf)。README 的结果以以上原始报告为依据。当前研究边界是可执行且动作合法的原型、方向不稳定的 article 比较，以及具有完整传输成本但未激活量子的健康任务比较；这些结果尚不足以支持稳定性能优势或独立创新性主张。

最终交付：6页工作稿含3图1表，已渲染并检查全部页面；所有A GPU作业及SSH连接已结束。[具体方向决定](output/RESEARCH_DECISION_20261002.json)记录停止范围、未完成证据与重新打开条件。本轮按目标允许的“具体证据足以换方向”终点完成。

GitHub 检出后的大型 JSON 恢复方式见[上级数据说明](../README.md#github-原始数据恢复2026-10-04)及[恢复脚本](../restore_large_data_20261004.py)。运行 raw 分析或冻结包校验前先恢复；本地原始结果未改写。

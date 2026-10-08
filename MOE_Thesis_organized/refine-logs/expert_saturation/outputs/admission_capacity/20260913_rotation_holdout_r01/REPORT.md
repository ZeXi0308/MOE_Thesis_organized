# 固定参数轮转：独立文本与同负载 A/A 实测

2026-09-13。**20/20 COMPLETE，640/640 次正式请求执行完成；NATIVE_SERVING / MEASUREMENT_ONLY。** 两组新文本中，轮转对 headroom 的吞吐、平均完成时间与最大 ITL 的改善方向在四个区组均保持；对 native 的吞吐差仍变号，平均完成更慢。支持这两个新文本 cohort 上的暂停—服务量权衡，不支持吞吐非劣、统计显著性或方法 GO。

本轮唯一问题：固定 KV、固定策略参数时，旧四臂观察能否迁移到两组不重复文本，并测得同负载 native A/A 漂移。主问题仍是减少已有请求的长暂停、保住完整服务量，没有事后更换目标。

## 1. 主比较

每行是一个 cohort/block 内的独立引擎对照；相对差均为 `(rotate / 指定基线) - 1`。四行分别保留，未从 native A/A 中反选主基线。

| Cohort/block | native 最大 ITL s | rotate 最大 ITL s | headroom 最大 ITL s | 吞吐 Δ/native | 平均完成 Δ/native | 吞吐 Δ/headroom | 平均完成 Δ/headroom |
|---|---:|---:|---:|---:|---:|---:|---:|
| cohort0/0 | 4.480222 | 1.012792 | 1.376210 | +0.475% | +1.109% | +2.440% | -4.728% |
| cohort0/1 | 4.446978 | 1.013891 | 1.373627 | -0.619% | +2.284% | +1.943% | -4.257% |
| cohort1/0 | 4.455265 | 1.005360 | 1.379451 | -0.269% | +1.901% | +2.492% | -4.824% |
| cohort1/1 | 4.465285 | 1.004141 | 1.382521 | +0.001% | +1.713% | +2.781% | -5.067% |

轮转最大 ITL 为 1.004–1.014 s，native 为 4.447–4.480 s，headroom 为 1.374–1.383 s。轮转相对 headroom 吞吐 +1.943%–+2.781%、平均完成 −4.257%–−5.067%；相对 native 吞吐 −0.619%–+0.475%、平均完成 +1.109%–+2.284%。这些是完整实测差值，尚未作统计确认。

保守 safe29 的代价同时报告：

| Cohort/block | safe29 最大 ITL s | safe29 最长 TTFT s | rotate 最长 TTFT s | 吞吐 Δ/safe29 | 平均完成 Δ/safe29 |
|---|---:|---:|---:|---:|---:|
| cohort0/0 | 0.093393 | 17.463326 | 0.715287 | +16.704% | +6.873% |
| cohort0/1 | 0.121338 | 17.422415 | 0.748203 | +16.071% | +7.645% |
| cohort1/0 | 0.121883 | 17.409083 | 0.723384 | +15.797% | +7.084% |
| cohort1/1 | 0.114482 | 17.396668 | 0.724133 | +15.956% | +7.039% |

safe29 的输出阶段暂停较短，但三条晚入场请求等待首 token 约 17.4 s。轮转比它吞吐高 15.797%–16.704%，平均完成仍慢 6.873%–7.645%，最大 ITL 也更高，故不存在所有指标上的支配。兼容参考 TTFT≤5 s / mean-TPOT≤0.2 s 下，native、native_aa、headroom、rotate 各格 32/32 通过，safe29 各格 29/32；这不约束最大 ITL。全通过格的 goodput 等于吞吐，不能据此声称长暂停安全。

全部 20 格原值、24 个四策略配对、4 个 A/A 和 10 个同角色跨 block 配对见 [analysis/report.md](analysis/report.md) 与 [analysis.json](analysis/analysis.json)。

## 2. 同负载 A/A：已观察漂移

| Cohort/block | 吞吐 Δ | 平均完成 Δ | 最大 ITL Δ ms | 完整输出序列相同 |
|---|---:|---:|---:|---:|
| cohort0/0 | +0.782% | -0.820% | -47.245 | 32/32 |
| cohort0/1 | -0.124% | +0.049% | -1.996 | 32/32 |
| cohort1/0 | -0.434% | +0.427% | +39.215 | 32/32 |
| cohort1/1 | -0.257% | +0.320% | +16.743 | 32/32 |

四对 A/A 的最大已观察绝对吞吐差为 0.782%；它不是总体噪声上界、显著性检验或非劣界。轮转对 native 的吞吐差变号且幅度很小，保留“未确认保住吞吐”的结论；不从策略差中减去 A/A，不把超过某个 A/A 最大值自动标为可分辨。两组文本各两个 block，不把 640 次请求或全部 token 当独立重复。

## 3. 请求代价与输出边界

| Cohort/block | 对 native 完成变慢数 | 对 native max-ITL 增大数 | 对 headroom 完成变慢数 | 对 headroom max-ITL 增大数 | 输出相同：native / headroom / safe29 |
|---|---:|---:|---:|---:|---:|
| cohort0/0 | 31/32 | 30/32 | 2/32 | 1/32 | 31 / 31 / 11（各32） |
| cohort0/1 | 32/32 | 2/32 | 2/32 | 1/32 | 31 / 31 / 11（各32） |
| cohort1/0 | 32/32 | 2/32 | 2/32 | 1/32 | 32 / 31 / 10（各32） |
| cohort1/1 | 32/32 | 30/32 | 2/32 | 1/32 | 32 / 31 / 10（各32） |

上表“增大/变慢”按原始差值严格 >0 计数，包含微小漂移，不是显著受损请求数。轮转改善原生长暂停的同时，每组均产生两条约 0.953–0.960 s 的新暂停：cohort0 的文章 7684/7818、cohort1 的文章 11623/11726；它们相对 native 的 max-ITL 增加约 0.861–0.877 s。相对 headroom，每区组 30 条完成更早、2 条更晚；不能只报告总体均值。

所有同角色跨 block 输出均 32/32 一致，四对 A/A 也均一致；跨策略输出如上表不同。每请求仍执行 1024 个输出 token，但没有语义质量参考与质量评估；既不把输出差异直接判为错误，也不把序列一致升级为质量保证。

## 4. 恢复路径与完整成本

四个轮转格均发生 8 次强制与 2 次自然抢占；每次受保护恢复用 4 个调用到首个新输出，held request-step 均为 0。每 cohort 两个 block 的离散恢复路径一致。保护条件在已测动作中成立，但没有证明保护机制是收益所必需的。

最长暂停仍来自 step931 自然抢占的请求：最后输出在930，首个恢复调度976，首个新输出979。等待恢复约0.888–0.898 s，恢复调用跨度约0.115–0.116 s；另含不足0.4 ms的调用边界。全局冷却20并不构成单请求20步等待上界：另一更早缺席请求先在956恢复，本请求再等到976，共45步。新数据继续否定旧0.39 s估算的完整性，不能据旧估算给策略硬保证。

完整时间守恒：`wall = scheduler-inclusive + engine-non-schedule + outside-engine`。decision 是 scheduler 的子集，不能重复相加。下表是 rotate−native；含重算调用也产生其它请求的新 decode，非纯重算/GPU时间。

| Cohort/block | 含重算调用非调度区间 Δ s | 纯 decode 调用非调度区间 Δ s | 全引擎非调度区间 Δ s | 完整 wall Δ s |
|---|---:|---:|---:|---:|
| cohort0/0 | +0.904287 | -0.959104 | -0.072613 | -0.107885 |
| cohort0/1 | +0.913905 | -0.811118 | +0.120849 | +0.141342 |
| cohort1/0 | +0.910176 | -0.853989 | +0.046687 | +0.061089 |
| cohort1/1 | +0.911092 | -0.922318 | -0.001947 | -0.000255 |

每格相对 native 多重算30,876个位置：含重算调用8→40，纯 decode 调用减少123，总调用1348→1257。新增重算与批宽/收尾变化抵销，净时间可变号；不能仅按重算量预测吞吐。以上是各策略真实演进的工作分解，不是从固定 trace 扣掉工作的可执行反事实。

相对 headroom，轮转 wall 缩短0.444–0.631 s，其中 scheduler 区间减少0.169–0.196 s、引擎非调度区间减少0.289–0.426 s，另有引擎外时间差；不能把整项收益归为纯模型执行加速。账本见 [rotation_accounting.json](analysis/rotation_accounting.json)，工作与配对分解在 analysis.json。

## 5. 范围、冻结与执行记录

- 两个32请求 cohort，共64篇新 WikiText 文章；与旧32篇全文、token前缀、文档及来源行区间不重叠。每格3072输入/1024输出、50 ms到达；20个fresh-engine执行共640次正式请求、655,360个输出token。仍是同数据集、同长度/到达/内存压力域，未覆盖新模型或新运行域。输入依据：[inputs_report.json](inputs/inputs_report.json)。
- OLMoE BF16、vLLM0.26、RTX5090；实际KV 16,089,350,144 bytes / 7,671 usable blocks。主引擎cap32、token budget1024；safe29只限制准入，共用引擎能力。七个策略/采集模块与旧包相同，轮转30/20/30等参数未重选。
- 冻结顺序和比较规则：[campaign.json](preparation/source/campaign.json)、[DECISIONS.md](preparation/source/DECISIONS.md)。包SHA256 `0ead2b4c9cf935ddda491646139e73073164f6e0250384066226f8b2fb9718f7`；预注册文件仍保留准备时状态，不能把它当作当前未运行状态。
- 当前完整执行：[execution03/execution.json](execution03/execution.json)，远端 `/root/autodl-tmp/moe-rotation-holdout-20260913-r02`。20格按原顺序完成、归档逐项回传核验；所有源码指纹、runtime指纹、GPU身份与边界进程检查见 [execution_checks.json](analysis/execution_checks.json)。这些是初始化、测量前后观察，非持续独占或恒频证明。
- 前两次因其它GPU任务在初始化前退出，分别保留 [execution](execution/execution.json)、[execution02](execution02/execution.json)；第二次首格无raw，不计作性能结果。新尝试在核实GPU及相关driver空闲后开始，没有重选已测数据。早期[范围拒绝](UPLOAD_APPROVAL_REJECTION.json)及后来明确的[20项授权](UPLOAD_EXECUTION_AUTHORIZATION.json)同时保留。

复算命令：

```sh
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_rotation_holdout.py --run-dir refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_holdout_r01/execution03 --output-dir /private/tmp/rotation-holdout-reanalysis-NEW
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_rotation_recovery.py --run-dir refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_holdout_r01/execution03 --analysis /private/tmp/rotation-holdout-reanalysis-NEW/analysis.json --output /private/tmp/rotation-holdout-reanalysis-NEW/rotation_accounting.json
```

输出目录必须新建，不覆盖本轮原始分析。独立完整性核查另记于同目录的 EXPERIMENT_AUDIT.md / .json，审阅状态以该记录为准；不以审计替代性能证据。

## 6. 当前裁决与唯一下一步

| 固定报告项 | 结论 |
|---|---|
| Verdict | MEASUREMENT_ONLY；固定参数权衡在两个新文本 cohort 保持，主问题 OPEN |
| Evidence type | NATIVE_SERVING，单卡原生进程内引擎完整请求与host逐token返回记录 |
| What was measured | 20格、640次完整请求；同block四臂、native A/A、同角色重复、实际恢复与完整成本 |
| What was not measured | 统计显著性/吞吐非劣、语义质量、客户端消费QoE、新长度/到达/KV/模型域、组件必要性 |
| Strongest baseline | 同底座headroom-fast；同时保留native32与safe29的完整权衡，尚缺正式最近邻策略比较 |
| Oracle/headroom status | 原生秒级暂停可被真实动作缩短；无经过验证的全动作Oracle/硬上界 |
| Claim ceiling | 这两组新文本的描述性迁移；不称MoE专属、方法GO或独立新颖性 |
| Failure category | 未确认native吞吐非劣；平均完成和部分请求承担代价；保护贡献与新颖性仍未分离 |
| Resurrection condition | 问题没有被判死；扩大主张需独立受控数据及明确目标约束，不能靠同数据选阈值 |
| One next smallest experiment | 同底座仅改变驱逐对象排序：现有最少进度 vs 合格集合内已交付输出最多者；其余触发、恢复与保护固定 |

当前轮转本身已含最长缺席恢复与aging，再加同义aging标签不构成独立基线。下一项仅为服务量排序消融，不能冒称VTC/FastServe复现或完整公平基线；两臂独立演进并保留真实成本，若选择相同则记录无动作差异。它用来分离本轮收益是否依赖特定驱逐规则；正式prior-art比较仍需另行定义服务实体、计费和触发语义。

**直接回答本轮问题：暂停—服务量权衡迁移到了这两组新文本；“保住native吞吐”仍未确认，且平均完成变慢的代价保持。**

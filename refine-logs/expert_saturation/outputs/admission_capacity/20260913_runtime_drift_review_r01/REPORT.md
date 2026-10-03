# 负控漂移批评的定向复核

2026-09-13，HEAD `7059fc98e0d98a127ff4edda86d44f7f634d26f4`，工作区含既有未提交研究改动。
本轮只读重析已有数据，未新增 GPU 测量，未修改原始报告、raw 或预注册门槛。

结论：需要把运行波动纳入门槛解释，但现有最大负控差不能称为统计噪声界，不能据此把 mixed 指标标为已确认真效应或已确认噪声。

## 原始数据核查

`waiting_bypass_limit_r01` 的 all_short 有两个 block、每 block 三臂，生成六个共享运行的两两差值。它们不是六个独立 A/A 配对。
六对首次入场顺序均相同，输出序列相同数依次为 6/5/6、6/10/4（各16请求）。但完整逐步请求/token分配全部不同：

| Block | FCFS steps | SPT steps | bounded steps |
|---|---:|---:|---:|
| forward | 306 | 301 | 304 |
| reverse | 309 | 305 | 313 |

因此可以称无目标排序动作时的运行和轨迹波动；仅靠首次入场顺序相同不能定位为纯 kernel 数值不确定性、纯计时噪声或完全相同调度路径。到达落在哪一步、后续 batch 和自由输出仍可能不同，策略路径与记录开销也未隔离。

以下是各 campaign 自己六个 all_short 差值的最大绝对百分比，仅作样本描述；相对差使用各配对声明的 baseline 作分母：

| 指标 | waiting | per_request |
|---|---:|---:|
| TTFT mean | 2.8241% | 5.9043% |
| TTFT p95 | 3.1017% | 6.3141% |
| TTFT p99 | 2.7343% | 6.2422% |
| 完成延迟 mean | 1.3675% | 3.3886% |
| pooled ITL p99 | 7.3252% | 12.0850% |
| 逐请求 max-ITL p99 | 21.3176% | 51.6044% |
| 逐请求 max-ITL max | 22.9686% | 60.9328% |

这两组描述本身也说明不能共用一个百分比常数；更不能直接把 all_short 的漂移搬到 mixed。固定长度、batch、生成轨迹及策略成本都影响波动。

## 哪些批评成立，哪些需要改写

1. 原报告已披露负控漂移和描述性边界，但没有把这些读数转为门槛的不确定性分析。这是可补的决策解释缺口；不宜写成完全忽略波动或原始数字错误。
2. `per_request/DECISIONS.md:57` 的约3%是考虑新输入受控确认的探索门槛，不是方法 GO。双 block 条件本身是设计；尚未校准误判风险不等于“全靠运气”。负控的 +3.39% 是变慢，也不能直接当作产生 −3% 改善的证据。
3. “最大 ITL 无可重复损害”缺少容忍损害量和可评估的不确定性。当前两块不足以确认小幅无损；高波动不意味着该指标永远不可用，也不意味着低于样本最大差的效应必然为零。
4. bounded 对 SPT 的 mixed TTFT p95 为 −6.3203%/−9.0982%，p99 为 −5.7019%/−9.5039%，是两块同向的探索性观察，应明确列出。但它们尚未经过同 mixed 负载 A/A 与更多独立区组验证，不能标为“已确认真信号”。p95/p99 也不是两份独立证据。
5. “最大 ITL 的 21.4→27.2 在噪声内”数值上亦不成立：reverse 对 FCFS 实际为 21.323→27.238 ms，即 +27.7407%；对 SPT 的 max-ITL p99 为 +27.5691%。均超过引用的相应样本最大差。但这同样不能直接判为可重复损害，应保留单块恶化与跨块不一致。
6. 已记录的 ITL p99 约29%–30%变化可以作为描述性测量；“只有样本最大漂移的4倍”既不是否认它的统计理由，也不是确认它的统计检验。原报告限于描述、不宣称显著性的边界必须一起读。
7. 全部请求通过时，本仓库 goodput = n_slo_pass / observation_duration = throughput。通过率饱和使延迟风险增量不可区分，但服务量比较仍有意义。门槛附近密集则使计数对阈值/运行波动敏感，需展示阈值附近分布、敏感性和重复；这两种现象不等于 goodput 一概不可用。
8. host_timing 的115/512来自对其他进程 paired-r01 的重分析；其报告明确初始新增GPU执行为0。它是独立分析，并非这条现象的新GPU复现。若与原 steady 引用同一批 raw，不能把两者当两份独立实验支持。
9. “3×观察最大差”不是有误判控制的门槛。改善目标应来自实际价值/业务要求，损害容忍量另行定义；由同负载重复评估差值的不确定性。样本不足时写未确认，不用提高任意百分比替代重复。

合理的主表标记是“描述性差值；两块同向/变号；缺同域 A/A；不确定性未估计”，不把与跨域最大差的大小比较标作可分辨/不可分辨。

## 证据与执行

复算命令：`python3 -B refine-logs/expert_saturation/outputs/admission_capacity/20260913_runtime_drift_review_r01/recompute.py`。
脚本拒绝覆盖已有 observations.json；重跑时用 `--output /tmp/新的文件名.json`。该文件保存六对全向量、比较顺序、逐步轨迹差异、输入摘要哈希与所读 raw 哈希。
来源是两项目 `analysis/summary.json`、waiting 六个 all_short raw，以及各自 REPORT/DECISIONS；host 边界来自 `refine-logs/independent_ideas_20260908/host_timing_boundary_r01/REPORT.md:3`。

统计解释参考：[NIST 样本极值容忍区间](https://www.itl.nist.gov/div898/handbook/prc/section2/prc264.htm)说明覆盖率、置信度与样本量均需明确；[NIST 随机区组设计](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm)说明区组与随机化对干扰因素的不同作用。

| 固定报告项 | 结论 |
|---|---|
| Verdict | MEASUREMENT_ONLY；批评核心成立，噪声界/真信号/独立复现表述需收紧 |
| Evidence type | CPU 重析已有 REQUEST_LEVEL / native in-process GPU 测量 |
| What was measured | 已有负控差向量、完整调度路径是否相同、mixed原值、指标分母 |
| What was not measured | 新 A/A、混合域噪声分布、显著性、非劣性、独立质量 |
| Strongest baseline | 各实验原有同块 FCFS/SPT 或 native1024，未反选基线 |
| Oracle/headroom status | 未测；负控最大值不是 Oracle 或误差上界 |
| Claim ceiling | 现有样本的描述性变动及解释边界，不改变旧 formulation 的探索停止决定 |
| Failure category | 不确定性校准不足；批评中存在跨域迁移和独立性过度解释 |
| Resurrection condition | 同域独立配对数据改变完整请求权衡证据；不以改门槛重开旧策略 |
| One next smallest experiment | 针对待确认的同一负载做同策略 A/A，并在随机区组内保留强基线和策略配对 |

这段结论可以作为研究流程的修正意见，但目前不能据它宣布已找回统计可分辨的真效应，或把已观察到的损害统一消解为噪声。

### 原审计范围补充核对（2026-09-13）

本次重跑 `recompute.py` 与保留的 `observations.json` 完全一致，未新增GPU数据。既有只读审查者另核对了原文（reused reviewer / same-family / provisional）：`per_request_prefill_share_r01/EXPERIMENT_AUDIT.md:23` 第5条已经明确给出负控完成延迟 +3.389%/+0.815%、每 arm/order 仅一个正式episode，并据此禁止显著性、稳定效应量和一般回归主张。因此“审计只检查数字、没有查噪声”与现有证据不符；准确缺口是未形成逐指标精度校准。该目录的审计md/json是同一次审计的两种表示，不能仅凭两个文件认定两轮独立审计。`DECISIONS.md:57–59` 的约3%明确只通往新输入的受控确认；`host_timing_boundary_r01/REPORT.md:4–6,40,75` 的115/512来自既有paired-r01重分析，新增GPU为0。没有重新扩展全面审计。

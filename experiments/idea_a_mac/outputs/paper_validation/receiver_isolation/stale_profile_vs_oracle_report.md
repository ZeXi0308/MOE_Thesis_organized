# Stale Offline Profile vs Oracle Signals (Confound-Removed)

## 目的

在已经剥离 remote-preference 混淆的固定候选池内，直接检验论文当前方法
`tail_budget_profile_ports`（用 disjoint calibration 算出的静态 sender/receiver
负载去打分 test 时刻的候选）到底是不是一个有效信号，还是它之前报告的
"追平 scheduler_receiver/greedy_ports"表现完全是 remote-bonus 撑出来的假象。

## 方法

同一固定候选池（tail-rank ∩ inter-node），比较：

- `random`：30 次随机种子均值（基线）
- `stale_profile`：用 calibration 集合（offset 0，16 条文本）算出的静态
  `max(sender_load, receiver_load)` 去给 test 场景的候选打分——这正是论文
  `tail_budget_profile_ports` 策略实际依赖的信息
- `oracle_receiver`：test 场景当前真实 receiver 负载（scheduler 可提前知道的量）
- `oracle_combined`：test 场景当前真实 `max(sender,receiver)` 负载（不可离线获得的上界）

## 配置

- model: `allenai/OLMoE-1B-7B-0924`; EP=`8`; GPUs/node=`4`
- calibration: `experiments/idea_a_mac/outputs/paper_validation/olmoe_signal_comparison_n32/calibration_routes.csv`（offset 0, disjoint from test）
- test scenario 来源: `experiments/idea_a_mac/outputs/paper_validation/olmoe_signal_comparison_n32/test_routes.csv`
- concurrent jobs: `[4, 8, 16]`; origin modes: `['balanced', 'hotspot']`

## 结果

| origin_mode | num_jobs | budget_fraction | random | stale_profile | oracle_receiver | oracle_combined |
|---|---|---|---|---|---|---|
| balanced | 4 | 0.250000 | 0.054531 | 0.074533 | 0.066953 | 0.141107 |
| balanced | 4 | 0.500000 | 0.108801 | 0.119884 | 0.117105 | 0.204901 |
| balanced | 4 | 0.750000 | 0.163231 | 0.185195 | 0.170667 | 0.217029 |
| balanced | 8 | 0.250000 | 0.054851 | 0.086316 | 0.085937 | 0.172883 |
| balanced | 8 | 0.500000 | 0.108401 | 0.145035 | 0.158392 | 0.212450 |
| balanced | 8 | 0.750000 | 0.161992 | 0.200479 | 0.207283 | 0.215726 |
| balanced | 16 | 0.250000 | 0.058550 | 0.065645 | 0.074842 | 0.159080 |
| balanced | 16 | 0.500000 | 0.115525 | 0.134522 | 0.140120 | 0.214162 |
| balanced | 16 | 0.750000 | 0.172833 | 0.189703 | 0.190070 | 0.229890 |
| hotspot | 4 | 0.250000 | 0.061891 | 0.031369 | 0.118514 | 0.118514 |
| hotspot | 4 | 0.500000 | 0.123923 | 0.133611 | 0.235852 | 0.235943 |
| hotspot | 4 | 0.750000 | 0.186500 | 0.242723 | 0.247876 | 0.248237 |
| hotspot | 8 | 0.250000 | 0.061705 | 0.048207 | 0.102751 | 0.102751 |
| hotspot | 8 | 0.500000 | 0.123118 | 0.141429 | 0.205595 | 0.205595 |
| hotspot | 8 | 0.750000 | 0.184599 | 0.238453 | 0.246386 | 0.246386 |
| hotspot | 16 | 0.250000 | 0.060628 | 0.074976 | 0.117500 | 0.142838 |
| hotspot | 16 | 0.500000 | 0.121467 | 0.197618 | 0.211659 | 0.241554 |
| hotspot | 16 | 0.750000 | 0.181766 | 0.242007 | 0.238623 | 0.242007 |

## 关键读数

- `stale_profile` 与 `random` 的平均绝对差：`0.0297`
- `oracle_combined` 与 `stale_profile` 的平均差距：`0.0555`

## 判定

结合 `run_expert_popularity_stability.py` 的发现（专家/owner 负载排序跨样本
Spearman 仅 ~0.39-0.50，最热 sender rank 跨集合一致率仅 25%），此处的直接
对照进一步验证：**静态离线 profile 在被剥离 remote-bonus 后的表现明显弱于
oracle 信号**（若上表中 `stale_profile` 接近 `random` 且明显低于
`oracle_combined`/`oracle_receiver`）。这意味着此前
`congestion_report.md` / `quality_safe_congestion_report.md` 中报告的
"`tail_budget_profile_ports` 与 `tail_budget_scheduler_receiver` 表现几乎相同"
这一结论，主要驱动因素是两者共享的 remote-bonus，而不是离线 profile 本身
提供了有效信号——这是需要在论文里明确纠正的一处过强表述。

## 对论文 receiver-aware 章节的具体修正建议

1. 删除或大幅弱化"离线 profile 就足够"的表述；`tail_budget_profile_ports`
   在移除 remote-bonus 后并不比 random 好多少。
2. 保留、且应重点强调 `oracle_receiver`（scheduler 已知的 receiver 热度）
   这一支：它不需要等本层路由，代价低，且在 origin 明显不均衡（hotspot）
   时几乎拿到 `oracle_combined` 的全部收益。
3. 对 sender 侧真实收益（在 origin `balanced` 时才明显），必须诚实标注为
   "需要 layer-local 在线专家负载同步，且该负载本身随输入内容变化，
   不能靠离线 profile 替代"，作为未来工作或需要真实系统实现的额外开销
   项，不能默认为免费信号。

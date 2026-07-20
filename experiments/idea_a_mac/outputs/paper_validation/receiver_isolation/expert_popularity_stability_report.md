# Expert Popularity Stability (Calibration vs Held-out Test)

## 目的

`run_receiver_sender_decomposition.py` 发现在 `balanced` origin 模式下，
sender 侧信号（expert-owner GPU 当前负载）贡献了 combined 收益的主要部分
（约 57%-99%，多数场景 > 75%）。但 sender 信号在当前实现中依赖"这一层
刚算出的路由结果"，这是逐层实时信息，同步代价高，几乎和要压缩的 combine
通信同时发生，可部署性存疑。

这里验证一个关键前提：expert 热度（谁更常被选中）是否在不同输入样本间
稳定？如果稳定，sender 信号可以退化为一次离线 profile（类似论文里已有的
PLTB layer sensitivity profile），而不需要逐层在线同步专家命中数。

## 方法

用 disjoint 的 calibration（offset 0）和 held-out test（offset 128）
WikiText-2 路由 trace，按 layer 比较：

- expert-id 级别命中次数分布的 Spearman 相关；
- 聚合到 sender_rank（owner GPU，按 `expert_id * ep_size // num_experts` 静态
  placement 规则）后的负载分布 Spearman 相关；
- 每层最热 expert / 最热 sender_rank 是否在两个集合中一致。

## 配置

- model: `allenai/OLMoE-1B-7B-0924`; EP size: `8`
- calibration: `experiments/idea_a_mac/outputs/paper_validation/olmoe_signal_comparison_n32/calibration_routes.csv`
- test: `experiments/idea_a_mac/outputs/paper_validation/olmoe_signal_comparison_n32/test_routes.csv`

## 结果（按层）

| layer | expert_popularity_spearman_cal_vs_test | sender_rank_load_spearman_cal_vs_test | cal_top_expert | test_top_expert | top_expert_matches | cal_hottest_sender_rank | test_hottest_sender_rank | hottest_sender_rank_matches |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.3915 | 0.2857 | 6 | 6 | True | 0 | 0 | True |
| 1 | 0.7379 | 0.6190 | 47 | 11 | False | 5 | 1 | False |
| 2 | 0.4565 | 0.3333 | 36 | 59 | False | 4 | 7 | False |
| 3 | 0.3075 | 0.4524 | 23 | 24 | False | 2 | 3 | False |
| 4 | 0.5860 | 0.4524 | 9 | 37 | False | 1 | 4 | False |
| 5 | 0.4263 | 0.6190 | 56 | 23 | False | 7 | 2 | False |
| 6 | 0.5207 | 0.4286 | 32 | 18 | False | 2 | 2 | True |
| 7 | 0.5442 | 0.4286 | 63 | 1 | False | 7 | 5 | False |
| 8 | 0.6856 | 0.7143 | 14 | 18 | False | 1 | 7 | False |
| 9 | 0.7830 | 0.5952 | 18 | 50 | False | 6 | 6 | True |
| 10 | 0.5881 | 0.0120 | 62 | 54 | False | 1 | 1 | True |
| 11 | 0.5396 | 0.5952 | 48 | 55 | False | 6 | 5 | False |
| 12 | 0.3694 | 0.0714 | 35 | 19 | False | 4 | 5 | False |
| 13 | 0.3351 | 0.0238 | 59 | 56 | False | 2 | 5 | False |
| 14 | 0.3941 | 0.3095 | 33 | 49 | False | 4 | 0 | False |
| 15 | 0.3813 | 0.2619 | 38 | 49 | False | 4 | 6 | False |

## 汇总

- expert-id 命中分布 Spearman 均值（跨层）：`0.5029`
- sender_rank 负载分布 Spearman 均值（跨层）：`0.3877`
- 最热 expert 跨集合一致率：`6.25%`
- 最热 sender_rank 跨集合一致率：`25.00%`

## 判定

**专家热度跨样本不稳定，sender 信号确实依赖当层在线路由，不能简单退化为离线 LUT**

## 意义

若上面判定为"可离线 profile"：这为论文提供一个比原始 receiver-aware 更
站得住脚、也更有新意的贡献点——**Two-Tier Congestion-Safe Budgeting**：

1. 离线阶段：像 PLTB 一样对每层做一次 expert-popularity profile，得到
   `LUT[layer, expert_id] -> owner_hotness_prior`（静态，随 checkpoint 固定，
   与 batch/request 无关）；
2. 在线阶段：调度器已知的 receiver/token-origin 热度（通常来自请求分布，
   不需要等路由结果）与①的静态 prior 相加，决定把有限 INT4 预算优先给
   哪些 remote (sender, receiver) pair；
3. 完全不需要在 dispatch 之后、combine 之前插入额外的跨 rank 专家负载
   同步，因为 sender 侧用的是离线 prior 而不是当层实时统计。

这比"receiver-aware"更准确地描述了真正驱动收益的机制，也更容易在论文里
写成一个可部署、有明确 offline/online 分工的系统设计，而不是笼统地说
"识别热点端口"。

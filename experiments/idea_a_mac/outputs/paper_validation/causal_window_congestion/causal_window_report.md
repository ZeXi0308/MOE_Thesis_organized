# Causal Sliding-Window Congestion Budgeting — Isolation Experiment

## 目的

在已经证伪"跨请求离线 profile"（`run_expert_popularity_stability.py`：Spearman 仅
0.39-0.50，无法跨样本迁移）之后，检验第三种、此前完全未测试过的信号：
**同一请求内部、因果的、滑动窗口 sender 负载估计**——只用该请求 REMOTE 流量里
已经解码过的最近 `window` 个 token 的 sender_rank 分布，去给接下来的 token 打分，
不使用任何未来信息，也不跨请求借用信息。

这个信号在真实系统里几乎零成本可得：EP dispatch 阶段本来就已经知道每个 token
被路由到哪个 sender rank（expert 所在 GPU），滑动窗口统计只是把这个信息按 causal
方式攒起来，不需要额外的跨 rank 同步或提前 profile。

## 方法

三方对照，固定候选池为 **tail-rank AND inter-node**（对三者完全相同）：

- `hot`（oracle，不可部署上界）：用当前 job 在该层**全部**（含未来）token 的真实
  remote 负载打分——这是`run_receiver_isolation_experiment.py`里已验证过的"热点
  识别本身有独立价值"的那个信号，这里作为上界参照，不代表可部署方案。
- `causal_window`（本实验新增，可部署）：只用同一 job、同一层、**该 token 之前**
  的最近 `window` 个 remote token 的 sender_rank 分布打分。
- `random`：30 个随机种子重复抽样，报告均值和 95% 分位区间（下界）。

baseline 为 uniform FP8（1 byte/elem），预算内 pair 升级为 INT4（0.5 byte/elem）。
瓶颈指标口径与 `run_ep_congestion_sim.summarize` 一致。

`causal_w{W}_pct_of_oracle_gap` = `(causal_window - random_mean) / (hot_oracle - random_mean)`，
即 causal_window 在"random 下界"和"oracle 上界"之间，拿到了多大比例的差距。这是
衡量"零成本可部署信号能追回多少不可部署上界收益"的核心指标。

## 配置

- model: `allenai/OLMoE-1B-7B-0924`; EP=`8`; GPUs/node=`4`
- concurrent jobs: `[4, 8, 16]`; origin modes: `['balanced', 'hotspot']`
- budget fractions (within tail∩remote pool): `[0.25, 0.5, 0.75, 1.0]`
- causal windows tested: `[8, 16, 32]` (tokens)
- inter-node bandwidth: `200.0 Gbps`
- random trials per config: `30`

## 结果

| origin_mode | num_jobs | budget_fraction | random_mean_saving | hot_oracle_saving | oracle_minus_random_gap | causal_w8_saving | causal_w8_pct_of_oracle_gap | causal_w16_saving | causal_w16_pct_of_oracle_gap | causal_w32_saving | causal_w32_pct_of_oracle_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| balanced | 4 | 0.250000 | 0.054531 | 0.139338 | 0.084807 | 0.091713 | 0.438431 | 0.102830 | 0.569513 | 0.110788 | 0.663357 |
| balanced | 4 | 0.500000 | 0.108801 | 0.204396 | 0.095595 | 0.150581 | 0.437054 | 0.159045 | 0.525592 | 0.166372 | 0.602238 |
| balanced | 4 | 0.750000 | 0.163231 | 0.217029 | 0.053798 | 0.189995 | 0.497495 | 0.193153 | 0.556199 | 0.192016 | 0.535066 |
| balanced | 4 | 1.000000 | 0.217155 | 0.217155 | 0.000000 | 0.217155 | nan | 0.217155 | nan | 0.217155 | nan |
| balanced | 8 | 0.250000 | 0.054851 | 0.171749 | 0.116898 | 0.094254 | 0.337070 | 0.106477 | 0.441630 | 0.115171 | 0.516007 |
| balanced | 8 | 0.500000 | 0.108401 | 0.212450 | 0.104049 | 0.150076 | 0.400533 | 0.160660 | 0.502261 | 0.166583 | 0.559180 |
| balanced | 8 | 0.750000 | 0.161992 | 0.215726 | 0.053734 | 0.190524 | 0.530994 | 0.194556 | 0.606035 | 0.192540 | 0.568514 |
| balanced | 8 | 1.000000 | 0.215726 | 0.215726 | -0.000000 | 0.215726 | nan | 0.215726 | nan | 0.215726 | nan |
| balanced | 16 | 0.250000 | 0.058550 | 0.158680 | 0.100130 | 0.093835 | 0.352391 | 0.104565 | 0.459549 | 0.115961 | 0.573364 |
| balanced | 16 | 0.500000 | 0.115525 | 0.214129 | 0.098604 | 0.158780 | 0.438681 | 0.169310 | 0.545470 | 0.178341 | 0.637052 |
| balanced | 16 | 0.750000 | 0.172833 | 0.230023 | 0.057190 | 0.203232 | 0.531541 | 0.207664 | 0.609035 | 0.211196 | 0.670797 |
| balanced | 16 | 1.000000 | 0.230190 | 0.230190 | -0.000000 | 0.230190 | nan | 0.230190 | nan | 0.230190 | nan |
| hotspot | 4 | 0.250000 | 0.061891 | 0.118514 | 0.056623 | 0.060477 | -0.024959 | 0.061743 | -0.002608 | 0.060387 | -0.026555 |
| hotspot | 4 | 0.500000 | 0.123923 | 0.235852 | 0.111930 | 0.122943 | -0.008749 | 0.122311 | -0.014403 | 0.125565 | 0.014672 |
| hotspot | 4 | 0.750000 | 0.186500 | 0.248237 | 0.061737 | 0.185952 | -0.008883 | 0.188212 | 0.027724 | 0.188302 | 0.029188 |
| hotspot | 4 | 1.000000 | 0.248237 | 0.248237 | -0.000000 | 0.248237 | nan | 0.248237 | nan | 0.248237 | nan |
| hotspot | 8 | 0.250000 | 0.061705 | 0.102751 | 0.041046 | 0.061491 | -0.005222 | 0.062805 | 0.026798 | 0.064072 | 0.057675 |
| hotspot | 8 | 0.500000 | 0.123118 | 0.205595 | 0.082477 | 0.122559 | -0.006773 | 0.124343 | 0.014854 | 0.126267 | 0.038188 |
| hotspot | 8 | 0.750000 | 0.184599 | 0.246386 | 0.061787 | 0.185505 | 0.014662 | 0.186209 | 0.026058 | 0.189354 | 0.076958 |
| hotspot | 8 | 1.000000 | 0.246386 | 0.246386 | 0.000000 | 0.246386 | nan | 0.246386 | nan | 0.246386 | nan |
| hotspot | 16 | 0.250000 | 0.060628 | 0.145982 | 0.085354 | 0.075988 | 0.179959 | 0.077161 | 0.193694 | 0.077854 | 0.201811 |
| hotspot | 16 | 0.500000 | 0.121467 | 0.241554 | 0.120087 | 0.138895 | 0.145127 | 0.138548 | 0.142242 | 0.135165 | 0.114064 |
| hotspot | 16 | 0.750000 | 0.181766 | 0.242007 | 0.060241 | 0.192955 | 0.185747 | 0.191650 | 0.164074 | 0.187440 | 0.094193 |
| hotspot | 16 | 1.000000 | 0.242007 | 0.242007 | -0.000000 | 0.242007 | nan | 0.242007 | nan | 0.242007 | nan |

## 判定

- balanced / jobs=4 / frac=0.25: W=8: 拿到oracle-random差距的43.8% (显著优于random); W=16: 拿到oracle-random差距的57.0% (显著优于random); W=32: 拿到oracle-random差距的66.3% (显著优于random)
- balanced / jobs=4 / frac=0.50: W=8: 拿到oracle-random差距的43.7% (显著优于random); W=16: 拿到oracle-random差距的52.6% (显著优于random); W=32: 拿到oracle-random差距的60.2% (显著优于random)
- balanced / jobs=4 / frac=0.75: W=8: 拿到oracle-random差距的49.7% (显著优于random); W=16: 拿到oracle-random差距的55.6% (显著优于random); W=32: 拿到oracle-random差距的53.5% (显著优于random)
- balanced / jobs=4 / frac=1.00: 
- balanced / jobs=8 / frac=0.25: W=8: 拿到oracle-random差距的33.7% (显著优于random); W=16: 拿到oracle-random差距的44.2% (显著优于random); W=32: 拿到oracle-random差距的51.6% (显著优于random)
- balanced / jobs=8 / frac=0.50: W=8: 拿到oracle-random差距的40.1% (显著优于random); W=16: 拿到oracle-random差距的50.2% (显著优于random); W=32: 拿到oracle-random差距的55.9% (显著优于random)
- balanced / jobs=8 / frac=0.75: W=8: 拿到oracle-random差距的53.1% (显著优于random); W=16: 拿到oracle-random差距的60.6% (显著优于random); W=32: 拿到oracle-random差距的56.9% (显著优于random)
- balanced / jobs=8 / frac=1.00: 
- balanced / jobs=16 / frac=0.25: W=8: 拿到oracle-random差距的35.2% (显著优于random); W=16: 拿到oracle-random差距的46.0% (显著优于random); W=32: 拿到oracle-random差距的57.3% (显著优于random)
- balanced / jobs=16 / frac=0.50: W=8: 拿到oracle-random差距的43.9% (显著优于random); W=16: 拿到oracle-random差距的54.5% (显著优于random); W=32: 拿到oracle-random差距的63.7% (显著优于random)
- balanced / jobs=16 / frac=0.75: W=8: 拿到oracle-random差距的53.2% (显著优于random); W=16: 拿到oracle-random差距的60.9% (显著优于random); W=32: 拿到oracle-random差距的67.1% (显著优于random)
- balanced / jobs=16 / frac=1.00: 
- hotspot / jobs=4 / frac=0.25: W=8: 拿到oracle-random差距的-2.5% (未显著优于random); W=16: 拿到oracle-random差距的-0.3% (未显著优于random); W=32: 拿到oracle-random差距的-2.7% (未显著优于random)
- hotspot / jobs=4 / frac=0.50: W=8: 拿到oracle-random差距的-0.9% (未显著优于random); W=16: 拿到oracle-random差距的-1.4% (未显著优于random); W=32: 拿到oracle-random差距的1.5% (未显著优于random)
- hotspot / jobs=4 / frac=0.75: W=8: 拿到oracle-random差距的-0.9% (未显著优于random); W=16: 拿到oracle-random差距的2.8% (未显著优于random); W=32: 拿到oracle-random差距的2.9% (未显著优于random)
- hotspot / jobs=4 / frac=1.00: 
- hotspot / jobs=8 / frac=0.25: W=8: 拿到oracle-random差距的-0.5% (未显著优于random); W=16: 拿到oracle-random差距的2.7% (未显著优于random); W=32: 拿到oracle-random差距的5.8% (未显著优于random)
- hotspot / jobs=8 / frac=0.50: W=8: 拿到oracle-random差距的-0.7% (未显著优于random); W=16: 拿到oracle-random差距的1.5% (未显著优于random); W=32: 拿到oracle-random差距的3.8% (未显著优于random)
- hotspot / jobs=8 / frac=0.75: W=8: 拿到oracle-random差距的1.5% (未显著优于random); W=16: 拿到oracle-random差距的2.6% (未显著优于random); W=32: 拿到oracle-random差距的7.7% (显著优于random)
- hotspot / jobs=8 / frac=1.00: 
- hotspot / jobs=16 / frac=0.25: W=8: 拿到oracle-random差距的18.0% (显著优于random); W=16: 拿到oracle-random差距的19.4% (显著优于random); W=32: 拿到oracle-random差距的20.2% (显著优于random)
- hotspot / jobs=16 / frac=0.50: W=8: 拿到oracle-random差距的14.5% (显著优于random); W=16: 拿到oracle-random差距的14.2% (显著优于random); W=32: 拿到oracle-random差距的11.4% (显著优于random)
- hotspot / jobs=16 / frac=0.75: W=8: 拿到oracle-random差距的18.6% (显著优于random); W=16: 拿到oracle-random差距的16.4% (显著优于random); W=32: 拿到oracle-random差距的9.4% (显著优于random)
- hotspot / jobs=16 / frac=1.00: 

## 解读边界

- 这仍是 bandwidth-only 解析回放，不含 collective、queueing、pack/unpack、kernel。
- `hot` oracle 使用了同 job 内的"未来" token 信息，不是可部署方案，只作为上界参照。
- `causal_window` 的 cold-start（job 开头 window 内的 token 无历史）会退化为接近
  random 的行为，这是真实系统里无法避免的启动代价，已如实计入总体指标（未做特殊剔除）。
- 若 `causal_window` 能稳定拿到 oracle-random 差距的可观比例（且显著超出 random
  95% CI），说明"因果同请求内滑动窗口"是一个真实、独立于此前两个信号（离线 profile
  已证伪、oracle 不可部署）的、可直接部署的新信号，可以作为 receiver-aware 支线的
  替代实现方式写入论文；如果始终落在 random CI 内或占比很低，则说明该信号在本场景
  的并发/热点结构下也不够用，需要另寻其他部署路径。

# Receiver-Aware Isolation Experiment

## 目的

剥离 `run_ep_congestion_sim.py` / `run_quality_safe_congestion_frontier.py` 中
"优先压缩跨节点流量"这一硬编码 bonus 造成的混淆，单独检验"识别热点
sender/receiver 端口"这个信号本身是否有独立价值。

## 方法

固定候选池为 **tail-rank AND inter-node** 的 pair（对 hot/cold/random 完全相同），
只改变在这个池子内选哪些 pair 获得 INT4 预算：

- `hot`：按当前真实 remote sender/receiver 负载（max(sender_load, receiver_load)）降序选择
- `cold`：同一负载指标升序选择（反向对照）
- `random`：30 个随机种子重复抽样，报告均值和 95% 分位区间

baseline 为 uniform FP8（1 byte/elem），预算内 pair 升级为 INT4（0.5 byte/elem）。
瓶颈指标口径与 `run_ep_congestion_sim.summarize` 一致：只统计 remote (inter-node)
ingress/egress 的 per-layer max，再除以带宽求和。

## 配置

- model: `allenai/OLMoE-1B-7B-0924`; EP=`8`; GPUs/node=`4`
- concurrent jobs: `[4, 8, 16]`; origin modes: `['balanced', 'hotspot']`
- budget fractions (within tail∩remote pool): `[0.25, 0.5, 0.75, 1.0]`
- inter-node bandwidth: `200.0 Gbps`
- random trials per config: `30`

## 结果

| origin_mode | num_jobs | budget_fraction | hot_saving_vs_fp8 | random_mean_saving_vs_fp8 | random_ci_low | random_ci_high | cold_saving_vs_fp8 | hot_minus_random | hot_within_random_ci |
|---|---|---|---|---|---|---|---|---|---|
| balanced | 4 | 0.250000 | 0.141107 | 0.054531 | 0.052318 | 0.057494 | 0.000000 | 0.086576 | False |
| balanced | 4 | 0.500000 | 0.204901 | 0.108801 | 0.105123 | 0.112547 | 0.000000 | 0.096101 | False |
| balanced | 4 | 0.750000 | 0.217029 | 0.163231 | 0.159951 | 0.166132 | 0.051668 | 0.053798 | False |
| balanced | 4 | 1.000000 | 0.217155 | 0.217155 | 0.217155 | 0.217155 | 0.217155 | 0.000000 | True |
| balanced | 8 | 0.250000 | 0.172883 | 0.054851 | 0.048759 | 0.059158 | 0.000000 | 0.118032 | False |
| balanced | 8 | 0.500000 | 0.212450 | 0.108401 | 0.102259 | 0.115171 | 0.000000 | 0.104049 | False |
| balanced | 8 | 0.750000 | 0.215726 | 0.161992 | 0.157664 | 0.165625 | 0.001764 | 0.053734 | False |
| balanced | 8 | 1.000000 | 0.215726 | 0.215726 | 0.215726 | 0.215726 | 0.215726 | -0.000000 | True |
| balanced | 16 | 0.250000 | 0.159080 | 0.058550 | 0.056506 | 0.060168 | 0.000000 | 0.100530 | False |
| balanced | 16 | 0.500000 | 0.214162 | 0.115525 | 0.112735 | 0.117788 | 0.000000 | 0.098637 | False |
| balanced | 16 | 0.750000 | 0.229890 | 0.172833 | 0.170799 | 0.175293 | 0.000000 | 0.057057 | False |
| balanced | 16 | 1.000000 | 0.230190 | 0.230190 | 0.230190 | 0.230190 | 0.230190 | -0.000000 | True |
| hotspot | 4 | 0.250000 | 0.118514 | 0.061891 | 0.052362 | 0.073004 | 0.000000 | 0.056623 | False |
| hotspot | 4 | 0.500000 | 0.235943 | 0.123923 | 0.113922 | 0.132600 | 0.011481 | 0.112020 | False |
| hotspot | 4 | 0.750000 | 0.248237 | 0.186500 | 0.181701 | 0.192158 | 0.129723 | 0.061737 | False |
| hotspot | 4 | 1.000000 | 0.248237 | 0.248237 | 0.248237 | 0.248237 | 0.248237 | -0.000000 | True |
| hotspot | 8 | 0.250000 | 0.102751 | 0.061705 | 0.054033 | 0.067600 | 0.000000 | 0.041046 | False |
| hotspot | 8 | 0.500000 | 0.205595 | 0.123118 | 0.118163 | 0.127566 | 0.040790 | 0.082477 | False |
| hotspot | 8 | 0.750000 | 0.246386 | 0.184599 | 0.181472 | 0.187985 | 0.143635 | 0.061787 | False |
| hotspot | 8 | 1.000000 | 0.246386 | 0.246386 | 0.246386 | 0.246386 | 0.246386 | 0.000000 | True |
| hotspot | 16 | 0.250000 | 0.142838 | 0.060628 | 0.057173 | 0.065813 | 0.000000 | 0.082210 | False |
| hotspot | 16 | 0.500000 | 0.241554 | 0.121467 | 0.117644 | 0.125716 | 0.000000 | 0.120087 | False |
| hotspot | 16 | 0.750000 | 0.242007 | 0.181766 | 0.179165 | 0.184431 | 0.001439 | 0.060241 | False |
| hotspot | 16 | 1.000000 | 0.242007 | 0.242007 | 0.242007 | 0.242007 | 0.242007 | -0.000000 | True |

## 判定

- balanced / jobs=4 / frac=0.25: hot-random=+0.0866 (端口感知有独立价值)
- balanced / jobs=4 / frac=0.50: hot-random=+0.0961 (端口感知有独立价值)
- balanced / jobs=4 / frac=0.75: hot-random=+0.0538 (端口感知有独立价值)
- balanced / jobs=4 / frac=1.00: hot-random=+0.0000 (端口感知无独立价值(落在random CI内))
- balanced / jobs=8 / frac=0.25: hot-random=+0.1180 (端口感知有独立价值)
- balanced / jobs=8 / frac=0.50: hot-random=+0.1040 (端口感知有独立价值)
- balanced / jobs=8 / frac=0.75: hot-random=+0.0537 (端口感知有独立价值)
- balanced / jobs=8 / frac=1.00: hot-random=-0.0000 (端口感知无独立价值(落在random CI内))
- balanced / jobs=16 / frac=0.25: hot-random=+0.1005 (端口感知有独立价值)
- balanced / jobs=16 / frac=0.50: hot-random=+0.0986 (端口感知有独立价值)
- balanced / jobs=16 / frac=0.75: hot-random=+0.0571 (端口感知有独立价值)
- balanced / jobs=16 / frac=1.00: hot-random=-0.0000 (端口感知无独立价值(落在random CI内))
- hotspot / jobs=4 / frac=0.25: hot-random=+0.0566 (端口感知有独立价值)
- hotspot / jobs=4 / frac=0.50: hot-random=+0.1120 (端口感知有独立价值)
- hotspot / jobs=4 / frac=0.75: hot-random=+0.0617 (端口感知有独立价值)
- hotspot / jobs=4 / frac=1.00: hot-random=-0.0000 (端口感知无独立价值(落在random CI内))
- hotspot / jobs=8 / frac=0.25: hot-random=+0.0410 (端口感知有独立价值)
- hotspot / jobs=8 / frac=0.50: hot-random=+0.0825 (端口感知有独立价值)
- hotspot / jobs=8 / frac=0.75: hot-random=+0.0618 (端口感知有独立价值)
- hotspot / jobs=8 / frac=1.00: hot-random=+0.0000 (端口感知无独立价值(落在random CI内))
- hotspot / jobs=16 / frac=0.25: hot-random=+0.0822 (端口感知有独立价值)
- hotspot / jobs=16 / frac=0.50: hot-random=+0.1201 (端口感知有独立价值)
- hotspot / jobs=16 / frac=0.75: hot-random=+0.0602 (端口感知有独立价值)
- hotspot / jobs=16 / frac=1.00: hot-random=-0.0000 (端口感知无独立价值(落在random CI内))

## 解读边界

- 这仍是 bandwidth-only 解析回放，不含 collective、queueing、pack/unpack、kernel。
- 若 `hot` 落在 `random` 的 95% CI 内，说明此前报告里 receiver-aware 相对 random 的
  "额外收益"主要来自 remote-vs-local 的选择，而不是"识别具体哪个端口更热"。
- 若 `hot` 稳定超出 `random` CI 且 `hot - cold` 差距明显，说明端口热度信息在
  "已经限定只压 remote"之后仍有独立、可复现的边际价值，receiver-aware 的
  claim 可以保留，但措辞必须改为"限定 remote 后进一步做端口热度选择"，
  而不是笼统的"receiver-aware 比 random 好 X%"。

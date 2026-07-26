# Bounded-Incast Gate 0：route-real 结构筛查

## 判定

`PASS_STRUCTURE_ONLY_NOT_MECHANISM_HEADROOM`

该结果允许继续捕获带时间戳的 expert-ready trace，但**不允许**声称存在真实
incast、receiver queue tail 或调度收益。

## 已验证问题

对 clean-v2 calibration 中完整校验过的 native route identity，按
`(token_position, token-owner receiver)` 构造一个请求波次，检查 bounded-incast
机制的必要结构：一个 receiver 是否同时对应多个 join，且这些 join 的 siblings
是否来自多个真实 expert sender rank。

| 模型 | receiver-wave | 合格比例 | 每 join sender fan-in（median / P95） | 每 wave join | 每 wave sender |
| --- | ---: | ---: | ---: | ---: | ---: |
| OLMoE | 1024 | 100% | 6 / 7 | 8 | 8（最小 7） |
| LLM-jp | 1024 | 100% | 8 / 8 | 8 | 8 |

每个 OLMoE receiver-wave 有 64 条 contributions，LLM-jp 有 128 条；单一 sender
在一个 wave 中的 contribution 数中位数分别为 12 和 20。这说明真实 routing
identity 中存在强 many-to-one fan-in，且不是 CRQM 已否定的“一个 sender 跨多个
receiver 排序”对象。

## 为什么还不能宣称 headroom

波次同步来自分析对齐，并非日志中的真实 ready timestamp。当前证据没有证明：

- 八个请求在真实 continuous batching 中同时处于该 MoE layer；
- 不同 expert sender 的结果在 receiver buffer 生命周期内重叠；
- NIC/QP/receiver ingress 存在可由 admission order 改变的排队；
- join-aware policy 能胜过 EDF、sender round-robin、receiver-credit 和 request-FCFS。

因此不进入纯合成 queue-depth sweep；那会把假设的拥塞反向当成收益证据。

## 下一道快速门

捕获每条 contribution 的
`(request, layer, token, slot, sender, receiver, expert_ready_ts, enqueue_ts)`，保留
自然时间关系，并对 buffer credit `B={1,2,4,8}` 回放：

1. `RR-credit`：receiver 对 sender round-robin 发 credit；
2. `oldest-ready`：最早 ready contribution；
3. `EDF-credit`：只读 deadline/slack，不读 sibling bitmap；
4. `request-FCFS`：强 grouping baseline，允许读 request identity但不读完成 bitmap；
5. `Join-Deficit Credit`：优先能闭合 join，其次 missing count、slack、fairness debt；
6. offline causal oracle：仅作上界，不作可部署 baseline。

继续条件同时满足：

- 至少 10% 自然 receiver busy periods 含 `>=2 joins` 与 `>=2 senders`；
- candidate 相对最强非 oracle baseline 的首 credit action flip `>=20%`；
- 两模型至少在 `B=2/4` 的一个共同区间，join-closure CVaR95 改善 `>=10%`；
- request-FCFS 未覆盖 candidate，且 starvation/总 work 不增加。

否则判 `NO_GO_TEMPORAL_INCAST_OR_JOIN_INFORMATION_VALUE`，不进入多 GPU/RDMA 实现。

## 证据文件

- runner: `experiments/ric_clean_v2/explore_incast_census.py`
- output: `outputs/incast_route_census_v1.json`


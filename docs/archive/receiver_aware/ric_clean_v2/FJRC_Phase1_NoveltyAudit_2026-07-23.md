# Fork-Join Risk Credits Phase 1 Novelty Audit

状态：**CONDITIONAL GO TO ORACLE EXISTENCE TEST**  
日期：2026-07-23

## 入选命题

receiver credit、deadline scheduling 和 backend pressure feedback 本身都不是创新。FJRC
唯一可辩护的增量是：在 deadline、bytes、route、remaining work、topology 与 aggregate
backend pressure 相同的情况下，**causal keyed receiver state 与完整 MoE fork-join DAG
共同产生的 request-level marginal deadline risk**，是否仍能改变早期 admission，并且
不能被 EDF、least-laxity、join-SRPT 或普通 receiver credit 吃掉。

若 risk statistic 退化为 slack、remaining work、qdepth 或普通 shadow-price 改名，novelty
判定为无。

## 最近邻与边界

- Gimbal 已用 KV/prefill/queue/MoE expert pressure 与 source-aware traffic 做跨层 MoE
  scheduling：<https://arxiv.org/abs/2606.15177>。
- SIRD 已使用 receiver-issued credit 主动调度 incast，并可表达 message-size/SRPT priority：
  <https://www.usenix.org/system/files/nsdi25-prasopoulos.pdf>。
- Pyrrha 已针对包括 MoE traffic 的 incast/last-hop congestion 做快速流控：
  <https://www.usenix.org/system/files/nsdi25-liu-kexin.pdf>。
- deadline-aware coflow 已覆盖 grouped-flow admission/deadline scheduling；FJRC 不能仅凭
  “一个 request 有多条 flow”声明新问题。
- SwiftEP、MegaScale-Infer 与 FinDEP 已分别优化 MoE communication primitives、
  disaggregated EP 与 fine-grained task overlap；FJRC 不声称新的 collective/kernel。

因此论文故事必须是 application-semantic receiver information：transport credit 只承载
一个经 necessity test 证明不可替代的 fork-join risk statistic。

## 最强简单基线

1. immediate/FCFS；
2. global request EDF、least-laxity；
3. task SRPT、request/join remaining-work SRPT；
4. EDF then SRPT；
5. causal join-deficit/last-missing-first；
6. c-mu、age-service DRR；
7. aggregate receiver qdepth/backpressure/ordinary credits；
8. Gimbal-like backend/expert pressure；
9. deadline-coflow heuristic；
10. exact joint B oracle，拥有上述全部非-keyed causal information。

full-future C 只作 ceiling。若任一简单基线捕获 exact B-to-R0 gain 的 90% 或以上，FJRC
novelty/no-simple-baseline gate 失败。

## 决策

| 候选 | Novelty | Necessity | 5090 路径 | 决策 |
| --- | --- | --- | --- | --- |
| 普通 slack credit | 低 | EDF/least-laxity 可替代 | 可做但意义弱 | 淘汰 |
| receiver qdepth + deadline | 低至中 | Gimbal/backpressure 邻近 | 可做 | 淘汰 |
| full-drain receiver queue-map reorder | 无剩余 headroom | CRQM 已实测 0 | 已完成 | 判死 |
| fork-join marginal deadline-risk information | 条件中高 | 待 exact B-to-R0 gate | route-real + 5090 LUT + L2 replay | 条件入选 |
| risk credit + Energy-SLO | 未证明 | 两命题会互相掩盖 | 暂不做 | 不合并 |

## 停止规则

Oracle miss-risk gap <10%、absolute gap <2pp、first action flip <25%，或
EDF/least-laxity/join-SRPT 捕获 >=90% gain，立即判死。禁止通过放大 synthetic tail/fanout、
选择 straggler、增加 seed、免费 feedback、换 predictor/bandit、降低 alpha、放宽门槛或
合并 Energy-SLO 抢救。


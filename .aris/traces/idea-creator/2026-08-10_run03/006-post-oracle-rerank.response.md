# Post-oracle rerank

## Hypothesis update

- H2 selective stabilization action value：SUPPORTED。oracle 恢复 `37/43=86.05%` route distance；33 positive actions 中 31 full restoration。
- H3 oracle upper bound：SUPPORTED。相对 budget-matched global random 优势 `36.690625`；即使 random 已知正确 33 cells、只随机 rank，oracle 仍领先 `17.5`。
- H5 sparse budget：SUPPORTED。`33/240=13.75%` cells 恢复 86.05%；B=32 已得 reward 36。
- H7 stability budget as runtime resource：SUPPORTED on bounded proxy。cell allocation 与 rank allocation 均有独立价值。
- H4 online decision 与 H6 natural batching：OPEN。

33 positive cells 分布于 13 layers、22 experts，任一 expert 最多 3 次，却在 8 victims 上聚集。因此不应转向静态 layer/expert score table，更值得研究 request trajectory 或 pack relation。

## Systems-paper Top 3

1. `C07+C12+C08` ShapeLane / ShapeABI scheduler。
2. `C01+C02` StabilityBudget + sparse action allocator；从 diagnostic 升至第二。
3. `C06` RouteStress WitnessPatch；最合理的 non-scalar controller path。

RouteGuard/precision-island 暂时降至第四；C8 与 M1 actions 已有更直接 evidence。

## Next mechanism: SparseShapeLane

默认 contribution 走 native fast lane；少量 contribution 走 fixed C8 canonical lane；stability budget 限制 protected actions；未来选择规则由 held-out witness/pack relation 决定，不使用 margin/entropy/norm scalar score。

### Smallest bridge experiment

只在 33 个 oracle-positive cell-ranks 上比较：

1. oracle-selected rank 使用 M1；
2. 同一 rank 使用 fixed C8；
3. budget-matched shuffled rank 使用 fixed C8。

保持相同 R/U 和 downstream reward。预计 5–10 GPU 分钟。

- 支持：C8 保留大部分 M1 oracle reward，且无明显新增 harm；随后进入 fresh natural-pack WitnessPatch。
- 否定：C8 reward 大幅消失；只否定 C8 作为 single-contribution protected path，不否定 global C8 scheduler 或 M1 WitnessPatch。

## P0/P1

`P0=0`  
`P1=1`

P1 只影响证据层级：oracle config 在已知 MaxGate/shuffle `-3/+3` 后冻结，不能称所有候选 outcome-naive preregistration；不改变 `37/43`、33-action budget、两种 matched-random advantages 或 Top-3 更新。

`reviewer_model=gpt-5.6-sol`  
`reviewer_reasoning=xhigh`  
`review_independence=same-family`  
`acceptance_status=provisional`

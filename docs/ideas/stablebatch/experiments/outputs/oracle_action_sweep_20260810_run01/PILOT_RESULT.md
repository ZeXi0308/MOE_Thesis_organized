# StableBatch Single-Contribution Oracle Action Sweep

**状态**：`COMPLETE`  
**判定**：`STRONG_ORACLE_ACTION_VALUE_SIGNAL`  
**审计**：`WARN / P0=0 / P1=1`；WARN 只影响全候选 outcome-naive preregistration 层级，不改变穷举数值。  
**GPU**：单张 NVIDIA GeForce RTX 5090，BF16 eager，430.21 秒。

## 本轮唯一问题

> 假如 hindsight oracle 知道 downstream outcome，在 frozen same-cell 的“一个 contribution 走 M1、其余七个走 M64”动作面上，单贡献保护是否明显优于 no intervention 和等动作预算 random？

## 最小设计

- 16 个文档、240 个 victim-layer cells。
- 每个 cell 全量执行 `R`（all-M1 proxy）、`U`（all-M64）和 `A0..A7`（恰好保护一个 top-k rank）。
- 共执行 1920 个 candidate actions；所有 action outcome 完成后才选 hindsight oracle。
- oracle 可 abstain；33 个正 action 全部复跑确认。
- random 对照同时包含：全局随机 B 个 cell-rank actions，以及在 oracle-selected cells 内随机 rank。
- MaxGate `-3` 与 frozen shuffle `+3` 只作为源结果 closure。

## 结果

| 指标 | 结果 |
|---|---:|
| `D(U,R)` / 可恢复 route distance | 43 |
| no intervention reward | 0 |
| uniform random：240 cells 各随机一个 rank | 2.25 (`9/4`) |
| forced oracle reward | 37 |
| abstaining oracle reward / action budget | 37 / 33 |
| recovery fraction | **86.05%** (`37/43`) |
| remaining route distance | 6 |
| budget-matched global random（B=33） | 0.309375 (`99/320`) |
| oracle − global random | **36.690625** |
| selected cells 内随机 rank（B=33） | 19.5 (`39/2`) |
| oracle − conditional random | **17.5** |
| positive cells / victims | 33 / 8 |
| full-restoration cells | 31 |
| MaxGate-v1 / frozen shuffle closure | -3 / +3 |

四个 frozen strong checks 全部通过。

## 解释

- **Supported**：在这个 frozen single-contribution action surface 上存在显著的 hindsight oracle action value。
- **Supported**：收益具有稀疏性；33/240 个正 actions 恢复了 37/43 的 route distance。
- **Supported**：cell selection 与 rank selection 都重要。只选中正确 cells、但随机 rank 的期望为 19.5；正确 rank 进一步提高到 37。
- **仍然失败**：MaxGate-v1 `-3 < +3`；该结果说明失败的是这个 selector，不是 action space。
- **未验证**：online selector、自然 prevalence、quality、serving latency/throughput、跨模型、EP/NCCL/RDMA。

## 唯一 P1 与证据层级

oracle config 在源 observable run 完成、MaxGate/shuffle `-3/+3` 已知之后冻结，并把这两个值写为 closure。因此它不能称为“全部候选 outcome 都未知时完成的 confirmatory preregistration”；准确表述是：

> 在 A0/MaxGate 与 frozen-shuffle 源结果已知、其余完整 rank-action sweep 执行前冻结的 retrospective oracle upper-bound experiment。

runner 仍然真实执行 `R/U/A0..A7` 全部动作，再读取 outcomes 选择 oracle；该 P1 不改变 37、33、86.05% 或两种 budget-matched random 数值，不需要重跑救结果。

## 下一研究动作

不要继续换 margin/entropy/norm 等 scalar score。下一步应寻找能表达 **cell × rank / pair-set relation** 的机制：优先利用 layer/expert stability profile、WitnessPatch relation 或 ShapeLane runtime state；任何 online selector 必须在 fresh held-out surface 上比较 conditional-random rank baseline。


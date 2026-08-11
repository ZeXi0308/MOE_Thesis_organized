# JoinStream 探索性 CPU Oracle 跟踪器

> 更新时间：2026-08-10 18:46:45 +08:00  
> 冻结计划：`refine-logs/EXPERIMENT_PLAN_20260810_182917.md`  
> 状态：`DONE / SUPPORT_ACTION_SPACE / ACTION_SPACE_FALSIFICATION_ONLY`  
> 权威边界：`EXPLORATORY / UNSEALED / NOT_A_PAPER_RESULT`

## Run ledger

| Run | Status | Evidence | Result |
|---|---|---|---|
| `JS-CONTRACT` | PASS | JoinStream 10/10 tests | atomic/stream milestone、final busy、curve/tax、baseline subset、M1、join release、replay、state/node closure 通过 |
| `JS-REGRESSION` | PASS | Python 3.14 下 38/38 tests | FrontierCredit 13 + CriticalSplit 15 + JoinStream 10 通过；未改 Critical/Bulk |
| `JS-ORACLE-8` | DONE | `artifacts/joinstream_pilot/20260810_184136/` | 8/8 strict flow improvement，4/4 tax=2 仍改善 |
| `JS-REPLAY` | PASS | 每 cell solved action trace replay | node exactly once，launch/service/objective/request completion 闭合 |
| `JS-NOVELTY` | CAUTION | same-family/provisional fresh review | C01 method 5/10, finding 6/10；C07/C12 不继续 |

## Mechanical result

| Metric | Result |
|---|---|
| strict improved cells | `8/8` |
| stream actually used | `8/8` cells，每个 `2` launches |
| cost-tolerant cells (`h=2 us`) | `4/4` |
| flow delta range | `0.100505 .. 30.000000 us` |
| deadline miss delta | `0` in 8/8 |
| launch count | baseline `4`, expanded `4` in every cell |
| h=0 total service | baseline == expanded |
| h=2 total service | expanded 多 `4 us`（两个 stream 各2 us） |
| max exact states | `3,872 < 500,000` |
| formal completion authority | none by design |

## Mechanism localization

8/8 expanded optimal traces 都是：

`layer0: atomic + atomic -> layer1: stream + stream`

因此本轮支持的是最后建模层的 early token/request completion，不是 next-layer compute overlap。不得把该信号改写为 GPU、自然 workload、EP/NCCL/RDMA、serving 或论文结果。

## Evidence and hashes

- Results：`artifacts/joinstream_pilot/20260810_184136/joinstream_results.json`  
  SHA256 `e4c6e2528a6397f7fe145529131c886ad13b40b7fd8ca02799676c165dd6234e`
- Summary：`artifacts/joinstream_pilot/20260810_184136/joinstream_summary.md`  
  SHA256 `63557a708d7f7c9495bff24914355897f368e8581885b4909844d1388d9edd74`
- Runner：`docs/ideas/bcrd/experiments/joinstream_full_dag_pilot.py`  
  SHA256 `fa10ef5c9729e29b06fc8261b6851b9db4a055e7038330436e1519a1287ab7af`
- Tests：`docs/ideas/bcrd/experiments/test_joinstream_full_dag_pilot.py`  
  SHA256 `04f03e8fca62536016bbb82002d3a529841b104559e3bac4eb9a718e286fdda3`
- Frozen plan：`refine-logs/EXPERIMENT_PLAN_20260810_182917.md`  
  SHA256 `bea88ade3c368a07becdd0cd1a09eacaabb131db906378e64639f397b4d5a8f4`

## P0/P1

1. **P0 physical-legality：resolved for abstraction, GPU unverified.** 已冻结 same-device release/acquire producer-consumer contract，不再是无法执行的虚构 state；真实性能仍需 GPU probe。
2. **P1 curve/order freeze：resolved.** curve、tax、canonical order 均在观察结果前冻结，Oracle 不选 milestone。
3. **P1 prior-art comparator：open for method claim, nonblocking for this Pilot.** FlashMoE/Event Tensor 强近邻使当前结果只能定位 action-space / join-aware finding。

## Bounded conclusion

JoinStream 在冻结 8-cell deterministic simulator 中创造了 atomic whole-batch completion 无法表达的 flow headroom，且在2 us emission tax 下仍存在。该 signal 只授权下一个最小 GPU completion-legality / measured-curve probe，不授权 online controller 或大规模评测。

## Next experiment

只测 M={2,4} 的 same-device persistent producer + per-row release flag + acquire consumer；采集真实 row milestones、notification tax、final finish 与 total service，再回灌同一 exact Oracle。若正信号消失，接受 GPU-level 负结果；不调 CPU curves 救结果。

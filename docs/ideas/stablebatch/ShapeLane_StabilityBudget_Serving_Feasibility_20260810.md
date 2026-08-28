## 1. State synchronization

本轮只使用已经封存的 StableBatch / ShapeLane / C8 结果和 264-action ledger，没有读取 fresh selector outcome，没有训练新 selector，也没有运行 GPU。

- 冻结并停止：FrontierCredit 当前 formulation、r2 forensic、A/B/C 探针、KV-axis 2x2、MaxGate-v1、C8 oracle sweep、fixed-C8 transfer。MaxGate-v1 的 `-3` 对 frozen shuffle `+3` 只是否定该 selector，不否定 action space。
- 当前 action-space 事实：240 cells、1920 个 M1 proxy actions、33 个正 oracle cell-rank actions，恢复 `37/43=86.05%`；它是单模型 prompt-forward upper bound，不是在线 policy。
- 当前 execution 事实：fixed-C8 correctness 在所测 contexts 上成立；C8 是 canonical numerical execution state，不是 M1、M64 或 ground truth。
- 当前 transfer 事实：C8 same-rank 为 `31 recovered / 6 harmed / +25`，exact-random 为 `22.25 / 4 / +18.25`，C8 oracle 为 `36 / 0 / +36`；transfer ratio `83.78%`、rank-specificity gap `6.75`，结论为 `GO / CASE_A`。
- 仍未成立：outcome-naive online selector、sparse native-runtime bridge、continuous-batching packing/fragmentation、真实 action latency、queue delay、controller overhead、TTFT/TPOT/P99、模型质量与 production deployability。

权威输入是 [C8 transfer result](experiments/outputs/c8_action_transfer_20260810_run01/PILOT_RESULT.md)、[264-action ledger](experiments/outputs/c8_action_transfer_20260810_run01/cell_results.jsonl)、[fixed-C8 correctness](experiments/outputs/shape_lane_correctness_20260810_run02/summary.json) 和 [dense continuous-cost result](experiments/outputs/shape_lane_continuous_cost_20260810_run02/summary.json)。

## 2. Runtime action contract

Action identity 是：

```text
a = (request/victim, target token, layer, top-k rank, expert_id)
```

这里的 `rank` 是 token 的 top-k rank `0..7`，不是 GPU/EP rank。

当前代码中的真实语义如下。

1. 输入是 target layer router 前的 focal BF16 hidden vector `[2048]`。runner 把它放到 slot 5，另加 7 个 zero rows，构造 `[8, 2048]` BF16 lane；随后对该 rank 对应的 expert 调一次 `expert(lane)`，只取 slot 5 的 raw output。实现见 [run_c8_action_transfer.py](experiments/run_c8_action_transfer.py) 的 `precompute_c8_replacements` 与 [run_shape_lane_correctness_pilot.py](experiments/run_shape_lane_correctness_pilot.py) 的 `build_lane_batch`。
2. C8 transfer 的候选 arm 是“一个 candidate rank 使用 C8 raw output，其余七个 target ranks 使用 M64 side-call outputs”；U 是八个 M64 side-call outputs。M64 是 operational proxy，不是 serving native/default batch。
3. 完整 prompt input shape 不变。patched forward 仍先执行原 native expert group，然后覆盖 target token/rank 的一个 raw row；原 router、expert identity、gate weight、其他 rows 和 `index_add_` combine 不变。因此当前实现是带重复计算的 offline output splice，不是 sparse serving dispatch。
4. 影响点位于 gate weight 之前的单个 raw expert contribution；恢复/伤害从 `layer+1` 的 downstream top-k membership 计算，相对 all-M1 operational reference，不是 ground truth 或质量恢复。

仓库中另有真正执行 zero-padding 的 universal executor：router/top-k 后把同一 decode epoch 的 `(layer, expert)` 所有 rows 在尾部 pad 到 C8，调用一次 expert，再取前 M 行 combine。它证明 fixed-C8 executor 可实现，但它 canonicalize 所有 occupied groups，不读取 per-row protection bit；因此不能直接实现 sparse rank-specific action。

需要补的最小 serving bridge 位于 router/top-k 与 expert dispatch 之间：

```text
RouteRow {
  request_id, decode_step, row_id,
  layer, expert_id, topk_rank,
  hidden, gate_weight,
  protected, ready_ts, deadline,
  dtype, device, backend_plan, plan_epoch
}

original expert group
  -> native default remainder
  + protected rows grouped by
    (layer, expert, C8, dtype, device, backend_plan, plan_epoch, readiness_window)
  -> zero-pad each protected group to 8
  -> expert(C8)
  -> discard dummy outputs
  -> scatter both paths by row_id/topk_rank with the original gate weights
```

若一个原 expert group 有 `P` 个 protected rows、`U` 个 default remainder rows，则结构上可推导：

```text
ShapeLane logical calls = ceil(P / 8)
dummy rows              = 8 * ceil(P / 8) - P
total logical calls     = 1[U > 0] + ceil(P / 8)
delta logical calls     = 1[U > 0] + ceil(P / 8) - 1
```

“logical expert call”不等于底层 CUDA kernel launch 数。只有其他已经被选中的 protected rows 才能作为可复用 companions；当前 correctness 不能授权把任意 unprotected row 当免费 filler 并复用其输出。

Preconditions：当前 stack/plan 与 ShapeABI certificate 相符；目标 row 尚未进入 default expert call；同 lane rows 的 layer/expert/device/dtype/plan/epoch 相同；原 gate weight 与 scatter identity 可保留；预算与 deadline 允许。

Fallback：任何 capability mismatch、预算不足、deadline 到期、`P>8` 无合法 chunk plan 或 lane 无法形成时，abstain 并走 native fast path。当前代码尚未实现这个 sparse split/merge/fallback bridge。

## 3. Cost decomposition

| Cost | 状态 | 当前能说什么 |
|---|---|---|
| direct execution | **measured only for other scopes; sparse delta unknown** | C8 side-call tensor shape是 `[8,2048]` BF16；input/output payload 各 32 KiB 可推导，但不等于实际 memory traffic。Correctness 的 zero-context median `0.583 ms` 是每 cell 顺序执行 8 个 rank-specific C8 calls 的总时间，不是单 action delta。dense universal replay 中 fixed-C8 expert GPU time 比 native 高 `13.61%`、比 serial-M1 低 `15.09%`；不能赋给 sparse action。 |
| padding | **derivable; dense measured** | singleton action 为 1 useful + 7 dummy，dummy ratio `87.5%`。dense universal replay 实测 padding `77.41%`，但不是 sparse workload。 |
| launch | **logical count derivable; CUDA launches unknown** | immediate singleton 至少一个 C8 logical call；若原 group 仍有 remainder，结构上相对原 group增加一个 logical call。现有 ledger 没有 CUDA launch trace。 |
| fragmentation | **structural formula derivable; realized cost unknown** | 原 group 变成 default remainder + C8 group + fallback。是否真的多一次 group/call取决于原 natural M、同 group selected rows 和 remainder，264 ledger 均未记录。 |
| queue delay | **unknown** | 三种动作必须区分：`wait-to-fill`、`pad-and-launch`、`split-and-launch`。ledger 没有 ready/enqueue/dispatch/deadline，因此不能估计等待。 |
| lost batching opportunity | **unknown** | protected row 从 natural group 抽离后对 default GEMM efficiency、stream overlap 和后续 batch 的影响未记录。 |
| controller | **complexity derivable; latency unknown** | feature attach/lane enqueue 可做 O(1)，预算分配可用 bounded heap 做 O(|A| log B)；没有 CPU timing，不能写成微秒或百分比。 |

已有 dense result 的 `NO_GO_D10_HEADLINE_COST` 是重要边界：全局 canonicalization 昂贵，正好支持研究 sparse exception path；它不能回答 sparse bridge 的 direct delta，更不能升级为 serving TTFT/TPOT/P99。

## 4. StabilityBudget formulation

避免把 harm double-count：令 `r_a` 为 gross recovered utility、`h_a` 为 harm，而不是先把 `r_a-h_a` 再减一次 harm。runtime 使用外部主 selector 提供的 outcome-naive estimates `r_hat_a, h_hat_a`；本轮 retrospective 才使用 ledger outcomes。

```text
x_a in {0,1}                         action selected or abstained
g(a)                                 mutually-exclusive cell
d(a)                                 request/document
k(a)                                 ShapeLane compatibility key
c_a                                  marginal runtime-cost estimate

maximize
    sum_a x_a * (r_hat_a - lambda * h_hat_a)
    - mu * RuntimeCost(x)

subject to
    sum_a x_a * c_a <= C             runtime-cost budget
    sum_{a:g(a)=g} x_a <= 1          at most one rank per cell
    sum_a x_a <= B                   cardinality ceiling, not a target
    sum_a x_a * h_hat_a <= H         harm budget
    Wait(k) <= W_k                    bounded queue wait
    Dummy(k) <= P_k                   padding/PadCap
```

由于 coalescing 让成本非加性，最终 `RuntimeCost(x)` 应按 lane group 计费：`ceil(P_k/8)` calls、dummy rows、default remainder split 和 wait；在 direct-cost 尚未测量前，`c_a` 只能是 normalized proxy。

当前不需要通用 MILP。可按以下顺序实现：

1. Top utility：按 `r_hat-lambda*h_hat` 排序；
2. Utility/cost：按 `(r_hat-lambda*h_hat)/c_hat` 排序；
3. Harm-constrained greedy：先拒绝超过 `H` 的动作，再按 ratio 入选；
4. 264-action exact multiple-choice knapsack 仅作 retrospective upper bound。

Fallback 是优化模型中的合法 `x_a=0`，不是 failure：没有正的 risk-adjusted marginal value、cost cap 不足、lane 过期或 ShapeABI 失效时都走 native。

## 5. Offline feasibility result

可复现 analyzer 与完整结果位于 [stability_budget_offline_cost_20260810](../../../artifacts/stablebatch/stability_budget_offline_cost_20260810/README.md)。它只读取 frozen 264-action ledger 与 summary；`derived_actions.jsonl` 恰好 264 行，所有 allocator 强制每 cell 至多一个 rank，未读取 fresh selector outcome。真实 cost 状态是 `UNKNOWN_NOT_MEASURED`。

五个场景均使用同一无量纲 proxy 单位：uniform `c=1`；rank sensitivity `c=1+rank/7`；fragmentation `c=1+alpha*d`，`alpha={0.25,1,3}`，其中 `d` 是忽略 readiness 的 structural dummy proxy。它们不是 ms、FLOPs、bytes 或 launch 数。

在 normalized `C=33`、`B<=33`、`lambda=1` 下的 exact multiple-choice upper bound：

| Scenario | Actions | Proxy cost | Recovered | Harmed | Utility | break-even `mu=utility/cost` |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 30 | 30.000 | 36 | 0 | 36 | 1.200 |
| rank-dependent | 29 | 32.571 | 35 | 0 | 35 | 1.075 |
| fragmentation-low | 27 | 32.031 | 33 | 0 | 33 | 1.030 |
| fragmentation-medium | 19 | 32.250 | 25 | 0 | 25 | 0.775 |
| fragmentation-high | 11 | 32.750 | 17 | 0 | 17 | 0.519 |

所以在所有离散 proxy 场景中都有非空 utility-cost Pareto；`mu` 也只具有“route points / proxy cost unit”的意义。Top utility、utility/cost、harm-constrained greedy 与 exact DP 在本数据和测试 caps 上得到相同 utility，这是 outcome surface 简单造成的 retrospective tie，不证明在线 greedy 最优。

Harm sensitivity：冻结 same-rank 33-action reference 的 `recovered=31, harmed=6`；`lambda={1,2,4}` 时 risk-adjusted utility 为 `{25,19,7}`。exact retrospective upper bound在三个 lambda 下都选到 harm-free actions，因此 selection 不变；这只说明 outcome-aware oracle 可以避害，不能替代 harm predictor 或 `H` constraint。

Sparsity必须用两个分母说明：

- 原 240-cell oracle surface：33/1920 actions=`1.72%`，33/240 cells=`13.75%`，8/16 request-documents=`50%`。per-layer positive-cell rate从 `0/16` 到 `6/16=37.5%`；22/64 experts 被命中，任一 expert 最多 3/33 actions。它看起来是 action/cell sparse，但仍是 prompt-forward oracle cohort，不是自然 serving prevalence。
- C8 transfer 是正例富集 cohort：uniform exact reference 选 30/264 actions=`11.36%`，却覆盖 30/33 cells=`90.91%` 和 8/8 requests；same-rank 覆盖 33/33 cells。因此不能从该 cohort 声称 workload-level low-frequency exception path。

Grouping/coalescing opportunity 当前很弱：same-rank 33 actions 只有 32 个不同 `(layer,expert,C8)` groups，忽略所有 readiness 后也只减少 1 个 lane（`3.03%`），fill `12.89%`、dummy lower bound `87.11%`；30-action C8 retrospective oracle 是 30 groups，完全无 reduction。ledger 没有 readiness window，所以这些已经是 structural optimistic upper bound，不是可执行 grouping。

对核心问题的答案是：fixed `B=33` 对 isolated C8 tensor shape 是合理的第一层 abstraction，但会掩盖 natural-M remainder、readiness、split 和 queue 导致的真实成本差异；系统接口下一步必须保留 `B` ceiling，同时以 `C/H/PadCap/WaitCap` 做真正约束。

## 6. Top system design

| Design | 优点 | 当前证据下的主要风险 | 选择 |
|---|---|---|---|
| Immediate Sparse ShapeLane | 不等待，控制路径最简单 | singleton `87.5%` dummy，可能多一个 remainder call | 只作高价值 escape path |
| Coalesced ShapeLane | 理论上可摊薄 dummy/launch | 当前 ledger 几乎无 shape-only overlap，readiness 完全未知 | 不作 headline/default |
| Opportunistic ShapeLane | 只在兼容 rows 已经 co-ready 时启用，成本最低 | 会 abstain，coverage 下降 | 作为默认保护策略 |
| Hybrid | opportunistic coalescing + cost-capped immediate escape + native fallback | 需要 sparse bridge 与 direct-cost calibration | **Top design** |

Top design 是 **Hybrid Opportunistic-First ShapeLane**：

```text
router/top-k rows
  -> main outcome-naive selector supplies candidate/risk only
  -> StabilityBudget allocator applies C/H/B
      -> unselected: native fast lane
      -> selected and compatible already-ready: coalesced C8 lane
      -> selected, not coalesced, high marginal value and PadCap/LaunchCap allow:
           immediate padded C8 lane
      -> otherwise: abstain -> native fast lane
  -> original gate-weight scatter/combine
```

当前 `WaitCap` 默认应为 0；只有 future runtime ledger 证明短 bounded window 有收益后才允许 wait-to-fill。这样不靠尚未观察的 grouping 撑论文，也不让 immediate singleton 成为常态。

最小 scheduler interface 只需四个操作：`submit(RouteRow)`、`quote_cost(action, current_groups)`、`allocate(C,H,B)`、`flush_or_fallback(deadline)`。真正的系统贡献是 risk/harm/cost-aware numerical-stability resource management 与双路径 expert batching，而不是另一个 rank predictor。

## 7. Classification

**SYSTEM_ACTION_VALID_COST_UNRESOLVED**

理由：fixed-C8 raw-contribution action 的输入、位置、shape、gate/merge 语义和 fallback 都能被精确映射；仓库中的 universal executor也证明 C8 zero-pad/expert/slice 原语可执行。与此同时，当前 sparse implementation 仍只是 offline splice，尚无 native-remainder split/merge bridge；action-transfer 的其他七 ranks 是 M64 proxy 而不是 native background，264 ledger 也没有 paired sparse cost/readiness。因而不能升级为 `SYSTEM_ACTION_SPACE_PLAUSIBLE` 的真实 serving 结论，也没有证据支持 `ACTION_VALID_BUT_SYSTEM_COST_DOMINATES`。

本轮离线结论没有 P0/P1：action/outcome 对齐、cost unit 隔离、每 cell 至多一 rank、fresh-outcome isolation 均满足。bridge 与真实成本是下一阶段未验证项，不是对既有 transfer 的否定。

## 8. Paper story

**Observation** → 在单模型、单 GPU、BF16 eager 的受控 prompt-forward surface 上，physical expert shape 能改变 downstream MoE route semantics；fixed-C8 在所测 contexts 中提供 canonical numerical state。

**Abstraction** → 把 numerical stability 从“全局模式开关”提升为受预算约束的 runtime resource：`StabilityBudget = action value + harm + execution cost + wait/padding caps`。

**Mechanism** → 外部 outcome-naive candidate signal、harm-aware allocator、ShapeABI certificate、protected-row lane former、opportunistic coalescer、cost-capped immediate escape、native fallback。

**System** → 在 router/top-k 与 expert dispatch 之间实现双路径 runtime：大多数 rows 保留 native continuous batching，少量 selected contributions 进入 canonical C8 ShapeLane，原 gate/identity/combine 语义保持不变。

**Evaluation** → action validity 用现有 C8 transfer；allocator tradeoff 用 frozen 264 ledger；直接成本用下一节唯一 microbenchmark；只有 bridge 实现后才测 continuous batching 的 padding/fragmentation/queue/TTFT/TPOT/P99。当前论文主张上限是“可实现的 sparse numerical-stability abstraction + 未解决的 serving cost”，不是已完成 serving system。

## 9. Next minimal experiment

唯一下一实验是 **direct expert-cost microbenchmark**；只在主 selector Gate 阳性后运行，不重跑 selector，不产生新 action outcome。runner 已写好但本轮未执行：[run_shape_lane_direct_cost_microbench.py](experiments/run_shape_lane_direct_cost_microbench.py)。代码同步到远端 `/root/autodl-tmp/moe_work` 后执行：

```bash
cd /root/autodl-tmp/moe_work
python3 docs/ideas/stablebatch/experiments/run_shape_lane_direct_cost_microbench.py \
  --model-path /root/autodl-tmp/models/olmoe \
  --ledger docs/ideas/stablebatch/experiments/outputs/c8_action_transfer_20260810_run01/cell_results.jsonl \
  --cell-ids cell-005 cell-012 \
  --layer 7 \
  --expert 43 \
  --canonical-m 8 \
  --focal-slot 5 \
  --warmup 100 \
  --repeats 1000 \
  --output docs/ideas/stablebatch/experiments/outputs/shape_lane_direct_cost_post_selector_run01/result.json
```

固定输入是 ledger 中共享 `layer=7, expert=43` 的 `cell-005` 与 `cell-012` 两个实际 hidden rows；runner 从各自 16-token prompt 重建 BF16 `[2,2048]` input并记录 input hash。四个 arms 是：native M2、singleton C8 direct call、native-M1 remainder + singleton padded-C8 full split、two-row coalesced C8。

指标只包括每 arm 1000 次 CUDA-event median/P90/P99、logical calls、ratio vs native、ms/protected-row，以及：

```text
Delta_single = median(padded_split) - median(native_M2)
Delta_coalesced_per_action = (median(coalesced_2) - median(native_M2)) / 2
```

若四臂均产生固定 input hash、有限输出和稳定 timing，则实验有效；`Delta_coalesced_per_action < Delta_single` 表示 coalescing 确实摊薄 direct cost。若 padded split 与 coalesced 两者都超过后续明确的 runtime cost budget，才支持“direct cost dominates”；否则只把实测 delta 写回 `c_a`。这个 microbenchmark仍不回答 queue、controller、TTFT/TPOT/P99 或 serving deployability。

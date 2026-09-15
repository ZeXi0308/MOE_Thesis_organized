## Gate classification

`HARM_DOMINATES`

## Frozen setup

- Train documents: old sealed indices 16–31; 240 unconditional cells; 1,920 broad fixed-C8 labels.
- Fresh documents: independently sealed indices 0–15; 16 document-disjoint windows; 240 cells and 1,920 actions.
- Features: layer/expert/rank one-hot, gate weight/share/gap-to-min, top-k mass, normalized entropy, and cutoff margin; no historical outcome-derived sensitivity.
- Ridge: L2 alpha `1.0`, unpenalized intercept, train-only population mean/std, no search or retuning.
- Budget: exact `B=33`.
- Constraint: choose maximum predicted-utility rank within each cell, tie by lowest rank; then global top-B cells, at most one action per cell.
- Freshness boundary: C8-outcome-naive. A separate fresh M1 result existed but was not bound, read, used as a feature, or used for tuning.

## Result table

| Policy | Actions | Recovered | Harmed | Net |
|---|---:|---:|---:|---:|
| Global matched random, exact expectation | 33 | 4.8296875 | 7.390625 | -2.5609375 |
| Selector-cell matched random-rank, exact expectation | 33 | 10.5 | 9.875 | 0.625 |
| Frozen ridge selector | 33 | 9 | 15 | -6 |
| C8 oracle exact-B | 33 | 58 | 0 | 58 |
| C8 oracle at-most-B | 33 | 58 | 0 | 58 |

## Generalization evidence

- Selector positive-net documents: `4/16`.
- Selector above cell-matched random-rank documents: `4/16`; above global matched random documents: `11/16`.
- Median document-level selector minus cell-matched rank gap: `0`.
- Selected document coverage: `15/16`; selected positive/zero/negative actions: `4/23/6`.
- Selected rank distribution for ranks 0–7: `[6, 8, 1, 5, 2, 5, 3, 3]`.
- Cell-selection gain: `+3.1859375`; rank-selection gain: `-6.625`.
- Rank-headroom denominator: `57.375`; rank-headroom capture: `-11.5468%`.
- Oracle gap: `64`; oracle positive-net document count: `11/16`.

## Mechanistic interpretation

The fixed-C8 action space generalizes because exact-B oracle obtains `+58` with zero harm across 11 positive-net documents. The full fresh C8 surface is risky rather than uniformly helpful: its aggregate net is `-149`, and global matched random is negative. The ridge identifies a cell set with positive average opportunity, as shown by the `+3.18594` cell-selection gain. It then chooses worse-than-random ranks inside those cells, producing rank gain `-6.625` and headroom capture `-11.55%`. Its 15 harms exceed 9 recoveries, so the frozen priority mechanically selects `HARM_DOMINATES`. This falsifies the current outcome-naive alpha-1 ridge policy, not the existence of useful fixed-C8 actions.

## System implication

**NO-GO for the current StabilityBudget scheduling prototype.** Retain the ShapeLane/fixed-C8 action primitive, but replace the current unweighted selector objective with a pre-registered risk-aware policy on new data; do not rescue this result by changing the current features, alpha, B, or harm weight.

## Scope of conclusion

This is single-RTX-5090, one-OLMoE-revision, same-cell route-proxy evidence. It does not establish model-quality improvement, complete output recovery, serving SLO gains, production readiness, multi-model generalization, or multi-GPU/EP/NCCL/RDMA behavior.

## Next minimal experiment

使用现有两套完整 C8 action surface，进行一次不占 GPU、不得形成确认性主张的 selector failure decomposition。将 action utility 分解为 cell opportunity 与 within-cell rank residual，并分别评估 cell head、rank-residual ridge、harm head 和 hierarchical static rank profile 的 train→fresh transfer。依据结果只选择一个新 policy：若 profile rank gain 为正，则在全新 document-disjoint cohort 上预注册 `Hybrid CellGate + ProfiledRank-v1`；若 profile 与 harm prediction 均无效，则停止监督式 selector 路线，转向 `WitnessPatch / budgeted probing`。

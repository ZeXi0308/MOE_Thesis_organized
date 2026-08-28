# Post-GPU rerank

固定 shape 分支从“可测试假设”推进为“已通过局部 correctness Gate 的机制候选”，但没有证明 serving 收益或模型语义质量。

## Hypothesis tree update

直接支持的是：对已有 M1/M64 shape sensitivity 的 focal contributions，固定 expert-row physical shape 为 `C=8`，可以消除 focal output 对 companion identity、row slot、zero padding 和本实验串行 expert call order 的依赖；相同 C8 output 注入后，下游 route 与 final logits 也保持一致。

证据为 21 cells、16 victims、32/32 prior-sensitive hashes reproduced、每 cell `4 contexts × 8 ranks × 3 repeats`，raw/post-combine/downstream-route/final-logit mismatch 均为 0。独立重算同样得到 21/16、32 targets、0 raw failures、0 downstream failures、0 native-noop failures。

H6 canonical lane scheduler 和 H7 runtime shape choice 被增强。H3 single-contribution oracle、H4 online selector、H5 sparse budget，以及 continuous-decode cost/quality 均未更新。

重要边界：32 个 prior-sensitive contributions 的 C8 hash 只有 1 个等于 M1、0 个等于 M64、31 个与二者都不同。当前证明的是 fixed-C8 context invariance，不是 M1 equivalence 或 canonical ground truth；下一实验必须同时观察语义差异，不能只测速度。

## Systems-paper Top 3

1. `C07+C12` ShapeABI / PadCap，吸收 C08 作为 scheduler 模块。固定 C8 lane + deterministic slot/padding + deadline-aware scheduler；唯一已有真实 GPU mechanism-correctness signal 的候选。
2. `C06` RouteStress WitnessPatch。若广覆盖 C8 成本过高，测试 relation-aware sparse pack surgery。
3. `C13+C05` RouteGuard stable-kernel mux / precision island。局部支付稳定执行成本，但仍缺 stable path/action-value evidence。

`C01` oracle sweep 降为重要 diagnostic，不再是 systems headline；C8 lane 已绕开 MaxGate/single-contribution selector。

## Experiment priority

1. Headline：C8 continuous-decode minimal Pareto Gate。比较 native、serial M1、global batch-invariant path、C8 lane，同时记录 policy 内 invariance、相对明确 behavior reference 的 route/logit或最小质量差异、padding/wait、TPOT/P99、goodput。
2. Diagnostic：C01 oracle single-contribution sweep。只用于判断 selective fallback 是否值得保留，不阻塞 C8 Gate；budget-matching P1 必须保持修正。
3. Backup headline：C06 natural-pack WitnessPatch。仅当 C8 lane 等待/填充成本不可接受时优先执行。

## P0/P1

`P0=0`  
`P1=0`

没有发现会改变“在这 21 个 enriched cells 上，固定 C8 对所测 context 保持 bitwise invariant”这一结论的问题。Enriched targets、非 M1 equivalence、side-call timing 和串行 call-order 都是 claim boundaries，不是当前 correctness Gate 的 P0/P1。

`reviewer_model=gpt-5.6-sol`  
`reviewer_reasoning=xhigh`  
`review_independence=same-family`  
`acceptance_status=provisional`

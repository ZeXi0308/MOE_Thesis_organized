# Focused Novelty Input — N09 / N06 / N01

> 截止：2026-07-29  
> 规则：未找到直接重合不等于已证明新颖；“把 X 用到 MoE”本身不构成新意。

## N09 MaxRoute

### Core claims

1. 用 MoE 每层 fan-out/join 与多层/decode 串联的 max-plus semantics，生成 expert/rank/link fault hypothesis 的 E2E latency signature，而不是 additive incidence + NNLS。
2. 在执行诊断前，对 exact natural routes 给出 k-diagnosability / indistinguishable-class certificate，并编译最小 canary codebook。
3. 目标 fault 是低于 fail-stop/liveness timeout 的 slow-but-responsive expert/rank/link；方法必须 fail-closed 区分正常 route/load skew 与不可分故障类。

### Closest primary work

| Work | Direct overlap | Remaining claimed delta |
|---|---|---|
| [NetBouncer, NSDI 2019](https://www.usenix.org/conference/nsdi19/presentation/tan) | active path probes、identifiability、gray device/link localization | binary/multiplicative network paths，非 learned MoE routes/fork-join latency |
| [deTector, ATC 2017](https://www.usenix.org/conference/atc17/technical-sessions/presentation/peng) | probe-matrix coverage/identifiability + greedy path selection | loss localization in known DCN topology，非 expert/rank/link hypothesis classes |
| [Gestalt, ATC 2014](https://www.usenix.org/conference/atc14/technical-sessions/presentation/mysore) | unified fault hypotheses under uncertain dependencies/noise/covering relations | 通用 networked systems，强烈压缩“只做 codebook/localizer”的新意 |
| [Link Delay Estimation via Expander Graphs](https://arxiv.org/abs/1106.0941) | sparse delay inversion + recoverability conditions | additive path delays，非 MoE max-plus/barrier/coalescing |
| [max-plus fork-join delay bounds](https://arxiv.org/abs/1512.08354) | multi-stage fork-join max-plus semantics | 分析 delay bounds，不做 fault localization/probe design |
| [GREYHOUND/FALCON, ATC 2025](https://www.usenix.org/conference/atc25/presentation/wu-tianyuan) | sub-job GPU/link fail-slow detection and active component benchmark | training repetition + intrusive benchmark，非 natural request route inverse problem |
| [StriaTrace, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/wu-haonan) | online LLM inference rank/phase/kernel tracing and diagnosis | 未展示 route-incidence EP expert localizer；额外 telemetry 可能使 N09 价值消失 |
| [Tarragon](https://arxiv.org/abs/2601.01310) | expert-worker delayed response、liveness probe、failure recovery | fail-stop/timeout，非 5–20% sub-timeout slowdown |
| [GEM](https://arxiv.org/abs/2605.19945) | route load + GPU latency variability in MoE inference | variability is directly profiled and used for placement，不做 online inverse localization |

### Decisive novelty risk

如果 max-plus compiler 只是已有 fork-join algebra + NetBouncer/Gestalt hypothesis search 的直接组合，或者需要 per-rank/per-kernel tracing 才能分开 fault classes，N09 应 `ABANDON`。

## N06 RECAP

### Core claims

1. 对 route-incidence inverse model 先做 exact-semantics model-adequacy audit，不在未检验 additivity 时直接拟合 localizer。
2. 从 natural cached-decode states 编译“同 per-expert marginals/异 co-activation”与“单 incidence difference”的 route-equivalent counterfactuals，分解 main、rows、interaction 与 barrier/max effects。
3. 输出可辨识模型成立的最大 route/load regime 与等价类，而不只是报一个回归误差。

### Closest primary work

| Work | Direct overlap | Remaining claimed delta |
|---|---|---|
| [Coz causal profiling](https://www.usenix.org/publications/login/summer2016/curtsinger) | active counterfactual performance experiments | virtual speedup/code regions，无 MoE route-equivalent adequacy boundary |
| [More intervention now!](https://www.usenix.org/legacy/events/hotos11/tech/final_files/Goldszmidt.pdf) | causal blueprint + controlled systems interventions | position/blueprint，无 exact top-k route matching theorem |
| [CausalSim, NSDI 2023](https://www.usenix.org/conference/nsdi23/presentation/alomar) | counterfactual trace validity under intervention | corrects trace bias，不检验 additive-vs-max inverse model adequacy |
| [Avoiding the Ordering Trap, ATC 2023](https://www.usenix.org/conference/atc23/presentation/duplyakin) | randomized order and state carry-over in systems measurements | experimental hygiene，不编译 route-equivalent pairs |
| [Network Delay Inference from Additive Metrics](https://arxiv.org/abs/math/0604367) | additive delay reconstruction | its additivity is the assumption N06 audits; no MoE grouping/barrier interactions |
| [Gestalt, ATC 2014](https://www.usenix.org/conference/atc14/technical-sessions/presentation/mysore) | uncertain dependencies and fault-hypothesis model | 通用 fault localization，不做 exact-semantics factorial adequacy gate |

### Decisive novelty risk

如果 natural states 没有足够 overlap，只能人工改 route/batch 才生成反事实，则因果对无效；若有效方法只是普通 factorial interaction test，N06 只能作为 N09 审计组件。

## N01 Frontier-Cut Bisimulation

### Core claims

1. 对一个 action pair 的 request-DAG prefix 构造 canonical frontier state，若 frontier 等价，则 suffix 可双模拟配对并保持 action ranking。
2. 输出 machine-checkable event pairing proof；失败时输出首个边界不匹配与可重放 witness。
3. 自然 trace 上的 certificate 必须在 sound 前提下相对 full DAG 压缩至少50%，否则无独立方法价值。

### Closest primary work

| Work | Direct overlap | Remaining claimed delta |
|---|---|---|
| [Frontier](https://arxiv.org/abs/2605.21312) | MoE/EP closed-loop event graph、explicit dependency、decision drift | full simulation，非 one-sided suffix proof |
| [CausalSim](https://www.usenix.org/conference/nsdi23/presentation/alomar) | intervention-valid counterfactual replay | learned causal correction，非 event bisimulation |
| [Unity, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/unger) | theorem-proved semantic-equivalent graph rewrite | DNN training computation semantics，非 online request queue suffix |
| [Bisimulation Learning](https://arxiv.org/abs/2405.15723) | classifier/ranking-function bisimulation with SMT counterexample loop | generic transition systems；强烈压缩“bisimulation + witness”的新意 |
| [Formal Methods for Network Performance Analysis](https://www.usenix.org/conference/nsdi23/presentation/tahmasbi) | formal query + synthesized bad-case workload | 生成 performance witness，非 action-pair suffix equivalence |
| [dPRO](https://proceedings.mlsys.org/paper_files/paper/2022/hash/b422680f3db0986ddd7f8f126baaf0fa-Abstract.html) | global dataflow graph + critical-path replay | training graph，非 continuous request DAG |

### Decisive novelty risk

双模拟是通用形式方法；若为 sound 必须把未来 queue/batch/request identities 几乎全部带进 frontier，方法就退化成 Frontier 式 full replay。


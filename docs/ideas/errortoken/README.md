# ErrorToken-MoE（RouteStress 组件试验）

> 状态：`STATIC_SIGNAL_WEAKENED / SELECTOR_NO_ENRICHMENT / NOT_STANDALONE`  
> 证据上限：单个 OLMoE + RTX 5090 已完成 expert-stage 结果的离线重算；不是 serving 或质量结论

## 当前裁决

ErrorToken 不再作为独立 Idea 推进，也不调 threshold/key 抢救。两个 CPU Pilot 从不同层面给出了一致负信号：

- calibration `(layer, expert, M)` risk 对 fresh raw-BF16 mismatch 的 AUC 为 `0.538943851250337`，相对 `M-only` 只增加 `0.007556757374986733`，裁决 `WEAKEN_STATIC_KEYED_RISK_TRANSFER`。
- 每 victim 只允许一次、按 layer 顺序 causal first-eligible 的风险预算 selector，选中 route-positive `6/16`；精确 `2^16` matched-null 均值同样是 `6`，gate-weight 和 top-k-rank baseline 也都是 `6/16`，裁决 `NO_RETROSPECTIVE_ENRICHMENT`。

保留的只是可复用工程资产：stack-bound input seal、请求预算状态机、causal first-eligible 动作计划、exact matched-null 枚举和 fail-closed 输出。主方向已转为 [RouteStress-WitnessPatch](../routestress/README.md)：先找到 held-out 可迁移的传播 witness，再编译 pack surgery，不再使用这张静态风险表。

## 原提议的一句话机制（已被当前证据削弱）

ErrorToken 把 execution-shape 带来的数值不一致从“全或无的 exact allowlist”改成 stack-versioned 风险 token；每个请求在 layer/decode 过程中累积 token，只有预计暴露会超过剩余预算时才拆分为 `M=1`，其余保留自然合批。

## 问题锚点

两组 2026-08-10 实验给出了一个相互约束的信号：

- StableBatch 在富集目标上观察到单个 execution-shape contribution 差异可以穿过 combine，并在 `12/32` targets、`8` victims 上改变下游 top-k membership；这只证明因果路径存在，不是自然发生率。
- SemanticFence 的 unrestricted arm 有 `7,584/8,192` raw-BF16 mismatch rows，但 calibration exact contract 的 `4,237` entries 中放行数为 `0`；处理臂退化成全 `M=1`，且比基线慢约 `0.2364%`。

因此不应该通过改 exactness 门槛救 SemanticFence。新问题是：**calibration 中观察到的 mismatch exposure 能否在 fresh data 上排序风险，从而支持有限的选择性隔离？**

## 原提议的完整系统机制（未被实验支持）

1. **Versioned risk table**：对 `(stack, layer, expert, M)` 记录 calibration 中的 non-exact row fraction。任一 model/backend/driver/kernel 绑定改变都使表失效。
2. **Request ledger**：runtime 对每个 request 记录已消耗与剩余 ErrorToken，避免同一请求在多层/多步中反复承担数值风险。
3. **One-step action**：对已经形成的 natural expert pack，若入账后超预算，则仅将该 pack 拆分成 `M=1`；不改 router、top-k、expert identity、gate weight 或输出 dtype。
4. **Canary invalidation**：以小额度同步 `M=1` canary 检查表的失配率；一旦越界，废弃整张版本表并 fail closed，而不在线调阈值。

## 与旧方向的实质区别

- **不是 SemanticFence**：SemanticFence 要求某个 class 对所有 calibration repeats 逐 bit exact；ErrorToken 承认 non-exact，控制的是请求级累积暴露。
- **不是 Quality Debt**：旧 Quality Debt 的 harm 是没有 BF16 shadow 时不可观测的 KL；ErrorToken 的入账量是版本绑定、可用 canary 核验的 execution-shape mismatch exposure。它仍不等于质量 harm。
- **不是 RouteContract**：RouteContract 检测跨实现 correctness bug/metamorphic relation；ErrorToken 处理同一合法 backend 内的数值 execution-shape 风险，并且有在线 pack/split action。
- **不是 margin fallback**：本轮不使用 future router margin，也不根据 low-margin token 做单次 rollback；核心状态是跨层/跨步的 request exposure ledger。
- **不复活 RouteShare/ShapeShare**：不计算 coalition cost、Shapley 或 virtual service units，也不使用公平指标抢救已冻结的 RouteShare NO-GO。

## Risk-transfer Pilot

[`experiments/risk_transfer.py`](experiments/risk_transfer.py) 只回答一个问题：

> calibration-only `(layer, expert, M)` non-exact fraction 能否在 fresh evaluation 的同 key natural packs 上排序 raw-BF16 mismatch，并且比 `M-only` 强 baseline 多提供信息？

这是已看过 aggregate evaluation verdict 后才实现的回顾性转移分析，所以最高只能授权一次新的 held-out selector Pilot，不能自己成为 confirmatory 证据。

### 输出边界

- 能说：calibration risk 对 fresh expert-stage mismatch 有或没有排序信号。
- 不能说：mismatch 一定传播，ErrorToken 改善质量/延迟，或 call-count proxy 等于 GPU latency。
- 正向后：冻结 held-out natural co-batch，实际执行 risk-ledger selector 与 matched-shuffle action。
- 负向后：停止静态 keyed-risk predictor；保留 StableBatch 已观察的因果现象，改用反例驱动 canary/版本回归线，不调风险阈值抢救。

## Cross-artifact selector Pilot

[`experiments/run_cpu_pilot.py`](experiments/run_cpu_pilot.py) 仅使用 SemanticFence calibration 生成 mismatch-onset risk，在读 StableBatch outcome 前冻结 16 个 victim 的 `B=1` first-eligible action plan，再与全部 `65,536` 个 matched assignments 及两个简单 baseline 比较。

- 输出：[`outputs/cpu_selector_20260810_run01/summary.json`](experiments/outputs/cpu_selector_20260810_run01/summary.json)
- 结果：`NO_RETROSPECTIVE_ENRICHMENT`
- 证据边界：动作只生成计划，`NOT_EXECUTED_PLAN_ONLY`；不能声称已阻止 route flip 或改善 GPU latency。

## 本地运行

```bash
python3 -m unittest discover -v \
  -s docs/ideas/errortoken/experiments -p 'test_*.py'

python3 docs/ideas/errortoken/experiments/risk_transfer.py \
  --config docs/ideas/errortoken/experiments/configs/risk_transfer_v1.json \
  --artifact-root artifacts/semanticfence_remote_20260810_run03/run \
  --output-dir docs/ideas/errortoken/experiments/outputs/risk_transfer_20260810_run01

python3 docs/ideas/errortoken/experiments/run_cpu_pilot.py \
  --config docs/ideas/errortoken/experiments/configs/cpu_selector_v1.json \
  --repo-root .
```

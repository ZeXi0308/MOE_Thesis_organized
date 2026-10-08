# Round 3 Final Method Review

> Reviewer: GPT-5.6-Sol, xhigh  
> Independence: same-family / provisional  
> Verdict: `REVISE`  
> Weighted score: `8.7 / 10`  
> Protocol status: `PROTOCOL_EXECUTABLE`  
> Scientific status: `UNVERIFIED / NO_METHOD_GO`

<details>
<summary>Full raw reviewer response</summary>

## 第三轮最终审查

**Weighted score：8.7/10**  
**Verdict：REVISE**  
**协议状态：可执行**  
**科学状态：UNVERIFIED / NO METHOD GO**  
**review_independence：same-family，provisional**

本轮三项 CRITICAL 已基本闭合。未到 READY 主要是因为尚无 native、paired-action 和 request-level 结果，不是方案仍需重构。

## 七维评分

| 维度 | 分数 | 权重 | 判断 |
|---|---:|---:|---|
| Problem Fidelity | 9.7 | 15% | Anchor 完整保留，无机制漂移 |
| Method Specificity | 9.0 | 25% | `Z_t`、`X_t(b)`、action、paired estimator 已可执行 |
| Contribution Quality | 8.2 | 25% | 条件容量边界足够锐利，但贡献必须由结果建立 |
| Frontier Leverage | 9.0 | 15% | 真实分支、最小信号和单旋钮机制合理 |
| Feasibility | 8.6 | 10% | Gate 化时间线适合三个月论文 |
| Validation Focus | 9.2 | 5% | 一个资格 Gate、一个决定性 Gate，无多余实验 |
| Venue Readiness | 5.0 | 5% | 尚无科学结果，且 Gimbal 等邻域碰撞强 |
| **加权总分** | **8.7** |  |  |

```text
9.7×15% + 9.0×25% + 8.2×25% + 9.0×15%
+ 8.6×10% + 9.2×5% + 5.0×5%
= 8.675 ≈ 8.7
```

## 上轮 CRITICAL 闭合检查

1. **共同 pre-state：已闭合。** `Z_t` 是 branch matching unit，`X_t(b)=g(Z_t,b)` 是候选 action 前特征，不再混淆不同 treatment 的条件变量。
2. **Paired risk-difference：已闭合。** 主结论来自 same-`Z_t` branch 的直接 risk difference；`gamma` 和 quantile model 已降为诊断与控制工具，不再冒充因果估计量。
3. **Request-level denominator：基本闭合。** 未完成和失败请求明确计为 miss，没有静默删除；TPOT、TTFT、arrival trace 和 branch state 均有统一规则。尚余一个 P1 级 cohort 截止定义。
4. **Previous-action controls：已闭合。** previous budget、scheduled tokens、total routed tokens、recent latency 均进入 `Z_t/M1`，显著降低 pressure 只是上一 action 记忆变量的风险。
5. **N0a/N0b：已闭合。** conformance transfer 与 harness eligibility 分开判定；N0 结果不升级为 capacity evidence，N0a 失败也不被误写成 pressure hypothesis NO-GO。
6. **Controller 合并：已闭合。** simple pressure threshold correction 就是最终 Controller，不再人为制造复杂 proposed method。
7. **C1 措辞：已闭合。** 已改成相对预注册强 route-free baseline 的 fresh-holdout incremental claim，并要求 frozen regime 或 EP confirmation 后才升级为 boundary。

## 仍存 P0/P1

### P0

**NONE。**

当前没有阻止执行 N0 的方法学问题，也没有需要重新设计研究问题的缺陷。

### P1：固定与 treatment 无关的 observation cohort 截止

当前写法同时包含：

- branch 运行至 pre-action cohort 全部完成；
- branch 后 arrivals 计入相同 observation cohort。

不同 budget 下 pre-action cohort 的完成时间可能不同。如果 arrivals 持续到各 branch 自己的结束时间，实际纳入的新请求集合会因 treatment 而变化。

在 I1 前冻结一种规则即可：

```text
方案 A：
C = pre-action cohort
    + 在预注册固定墙钟窗口 [t, t+W] 到达的 request IDs
每个 branch 运行到 C 全部完成或统一 timeout。

方案 B：
主 risk 只评分 pre-action cohort；
固定 post-branch arrivals 继续作为干扰 workload，
TTFT guard 使用预注册且 treatment-independent 的 arrival-ID 集合。
```

这属于 I1 协议的小修订，不阻塞当前 N0，也不需要新增实验。

## Venue Readiness 低于 7 的原因

- **弱点：** 当前仍没有 native residual、pressure-conditioned paired effect、Oracle headroom 或 Controller gain。
- **修正：** 不是继续改提案，而是按顺序取得 N0、I1 数据。只有 I1 证明 stable/material interaction，且 EP confirmatory 保留结果后，才可能接近 top-venue contribution。
- **优先级：科学证据缺口，不是协议 P0。**

## 最终方法判断

dominant contribution 已经足够清楚：

> 在固定 placement/router/scheduler order 下，completed expert/rank pressure 是否稳定修饰 decode-budget 的 request-level treatment effect，并形成强 route-free counters 之外的 empirical feasible-capacity boundary。

Action endogeneity 已通过 same-prestate、policy-specific branch 独立演化关闭；完整请求分母也已达到执行计划所需程度。唯一 P1 可在 I1 冻结协议时解决。

因此本次 `REVISE` 不表示方法仍不可执行，而表示：

```text
PROTOCOL_EXECUTABLE
SCIENTIFICALLY_UNVERIFIED
NO_METHOD_GO
NOT_VENUE_READY
```

## 唯一下一步

**只执行 N0 native qualification：**

- N0a：一 steady + 一 bursty native A/C/D transfer；
- N0b：telemetry OFF/ON parity、overhead、snapshot/replay 和 branch actuator eligibility。

不要提前运行 I1、实现 Controller、扩展 operator localization 或申请 8×A100。

</details>

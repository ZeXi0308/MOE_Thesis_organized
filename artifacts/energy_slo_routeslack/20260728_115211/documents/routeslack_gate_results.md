# RouteSlack-MoE Gate 结果

> 证据截止：2026-07-28  
> formal result：`false`  
> 主 artifact：`artifacts/energy_slo_routeslack/20260728_114917/`

## 1. 证据边界

- `[Observed]` 本机不存在 `nvidia-smi`；`.venv` 为 PyTorch 2.8.0，CUDA available=false、CUDA version=null、device count=0。
- `[Blocked]` 已知 5090 远端 SSH 探测返回 connection closed，不能替代本机执行。
- `[Observed]` protocol-critical CPU tests：`95/95 PASS`；另行 broad regression：`113/113 PASS`，均不是物理 Gate。
- `[Observed]` dry-run 每阶段 16 个 contribution，`routed=dispatched=executed=combined=16`；raw ledger 共 64 行。
- `[Observed]` 10 个 simple baseline 与 1 个 future-known Oracle 完成接口级执行；online/Oracle 输入类型隔离。
- `[Observed]` out-of-range surface 请求得到 `FALLBACK_DEFAULT`、`action_eligible=false`。
- `[Observed]` physical latency samples=0、physical energy samples=0、formal 95% CI=`N/A`。
- `[Inferred]` 单元测试与 synthetic dry-run 只证明 contract、fallback、隔离和 artifact 流程，不证明 Energy–SLO 收益。

## 2. 真实运行数字

| 项目 | 结果 | independent unit / 含义 |
|---|---:|---|
| protocol-critical tests | 95，0 fail | 单元测试；代码健康度 |
| broad regression tests | 113，0 fail | 单元测试；包含历史 policy tests |
| identity | 16/阶段 × 4 阶段 | synthetic contribution |
| policy rows | 10 baseline + 1 Oracle | synthetic interface invocation |
| completed output tokens | 每策略 4 | synthetic matched set |
| output hash | 11/11 相同 | fixture exactness |
| physical latency samples | 0 | CI=N/A |
| physical energy samples | 0 | CI=N/A |
| natural models completed | 0/2 | 两模型 AND 未测试 |
| formal CaptureRatio | N/A | 无 matched physical energy |

## 3. Host-only no-op 诊断

以下数字来自本机 CPU，25 个 outer trials、每 trial 2000 次调用；它们不含 CUDA、NVML、真实 route hook 或 GPU energy，不能与 controller gross saving 相除：

| operation | host P50 µs/call | host P99 µs/call | P50 increment over empty µs |
|---|---:|---:|---:|
| empty loop | 0.027167 | 0.028630 | 0 |
| instrumentation | 0.042396 | 0.048079 | 0.015230 |
| route-hook fixture | 0.063063 | 0.069054 | 0.035896 |
| JSON logging | 2.926958 | 3.119585 | 2.899792 |
| decision framework | 0.110396 | 0.118829 | 0.083229 |

paired mean increment over empty-loop 及 percentile-bootstrap 95% CI 分别为：instrumentation 0.015563 µs `[0.015077,0.016212]`、route-hook fixture 0.036489 µs `[0.035894,0.037217]`、logging 2.887233 µs `[2.850043,2.926636]`、decision 0.084683 µs `[0.083474,0.086022]`。该诊断没有 GPU paired baseline，故 no-op **energy** tax 和 `tax/gross` 均为 `N/A`。

## 4. Gate table

> Gate 1–3 的 FAIL 表示“未达到 PASS 条件且因 Gate 0 顺序停止而未运行”，不是物理假设已被反证。

```text
Gate 0: FAIL
Gate 1: FAIL
Gate 2: FAIL
Gate 3: FAIL
Gate 4: NOT RUN
```

| Gate | 依据 |
|---|---|
| Gate 0 | natural continuous-batching decode、双模型 instrumented exactness、CUDA/energy window、thermal state 均无当前物理闭合证据 |
| Gate 1 | 0/2 模型、0 physical sample；无 rows×tier surface、natural energy-mass coverage 或 95% CI |
| Gate 2 | Oracle 仅接口 dry-run；gross/net、switch/hold/controller/idle tax 和 SLO violation 全部 N/A |
| Gate 3 | baseline 仅 synthetic plumbing；formal CaptureRatio=N/A |
| Gate 4 | Gate 0–3 未全 PASS；禁止 controller feasibility |

## 5. Simple baseline versus Oracle

正式比较不存在，CaptureRatio=`N/A`。dry-run 的 synthetic cost 使用 immediate=100、最强 simple=94、Oracle=90，因此公式检查值为 `(100-94)/(100-90)=0.60`；所有行均标记 `scientific_result_eligible=false`，**60% 不是实验收益**。

## 6. Artifact 完整性

主 artifact 保存 manifest、commit、dirty status、command、seed、环境、diff、raw JSONL、processed summary、test log、no-op raw trials 与逐文件 SHA-256。`figures/README.md` 明确拒绝在 0 physical sample 时生成 12 张科学图，避免把 synthetic fixture 包装成结果。

主 manifest SHA-256 为 `34d7865f843c07d7126302cfdc3860d3905243f4d21219d4e20903a26d42d376`。`20260728_114933/` 另保存 BCRD `SMOKE_ONLY` Gate 1–3 打包链；它不改变正式 Gate。`20260728_114153/` 的首轮 provenance 因 `git-ai` wrapper 在中文路径下崩溃而无效，其余早期目录均为中间 dry-run。

## 7. 被阻断/未授权的 GPU 项

- Experiment A：两模型 BF16 rows×power/clock tier latency-energy surface。
- Experiment B：真实 continuous-decode natural route census。
- Experiments C–E：actionability、conservative Oracle、matched baseline comparison。
- GPU no-op/controller tax、thermal/clock/power timeline 与 12 张图。

缺失原因同时包括硬件不可用和 Gate 0 P0 未闭合；不能用旧 FP8、full-forward、CPU replay 或 synthetic cost 代替。

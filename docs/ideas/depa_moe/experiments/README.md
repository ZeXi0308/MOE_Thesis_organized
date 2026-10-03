# DEPA-MoE v1 实验代码

这套代码实现的是 **Deadline- and Expert-Pressure-Aware Admission** 的 CPU 回放与三道串行门禁，不是 GPU 实验结果。它解决四个实验协议问题：

1. 每个到达请求在账本中只能终结为 `completed`、`rejected` 或观测窗边界上的 `pending`，不能静默丢失；
2. 在线策略接口只接收当前时刻和已经到达的请求，不向策略泄漏未来到达；
3. 服务时间只使用实测曲线的区间内保守插值，越界直接失败；
4. 离线 oracle 只作为小规模 SLO-goodput 上界，允许看未来，但不会被包装成线上策略。

## 文件

- `depa_policy.py`：请求/曲线数据模型、基线、DEPA rolling 策略、因果回放、指标和精确 oracle。
- `run_depa_gates.py`：Gate 1 瓶颈占比、Gate 2 oracle 空间、Gate 3 简单策略差距的串行 fail-closed runner。
- `configs/depa_v1.json`：当前冻结阈值和缺失的正式能力。
- `test_depa_policy.py`：账本、因果性、oracle、公共 cell 和 fail-closed 测试。

## 本地逻辑验证

```bash
cd docs/ideas/depa_moe/experiments
python3 -m unittest -v test_depa_policy.py

python3 run_depa_gates.py make-development-fixture --output-dir /tmp/depa-dev
python3 run_depa_gates.py run \
  --config configs/depa_v1.json \
  --breakdown /tmp/depa-dev/breakdown.json \
  --episodes /tmp/depa-dev/episodes.json \
  --surface /tmp/depa-dev/surface.json \
  --output /tmp/depa-dev/verdict.json \
  --development
```

开发夹具会固定写出 `scientific_result_eligible=false` 和
`DEVELOPMENT_ONLY_NOT_SCIENTIFIC`。去掉 `--development` 时，只要配置中的任一正式能力仍为 `false`，runner 会在 Gate 1 前失败。
正式模式还要求配置本身及 breakdown、episodes、surface 三个输入都显式声明
`scientific_result_eligible=true`；缺失该字段同样 fail closed。

## 输入契约

Gate 1 的 `depa-breakdown-v1` 要求每条重复记录提供：

- `model`、`cell`、`seed`；
- `total_critical_path_us`；
- 已去重归因后的 `target_exposed_us`，表示 expert fragmentation、route-conditioned straggler 与 HoL 的目标暴露时间。

Gate 2/3 的 `depa-episodes-v1` 每个 episode 必须是一个模型、一个自然 workload/load cell，包含固定观测窗及不超过 oracle 上限的请求。每个请求提供到达、deadline、按 expert 聚合的 rows 和可选请求类别。

`depa-service-surface-v1` 当前只支持单 5090 的 `serial_experts` 执行模型。它不会把 LUT、单卡回放或模拟路由解释成 EP、网络、TPOT/P99 或多卡 serving 证据。

## 串行裁决

- Gate 1 未 PASS：不运行 oracle 或策略比较；
- Gate 2 未 PASS：不运行复杂策略价值判断；
- Gate 3 只使用 Gate 2 已通过的公共 workload/load cell；
- Gate 3 中任一简单策略捕获至少 90% oracle 空间：`FAIL_ABANDON_COMPLEX_POLICY`；
- DEPA 只有同时满足净增益、oracle 捕获率、相对简单策略差距、决策开销、P99 和公平性约束才 PASS。

把正式能力开关改为 `true` 只是声明能力已具备，不会自行生成 GPU 证据；正式运行前仍需代码评审、真实连续 decode producer、5090 实测 surface、exact-output replay 和自然 workload manifest。

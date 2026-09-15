# RouteSlack-MoE 最终裁决

## Verdict

```text
MEASUREMENT_ONLY
```

Gate 0 仍有 14 个 P0 未关闭、当前环境有 0 个 physical energy/latency sample，因而只能保留 measurement/protocol characterization；不得实现 controller，也不得进入 8×A100 Gate。

## 裁决边界

- `[Observed]` 96 个 CPU tests 全部通过，cached-decode 开发路径、route-v2 identity、fail-closed provenance、counter wrap 和 sampler exception 等代码条件得到覆盖。
- `[Observed]` tiny-Mixtral development capture 有 8 contribution，但 metadata 明确 formal/scientific false。
- `[Observed]` 当前无 CUDA/NVML GPU；Gate 1–3 没有真实数字，Gate 4 未运行。
- `[Blocked]` H1、H2、H3 都没有被物理证实或证伪。
- `[Inferred]` 当前唯一可发表/保留的内容是“如何避免 RouteSlack Energy–SLO 假阳性”的 measurement methodology；把它写成主系统会越过证据边界。

## Gate table

```text
Gate 0: FAIL
Gate 1: FAIL
Gate 2: FAIL
Gate 3: FAIL
Gate 4: NOT RUN
```

Gate 1–3 的 FAIL 是“没有获得运行/晋级资格”，不是将缺硬件包装成物理负结果。

## Oracle / simple baseline

```text
Oracle net saving = NOT MEASURED
Best simple saving = NOT MEASURED
CaptureRatio = NOT COMPUTABLE
```

0 physical sample 时不存在合法分母；synthetic dry-run cost、历史 FP8 数字、BCRD latency replay 和 CPU no-op 微基准均不得代入。

## 为什么不是 8xA100_CANDIDATE

该 verdict 要求两模型 natural workload、Gate 0 accounting、Oracle net ≥10%、simple capture <90% 与可接受 tax 同时成立；本轮这些条件全部缺少 physical evidence。单卡本可验证的 service–energy surface 与 action timing 都尚未完成，直接上 8×A100 会把 measurement bug、baseline bug 与 EP 复杂度混在一起。

## 为什么保留为 measurement-only

当前代码修复和 protocol 能作为 characterization 的方法学部分：

1. cached one-token decode 与 full recomputation 对照；
2. routed→dispatched→executed→combined exact identity；
3. raw board energy 与 idle-adjusted sensitivity 分离；
4. fixed-overhead/repeat/ABBA/thermal/future-leak 反例测试；
5. formal provenance 与 out-of-range default fallback。

这不是对 RouteSlack 节能的正面结论，也不是允许更换指标、增加 synthetic skew 或复活旧 FP8/JouleQueue NO-GO。

## Reproduction

当前 working tree 的自包含审计命令：

```bash
cd '/Users/leandrozhao/Desktop/毕设论文资料'

'./.venv/bin/python' -B docs/ideas/energy_slo/routeslack/experiments/run_routeslack_dry_run.py \
  --output-dir artifacts/energy_slo_routeslack/REPRO_TIMESTAMP \
  --seed 20260728 \
  --run-tests

# 单独复核关键 suite
PYTHONDONTWRITEBYTECODE=1 './.venv/bin/python' -B -m unittest \
  discover -s docs/ideas/bcrd/experiments -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 './.venv/bin/python' -B -m unittest \
  discover -s docs/ideas/energy_slo/routeslack/experiments -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 './.venv/bin/python' -B -m unittest \
  docs.ideas.energy_slo.route_row_fp8.experiments.test_continuous_decode_harness \
  docs.ideas.energy_slo.route_row_fp8.experiments.test_power_accounting
PYTHONDONTWRITEBYTECODE=1 './.venv/bin/python' -B -m unittest \
  discover -s docs/ideas/energy_slo/joulequeue/experiments -p 'test_*.py'
```

最终可追溯快照、测试日志、raw dry-run ledger、development capture、环境与 SHA-256 位于 `artifacts/energy_slo_routeslack/20260728_115300/`。该目录内 `source_snapshot/` 保存本轮相关代码/文档，不需要把 synthetic artifact 当 formal 输入。

## Next action

```text
保留为 characterization。
```

只有未来获得 NVIDIA GPU 后，另起一次全新 timestamped run，先关闭 Gate 0 并完成双模型 Experiment A/B；在此 verdict 下不直接进入 8×A100，也不实现 controller。

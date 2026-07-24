# RIC-v1 Phase 4 严格代码审查

STATUS: SIGNED-OFF
OPEN_P0: 0
OPEN_P1: 0
OPEN_P2: 0
OPEN_P3: 0
REVIEW_PENDING: 0

审查对象：[`RIC_Phase2_冻结实验协议_2026-07-22.md`](RIC_Phase2_冻结实验协议_2026-07-22.md) Amendment A–P、[`configs/ric_v1.json`](configs/ric_v1.json) 与 [`experiments/ric/`](experiments/ric/) 的全部 formal runtime/provenance/test 路径。

## 最终门禁结论

允许在本签字绑定的 exact scope 上执行全新 formal calibration。v4 保持 `BLOCKED_G1_DESIGN / SUPERSEDED`；Amendment P 前的 dev-v5 oracle 数字也已 `SUPERSEDED / NOT_TESTED`。本签字不构成 NCCL、RDMA、真实 serving TTFT/TPOT/P99 或 scientific Go/No-Go。

独立 Reviewer 已对下列最后修复复签，结论为 `SIGNED-OFF`，P0–P3 全部关闭：

- observation-history builder 真正驱动 nonanticipativity node；
- starvation 与 FCFS/no-overtake 服务纪律显式入实例，并与 independent literal LP 核对；
- reviewed scope 每一行对 current bytes 重算，test report 绑定 scope file SHA；
- data producer signoff 显式贯穿 route/capability/LUT/scenario。

## 审查中发现并关闭的关键缺陷

| 缺陷 | 级别 | 修正与验收 |
| --- | --- | --- |
| full-barrier 不同 kernel顺序被误作逐 trial timing等价负对照 | P0 | Amendment O 改为四个 paired LCB与全 trial事件前驱门；barrier跨策略 timing仅诊断 |
| type-1 5% LCB受 IEEE浮点影响取第501个而非第500个 | P0 | config显式冻结 one-based rank 500；producer/consumer不再用浮点推 rank；边界向量测试覆盖第500≤0、第501>0 |
| checksum与tensor hash在 stream context外回落 default CUDA stream | P0 | 两者移入唯一 persistent sender-local stream；两份 profiler逐GPU activity复算唯一stream ordinal并要求一致 |
| `canonical_equal=true` 可掩盖 output/reference hash不等 | P0 | consumer逐raw row验证 `output == row reference == artifact reference`；artifact bool不参与自证 |
| raw `physical_frontier` 可早于 closing task开始 | P0 | 所有 policy×mode×trial要求 frontier位于 closing `x` 的 service区间 |
| custom stream pointer可重签为0 | P0 | producer与consumer都要求非默认 stream pointer `>0`；zero/mixed-stream mutation均拒绝 |
| capability producer source/signoff未被 formal calibration完整绑定 | P0 | runner source closure纳入 `measure_capability_gpu.py`；逐model绑定 artifact self-hash与 producer-signoff SHA |
| root request/layer/block/warmup与raw/action未交叉绑定 | P1/P0边界 | consumer绑定 request、assigned layer、selected layers、task contribution、raw rows；formal固定32/10/30 |
| 五个辅助median只由producer报告 | P2 | consumer从raw重算；`total_us`纳入finite/non-negative校验 |
| 缺旧v4 schema显式回归 | P2 | 单-profiler v4 schema专门拒绝测试 |

## 冻结后的 G1 acceptance

同一30个 trial、同一10,000次 paired bootstrap索引、seed `2026072226`、type-1 one-sided 95% LCB one-based rank 500：

1. streaming application release 的 `B-C` LCB严格 `>0`；
2. streaming downstream start 的 `B-C` LCB严格 `>0`；
3. release 的 `streaming(B-C)-barrier(B-C)` interaction LCB严格 `>0`；
4. downstream 的同类 interaction LCB严格 `>0`；
5. canonical exactness、persistent non-default stream、双 profiler、全 trial事件前驱全部通过。

任一失败即 G1不通过；禁止删 outlier、换 seed、放宽阈值或重复运行求过。

## 最终本地验证

```text
TEST_STATUS: PASS
TEST_TOTAL: 185
TEST_PASSED: 185
TEST_FAILED: 0
TEST_ERRORS: 0
TEST_SKIPPED: 0
```

命令：

```bash
.venv/bin/python -m py_compile docs/ideas/receiver_aware/experiments/ric/*.py
.venv/bin/python -m unittest discover \
  -s docs/ideas/receiver_aware/experiments/ric -p 'test_*.py'
git diff --check
```

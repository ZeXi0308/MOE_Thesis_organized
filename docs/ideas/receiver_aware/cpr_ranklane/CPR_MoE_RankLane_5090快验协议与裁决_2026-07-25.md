# CPR-MoE RankLane：RTX 5090 快验冻结协议、实现与裁决

> 日期：2026-07-25  
> 阶段：快速验证；不是 8×A100 formal/sealed serving 实验  
> 冻结对象：CPR-MoE 中的固定 RankLane 返回表示执行器  
> 裁决：`NO_GO_RANKLANE_ACTUATOR_UNDER_P_RETURN_MAX_0_20`

## 0. 直接结论

在原始 BF16 EP return path 占端到端时延不超过 20% 的冻结域内，当前固定 RankLane 执行器已经被上界判死：两个模型的最乐观 tail policy 都只能把字节节省率从 uniform FP8 的 50% 提到 68.75%。即使假设 codec、launch、排队和 metadata 开销全部为零，`p_return=20%` 时相对 uniform FP8 的端到端改善上界也只有：

\[
\frac{0.2(0.6875-0.5)}{1-0.2\times0.5}
=\frac1{24}=4.1667\%<5\%.
\]

达到 5% 至少要求：

\[
p_{return}\ge
\frac{0.05}{(0.6875-0.5)+0.05\times0.5}
=\frac4{17}=23.5294\%.
\]

这是 quality-free、zero-codec、线性 wire scaling 的候选最有利上界；增加真实质量约束或执行开销只会降低收益。因此当前不需要再消耗一次 5090 GPU run 来确认同一个负结论。

该裁决**不否定**“优化后 EP return path 是否真实暴露”这个 P1 问题，也不证明 receiver ordering 无效。二者仍是 `NOT_TESTED_REQUIRES_8XA100` / `NOT_TESTED`。它只否定 `p_return≤20%` 时当前固定 RankLane 相对 uniform FP8 达到 5% E2E 改善的机制假设。

---

## 1. 研究审计

### 1.1 可证伪假设

> H1：在原始 BF16 EP return path 占端到端时延不超过 20% 时，RankLane 相对 uniform FP8 仍能在零编解码成本的乐观上界下，为 OLMoE 与 LLM-jp 同时带来至少 5% 的端到端时延改善。

- 可证伪：是；任一模型在 `p_return=20%` 的最乐观上界低于 5% 即 FAIL，跨模型规则为 AND。
- 冻结自变量：`p_return∈{5%,10%,15%,20%}`。
- 冻结控制：uniform FP8，字节节省 50%。
- 候选选择：每个模型所有 `fp8top*_rest_int4` 中字节节省最高者；主门槛故意不加质量预算，避免通过挑阈值制造结论。
- 主终点：相对 uniform FP8 的 exact zero-codec E2E improvement upper bound。

### 1.2 因果链逐环审计

```text
gate-rank 质量不对称
  → 可以让更多 tail contribution 用 INT4
  → 相对 uniform FP8 进一步减少 return bytes
  → return service time 线性下降且位于暴露关键路径
  → 降低 request/token completion time
```

| 环 | 当前证据 | 状态 | 对裁决的影响 |
|---|---|---|---|
| rank 与质量敏感度相关 | 两模型各 128 文档、single-forward combine-output KL + document bootstrap CI | `[Observed]`，但不是任务质量/decode 证据 | 提供 policy catalog，不证明系统收益 |
| tail INT4 增加 byte saving | 两模型 catalog 最大均为 68.75%，uniform FP8 为 50% | `[Observed/accounted]` | 给出最大额外 saving 18.75pp |
| bytes 线性转化为 return service | 本快验直接假设成立，且所有 codec tax 设零 | `[Optimistic upper bound]` | 对候选最有利，不是硬件预测 |
| return service 是暴露关键路径 | 5090 无 EP；尚无优化后 8 卡同轴 DAG | `NOT_TESTED_REQUIRES_8XA100` | 只能做条件裁决 |
| service reduction 改善 E2E | 由 Amdahl 精确换算 | `[Derived]` | 在 `p≤20%` 时上界仍 FAIL |

最可疑的一环仍是“真实暴露 return fraction”，但无需在 5090 上伪造它；条件上界已经说明，若未来 F2 只测得 10%–20%，当前 RankLane 执行器不值得进入实现。

### 1.3 上界与 Captured Headroom

将 BF16 E2E 时间归一化为 1，原始 return fraction 为 `p`，表示的字节节省率为 `s`：

\[
T(s)=1-ps,
\qquad
G(c\mid b)=\frac{p(s_c-s_b)}{1-ps_b}.
\]

两个模型都取 `s_b=0.5, s_c=0.6875`。相对“把 uniform FP8 剩余 return bytes 全部消除”的 headroom，RankLane 最多捕获：

\[
CH=\frac{0.6875-0.5}{1-0.5}=37.5\%.
\]

这里绑定结论的是绝对 E2E 上界而不是 CH：即使允许 37.5% 的剩余 return headroom，在 `p≤20%` 时仍过不了 5%。

| 原始 `p_return` | OLMoE 上界 | LLM-jp 上界 | 5% Gate |
|---:|---:|---:|---|
| 5% | 0.9615% | 0.9615% | FAIL |
| 10% | 1.9737% | 1.9737% | FAIL |
| 15% | 3.0405% | 3.0405% | FAIL |
| 20% | 4.1667% | 4.1667% | FAIL |

---

## 2. 最小实验矩阵

最多保留两个串行实验。E1 已执行并 FAIL，因此按预声明决策树停止；E2 不实现、不运行。

| 顺序 | 实验 | 资源 | 最高信息增益 | 当前状态 |
|---:|---|---|---|---|
| E1 | 跨模型 policy catalog + zero-codec Amdahl upper-bound gate | CPU；复用既有 5090 产物 | 在不需要 EP 的情况下直接淘汰不可能达到 5% 的执行器 | **已执行，FAIL** |
| E2 | fused FP8→INT4 incremental codec microbench | 1×RTX 5090 | 仅在 E1 PASS 后判断真实 executor 是否还有净快区 | **按协议未运行** |

### 2.1 E1 的 17 项冻结设计

| 字段 | 冻结值 |
|---|---|
| 1. 验证的具体假设 | H1；见 1.1 |
| 2. 为什么必须先做 | 若对候选最有利的 Amdahl 上界也过不了 5%，任何 kernel/controller 实现都没有信息价值 |
| 3. 最小实验对象 | 两个已有 rank-quality policy catalog + 一条封闭公式；不加载模型、不执行 GPU |
| 4. 自变量 | 只改变原始 BF16 `p_return={0.05,0.10,0.15,0.20}` 与模型；每模型候选固定为最大 saving policy |
| 5. 因变量 | 相对 uniform FP8 的 exact zero-codec E2E improvement upper bound |
| 6. 控制变量 | baseline saving、候选选择规则、5% 阈值、zero-tax 假设、输入 artifact、跨模型 AND 全部冻结 |
| 7. Baseline | deployable simple baseline：`uniform_fp8`，byte saving=0.5 |
| 8. Oracle upper bound | theoretical upper bound：每模型 catalog 内最大 saving RankLane + zero codec/launch/queueing/metadata tax |
| 9. Workload | 上游 OLMoE 与 LLM-jp 各 128 文档的 single-forward combine-output quality artifact；`p_return` 为条件扫描而非合成实测 |
| 10. 指标 | 主终点、达到 5% 所需 `p_return`、CH；KL CI-high budget 仅作敏感性 |
| 11. 重复次数 | 确定性分析一次即可；用有理数独立复算一次；上游 quality 每模型 128 docs |
| 12. 统计方法 | 主终点无采样噪声；quality 使用上游 document bootstrap CI；跨模型 AND；不使用 codec 镜像 p95 |
| 13. 成功阈值 | 两模型均在 `p_return=0.20` 时 `upper_bound≥5%` |
| 14. 淘汰阈值 | 任一模型 `<5%` 即淘汰冻结域内 fixed RankLane executor，不进入 E2 |
| 15. 可能的混杂因素 | 真实 `p_return` 未测；KL 不等于 task quality；wire scaling/zero-codec 是乐观假设；两模型 top-k 不同 |
| 16. 能得出的结论 | `p_return≤20%` 时当前 catalog 的 RankLane 是否理论上可能胜 uniform FP8 达到 5% E2E |
| 17. 不能得出的结论 | 不能证明 EP criticality、NCCL/RDMA、decode TPOT/P99、真实 codec、receiver ordering 或生产收益 |

### 2.2 E2 的 17 项条件设计（未触发）

| 字段 | 条件冻结值 |
|---|---|
| 1. 验证的具体假设 | producer-fused FP8→INT4 增量 codec 在真实常见 shape 中具有净正收益 |
| 2. 为什么必须先做 | 只有 E1 上界 PASS 后，才值得判断该理论空间能否跨过真实 executor tax |
| 3. 最小实验对象 | 一对公平的 fused uniform-FP8 / RankLane-INT4 producer-consumer kernel；不改完整 serving runtime |
| 4. 自变量 | 最多 rows 与 hidden 两个变量；link/backend 预算按真实目标固定 |
| 5. 因变量 | `(FP8 wire - INT4 wire) - incremental pack/unpack/layout tax` 的分布 |
| 6. 控制变量 | 相同输入、shape、dtype、layout/fusion 机会、message count、warmup、stream、clock/power |
| 7. Baseline | deployable baseline：同 fusion/layout 的 uniform FP8；不能给候选单独优化 |
| 8. Oracle upper bound | theoretical zero-codec wire saving；只作上界，不作为可部署方法 |
| 9. Workload | 未来 8 卡 trace 的 P10/P50/P90 common shapes；当前无此输入，不用有利 hotspot 代替 |
| 10. 指标 | net delta P50/P95/P99/mean/std、gross saving、codec tax ratio、峰值显存 |
| 11. 重复次数 | 20 次同 shape warmup；每 cell 至少 200 个原始 timing samples；多个独立 run |
| 12. 统计方法 | CUDA Events + 完整 stream sync；paired samples；保存原始值；bootstrap 95% CI/LCB；不删异常值 |
| 13. 成功阈值 | 两模型 common shapes 净收益>0、codec tax≤30% gross saving、95% LCB>0 |
| 14. 淘汰阈值 | E1 FAIL 不运行；任一 common bucket 无净区或只在罕见 hotspot 正则停止 executor |
| 15. 可能的混杂因素 | PCIe H2D 不等于 NCCL、A100/5090 代际差异、fusion 公平性、analytic wire 假设、Triton compile/cache |
| 16. 能得出的结论 | 当前 5090 fused primitive 在冻结 common shapes 是否存在净快区 |
| 17. 不能得出的结论 | 不能证明 EP collective、8×A100、TPOT/P99、跨节点 RDMA 或生产 E2E 收益 |

既有 unfused codec artifact 的 FP8→INT4 增量 gate 为 0/8 viable，但其 p95 从 p50 镜像、不是独立样本，因此只作方向性旁证，不把它冒充 E2 formal 结果。

### 2.3 紧凑实验矩阵

| 实验 | 核心假设 | 自变量 | Baseline | Oracle | 指标 | 成功阈值 | 失败结论 |
|---|---|---|---|---|---|---|---|
| E1 catalog+Amdahl | zero-tax RankLane 在 `p≤20%` 仍可改善≥5% | 模型、`p_return`；每模型 policy 固定 | uniform FP8 | catalog 最大 saving + zero tax | exact E2E upper bound | 两模型均≥5% | 淘汰冻结域内 fixed RankLane |
| E2 fused codec | common shapes 中真实增量 codec 有净区 | rows、hidden | 公平 fused uniform FP8 | zero-codec wire saving | net P50/P95/P99、LCB、tax ratio | common shapes LCB>0 且 tax≤30% | 淘汰当前 executor |

---

## 3. 预声明决策树与当前落点

```text
E1: 两模型 quality-free + zero-codec 上界在 p_return≤20% 是否均 ≥5%？
  ├─ NO  → STOP fixed RankLane executor
  │        P1 return-path existence 仍为 NOT_TESTED_REQUIRES_8XA100
  │        若未来 p_return <23.53%，不得重开当前执行器
  └─ YES → E2: 公平 fused codec common-shape 净收益是否通过？
           ├─ NO  → STOP executor
           └─ YES → 才允许 8×A100 E2E simple-baseline Gate
```

当前落点：E1 的两个模型都为 4.1667%，走 `NO` 分支。E2 未触发。

若未来 8×A100 的优化后同轴 DAG 在**两个模型共同自然 cell** 测到原始 BF16 exposed return fraction 均至少 23.53%，且出现公平 fused codec 净快区，可新建协议重新开放；不能修改本次冻结输出，也不能用 barrier、sleep、关闭 overlap 或慢链路制造 `p_return`。

---

## 4. 最小实现与复跑命令

- 实现：[快验分析器](./experiments/cpr_moe_quick/run_gate.py)
- 冻结配置：[quick_validate.json](./experiments/cpr_moe_quick/configs/quick_validate.json)
- 单元测试：[test_run_gate.py](./experiments/cpr_moe_quick/test_run_gate.py)
- 结果：[report.md](./outputs/cpr_moe_ranklane_quick_gate_2026-07-25/report.md)
- 机器裁决：[decision.json](./outputs/cpr_moe_ranklane_quick_gate_2026-07-25/decision.json)
- 输入与代码哈希：[source_manifest.json](./outputs/cpr_moe_ranklane_quick_gate_2026-07-25/source_manifest.json)

从仓库根目录执行：

```bash
# 环境检查；E1 不要求 CUDA，nvidia-smi 仅记录机器状态
python3 --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 依赖安装：E1 仅使用 Python 标准库，无 pip 依赖

# smoke / 静态语法检查
python3 -m py_compile \
  docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick/run_gate.py \
  docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick/test_run_gate.py

# 单元测试
python3 -m unittest \
  docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick/test_run_gate.py -v

# 冻结正式运行
python3 docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick/run_gate.py \
  --repo-root . \
  --config docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick/configs/quick_validate.json \
  --output-dir docs/ideas/receiver_aware/cpr_ranklane/outputs/cpr_moe_ranklane_quick_gate_2026-07-25

# 结果分析：人读报告与机器裁决
sed -n '1,220p' \
  docs/ideas/receiver_aware/cpr_ranklane/outputs/cpr_moe_ranklane_quick_gate_2026-07-25/report.md

python3 -m json.tool \
  docs/ideas/receiver_aware/cpr_ranklane/outputs/cpr_moe_ranklane_quick_gate_2026-07-25/decision.json

# 可恢复清理：若确需重跑，先把旧证据目录归档；不要覆盖
mv \
  docs/ideas/receiver_aware/cpr_ranklane/outputs/cpr_moe_ranklane_quick_gate_2026-07-25 \
  docs/ideas/receiver_aware/cpr_ranklane/outputs/cpr_moe_ranklane_quick_gate_2026-07-25.archived
```

E1 是确定性上界计算，没有随机 seed 或 GPU timing，因此不存在有意义的“多 seed timing run”；虚构多个 seed 只会重复相同数字。随机性已经留在上游 128-document quality 估计及其 bootstrap CI 中。E2 若未来触发，才需要多个独立 run 与 200+ 原始 CUDA samples。

输出目录必须不存在。已生成的证据包默认不可覆盖；复跑应使用新目录并比较 source manifest。预期文件为 `decision.json`、`matrix.csv`、`report.md`、`environment.json` 与 `source_manifest.json`；本次已经真实生成，文中数值来自该产物而非预期值。

---

## 5. 严格 Code Review 与修复记录

Review 与实现分角色进行：实现阶段只编码冻结公式与输出；随后以“代码默认有错”为前提审查关键调用链，且没有在 Review 前启动任何新 GPU 实验。

关键执行路径逐段审查如下：

```text
main
 → parse_args
 → load_json(config)
 → analyze / validate_config
 → read_quality_input × 2
    → CSV schema + metadata identity + byte-saving cross-check
 → choose_max_saving（quality-free theoretical upper bound）
 → exact_relative_improvement / required_exposed_fraction
 → cross-model AND decision
 → load codec metadata（descriptive only）
 → write_outputs
    → matrix / decision / environment / code+input hash manifest / report
```

- **假设与代码一致性：** 代码实际验证的是“给定 `p_return≤20%` 时 RankLane catalog 相对 uniform FP8 的 zero-tax E2E 上界”，而不是“真实 EP return path 是否暴露”或“5090 能模拟 8 卡通信”。
- **变量与公平性：** 同一模型的 baseline/candidate 共用相同 `p_return`、输入 catalog 和线性 wire 假设；候选只改变 byte saving。主门没有使用质量阈值挑 policy，反而选最大 saving，避免让候选因 cherry-picking 获利。
- **baseline / method / oracle：** `uniform_fp8` 是 deployable simple baseline；`fp8top*` 是待评 mechanism catalog；最大 saving + zero tax 是 theoretical upper bound，不是 deployable proposed method；当前没有把 offline oracle 当在线算法。
- **routing / quantization / tensor 语义：** E1 不生成 token、不执行 routing/top-k/capacity/combine tensor，也不声称实际 INT4 kernel；因此 shape/dtype/device/stride/fallback 审查对当前执行路径不适用。上游 artifact 的边界被原样保留，不能由 E1 扩成 decode 或 task-quality 结论。
- **GPU 测量：** E1 没有 CUDA op、stream、Event、warmup 或显存测量，`environment.json` 明记 `gpu_timing_performed=false`。因此没有用 CPU wall time冒充 GPU latency。既有 codec p95 镜像问题被显式降级；未来 E2 的 CUDA Event/sync/warmup/compile/cache/H2D 边界已在 2.2 冻结。
- **统计：** 主终点是精确公式，无需 mean/P50/P95/P99；强行生成分布会伪造不确定性。上游 quality 的样本量与 CI 只用于 sensitivity；主裁决使用更宽松的 quality-free maximum。两模型用 AND，不报告最佳单模型。
- **系统有效性：** 该实验能直接证伪冻结域内 5% 机制假设，但不能把局部 bytes 变成 EP/TPOT/P99 结论。Amdahl 正是防止局部收益外推；结果只在 `p≤20%` 有效，未来若真实共同 `p≥23.53%` 必须新协议重开。

| 级别 | 发现 | 风险 | 修复/处理 |
|---|---|---|---|
| High | 初版 source manifest 未包含分析器自身哈希 | 代码漂移后结果不可完整复现 | 已加入 `run_gate.py` SHA-256，并重新生成全部输出 |
| Medium | 初版 metadata/config 的部分错误类型可能逃逸为普通异常 | 输入损坏时不能统一 fail closed | 已补非数值/非有限 byte saving、字符串、budget 与 input object 校验 |
| Medium | 初版 NO-GO code 将 `0.20` 写死在字符串 | 修改 config 后决策标签与真实域可能不一致 | 已从冻结 `p_max` 动态构造标签 |
| Medium | codec artifact 的 p95 是 p50 镜像 | 把伪 p95 当 tail 证据会过度声明 | 主门完全不使用 codec timing；report 显式标为非独立，仅用 p50 方向性旁证 |
| Low | 重跑可能覆盖旧证据 | provenance 丢失 | runner 拒绝已有 output directory，返回 `EVIDENCE_ERROR` |

审查后验证：

- `py_compile`：PASS；
- 7 个单元测试：PASS；覆盖公式、反函数、候选方向、quality-free 选择、CI-high budget、跨模型 AND、CSV/metadata 不一致 fail closed；
- 独立有理数复算：`gain=1/24`、`required_p=4/17`，PASS；
- 输入/配置/runner 哈希已固化；
- 未安装 `ruff`，因此没有声称通过 ruff；这不影响数学与证据裁决。

---

## 6. 今天的 PASS/FAIL 清单

- [x] 假设可证伪，主终点与跨模型 AND 在运行前写入冻结 config。
- [x] 因果链逐环标注 `[Observed] / [Derived] / [Optimistic upper bound] / NOT_TESTED`。
- [x] 给出 Amdahl 精确上界、CH 和 5% 所需 `p_return`。
- [x] 实现 stdlib-only、可复跑、输入校验、拒绝覆盖的最小分析器。
- [x] 执行 E1；两模型均 4.1667%，主门 `FAIL`。
- [x] 按预声明停止；没有启动无信息增益的新 5090 GPU run。
- [x] 既有 codec 0/8 只作旁证，未使用镜像 p95 作统计结论。
- [x] 严格 Code Review 后修复 3 个实现问题并重新生成证据。
- [x] P1 EP return-path existence 保持 `NOT_TESTED_REQUIRES_8XA100`。
- [ ] 8×A100 formal Gate：本阶段未授权、硬件信息未冻结，未运行，也未伪造。

最终状态：**今天的 5090 快验目标已完成；RankLane 执行器在冻结域内 FAIL，P1 问题存在性仍待真实 8×A100。**

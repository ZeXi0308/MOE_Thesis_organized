# 原生 MoE decode 的捕获桶成本阶梯，与接纳上限无法对齐执行宽度

2026-09-08，`agent/publish-current-moe-code@2a37765` 加本轮未提交实现。
本轮分两段：先对 2026-09-06 已封存的 44 个 episode 做零新执行重建，再据此在
RTX 5090 上运行 **24 个新 GPU episode**（768 次完整请求执行）检验冻结预测。

**结论：`MEASUREMENT_ONLY`。**

- **成立且跨引擎独立复现：** pure-decode step 成本是 CUDA-graph 捕获桶的阶梯，
  桶内平坦、跨桶跳变。两次独立运行的逐宽度中位数最大差 **0.125 ms**。
- **被本轮实验推翻：** 把反馈 ladder 的档位放在捕获点上，**不能**把执行宽度放到
  捕获点上。steady 域有 **84.5%** 的 step 目标在捕获点、实际宽度不在。
- **不可主张：** steady 的 `+21.26%` goodput 正信号落在噪声内
  （同组 static16 两次重复差 **58.36%**），按预冻结规则不作为方法信号报告。

因此 `capture-aligned admission ladder` 在本运行域为 **`CONDITIONAL_NO_GO`**，
失败位置精确：不是成本模型错，而是**接纳上限控制的是宽度上界，不是执行宽度**。
U/C 专家信号本轮完全未触及，仍为 `UNRUN`。

## 第一段：零新执行的重建

### 重建恒等式与可核对性

引擎 `async_scheduling=False`、`stream_interval=1`，一次 `schedule()` 严格对应一次
输出交付，故 `step_exec_ms[k] = received_s[k] - scheduler_steps[k].start_s`，两个
时间戳同属原始运行的 `perf_counter` 原点。**44/44 cell 通过对齐检查**，零排除。

非抢占下每个 running 请求每 step 推进一个 token，故请求平均 TPOT 应等于其所跨 step
的平均执行时间。**重建值与实测 TPOT 最大偏差 0.1534 ms**（相对 8–10 ms TPOT < 2%），
这是本轮全部推断的误差上界。

### 阶梯与其请求级后果

| 捕获桶 | 宽度 | 桶内中位数 (ms) | 离散度 |
|---:|---|---|---:|
| 8 | 5–8 | 6.763–7.032 | 3.97% |
| 16 | 9–16 | 8.410–8.672 | 3.11% |
| 24 | 17–24 | 9.300–9.508 | 2.23% |

捕获边界经两源确认：从耗时反推得 `[1,2,4,8,16,24,32]`，引擎日志打印
`cudagraph_capture_sizes: [1,2,4,8,16,24,32,40,...]` 且 `decode, FULL: 7/7`。
KV 长度非解释：w1 在 ctx128→224 为 2.807→2.809 ms，w16 为 8.325→8.408 ms。

已测静态 cap 的请求级后果（cap12 与 cap16 同处桶 16）：

| | steady | bursty |
|---|---:|---:|
| pure-step 成本比 (16/12) | 0.9949 | 0.9764 |
| 请求 TPOT 比 | 1.0049 | 0.9943 |
| goodput | +3.44% | +142.58% |

接纳干扰税：`tax = 3.853 + 0.008826 × prefill_tokens`（宽度≥7，n=268，R²=0.8498），
128-token prompt 中位税 5.261 ms，约 73% 为固定成本。

诊断出被判 `CONDITIONAL_NO_GO` 的四步 ITL 反馈控制器 padding waste 为 **0.301**，
是 44 个 episode 最高值 —— 这构成了第二段实验的动机。

## 第二段：24 个新 GPU episode

一张 RTX 5090（32607 MiB，启动前 0 计算进程），软件栈与封存 `environment.json`
逐项一致（python 3.12.3 / vLLM 0.26.0 / torch 2.11.0+cu130 / CUDA 13.0 /
transformers 5.15.1），模型 `OLMoE-1B-7B-0924@6d84c485...`。两个全新引擎各 12
episode，正反序，24/24 全部 `COMPLETE`。

### 唯一改变：ladder 坐标

`{8,12,16,32}` → `{8,16,24,32}`（12 在桶内部，24 是捕获点）。其余全部相同。

工作区代码与 09-06 实际执行代码的 sha256 **不同**（`admission_feedback.py`
08ead8c2 → 279c3e0e 等），差异来自另一个 probe 新增的 `single_down`/`single_shadow`
分支。为此新增 [`test_sealed_feedback_equivalence.py`](../../../experiments/admission_capacity/test_sealed_feedback_equivalence.py)：
从保留的 `deployment/stage2.tar.gz` 直接加载封存模块，用确定性分段信号驱动两份
实现，逐字段断言 decisions 一致（两个 ladder × feedback/shadow × 3 seed，7 项通过）。
**"只改 ladder"因此是被证明的，不是被声称的。**

该测试还记录了一个此前未写明的边界：**shadow 的升档分支结构性不可达**。shadow 不
应用目标，故 `active` 永不跟随 intent，`active == previous` 在首次降档后永假，其
可达 reason 集止于 `natural_drain`。shadow 不是 feedback 的零成本孪生。

### 冻结预测裁决

预注册见 [`next_experiment/DECISIONS.md`](next_experiment/DECISIONS.md)，未修改。

| | 预测 | 结果 |
|---|---|---|
| F1-as-stated | static24 的 pure-step 中位数落在 [9.30, 9.51] ms | **False**：steady 8.678、bursty 8.191 |
| F1-mechanism | 实际宽度 17–24 的 step 落在该平台 | **True**：8 个宽度全部在带内 |
| F2 | aligned feedback 的 padding waste 更低 | **实质失败**：中位数 0.229 vs 0.229 |
| F3 | TPOT 不恶化 | steady True (8.661 vs 8.888 ms)、bursty False |
| F4 (null) | 仍不超过最好静态点 | steady 表面 +21.26%、bursty −4.01% |

**F1-as-stated 的失败源于我的预测缺陷，不是阶梯错误。** 我把接纳 cap 当成了执行
宽度：cap24 的实际宽度中位数只有 14–16，从未达到 24，所以它付的是桶 16 的价格。
按实际宽度分组后，阶梯在全新引擎上完美复现：

```text
aligned  w17-24: 9.284 9.222 9.293 9.244 9.432 9.379 9.403 9.344
sealed   w17-24: 9.360 9.304 9.364 9.369 9.477 9.450 9.508 9.463
最大逐宽度差: 0.125 ms
```

两种形式都报告，只报后者等于事后移动标靶。

### 机制为何失败：cap 是上界，不是执行宽度

| 运行域 | 目标在捕获点 | 实际宽度也在捕获点 | **目标对齐但宽度未对齐** |
|---|---:|---:|---:|
| aligned steady feedback | 目标中位 8 | 实际中位 9 | **0.845** |
| sealed steady feedback | 目标中位 12 | 实际中位 10 | 0.750 |
| aligned steady static | 目标中位 20 | 实际中位 14 | 0.463 |
| aligned bursty feedback | 目标中位 12 | 实际中位 16 | 0.006 |

**非抢占 cap 只限制新接纳。** 已有请求继续 decode，实际宽度由到达、完成与 cap 三者
平衡决定，因而遍历所有整数并停在平衡点上，而非停在档位上。steady 域 84.5% 的 step
目标已在捕获点、宽度却不在。bursty 的 0.006 不是 ladder 的功劳：到达本就是 8 个一组，
恰好落在捕获点。

两个 aligned feedback episode 的**动作序列完全相同** `[24,16,8,16,8]`，
waste 却为 0.2066 与 0.2516、goodput 3.9721 与 3.6275。相同动作、不同结果，
这是执行噪声的直接测量。

### steady 正信号为何不可主张

| arm | goodput 两次重复 | 相对差 |
|---|---|---:|
| steady_static16 | [3.8415, 2.4258] | **58.36%** |
| steady_shadow32 | [1.1443, 1.5355] | 34.19% |
| steady_static24 | [1.8909, 2.2955] | 21.39% |
| steady_feedback32 | [3.9721, 3.6275] | 9.50% |
| bursty 全部 arm | — | 0.45–1.22% |

"最好静态点" 3.134 是 static16 两次重复的中位数，而其单次重复 3.8415 已落在
feedback 区间 [3.6275, 3.9721] 内。`+21.26%` 完全可由 static16 的重复方差解释。
DECISIONS.md 预先写明"正 F4 需受控重复才能报告"，据此**不主张该收益**。

bursty 噪声底仅 0.45–1.22%，故 bursty 的 **−4.01%** 是可信负结果。

## 明确的解释边界

- 第一段为观测性重建，未分叉执行任何策略；`interference.json` 中 group2/4/8 是
  纯算术上界，不含 hold 的 TTFT 代价与队列动力学；"去干扰后达标数"是减法。
- **padding waste 不单调预测 goodput。** steady cap32 waste 0.161 却曾 goodput 最优。
  阶梯是一个新的成本项，**不是新的目标函数**。
- 单模型、单卡、32 条重复文本、固定 128/128 token、无 EOS；不涉及 EP/NCCL、多卡、
  生产 serving，也不主张 vLLM 默认捕获列表不合理。
- CPU 可行性检查另证：bursty 域无合并动作空间（到达已成 8 组）；steady 域 hold 必须
  ≥ 到达间隔（25 ms 完全无效，50 ms 得 16/32 个事件但花掉 TTFT SLO 的 25.5%）；
  `group_size` 在此 trace 上是死参数（hold 总先到期，g4 与 g8 结果相同）。
- **本轮不含任何专家感知贡献。** 未触及 U/C、实测 HBM 或任何专家信号。

## 文献边界

不主张首次发现 CUDA graph padding。[Chiron](https://arxiv.org/html/2501.08090v1)
已用 ITL 反馈调 batch 上限，[Gimbal](https://arxiv.org/html/2606.15177v1) 已用专家
压力做引擎分配，[DA-MoE](https://arxiv.org/html/2607.23099v3) 已按 histogram 选
kernel。本轮的候选残差是那条被自己实验证实的**边界**：*在非抢占接纳下，量化的
执行成本阶梯真实存在且可复现，但接纳上限这一动作无法把批次放到量化点上*。
出现可主张的正收益前不扩张查新。

## 唯一下一最小实验

失败位置指出唯一有意义的下一步：**换动作，不换参数。** 需要一个能直接决定执行宽度
的动作，而不是宽度上界 —— 即在 step 边界按当前 running 集合决定"本 step 是否延迟
接纳到下一个捕获点"，动作对象是单个 step 的 batch 组装，而非 cap。

在实现之前，先做一个更便宜的判定：从本轮 48 个 episode 的原始 step 序列，统计
**存在多少 step 只需 ±1~2 个请求即可落到捕获点，且该调整不违反非抢占语义**。
若这类 step 占比过低，则该动作空间在此运行域不存在，应停止本机制族，
不再更换 ladder、阈值或反馈规则。该检查为 CPU-only、零 GPU 成本、**UNRUN**。

不实现第三个 selector，不给失败的 ladder 加预测器，不把本轮负结果外推为所有
MoE 并发控制 NO-GO。

## 复跑与保留

```bash
cd refine-logs/expert_saturation/outputs/admission_capacity/20260908_step_cost_surface_r01
S1=../20260906_native_knee_r01/gpu_results
S2=../20260906_native_knee_r01/policy_probe/gpu_results
python3 analyze_step_cost.py     --results-dir $S1 --results-dir $S2 --output-dir /tmp/a1
python3 analyze_interference.py  --results-dir $S1 --results-dir $S2 --output-dir /tmp/a2
python3 analyze_alignment.py     --results-dir $S1 --results-dir $S2 --output-dir /tmp/a3
python3 analyze_step_cost.py     --results-dir next_experiment/gpu_results --output-dir /tmp/a4
python3 adjudicate_predictions.py --aligned-dir next_experiment/gpu_results \
  --sealed-dir $S2 --output-dir /tmp/a5
python3 analyze_width_mismatch.py --run aligned=next_experiment/gpu_results \
  --run sealed=$S2 --output-dir /tmp/a6
python3 ../../../experiments/admission_capacity/check_coalescing_feasibility.py \
  --prepared-dir ../20260906_native_knee_r01/prepared --output-dir /tmp/a7
python3 -m unittest discover -s ../../../experiments/admission_capacity -p 'test_*.py'
```

测试 55/56 通过（`test_runtime` 需 torch，本机无 CUDA；与本轮改动无关）。
`next_experiment/gpu_results/*/commands.txt` 为实际执行命令。全部 24 个 episode
及两次重复保留，未按结果挑选或替换 canonical（预声明 forward 为 canonical、
reverse 为受控重复）。原始封存 cell 未被修改，`DECISIONS.md` 未在见到结果后改动，
未 push，未改 `docs/current/README.md`。GPU 任务已自然退出，设备已释放。

| 结束字段 | 当前边界 |
|---|---|
| Verdict | `MEASUREMENT_ONLY`；capture-aligned ladder 在本运行域 `CONDITIONAL_NO_GO` |
| Evidence type | `REQUEST_LEVEL` / 原生 vLLM 进程内；单模型单卡、有限 episode |
| 成立 | 量化成本阶梯 + 跨引擎独立复现（0.125 ms）+ cap 无法对齐执行宽度（steady 0.845） |
| 不成立 | F1-as-stated（预测混淆 cap 与 width）；F2 实质失败；steady F4 在噪声内 |
| Strongest baseline | 同引擎实测 static8/16/24/32 与 shadow32；noise floor 已量化 |
| Oracle / unmeasured | step 级宽度动作、U/C 增量、第二模型、EP 全部 `UNRUN` |
| Reopen | 先证明 ±1~2 请求即可落到捕获点的 step 占比足够；否则停止本机制族 |
| Claim ceiling | 一条被实验界定了适用边界的量化成本规律；**不是**已验证的调度方法 |

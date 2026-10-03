# 准入可行性包络：一个闭式模型解释了四次失败，并给出可前置检查的判据

2026-09-12。分支 `agent/publish-current-moe-code`，HEAD `2a37765`。
**本轮 GPU 新执行数为 0。** 建模与验证全部基于 64 个已封存 episode，原始数据未修改。
GPU 机器已就位，环境搭建进行中（见末节）。

## 本轮研究问题

上一轮（[`20260911_aimability_r01`](../20260911_aimability_r01/REPORT.md)）否证了
两个解释——信息不足、通道不对——并观察到第三个：短文本 steady 域的联合 SLO 是
刀口统计量，cap 只是把 TTFT 失败换成 TPOT 失败。但那只是**观察**。

> 能否用系统侧实测量建一个闭式模型，**从第一性原理推出**该运行域为何无判别力，
> 并给出可在跑任何机制之前使用的判据？

## 两个实测原语

从 64 个配对 episode 重建（收据对齐法，覆盖墙钟 99.2%）：

**1. 纯 decode 成本阶梯 c(w)**，按 CUDA graph 捕获桶量化：

| 桶 | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| c(w) ms | 2.843 | 4.070 | 5.247 | 6.759 | **8.432** | **9.319** | 9.933 |

**关键巧合：TPOT SLO = 9.0 ms 恰好落在桶 16（8.432）与桶 24（9.319）之间。**
桶 24 及以上在**零负载下**就已违反 SLO。

**2. prefill 干扰税**（1090 个配对 mixed/pure 样本）：

```
tax = 5.2192 + 0.007780 x prefill_tokens   [ms]
```

prompt=128 时为 **6.215 ms/次接纳**。这是**每个驻留请求**都要付的，不只是新到者。

## 闭式模型

稳态下到达率 λ 的请求在自身驻留期内被 `λ·(L·τ)` 次接纳打断，故

$$\tau = c(w) + \lambda\,\tau\,(a + bP) \quad\Longrightarrow\quad \boxed{\tau = \frac{c(w)}{1 - \lambda\,(a+bP)}}$$

极点在 `λ = 1/(a+bP) = 160.9 req/s`。联合 SLO 可行需**同时**满足：

| 约束 | 形式 | 含义 |
|---|---|---|
| (T) | `τ ≤ TPOT_SLO` | 逐 token 够快 |
| (Q) | `λ ≤ μ(w) = w/(τL)` | 服务率覆盖到达率（TTFT 有界的必要条件） |

解出每桶的可行到达率上限：

| 桶 | c(w) ms | λ_max from (T) | λ_max from (Q) | **可行 λ** | 绑定约束 |
|---:|---:|---:|---:|---:|---|
| 1 | 2.843 | 110.07 | 2.70 | 2.70 | throughput |
| 4 | 5.247 | 67.10 | 5.74 | 5.74 | throughput |
| 8 | 6.759 | 40.07 | 8.74 | 8.74 | throughput |
| **16** | **8.432** | **10.15** | 13.57 | **10.15** | **TPOT** |
| 24 | 9.319 | **0**（c ≥ SLO） | n/a | **0.00** | TPOT_at_zero_load |
| 32 | 9.933 | **0**（c ≥ SLO） | n/a | **0.00** | TPOT_at_zero_load |

**临界工作点：桶 16，λ_max = 10.15 req/s（到达间隔 98.5 ms）。**

## 留出验证：这是预测，不是拟合

`c(w)` 与 `(a,b)` 是拟合量，因此**在同一批 episode 上验证会循环论证**。
原语在一个 campaign 上标定、预测另一个，反之亦然。**τ 公式的结构不是拟合的——是推导的**，
所以用实际宽度与实际接纳率预测实测 TPOT 是真检验。

| 指标 | 值 |
|---|---|
| 中位相对误差 | **3.25%** |
| p90 相对误差 | 6.09% |
| 最大相对误差 | 16.38% |
| TPOT 判决一致 | **60/64** |

### 模型独立命中了分界点

| steady 分组 | 实测接纳率 λ 中位 |
|---|---:|
| TPOT 通过（20 个） | **8.19** req/s |
| TPOT 失败（12 个） | **12.34** req/s |
| **模型预测的可行上限** | **10.15** req/s |

模型的可行上限恰好落在通过组与失败组之间。它**没有**见过这个分界。

## 这解释了全部四次失败

冻结负载的名义到达率是 20.0 req/s（50 ms 间隔）。

$$\frac{20.0}{10.15} = \mathbf{1.97\times}$$

**该运行域从一开始就在可行包络的两倍处运行。** 系统饱和，没有调度余量可找。
四个机制不是控制得不好，是在一个**结构上不可能有收益**的工作点上测量。

### 而且刀口打败了已验证的模型

4 个判决不一致**全部在 cap16**，即临界桶：

| | 值 |
|---|---:|
| 模型在这 4 例的相对误差 | 2.0% |
| 判决所需精度 `\|pred − SLO\|/SLO` | **0.68%** |

模型精度 3.25% 优于本仓库任何已有预测，**但决策需要 0.68%**。
这独立地确认了上一轮的刀口诊断：**在该工作点，不存在足够精确的预测器**——
不是因为模型差，而是因为决策边界比任何可达精度都窄。

## 交付：可复用的前置检查

`experiments/admission_capacity/check_regime_discriminability.py`。
只需静态 cap 扫描，**不需实现任何控制器**。

用已封存数据自检，独立重现了诊断：

| campaign | 引擎 | 域 | C1 裕度 | C1 | C2 翻转/1% | C2 | C3 摆幅 | C3 | 裁决 |
|---|---|---|---:|---|---:|---|---:|---|---|
| r01 | forward | bursty | 8.0% | **NO** | 0.0% | yes | 75.0% | yes | NON_DISCRIMINATING |
| r01 | forward | steady | 7.2% | **NO** | 6.2% | yes | 6.2% | **NO** | NON_DISCRIMINATING |
| r01 | reverse | steady | 6.2% | **NO** | 6.2% | yes | 6.2% | **NO** | NON_DISCRIMINATING |
| repeat | forward | steady | 9.4% | **NO** | 6.2% | yes | 28.1% | yes | NON_DISCRIMINATING |
| repeat | reverse | steady | 8.3% | **NO** | 3.1% | yes | 18.8% | yes | NON_DISCRIMINATING |
| （另 3 行 bursty） | | | 8.6–8.9% | **NO** | 0.0% | yes | 75.0% | yes | NON_DISCRIMINATING |

**8/8 全部无判别力**，且交换签名被自动检出：
`forward/steady 联合摆幅 6.2%，而 TTFT 失败摆幅 75%、TPOT 失败摆幅 81%`。

## 可行运行点的推论（未执行，待验证）

模型给出一个**可证伪的处方**：把到达间隔从 50 ms 放到 **≥98.5 ms**，
工作点进入桶 16 的可行区，此时：

| 预测 | 证伪条件 |
|---|---|
| P1 到达间隔 125 ms（λ=8.0）时 C1/C2/C3 全部通过 | 任一不通过 → 包络未捕捉到真实约束 |
| P2 桶 24/32 即使在 λ→0 也无法联合达标 | 出现达标 → c(w) 阶梯或 SLO 口径有误 |
| P3 λ=8.0 处最佳 cap 落在 16，而非 engine max | 最佳为 32 → (Q) 约束被高估 |

这三条只需一次静态扫描即可判定，**不需要控制器**。

## 结论边界

| 字段 | 结论 |
|---|---|
| Verdict | `MODEL_VALIDATED_OUT_OF_SAMPLE`；`REGIME_INFEASIBLE_BY_CONSTRUCTION (short-context steady @ 20 req/s)` |
| Evidence type | 64 个封存 episode 的建模与留出验证；零新 GPU 执行 |
| 成立 | 闭式 τ 模型留出误差中位 3.25%、判决一致 60/64；可行上限 10.15 req/s 独立命中通过/失败分界；冻结负载在 1.97× 可行包络处 |
| **收窄的旧结论** | 四个机制的 NO-GO 现有**结构性原因**：工作点本就不可行，不是控制器缺陷 |

### 明确不主张

- **不主张任何策略收益。** 本轮零执行，模型不产生调度动作。
- (Q) 是 TTFT 有界的**必要**条件，不是充分条件；它不说明哪个具体请求超时。
- 稳态均值模型：**不预测尾部**，把到达当速率而非序列，**无法描述 bursty**。
  bursty 行的 C1 失败应按此理解，不可当作 bursty 的可行性结论。
- `c(w)` 与 `(a,b)` 是单模型、单卡、单引擎构建、固定输出长度、固定 prompt 长度下的实测；
  不外推到其他模型、EP/多卡、变长上下文。
- 极点 160.9 req/s 是模型的数学极点，**不是实测容量**，远超已测范围，不得引用。
- P1–P3 是模型的预测，**目前全部 UNRUN**。
- C1–C3 阈值是本仓库证据支持的经验值，不是通用常数。

## 唯一下一步

**不改变下一个实验的身份**——仍是已冻结的
[batched 专家并集测量](../20260910_expert_union_r01/DECISIONS.md)。
它是纯结构测量，`U` 的值与运行域判别力无关，因此本轮结论不影响其有效性。

**新增的零成本一步**：该实验的同一次引擎启动内，顺带跑 P1–P3 的静态扫描。
理由是它能一次性判定"是否存在可判别的运行点"，而这是此后**任何**机制实验的前提。

## GPU 环境状态

新 5090 机器 `connect.weste.seetacloud.com:23478`，RTX 5090 / 32607 MiB / 空闲，
驱动 595.71.05，Python **3.12.3**（与全部封存基线逐位一致）。全新空盘。

按 [`20260910_expert_union_r01/SETUP_STATUS.md`](../20260910_expert_union_r01/SETUP_STATUS.md)
记录的三个坑规避：全局换清华源（阿里源对 303 MB wheel 挂死）、
`HF_HUB_DISABLE_XET=1`（xet 返回 401）、带重试的下载脚本（孤儿锁）。

| 项 | 状态 |
|---|---|
| venv `/root/autodl-tmp/expert-saturation/vllm-0.26` | 已建（路径与封存部署一致，脚本零改动可复用） |
| `vllm==0.26.0 transformers==5.15.1` | 安装中（torch 2.11.0 已过，triton 3.8.0 248 MB 中） |
| 模型 `OLMoE-1B-7B-0924@6d84c485` | 下载中，12 文件，无锁死 |
| **GPU 采集** | **UNRUN** |

## 复算

```bash
cd refine-logs/expert_saturation/outputs/admission_capacity/20260912_feasibility_envelope_r01
python3 analyze_envelope.py \
  --campaign ../20260908_capture_ladder_paired_r01 \
  --campaign ../20260908_capture_ladder_paired_repeat_r01 \
  --output-dir analysis

# 前置检查（可用于任何新运行域）
cd ../../../
python3 experiments/admission_capacity/check_regime_discriminability.py \
  --paired-dir outputs/admission_capacity/20260908_capture_ladder_paired_r01 \
  --paired-dir outputs/admission_capacity/20260908_capture_ladder_paired_repeat_r01 \
  --output-dir /tmp/disc
```

原始 raw 未修改，未 push，未改 `docs/current/README.md`。
本轮为已有负结果提供**结构性解释**并收窄其适用范围，不推翻其测量事实。

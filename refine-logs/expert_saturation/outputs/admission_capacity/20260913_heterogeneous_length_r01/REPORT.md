# 异构输出长度：模型前提被证实，但机制在该域无动作可做

2026-09-13。HEAD `7059fc98e0d98a127ff4edda86d44f7f634d26f4`，未提交。
本会话新增 GPU 执行 **8 格**（4 有效 + 4 保留失败），远端输出在 `/root/het-r01`
（系统盘，避开已 97% 满的 `/autodl-tmp`，未动任何其他会话的数据）。

设计与四条预注册判据在见到任何 GPU 数据前冻结于 [DECISIONS.md](DECISIONS.md)。

## 1. Verdict

**MEASUREMENT_ONLY。H1 成立；H2/H3/H4 在异构臂上因我的设计缺陷不可测，按预注册记为
`INVALID_NO_ACTION`。** 另外测到两个新结果和一个模型限制。研究问题仍 `OPEN`，
但**适用范围被本轮收窄**。

## 2. 结果

四个 native 格全部 42/42 完成，两 block 高度一致：

| 格 | wall s | 吞吐 | max-ITL s | 完成跨度 s | 峰值 KV | 抢占 | 步数 | 均宽 | 步均 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hom block0 | 40.271 | 1.0429 | 1.3903 | **3.508** | **102.0%** | 1 | 2189 | 38.4 | 17.73 |
| hom block1 | 40.001 | 1.0500 | 1.3322 | 3.501 | 101.9% | 1 | 2189 | 38.4 | 17.61 |
| het block0 | **49.695** | 0.8452 | 0.7892 | **33.955** | **68.7%** | **0** | 3156 | 26.7 | 15.32 |
| het block1 | 49.561 | 0.8474 | 0.7539 | 33.800 | 68.7% | 0 | 3154 | 26.7 | 15.28 |

两臂的预留块数（8064）、总输出 token（86,016 计划 / ~84,200 实测纯 decode）、
均值输出长度（2048）完全相同，唯一差别是 CV 0 vs 0.500。

### H1 成立且幅度很大

完成跨度 **3.51 → 33.96 秒（9.7 倍）**。异构确实制造了完成离散，负载设计在这一点上有效。

### 新结果一：异构在同等 token 下慢 23.4%

`49.63 / 40.14 − 1 = +23.6%` wall，吞吐 −19.1%。**这不是策略差异**，两臂都是原生 native。

机制是步数，且可无拟合陈述：

| | 步数 | 纯 decode token | 均宽 | 步均成本 |
|---|---:|---:|---:|---:|
| 同构 | 2189 | 84,130 | 38.4 | 17.73 ms |
| 异构 | 3156 | 84,251 | 26.7 | 15.32 ms |

**均宽降 30.5%，但步均成本只降 13.6%，于是步数涨 44.2%。**
步成本对宽度强次线性——这是同一条规律的第三个实例
（前两个：safe29 的 941 步跑在宽度 3；轮转靠维持高宽度换来免费）。

逐宽度实测（异构臂同一 episode 内）：`g(42)=0.374`、`g(21)=0.697 ms/token`，**1.87×**。

### 新结果二：异构自行解除了 KV 压力

峰值 KV **102.0% → 68.7%**，抢占 **1 → 0**。

原因是我的赤字对齐只对齐了**预留总和**，没有对齐**时间上的峰值**。
`full_sequence_must_fit` 在准入时按 `min(num_tokens, max_model_len)` 预留，而准入时
`num_tokens` 只是 prompt（1024 → 64 块），真正的占用随生成增长。异构下 21 条短请求
在 L=1024 退场，此时长请求才到 ~1024，全体峰值为 `42 × ceil(2048/16) = 5376` 块
= 70.1%，与实测 68.7% 吻合。

**这是一个关于 runtime 的结构事实**：在 `reserve_full_isl` 下，输出长度异构会
自行削平 KV 峰值。它不是我要的实验条件，但它本身值得记录。

### H2/H3/H4：不可测，按预注册作废

异构臂无抢占 → 轮转无动作。两个 `het-rotate` 格跑完 42 请求后由实现自身报
`rotation never applied a forced preemption`，即 `INVALID_NO_ACTION`。
冻结设计 §"证据上限"已写明这一情形须如此处理，**不解释为科学负结果**。

## 3. 模型限制：三项线性成本模型不可辨识

我尝试把 `c(w)` 升级为物理分解 `t = γ + α·w + β·C`（γ 每步固定、α 每请求
dense、β 每上下文 token 的 attention 扫描）。

**域内表现很好**：在 `hom block0` 标定，预测 `hom block1` 总时间 +0.67%、
`het block0` +3.94%（中位相对误差 4.3% / 10.1%）。并且 `γ × Δ步数 = 9.135 s`
解释了两臂 decode 时间差 9.527 s 的 **95.9%**。

**但跨域失败，且系数不稳定**：

| 标定集 | γ ms/step | α μs/req | β ns/ctx |
|---|---:|---:|---:|
| 新域 hom | 9.447 | 3.70 | 102.6 |
| 新域 het | 8.552 | 18.32 | 101.0 |
| 原域 native | 4.393 | 281.49 | 54.6 |
| 原域 safe29 | 5.430 | 96.15 | 93.9 |

用新域系数预测原域四臂，总时间高估 **+11.8% ~ +24.8%**，p90 逐步误差 63–116%。
同一 trace 内把主宽度的步按上下文三分，β 的分段斜率比值在 0.48–1.23 之间摆动，
safe29 甚至给出负斜率。

**诊断：在 serving trace 里 w 与 C 强共变**（`C ≈ w × 每请求上下文`，而后者随
episode 单调增长），三项无法分别识别；拟合只是在各域内插值。丢掉常量项后更差
（总误差 −9.5% ~ −21.6%），说明常量项必要但不可外推。

**结论：保留无拟合的口径（步数 × 步均成本），不主张物理系数。** 要分离宽度与
上下文需要固定上下文、只变宽度的受控微基准，那不是 serving episode 能做到的。

## 4. 这对研究问题意味着什么

**收窄（明确承认）**：我一直研究的"已有 decode 被不成比例暂停"现象，
**需要同构长输出**。一旦输出长度离散，KV 峰值自行下降、抢占消失、轮转无对象。
原域四臂的结论（轮转把 max-ITL 从 4.46 降到 1.01 秒）**只在完成时刻聚集的域内成立**。

**未被否定**：同构长输出域本身是真实场景（批量摘要、固定长度生成、
beam/self-consistency 的等长采样）。原域结论未被本轮推翻，只是范围被钉住。

**新增一条待解释的量**：异构负载在同等 token 下贵 23.4%，且完全由步数解释。
这与策略无关，属于负载—执行效率问题，是本研究问题内的自然延伸。

## 5. 上机定位的四处实现阻塞（非科学结果）

冻结 runner 对原域形状有硬编码，均已修复并记录：

| # | 位置 | 症状 |
|---|---|---|
| 1 | `safe_static.qualify` 断言 `(3072,1024,32)` | `QUALIFICATION_FAILED` |
| 2 | `run_probe --cap choices=[29,32]` | 准入卡 32，峰值 KV 仅 78.2%、抢占 0 |
| 3 | `rotation_native.install(expected_requests=32)` 两处 | cohort 永不形成，四个 rotate 格 INCOMPLETE |
| 4 | 缺 `HF_HOME` / `VLLM_USE_FLASHINFER_SAMPLER=0` | 模型找不到 / FlashInfer JIT 报 sm75 |

全部失败原件保留在远端 `/root/het-r01/results/FAILED-*` 与 `DIAG-cap32-no-pressure`。
其中 #2 的诊断格本身有用：它是"同负载、并发受限"的对照（峰值 78.2%、抢占 0、
完成跨度 21.56 秒），说明本域的压力确实来自并发而非赤字算术。

另有一格 `hom-rotate` 因与其他会话共占显存 OOM，同样保留，不重跑充数。

## 6. 上机前的 CPU 验证（10/10 通过）

`test_heterogeneous_length.py` 覆盖逐请求长度的实际请求值、warmup 标量回退、
长度表不全的拒绝、标量与表同时给出的拒绝，以及冻结负载的赤字对齐与文档复用声明。
**它抓到了一个会在 GPU 上立即崩溃的真实 bug**：我把 `arrival_traces_s` 写成列表，
而 `capture_episode` 按 regime 名索引字典。

`analyze_length_dispersion.py` 在同构四臂上逐项复现旧值 **5/5**（偏差 ≤20 ms），
确保新分析器无回归。

## 7. 不主张

- 每格 n=1，重复仅 forward/reverse 一对，与顺序效应混淆。两 block 的 wall 差
  0.27 / 0.13 秒只是**已观测重复差**，不是噪声界或显著性。
- 42 个 prompt 来自 32 篇文档的不重叠切片，**不是 42 篇独立文档**，
  不支持任何总体或泛化主张。两臂消费完全相同的 prompt，故臂间可比。
- 新域与原域**不可按绝对数字混排名**（prompt 3072→1024、输出 1024→2048、
  N 32→42、足迹 256→192 块）。
- 步成本的任何分项都**不是 kernel 时间**；步时长用相邻 `start_s` 差，
  区间内含执行、采样、交付与记账。
- 未测：任务质量、输出一致性、Oracle/上界、HTTP 服务、多卡 EP、专家信号增量。
- 参考 SLO（TTFT 5 s / 平均 TPOT 0.2 s）四格均 42/42 通过，该口径不约束最长停顿。

## 8. 唯一下一步

**在异构域重建 KV 压力，否则该域无法检验任何恢复机制。**

本轮已证明：仅对齐预留总和不足以对齐峰值。正确做法是**对齐时间峰值**——
按"短请求退场时刻的全体占用"反解请求数，而不是按预留总和。由本轮实测，
异构 42 请求峰值 68.7%，要到 ~100% 需约 `42 × 1.0/0.687 ≈ 61` 请求
（须重新核算并受 `max_num_seqs` 与图捕获上限 64 约束）。

若该配置能让异构臂出现抢占，H2/H3/H4 即可按原判据检验，且冻结阈值不变。
若受 64 的捕获上限阻挡，则如实记录"异构域在本硬件下无法复现该压力"，
并把原域结论的适用范围永久钉在同构长输出上。

## 复现

```bash
# 输入（本地，无需 GPU）
python3 refine-logs/expert_saturation/experiments/admission_capacity/prepare_heterogeneous_length.py \
  --sealed-prepared refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_fourarm_r01/execution/frozen/inputs_preparation/prepared/long \
  --output-dir <新目录>/prepared
# 执行包
python3 .../prepare_heterogeneous_package.py --sealed-package <四臂frozen> --prepared <新目录>/prepared --output-dir <新目录>/execution
# CPU 检查
python3 -m unittest refine-logs/expert_saturation/experiments/admission_capacity/test_heterogeneous_length
# 分析
python3 .../analyze_length_dispersion.py --cell hom=<...>/block0-hom-native --cell het=<...>/block0-het-native --output-dir <新目录>
python3 .../analyze_step_cost_model.py --calibrate hom=<...> --holdout het=<...> --attribute hom:het --output-dir <新目录>
```

执行包 SHA256 `721092784176ce050d9e7c8b4885b8470e1bd0badd0487dca00f284e0c6d6453`，
负载 `workload_sha256 f6a6b55cea0139fa8fc79b1caf4d45334519ed0f2df45373aaf0317bb3d384ba`。
回传归档 `het-full.tar.gz` SHA256 `e7eb84e017a62e1a14486587ed8b53e16c706ad47240b2373261e5281b5352bf`，
本地与远端双向一致，远端原件保留。

| 固定报告项 | 结论 |
|---|---|
| Verdict | `MEASUREMENT_ONLY`；H1 成立，H2/H3/H4 `INVALID_NO_ACTION` |
| Evidence type | `REQUEST_LEVEL` / 原生 in-process GPU，新运行域 |
| Measured | 完成离散度、异构的 23.4% wall 代价及其步数归因、峰值 KV 自削平 |
| Not measured | 异构域下的轮转/恢复机制（无动作）、质量、Oracle、多卡 |
| Strongest baseline | 同域同构 native，两 block 各一次；原域结果未混排 |
| Claim ceiling | 长度离散在本域使 KV 压力消失并使同等 token 贵 23.4% |
| Failure category | 设计缺陷（对齐了预留和而非时间峰值），非机制失败 |
| Reopen | 按时间峰值重算请求数后，异构域能否复现抢占 |

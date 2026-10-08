# 同后端 LTR 风格强基线：CPU 交付与比较合同

## 接续与证据边界

开始时本地 `agent/publish-current-moe-code` 的 HEAD 为 `76d6d888de42081c63cd440a8a67623161d8181f`。G/H 实际完整请求说明 selected/eager 缩短了最大生成停顿，但原生 full-save 在输出吞吐、平均完成和多数请求自身停顿上保有优势；不能称 eager 全面获胜。cooldown/window/headroom/growth-predictor 的旧 formulation 已停。决定性近邻问题是：在同一 selected-save 后端上，经有限校准的 LTR 风格等待提权与服务量子能否覆盖 eager 的取舍。

这是 **CPU 可检查的基线实现与事前比较合同**。本轮没有启动 SSH、GPU、远端模型或完整请求新运行；所有新策略的原生 GPU 生命周期、吞吐、goodput 和服务收益均为 **UNRUN**。历史 r02 的 `pkg/` 和 `cpu_checks.json` 在当前 sparse checkout 中缺失，不能把其文档或回执当作本地已运行的代码。新实现位于本目录，封存的 r02 以及 `CURRENT_EXPERIMENT.json`、`RESULT_LEDGER.md` 均不属于本交付。

## 原论文、官方代码与移植范围

| 近邻 | 已覆盖的决策 | 此处的可比范围 |
|---|---|---|
| [LTR §4.3 / Algorithm 1](https://arxiv.org/pdf/2408.15792)、[官方实现](https://github.com/hao-ai-lab/vllm-ltr) | waiting/running/swapped 的饥饿提权、优先级排序及 `PriorityQuantum`；原目标含 TTFT 与后续 token 等待 | 移植在线计数和正计算调用量子，共用本地 selected `OffloadingConnector`。缺官方预测器、全队列目标、多 victim 及 CPU-SWAP；只能称 **LTR-style 组件**。 |
| [Andes §§4.2–4.3](https://arxiv.org/html/2404.16283v2)、[作者实现](https://github.com/AmberLJC/vllm-0.6.1) | 目标/peer QoE、KV 占用和交换/重算成本的净收益选择 | 证明成本感知恢复不是新颖性空白；其客户端消费模型和完整 refiner 未在此复现。 |
| [FastServe](https://arxiv.org/html/2305.05920)、[官方实现](https://github.com/LLMServe/FastServe) | 多级反馈队列、饥饿提权与主动状态交换 | 证明单纯等待提权及交换不是新动作。 |
| [TokenFlow](https://arxiv.org/html/2510.02758)、[官方实现](https://github.com/SJTU-RTEAS/TokenFlow) | 缓冲感知工作集、host KV 写入及重叠 load/evict | 它面向消费速率；不把此处“首个新输出和连续交付”误写成其遗漏。 |

官方 LTR 的优先级扫描在等待阈值达到时重置等待计数并重新授予完整量子；量子耗尽在下一次排序时降权。此移植把一次实际 `num_scheduled_tokens > 0` 的调度调用计为一次服务，prefill、重算、decode 均计；纯 pending load、零分配不计。一次调用分配 1 或 128 个计算位置都只扣 1。首个新输出不释放量子。没有 predictor score 的同优先级 tie 采用稳定到达/原队列顺序，并在结果中披露该退化。

## 本地代码与动作闭环

- `ltr_fair_policy.py`：`observe → propose → accept → feedback`，返回 `no-op` 或可执行的直接恢复/单 victim 恢复意图；扫描下一个可执行 boosted target，避免首个不可筹措目标造成队首阻塞。只读当前请求状态、实际持块、空闲块和序列槽；不读取未来 EOS、route 或输出长度。
- `ltr_fair_native.py`：在固定 SHA 的 vLLM 0.26 native `Scheduler.schedule` 上安装互斥钩子，与 `staged_save_contract.py`、`native_store_delta.py`、`rotation_native.py` 共享物理/保存合同。接受 intent 后进行 victim 的一调用准备、当前前缀/块归属/新 store 注册检查、下一调用状态重检、真实抢占、native flush、load 等待与实际分配反馈。direct-free 动作没有无谓 victim。状态变化、目标结束、不可筹措、外部 pending 队列或自然抢占时取消/重评估，保留计数；无干预时原生路径继续。
- `test_ltr_fair_policy.py`、`test_ltr_fair_native.py`：只针对计数、扫描、量子、块归属、保存/加载、EOS/取消、资源不足、回退和原生事件反馈；allocator、transfer、输出仍为 CPU 替身。实际 tensor DMA、安装时 pinned 源和真实 EOS 需要原生 GPU 资格。
- `ltr_baseline_compare.py`：从每个策略独立运行的 `raw.json` 读取全部计划请求，验证 cohort/prompt/arrival/cap 与运行时物理资源回执，计算固定时域 goodput、真实输出吞吐、完成与停顿分布、逐请求改善/恶化、输出长度/序列/stop 变化，并按冻结规则选择合格校准点。失败或未完成保留且不能赢得校准。

当前 native runner 的接点是 warmup 与 connector cache reset 完成、scheduler 已 drain 后，且安装任何测量包装器之前：

```python
from ltr_fair_native import install_on_engine
from request_measurement import measure_episode

policy_data, uninstall = install_on_engine(engine, threshold=30, quantum=10)
try:
    raw = measure_episode(engine, workload, config, regime, 1.0, run_id,
                          max_seconds=180, record_preemptions=True)
finally:
    policy_data = uninstall()
```

`install` 会校验 native scheduler 源 SHA、connector 类型及无共享块所有权结构，并直接替换本实例 `schedule`；实际 GPU/host 字节数还需运行时回执。`uninstall` 必须在测量包装器先卸载后调用。这个调用链只说明如何接入现有 `LLMEngine`，不构成已经运行的 GPU 资格或结果。

`prepare` 只要求当前 target history 可由真实 free + 当前单 victim 可回收块筹措，并由原生 allocator 处理后续增长。此前对 **所有 running 下一块增长求和预留** 的过滤器已被[同前态 pinned-native CPU 反例](../../outputs/admission_capacity/20260915_joint_growth_decision_r01/I_PREPARE_SUM_GUARD_ADDENDUM.md)推翻为非必要限制，本实现不复活它。单 victim 是共同 selected-save 组件动作空间，不能用于宣称完整 LTR 劣于新机制；必须报告因多 victim 需求被 censor 的 boosted 机会。

## 冻结校准与消融

[配置 JSON](ltr_baseline_calibration_20260929.json) 固定 G64 开发输入、0.2 s 到达、自然 EOS/cap1024、4096 个 usable 16-token GPU KV 块（实际 8,592,031,744 B）及 16 GiB/8192 host KV 块；同一模型修订、vLLM 固定源、32 running、1024 scheduled、FCFS、full-history、APC off、`OffloadingConnector`。必须用运行时回执确认真实物理占用，不能只抄配置。父 cgroup 上限不是独立 host KV 预算，host KV 已包含在进程物理内存，不重复相加。

每格回执还必须记录实际 `policy_id`、完整 `policy_config`、模型修订、调度器源 SHA、输入 SHA、输出 cap/EOS 和调度容量，并与 JSON 严格一致；只换结果文件名不能伪造一格校准。`selected_eager` 的30次 absence/30次 residency、0.9 progress、每请求最多8次 absence、0全局 cooldown 及 most-output victim 明写在配置中。这个共同 selected-save 组件对照与另列的 native full-save 系统参照分开解释。

G 开发只给 LTR 风格组件 `T∈{30,200} × Q∈{1,10}` 四点，200/10 是[官方脚本](https://github.com/hao-ai-lab/vllm-ltr/blob/main/benchmarks/fair-lmsys-70B.sh)来源锚点。`T` 是无正分配的调度调用数，不是秒；G 的持续 waiting 区间中 200 调用约 3.2 s，单测 200/10 会无根据地削弱基线。与同窗口 selected/eager 相比，完整合法运行需实际输出率 ≥97%、已完成 cohort 的平均 arrival-to-completion ≤105%；在合格点中取每请求最大生成返回间隔的全局最大值最小者，平手按输出率、平均完成、低 T、低 Q。无合格点就报告无胜者，不扩网格。已有 G/H 的旧 3%/5% eager/current 判据保持原义，不追溯改为 native 或 LTR 对照。

| 对照 | 只变的因素 | 解释边界 |
|---|---|---|
| `ltr_t30_q10` vs `ltr_t200_q10`；`ltr_t30_q1` vs `ltr_t200_q1` | 等待提权阈值，Q 固定 | 比较整个策略轨迹，不能把结果当一个动作的反事实。 |
| `ltr_t30_q10` vs `ltr_t30_q1`；`ltr_t200_q10` vs `ltr_t200_q1` | 实际正分配调用量子，T 固定 | 1 次可能只够加载后的首输出，不保证冷重算成功。 |
| `selected_eager` vs 同底座 `recovery_compute_share.install(enabled=True)` | 仅打开已有默认关闭的冷重算 compute-share 钩子 | 只有固定输入自然出现该动作才运行 on/off；无动作就记录边界，不降阈值或造压力。此 pair 不参与 LTR 参数选择。 |
| `native_full` vs selected 各臂 | 保存范围和额外轮转均不同 | 完整系统价值参照，不能作单因素归因。 |

冷重算钩子的[独立 on/off 清单](ltr_compute_share_ablation_20260929.json)逐字段固定同一底座，两个 `policy_config` 仅 `compute_share_enabled` 不同；它不参与四点 LTR 选择。比较器也可读取该清单并输出逐请求差异，`selected_ltr_status` 为 `NOT_APPLICABLE`。若 on 臂没有实际 compute hold，结论为 `NO_ACTION`，不能拿相同结果宣称机制无效。

初步 goodput 前沿固定开发阈值 `(TTFT, max_gap)=(10,1.5),(20,3),(30,5)` 秒，以共同 180 s 计时窗、同一到达 cohort 计数；没有业务 SLO 主张。主校准仍按上述 max-gap 与完整服务预算。H128 已见过旧 eager/current 结果，只能称未用于 LTR 参数选择的迁移输入，不能称完全盲测。若正式确认，需要在新输入前冻结选中点与判据。

## 本地复核与之后的比较入口

从本目录执行，无需 GPU：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest \
  test_ltr_fair_policy test_ltr_fair_native test_ltr_baseline_compare
```

配置里的 `results/*/raw.json` 与 `resource_receipt.json` 是 **待执行路径**。仅在各臂取得独立原生结果且回执真实确认 GPU/host/保存范围后，运行：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B ltr_baseline_compare.py \
  --manifest ltr_baseline_calibration_20260929.json \
  --output /private/tmp/ltr-G-comparison-new.json

PYTHONDONTWRITEBYTECODE=1 python3 -B ltr_baseline_compare.py \
  --manifest ltr_compute_share_ablation_20260929.json \
  --output /private/tmp/ltr-share-ablation-new.json
```

输出使用新路径且拒绝覆盖。若结果缺失，比较器会直接失败；不得拿 G 旧 eager 轨迹离线回放成 LTR 的反事实运行。诊断先验证真实 store/load/flush、pending load 不扣 Q、首输出后的 Q 演进、自然抢占及 EOS；性能组待诊断通过并另行冻结后才可执行。本轮用户明确禁止 GPU 实验，因此结果状态保持 **UNRUN**。

## 当前决策

基线的新实际动作是：当可见 idle 达 T 时，即使 eager 的 absence/residency/progress 等门不触发，也可把首个可筹措的 PREEMPTED 请求提权，并在正计算分配的 Q 次调用内保持优先级；必要时通过当前真实 KV 所有权选一个合法 peer 做 selected-save/换出。它可能降低长等待，也可能增加 peer 停顿、保存及完成成本。CPU 测试只验证可执行意图与状态推进；尚无同后端完整请求结果支持服务收益。若校准后 LTR 已覆盖 eager 的取舍，当前主线没有独立方法增量；若仍有残差，先定位实际不同动作及完整请求代价，再决定是否测试默认关闭的新钩子。

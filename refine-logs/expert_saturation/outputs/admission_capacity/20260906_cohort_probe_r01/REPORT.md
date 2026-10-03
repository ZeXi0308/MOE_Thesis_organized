# 固定长度自然文本 cohort 的并发响应与受控复测

2026-09-06，分支 `agent/publish-current-moe-code`，HEAD `2a37765`，工作区新增实现未提交。

**Verdict：`MEASUREMENT_ONLY / CAP_RANKING_UNSTABLE`。并发确实改变请求级 SLO；
当前数据没有证明 U/C 提供稳定的动作选择增量，不能据此实现专家感知 Controller。**

本轮问题是：保持长度和到达序列，换自然文本后，cap 6 与 8 的相对收益是否变化？
新增两批文本的 32 个 GPU cell，并对发生排名反转的一批原配置复测 16 个 cell。
**48/48 新 cell 完成，48/48 前后进程检查通过，所有请求均完成，无失败或未完成请求。**
这不是连续 GPU 隔离证明，也不是 768 个独立文本样本。新数据只有两个各 16 行的
WikiText cohort；与前序第一批合计 48 个不同源行，未认证 article-disjointness。

## 已执行内容

- 单 RTX 5090，32,607 MiB；OLMoE-1B-7B-0924 固定 revision
  `6d84c48581ece794365f2b8e9cfb043c68ade9c5`，BF16。
- Torch 2.8.0+cu128、Transformers 4.57.6、eager attention、25 个 Torch CPU threads。
  保持一轮一个完整 prefill、全部 active decode 的 FCFS 非抢占规则。
- 每 cell 16 请求，128 prompt tokens、16 固定输出 tokens；steady/bursty 到达，
  首尾到达均为 0–1.5 秒。TTFT <= 5 秒、请求平均 TPOT <= 0.2 秒沿用探索配置。
- cap 6/8、OFF/ON、两次反序重复。每个策略都独立生成未来 KV、route 和 completion。
  cohort offset 16、32 在长度过滤后按源顺序选择，没有按 U/C 或收益选择文本。
- 原配置的 offset 32 复测使用新模型进程，安排在另一接纳动作复测任务退出后执行；
  本轮各个计算任务顺序运行，未杀其它进程。原始运行与复测分别保留。

执行前决策及追加复测的理由见 [DECISIONS.md](DECISIONS.md)。

## 请求级结果

下表为 telemetry OFF 的有限 episode goodput，单位为达标请求/秒。每格为四个
steady/bursty × repeat 的范围，不是置信区间。goodput 的 episode 墙钟分母包含
prefill、decode、KV 整理和循环成本；模型加载及 warmup 在服务计时前完成。

| 文本 / 执行 | cap 6 goodput | cap 8 goodput | 四组比较中 cap 6 / cap 8 胜出 |
|---|---:|---:|---:|
| offset 0，前序 bracket | 1.790–1.955 | 1.239–1.562 | 4 / 0 |
| offset 16，新文本 | 1.826–1.972 | 0.929–1.265 | 4 / 0 |
| offset 32，首次 | 1.379–1.959 | 0.926–2.257 | 3 / 1 |
| offset 32，原配置新进程复测 | 1.932–2.009 | 1.706–2.358 | 1 / 3 |

offset 32 的同一到达方式在重复间仍发生反转：

| 执行 / 到达 / repeat | cap 6 | cap 8 | 较高者 |
|---|---:|---:|---|
| 首次 / steady / 0 | 1.9594 | 1.7988 | 6 |
| 首次 / steady / 1 | 1.6288 | 1.0934 | 6 |
| 首次 / bursty / 0 | 1.3794 | 0.9257 | 6 |
| 首次 / bursty / 1 | 1.6372 | 2.2571 | 8 |
| 复测 / steady / 0 | 2.0040 | 1.7056 | 6 |
| 复测 / steady / 1 | 2.0089 | 2.3288 | 8 |
| 复测 / bursty / 0 | 1.9320 | 2.3575 | 8 |
| 复测 / bursty / 1 | 1.9779 | 2.3354 | 8 |

这证明“该运行域里选择哪个 cap”有实际请求后果，但尚未证明内容改变最佳 cap。
cap 8 的请求平均 TPOT 位于阈值附近，较小的执行时间差异会改变达标人数。
固定 cap 6 是当前应保留的强简单对照，不能称为所有时段或所有配置的最优策略。

## U/C 与解释边界

- 前序数据的 width 8 中 U 最大仅 0.734375，未出现专家全覆盖。相同 width 下
  U/C 仍有变化；C 的小 batch 取值离散，不能把数值上限解释为硬件拥塞。
- 新 cohort 每个 cap 的 ON 请求成员、decode index 与逐层 U/C 轨迹在四个
  steady/bursty × repeat 中均相同。offset 32 复测的八个 ON cell，与首次对应
  cell 的这些可见轨迹 **8/8 完全一致**；输入 workload 文件也逐字节相同。
- 尽管如此，ON 和 OFF 的 cap 排名都出现过反转。因而不能用这些重复作为
  多个独立的“压力决定最佳 cap”样本，不能把一次 cap 8 正收益归因为专家机制。
- 预先选定的首个满 batch、无 prefill、首个 completion 之前的窗口，也有相同
  U/C、不同执行时间。近期延迟、prefill 时间及运行时波动尚未被充分控制。
- ON 与 OFF 是各自执行的轨迹；观测成本已进入各自的 episode 时间。两者的
  goodput 差值不是纯遥测税，尤其不能在 SLO 门槛附近用该差值推算固定开销。

本轮没有定位时间波动来自 GPU clocks、host 调度还是其它运行时因素。
GPU 温度/功耗/频率只在 cell 边界采样，不能据此声称某个因素是根因。

## 代码、检查与复现

新增 `--request-offset`，输入准备和复用验证共用确定性选择函数；旧 prepared
无该字段时按 0 处理。四项定向检查通过，覆盖默认兼容、过滤后切片、越界/负值、
prepared 选择漂移及执行时覆盖拒绝。未改接纳、KV 或 decode 动作。
三个新运行实际执行的六个 Python 源文件均与当前本地字节一致；统计从 raw
请求时间重算，输入身份及已有 cell 检查全部通过。未扩展全仓审计。

- [首次两批 GPU 原始输出及日志](gpu_results/)
- [三 cohort 逐 cell 汇总](summary.json)；[汇总脚本](summarize_cohorts.py)
- [原配置复测重算](repeat_analysis/analysis.json)
- [顺序执行脚本](run_remote_cohorts.py)
- [前序容量曲线](calibration_capacity.png)；[绘图脚本](plot_calibration.py)

每个 raw 目录的 `commands.txt`、`config.json`、`environment.json` 保存实际命令、
配置和执行环境。复测使用相同 offset32_inputs，输出另存为 offset32_repeat。

## 判断与唯一下一步

| 项目 | 当前判断 |
|---|---|
| Evidence type / claim ceiling | 单模型单 GPU custom continuous runtime 的有限请求测量 |
| Measured | cap 的真实接纳/并发、完整请求 TTFT/TPOT/goodput、两批文本及原配置复测、ON 轨迹 |
| Unmeasured | U/C 超出普通状态的动作增量、动态 Oracle、在线 policy、native transfer、自然停止与 EP |
| Strongest baseline | 实跑固定 cap 6；普通状态动态规则尚未在本实验比较 |
| Oracle/headroom | 只有已测静态 cap 的 hindsight 排名；动态 Oracle 未运行 |
| Failure category | cap 排名的运行稳定性未解决，不能解释专家机制；不是 problem family NO-GO |
| Reopen / continue condition | 代表性运行时中的可重复 cap 响应，随后检查动作前 U/C 是否超出普通延迟反馈 |
| One next smallest experiment | 在已有 vLLM 环境做一个能触及 cap 的自然文本负载下的 cap 6/8 请求级对照，确认容量边界与重复稳定性 |

远端已只读核实 `/root/autodl-tmp/expert-saturation/vllm-0.26` 的包版本为
vLLM 0.26.0 / Torch 2.11.0 / Transformers 5.15.1；**本轮尚未运行该后端**。
原生对照先确认实际并发和排队是否暴露；若负载过低，另起并保留校准配置。

直接回答：**现在有真实的并发–SLO 权衡证据，但还没有能据此成立的专家感知
并发方法。原配置复测保留了不稳定性，下一笔计算优先验证代表性 runtime，
当前不增加 predictor，也不把未验证写成系统 NO-GO。**

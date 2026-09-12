# OLMoE 非抢占并发容量实验

本入口执行用户 2026-09-05 确定的容量主线，复用旧 continuous-decode 的 KV
拼接/拆分函数。它不改变旧 N0d 或历史容量记录，不要求先完成 N0e。

早期 custom 实验：**2026-09-06 已完成 64 个预训练 OLMoE / RTX 5090 GPU episodes：
静态扫描、cap=6 局部重复及真实非抢占 pulse 对照。`MEASUREMENT_ONLY`；U/C 方法增量未验证。**
本轮结果和唯一下一实验见 [GPU 实验记录](../../outputs/admission_capacity/20260906_gpu_pilot_r01/REPORT.md)。
其中 64 次执行复用同一组 16 条文本，不是 1,024 条独立请求。
首次阻塞记录见 [REPORT.md](../../outputs/admission_capacity/20260905_local_blocked_r01/REPORT.md)。
最新进入 GPU 前准备见 [准备记录](../../outputs/admission_capacity/20260905_pre_gpu_r01/REPORT.md)。

另已完成 [自然文本 cohort 与受控复测](../../outputs/admission_capacity/20260906_cohort_probe_r01/REPORT.md)：
两批新文本及一次原配置复测共 48 个 GPU episodes。相同输入与 U/C 轨迹下 cap 排名仍会变化，
当前不据此主张专家信息的动作增量；下一步优先做代表性 runtime 对照。

原生 vLLM 的 32 个 cap 6/8 episode 已完成，当前为 `MEASUREMENT_ONLY`：快速负载下
cap 8 的八组配对均更高，但原 SLO 全通过，且引擎 cap 改变了 CUDA graph 捕获计划。
见 [原生结果与下一步](../../outputs/admission_capacity/20260906_native_transfer_r01/REPORT.md)。

共同引擎配置下的 [接纳上限对照](../../outputs/admission_capacity/20260906_native_fixed_engine_r01/REPORT.md)
已完成 16 个 episode：cap 8 在 8/8 配对中 goodput 更高，主 SLO 达标数为 117/128，
cap 6 为 89/128。随后 [新输入复测](../../outputs/admission_capacity/20260906_native_fresh_cohort_r01/REPORT.md)
又完成 16 个 episode，保持相同执行代码、引擎和 SLO；cap 8 再次 8/8 胜，达标数
112/128，cap 6 为 82/128。两组各有 16 个不重叠输入；结论仍为静态测量，U/C 和动态增量未验证。

既有动作数据的 [信号窗口补充分析](../../outputs/admission_capacity/20260906_gpu_pilot_r01/SIGNAL_WINDOW_ADDENDUM.md)
发现八次升档前的四步 U/C 均混合 batch 宽度；这尚不能证明信号存活或动作选择增量。

最新 [固定 decode 边界对照](../../outputs/admission_capacity/20260906_step_action_r01/REPORT.md)
完成 32 个 GPU episodes：升档均缩短 episode，却全部恶化 TPOT 中位数，goodput 仅
4/16 配对改善、第二重复全部为负。同一文本组的动作前逐层 U/C 完全重现，尚无稳定动作分界。

原生32请求、128输出token的 [容量扫描与普通反馈pilot](../../outputs/admission_capacity/20260906_native_knee_r01/REPORT.md)
已完成44个episode（1408次执行、32条重复文本）。存在吞吐—TTFT—TPOT权衡，但四步ITL反馈
在四组对照中均未超过最好已测静态点；steady静态排名仍波动，当前规则停止，U/C增量未验证。
`run_native_capacity.py --caps ...`在同一共同配置引擎内、排空后依次改接纳上限；
`--include-feedback`加入同历史观测的shadow和真实非抢占反馈，全部执行命令见上述记录。

## 动作与测量

- cap 限制接纳；每轮最多接纳一个已到达的 FCFS 请求并完整 prefill，再推进全部
  active 请求。所有 cap 使用相同 prefill 规则。降 cap 不截断 decode、不暂停请求、不驱逐 KV。
- `target_cap / actual_active / decode_requests / waiting_requests / future_requests`
  分别记录；等待数不包括未来到达。每个 cap/重复/统计开关都重新执行自己的 KV、路由和输出。
- `--cap-schedule '[[0,4],[2,2],[5,4]]'` 可执行局部升降；记录请求时间、实际应用时间、
  自然达到目标的时间和被后续动作替代的情况。它不是共享历史快照的严格因果配对。
- `--cap-schedule '[[0,6]]' --step-action '[6,8]' --warmup-caps 6,8` 在完成六次
  decode 后、下一次 prefill 前升档；`[6,6]` 为同边界 hold。两臂使用共同预热宽度。
  `pre_action.latest_completed_step` 保留最新单步逐层 U/C、身份与时间；最近四步仅作历史诊断。
- 每个 decode step、每层记录 `U=非零专家数/E`、`C=max(n_e)/(sum(n_e)/E)`、
  routed tokens 和最大专家 token 数；零 token 时 U/C 为 null。
  过去四个已完成 step 的统计做均值，不求长窗口 expert union。
- 统计使用 OLMoE 返回的 router logits 按原 softmax/top-k 算法重算，再在 GPU 上归约、
  一次导出紧凑结果。没有逐 token route 导出。它不等于实际 HBM 读量、硬件拥塞或
  fused backend 的 dispatch trace。复用 helper 直接调用 base model + lm_head，
  不计算辅助负载均衡损失；logits 返回和统计归约的额外成本仍需要 OFF/ON 测量。
- 普通状态含 batch、逻辑 KV 长度、padding、队列/等待、prefill、近期 model/iteration/ITL
  和此前 cap；没有声称测到原生 KV allocator bytes。

请求时间使用统一 host 墙钟：

```text
TTFT = first token time - arrival time
TPOT = (last token time - first token time) / (output tokens - 1)
goodput = completed requests passing both TTFT and mean-TPOT / episode wall duration
attainment = those passing requests / all arrival requests
```

完整 episode 分母包含排队、prefill、decode、KV 整理、统计和循环开销，不重复累加
局部耗时。模型加载、离线 tokenization 和 warmup 在服务计时前完成。每次 prefill
产生的首 token 计入输出数；单 token 的 TPOT 为 null 并单列。
失败与未完成请求保留，不能成为 SLO pass。若提前 OOM，全部计划请求仍保留在 raw，
尚未到达的请求另列 `n_not_yet_arrived`，不会人为延长测量终点，也不会混入已到达分母。ITL 另报分位数；平均 TPOT 达标并不保证
每个 ITL 达标。小样本 P99 只是描述统计。这里测的是有限 cohort 的有效吞吐，不能
直接宣称稳定容量或生产 SLO。

## 运行

在已有 CUDA 环境、固定 OLMoE revision 已缓存的完整仓库内运行。当前复用接口在
Torch 2.8.0 / Transformers 4.57.6 上完成上述 custom-runtime CUDA 实验；原生入口另行验证。
默认只读本地模型缓存，不下载模型或安装依赖。需要一个 GPU，检查到其它计算进程即停止；
进程检查仅覆盖 cell 前后，不声称连续隔离。每个输出目录必须全新。

已经准备好的默认配置在 `outputs/admission_capacity/20260905_pre_gpu_r01/prepared/`。
GPU 可用后，从仓库根目录只需执行：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python \
  refine-logs/expert_saturation/experiments/admission_capacity/run_capacity.py \
  --prepared-dir refine-logs/expert_saturation/outputs/admission_capacity/20260905_pre_gpu_r01/prepared \
  --output-dir /tmp/olmoe-admission-capacity-r01
```

`--prepared-dir` 复用明确的 token IDs、文本身份、到达时刻与执行顺序；不接受同时覆盖
工作负载参数。入口会先检查离线缓存、接口、GPU/BF16/可用显存和进程，再加载模型。
默认单 cell 600 秒、整次执行 1200 秒预算；在调用边界检查，无法中断一个正在执行的模型调用。
预算耗尽保留此前全部结果，标为未完成。

如需调整到达率、SLO 或输出长度，先准备一个新目录并说明原因：

`--request-offset 16` 或 `32` 可选择长度过滤后的后续自然文本 cohort，默认值为 0。
准备与复用共用确定性选择规则；旧 prepared 缺该字段时按 0 处理。
执行已有 `--prepared-dir` 时不能另外覆盖 offset。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  refine-logs/expert_saturation/experiments/admission_capacity/run_capacity.py \
  --prepare-only --output-dir /tmp/olmoe-admission-inputs-r02 \
  --caps 2,4,8 --requests 16 --prompt-tokens 128 --output-tokens 16 \
  --repeats 2 --arrival-gap-s 0.1 --burst-size 4 \
  --ttft-slo-s 5 --tpot-slo-s 0.2 --max-run-seconds 1200
```

此命令中的到达率和 SLO 是**待 GPU 校准的探索起点**，不来自测量或业务承诺。
先检查实际活跃数是否覆盖目标、队列是否暴露、SLO 是否全部通过/失败，再据此说明
调整原因并保留新运行。不要根据 U/C 或收益挑选请求：固定取源 WikiText manifest 中
按原顺序前 16 个长度足够的文本，各截到 128 tokens。

每轮包括 steady 和 bursty，同一有限 cohort 的首尾到达时间相同；cap=2/4/8 各做 OFF/ON，下一重复反转
cap 及 OFF/ON 顺序，共 24 个小 cell。输出长度固定只用于性能隔离；
`--natural-stop` 启用模型 EOS，保留同一最大输出上限，供后续补充。
`--model-path` 允许指定本地 snapshot；如果覆盖默认模型路径，结果会明确标记 revision
另由离线预检检查 resolved revision 与缓存 shard 完整性；未知 revision 会在加载模型前停止。
权重只检查文件存在与大小，不声称完成全量权重内容校验。

输出只有一个实验目录：config/commands/environment/workload、各 cell 原始结果与
请求指标、逐 cell 更新的 `curves.csv`、status。原始 cell 不改写；失败也先落盘。
OFF/ON 是各自独立的真实执行，不能把 ON 的 U/C 附到 OFF 的未来请求状态上。
同次扫描里最高的 cap 只能叫探索性静态最优，不是调优后 holdout 基线。

```bash
.venv/bin/python -m unittest discover \
  -s refine-logs/expert_saturation/experiments/admission_capacity -p 'test_*.py' -v
```

## 原始结果分析

```bash
.venv/bin/python refine-logs/expert_saturation/experiments/admission_capacity/analyze_capacity.py \
  --run-dir /tmp/olmoe-admission-capacity-r01 \
  --output-dir /tmp/olmoe-admission-capacity-r01-analysis
```

分析器忽略已有 `curves.csv` 的指标值，从原始请求 token 时间重算；校验 request、document、
prompt token hash 和具体 arrival。输出 `analysis.json` 与 `report.md`，含逐 cell 请求指标、
每轮 OFF 静态 cap 比较、相邻 cap 的 goodput/TTFT/TPOT/SLO 请求数变化，以及 ON 自有 U/C
和 OFF/ON 的输出、batch membership、时间差。缺 cell 返回 PARTIAL；没有 cell 返回 UNRUN。
同次扫描的最佳 cap 只用于观察响应，不是 holdout 选择器或 action Oracle。
同时汇总实际活跃数、decode 宽度、等待/未来请求、目标达到情况、请求等待和 TTFT/TPOT
分别达标情况。decode 快照比例不是时间占比；`telemetry_s` 已包含在 episode 内，不能再加一次。
新运行将 post-cell GPU 进程检查单独写入 `checks-NNN.json`；缺失或失败时保留原始指标，
但分析器将该 cell 标为 INVALID 并排除比较。源码可最小导出到无 Git 的执行目录，
此时 Git 字段为 null；实际运行源码 hash 仍保留。
若所有 cap 均无达标请求或全部达标，报告提示检查校准条件；不修改原有 SLO 或隐藏该次运行。

## 后续解释边界

有真实曲线后，只判断静态 cap 是否出现可重复的响应差异。压力是在动作后观察到的状态，
分组相关性不能证明压力造成性能变化；动作增量需要普通状态有重叠的局部接纳对照。
本轮没有 ordinary-state controller、U/C 修正策略或 action Oracle。

| 相关工作 | 原文动作 | 本实验边界 |
|---|---|---|
| [Gimbal](https://arxiv.org/html/2606.15177v1) §4–5 | pressure-aware DP engine 选择、队列排序和 expert placement/migration | 固定 placement、单 engine 的 cap；不主张首次使用 expert pressure |
| [ELDR](https://arxiv.org/html/2607.00466v1) §4.2–4.3 | prefill expert signature 与负载驱动 request→decoder handoff | 不按专家信息挑选/分配请求；不主张首次使用 expert union |
| [SCORPIO](https://arxiv.org/html/2505.23022v1) §3 | VBS feasibility admission、credit batch selection、TTFT deadline 处理 | 最接近 admission 基线；不能将原算法削弱为固定 cap 后称已公平比较 |

原生确认优先复用上级目录已有 `run_vllm_decode_cap_branch.py`，它支持初始 cohort 的
`max_num_seqs` 对照，但不提供当前 arrival replay、动态降 cap 或可信的实际调度逐层
U/C 联合记录。原生环境恢复后先选一对代表性 cap 确认请求级方向，再针对缺失字段
接通接口；不把该旧脚本标为已完成本轮原生确认。


## 2026-09-08：对齐档位的已完成原生对照

[24-episode回传与重算记录](../../outputs/admission_capacity/20260908_aligned_ladder_r01/REPORT.md)
保留了同一32条文本的768次完整测量请求执行。aligned feedback相对同引擎最好已测static，
steady两次为+3.40%/+49.54%，bursty为−4.08%/−4.18%。这是请求级探索信号，
尚未隔离旧/新ladder，也不含专家信号；padding没有一致下降，收益幅度随重复波动。
两引擎测量raw已完整回传并校验；继承runner的成功warmup仅有摘要，完整raw缺失。
唯一下一项为[同引擎配对方案](../../outputs/admission_capacity/20260908_capture_ladder_paired_r01/DECISIONS.md)，
其执行状态须查看对应目录，不能由本条已完成结果推断。

## 2026-09-08：配对与一次固定重复完成，短文本档位方法停止

[合并报告](../../outputs/admission_capacity/20260908_capture_ladder_paired_repeat_r01/REPORT.md)
记录 64 个测量 episode、2048 次请求执行和另外 60 个完整 warmup raw；所有四个引擎的
数据、环境、日志和退出记录已完整回传本地并核对归档摘要。原配对 forward 保持 canonical，
所有重复与负结果保留。仍是同一 32 条文本的重复执行。

steady 四个引擎只有 1/4 同时超过旧反馈和当次最好 static；固定重复相对 static 为
−59.89%/−38.73%。bursty 虽 4/4 改善旧反馈，却 0/4 超过最好 static。当前运行域的
`[8,12,16,32] → [8,16,24,32]` 替换没有稳定方法收益，停止第三次同配置 campaign，
不添加专家特征或调整阈值抢救。结论不扩写为整个准入调度家族失败。

四条 aligned steady 轨迹的三组首次 scheduled 内容差异，都早于首次 cap 动作，
并跨在请求到达边界两侧；这是 batch/prefill 轨迹先分叉的记录证据，没有定位 kernel
或专家原因。目标写入也不等于立刻限制接纳，具体窗口与首次 binding 已归档。

唯一下一项为既有[自然长上下文容量对照](../../outputs/admission_capacity/20260906_native_memory_pressure_r01/DECISIONS.md)：
同文短/长输入、cap16/32、独立正反序，共八个 episode，先检验持续 KV 占用是否造成
真实容量约束。它仍为 GPU UNRUN，不继承短文本 9ms 阈值或任何方法 GO。

## 2026-09-08：自然长上下文已实跑，decode KV 容量边界复现

[最新执行报告](../../outputs/admission_capacity/20260906_native_memory_pressure_r01/EXECUTION_REPORT_20260908.md)
取代上一条中的“GPU UNRUN”作为当前执行状态；旧准备报告与冻结规则保留。
八个新进程全部执行并逐项回传：六个完整 episode 共192次完成请求，两项容量保护
保留64条部分轨迹，另有24个完整warmup（528次请求执行）。归档摘要、raw、环境、
日志和退出码全部在本地，远端原件保留。

短输入cap16/32与长输入cap16各两次全部完成。两次长输入cap32都在全部prefill结束后、
decode增长至attempt809时耗尽7677个可用KV块；保护钩子未调用原抢占或执行失败步GPU
forward。两次长输入cap16的KV峰值为53.13%，完整时长28.665/28.399秒。边界行不参与
完整吞吐比较，不能把0条完成当成cap16的“加速”分母。

当前认识是：固定请求数上限在短输入可完成，不保证长输入的后续KV增长仍可容纳。
这证明运行域与容量约束存在，尚未证明专家信号或动作增量。唯一下一项是按真实KV池
与每请求最大长度计算安全静态cap，与cap16做四项正反序完整请求对照；相同长度下预算
预留退化为常数cap，先量清这项简单基线，不实现动态专家控制器。

## 2026-09-08：安全静态cap29实测完成，容量与时延存在权衡

[最新结果](../../outputs/admission_capacity/20260908_kv_safe_static_r02/REPORT.md)：
live单组KV池7677可用块、16tokens/块、每请求256块，公式给出cap29。
四个独立进程128次测量请求及12个完整warmup全部完成并回传。r01曾因错误要求
Unitary coordinator在测量前停止；实际NoPrefixCache接口修正后在新r02执行，旧失败保留。

cap29两次完整吞吐相对cap16提高3.75%/4.53%，但TTFT p99由约13.6s升至18.0s，
请求平均TPOT p50由约13.55ms升至18.6ms；KV峰值95.83%，无抢占或分配失败。
容量可行不等于时延全面改善，参考SLO达标数增加不能替代尾部代价。
本轮没有专家信号或可回收专家内存证据，单token top8的结构估算不代表batch闲置容量。

唯一下一项是让原生cap32在抢占后继续重算、完成全部请求，与公式cap29做完整请求对照。
此前cap32保护终止的0完成不能充当默认策略吞吐基线。先补这项真实默认成本，不扫新cap，
不增加专家预测器或动态预算控制器；该对照当前GPU UNRUN。

## 2026-09-08：原生cap32恢复完成，零抢占不能代替完整目标

[最新完整对照](../../outputs/admission_capacity/20260908_native_preemption_r01/REPORT.md)
已补齐此前保护终止遗漏的恢复过程：四个新引擎、128次请求、12次暖机全部完成并回传。
native32两次各抢占2条请求、重算7691个token位置，仍比safe29吞吐高16.99%/17.08%，
TTFT p99为0.760/0.795s（safe29为17.900/18.107s）。代价是平均TPOT中位数高约7.5%–7.9%，
两条victim分别出现约2s与4.5s暂停；safe29最长ITL约0.108s。

当前中心问题仍是SLO下的MoE资源管理。最新认识是：提前准入限制把等待移到首token
之前，原生恢复把部分等待留在生成期间；零抢占并不保证更好完整吞吐或TTFT。
长暂停主要发生在victim重新执行之前，pooled token ITL p99约28ms会掩盖少数秒级暂停。
这些是普通KV/queue策略的测量规律，尚无专家信号或动作增量。

唯一下一项为固定native32的0.90/0.95显存预算对照，先在0.95首引擎核对live可用块
是否达到8192的完整预留充分条件，再执行同配置正反序。记录真实KV增配量及完整请求，
检验普通预算调优是否已消除暂停；不称同预算调度收益，不扫新cap。该预算实验GPU UNRUN。

## 2026-09-08：预算对照在资源预检处停止，保留UNRUN

[首项尝试](../../outputs/admission_capacity/20260908_kv_budget_r01/REPORT.md)已执行并完整
回传失败bundle：启动前检查曾空闲，首个0.95 child随后在import torch/模型初始化前
报告另一个GPU计算进程，退出1、0暖机、0测量，余下三项未启动。之后同端点SSH连续
关闭连接，无法确认当前资源；不能将此失败写成0.95预算或方法NO-GO。

[新r02](../../outputs/admission_capacity/20260908_kv_budget_r02/ADDENDUM.md)已封包，唯一
代码修复是在资源占用异常中保留实际进程CSV，科学方案及95→90→90→95顺序不变。
r01原尝试不改，r02尚未上传或运行。唯一下一步仍是端点恢复、GPU空闲后完成预算对照，
不以条件候选替代这项未回答问题。
[固定运行入口](../../outputs/admission_capacity/20260908_kv_budget_r02/COMMANDS.md)
已保留上传校验、逐项回传和分析命令。

[局部文献核对](../../outputs/admission_capacity/20260908_kv_budget_r01/PRIOR_ART.md)将
WiSP的专家/KV联合分配、ELDR的decoder选择与FluxMoE的按层流式动作区分开。
本链条尚缺普通KV配置之外的MoE增量；若之后检查稀疏专家回收，先量实际batch专家并集、
跨step复用和KV压力的交集，不能使用单token未选中比例推断可回收HBM。该结构诊断未运行。

## 2026-09-08：已有请求轨迹的SLO口径诊断

[只读分析](../../outputs/admission_capacity/20260908_native_preemption_r01/slo-readout/report.md)
保留四份原始轨迹的128条请求值和138个联合观测断点。固定TTFT≤5s、生成阈值200ms，
从平均TPOT改为请求最大ITL判定后，native32每轮达标数从32降为30，safe29仍为29；
native32的完整episode goodput为1.2953/1.2869 req/s，仍高于safe29的1.0703/1.0626。
在这四份固定轨迹和TTFT约束下，最大ITL规则没有safe29胜出的阈值区间。
这是事后描述：平均TPOT会掩盖少数长暂停，但该事实不自动支持准入上限更优。
未选择新SLO、重跑GPU或产生预算对照结果；唯一下一项仍为已冻结的0.95/0.90对照。

## 2026-09-10：迁移至用户提供的新5090实例

[新实例运行入口](../../outputs/admission_capacity/20260910_kv_budget_r01/COMMANDS.md)
已准备，科学包逐字继承此前r02。新端点weste:11155连接成功，5090为不同UUID；
旧实例的吞吐和KV池资格不沿用为本次结果。用户已明确授权后续操作，冻结包已上传并校验。
软件安装与固定revision权重下载仍在进行，本地runner21199已启动，待环境完整后自动执行
四项并逐项回传。本次GPU测量仍UNRUN；现场状态见同目录ADDENDUM、SETUP_OBSERVATIONS
和await-readiness-state-20260910.json。唯一实验仍为同一新GPU上的95→90→90→95对照。

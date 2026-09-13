# 固定 KV 四臂：轮转的实测权衡

2026-09-13；HEAD `7059fc98e0d98a127ff4edda86d44f7f634d26f4`，未提交工作区。**8/8 COMPLETE，256/256 请求完成；MEASUREMENT_ONLY。**

问题：固定 KV 预算下减少已有请求长暂停，并保住完整服务量。本轮轮转相对保 KV 两块都观察到更短的最坏停顿和更高吞吐；相对原生的吞吐差接近零且变号，平均完成仍变慢。当前是探索性候选，未证明吞吐非劣或方法 GO。

## 同轮结果

各格独立引擎；OLMoE BF16 / vLLM 0.26 / RTX 5090，同32篇3072输入、1024输出、50ms到达、budget1024。实际 KV均为16,089,350,144 bytes / 7,671 usable blocks；engine能力均为32，safe29只改准入29。所有原始结果与失败状态永久保留。

| 顺序 | 策略 | 墙时 s | 请求/s | 平均完成 s | 最大 ITL s | 自然/强制抢占 |
|---|---|---:|---:|---:|---:|---:|
| repeat0 | native | 22.804285 | 1.403245 | 20.443663 | 4.443723 | 2/0 |
| repeat0 | safe29 | 26.832109 | 1.192601 | 19.810509 | 0.308060 | 0/0 |
| repeat0 | headroom | 23.488375 | 1.362376 | 21.934057 | 1.381525 | 0/0 |
| repeat0 | rotate | 22.929957 | 1.395554 | 20.902513 | 1.006017 | 2/8 |
| repeat1 | rotate | 22.902337 | 1.397237 | 20.867643 | 1.006155 | 2/8 |
| repeat1 | headroom | 23.415711 | 1.366604 | 21.863351 | 1.370199 | 0/0 |
| repeat1 | safe29 | 26.563681 | 1.204652 | 19.543185 | 0.095501 | 0/0 |
| repeat1 | native | 22.914865 | 1.396473 | 20.539101 | 4.483895 | 2/0 |

以下全部是同 block 配对，数值按 forward / reverse：

- **rotate vs native**：吞吐 −0.548% / +0.055%；最大 ITL 4.444/4.484→1.006/1.006s；平均完成 +2.244% / +1.600%，32/32和31/32请求完成更晚。原两条长暂停请求改善，另外两条请求自身最大 ITL从约0.079/0.091s升至约0.95–0.96s。
- **rotate vs headroom**：吞吐 +2.435% / +2.242%；最大 ITL低0.376/0.364s；平均完成 −4.703% / −4.554%，两块均30/32请求完成更早。首请求仍晚约3.37/3.39s，另一请求晚0.52/0.55s，不能称每请求支配。
- **rotate vs safe29**：吞吐 +17.018% / +15.987%；最大 ITL从0.308/0.096s升至1.006s；平均完成 +5.512% / +6.777%。safe29最大 TTFT约17.92/17.64s，轮转约0.76/0.75s。两者存在等待位置与服务量权衡，不称轮转全指标支配safe29。

每请求最大 ITL >1s 的计数：native 2/2、headroom 31/31、rotate 1/1、safe29 0/0（每格32请求）；1s是已冻结的描述性切点，非业务SLO。历史TTFT5s/mean-TPOT0.2s下，safe29为29/32通过，其余32/32，不能反选这些门槛宣布安全容量贡献。safe29自身最大ITL在两次运行中0.308→0.096s，说明尾指标必须保留原波动。

## 动作与成本解释

轮转参数保持初始等待30步、交换间隔20步、驻留30步、保护进度0.90及每请求最多8次缺席。每轮10次真实抢占=2次自然+8次强制，8次受保护目标均经4次调用恢复首个新token；两次抢占/恢复步骤一致。原生抢占事件与SchedulerOutput恢复记录吻合，恢复后真实返回新token；未独立捕获完整worker块表。未观察到无新token的反复抢占。

本轮held为0，保护余量没有阻止其它running请求执行。因此不能把收益归因于保护本身，更不能证明保护必不可少；本次实际动作主要是驱逐/恢复顺序改变。

相对native，每轮重算7,685→38,561位置（+30,876）；含重算调用8→40，纯decode调用减少123，总engine调用1348→1257。纯decode宽度1调用132→15，末尾小batch缩短。互斥的host engine非调度桶中，含重算桶增加0.913/0.913s，纯decode桶减少0.829/0.904s；轮转含重算桶还同时完成1,185个新decode位置，其时间不是纯重算或纯GPU时间。

总墙时 = scheduler inclusive + engine non-schedule + outside engine；decision只是scheduler内的子集。rotate−native总墙时+0.12567/−0.01253s，分解为scheduler +0.02076/−0.00311、engine其余 +0.09267/+0.00876、engine外 +0.01224/−0.01818s。相对headroom，轮转减少183次engine调用、278次宽度1调用，墙时低0.55842/0.51337s。**额外重算量本身不足以预测完整吞吐损失，必须同时计入实际batch宽度和完成尾部。**

## 旧0.39s估算为何不成立

两次最长间隔均来自请求`memory-train-article-0003571`：step930最后返回token，931自然抢占，976开始重算，979返回下一token。更早在926被驱逐的0003640到956才满足30步等待并先恢复；这次交换再触发20步全局冷却，3571实际缺席45步。forward从抢占所在调用开始到首恢复调用约0.890259s，四次恢复调用跨度0.115394s，加边界间隙约0.000365s得到1.006017s。

旧模型将20步交换间隔误当成每个请求的停顿上界，遗漏了多个缺席者的排队与初始等待。这个反例修正该估算，不否定轮转动作。完整恢复诊断见[动作账本](analysis/rotation_accounting.json)及可重跑脚本。

## 证据边界与唯一下一步

两块均真实独立推进KV、队列和输出；每个策略的两次输出32/32相同；rotate/native同块32/32相同，rotate/headroom为31/32，rotate/safe29为7/32。输出一致不替代质量评价或通用语义保证。

两次执行及共享cell的pairwise差值不构成独立噪声界。native对照的吞吐差变号，本轮不证明零吞吐损失；对headroom的约2.3%优势仍需独立重复。raw逐项身份/指标/块守恒及成本守恒已复算；fresh同族审计状态见[EXPERIMENT_AUDIT](EXPERIMENT_AUDIT.md)，审计完整性结论不代表统计确认。

| 固定报告项 | 本轮结论 |
|---|---|
| Verdict | MEASUREMENT_ONLY；轮转进入独立确认候选，主问题仍OPEN |
| Evidence type | NATIVE_SERVING / native in-process，单模型单卡固定输入 |
| What was measured | 同池四臂正反序8格、256请求，完整token/请求时间、实际抢占与重算、成本和逐请求转移 |
| What was not measured | 独立请求/到达过程、统计非劣、质量、第二模型、生产服务、EP |
| Strongest baseline | native32、safe29及同轮headroom-fast，分别保留吞吐/停顿/等待位置代价 |
| Oracle/headroom status | exact Oracle未测；旧0.39s与重算比例仅为已被实测修正的估算 |
| Claim ceiling | 本固定负载下轮转实测改善headroom的最坏停顿—吞吐读数；不保证逐请求获益或泛化 |
| Failure category | 旧停顿上界漏算缺席者排队；重算单项成本漏算batch尾部抵销；吞吐非劣未确认 |
| Resurrection condition | 主问题未判死；未来负结果只限定相应负载/参数/成本区间 |
| One next smallest experiment | 保持轮转参数，在新独立请求episode上做native/headroom/rotate与同负载native A/A的随机区组重复，事前冻结实际意义与可容忍损害，检验完整权衡而非继续c20阈值扫描 |

## 重跑与原件

执行原件：[execution.json](execution/execution.json)；上传包SHA256 `e0f95c0657afb7815b48b95de1e8192b53ddf1a24be8df2f01ecd443ae787051`；远端目录`/root/autodl-tmp/moe-rotation-fourarm-20260913-r01`。全8项归档SHA均在执行记录内，driver退出0。准备状态文件是执行前快照，当前状态以本报告及execution记录为准。

从仓库根目录执行，下列输出路径必须尚不存在：

```sh
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_completion_headroom.py --four-arm --run-dir refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_fourarm_r01/execution --output-dir /tmp/rotation-fourarm-reanalysis
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_headroom_cost.py --four-arm --run-dir refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_fourarm_r01/execution --output /tmp/rotation-fourarm-reanalysis/cost_breakdown.json
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_headroom_work_cost.py --four-arm --run-dir refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_fourarm_r01/execution --output /tmp/rotation-fourarm-reanalysis/work_cost.json
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_rotation_recovery.py --run-dir refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_fourarm_r01/execution --analysis /tmp/rotation-fourarm-reanalysis/analysis.json --output /tmp/rotation-fourarm-reanalysis/rotation_accounting.json
```

本轮回答：轮转确实能把约4.5s最坏生成停顿压到约1.006s，并在两块中取得优于保KV的最坏停顿—吞吐读数；它仍转移了部分请求等待，保住原生吞吐尚待独立确认。

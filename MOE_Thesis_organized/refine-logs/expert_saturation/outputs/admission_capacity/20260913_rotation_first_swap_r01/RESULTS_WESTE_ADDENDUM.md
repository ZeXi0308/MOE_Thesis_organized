# 首次交换诊断：weste 六项实测补充

六项已在 `connect.weste.seetacloud.com:23478` 完成并逐项回传，192/192 请求、196,608 个新输出 token，资格全部成立。**MEASUREMENT_ONLY：只改首次交换对象没有保留持续 most_output 的吞吐结果；它减轻了早完成请求的拖延，但这条简化路径没有实现预期组合。** 原始 REPORT 和 westc 暂存记录保留。

| 固定报告项 | 结论 |
|---|---|
| Verdict | MEASUREMENT_ONLY；当前 first_most_then_least 的组合目标未获支持，主问题仍 OPEN |
| Evidence type | NATIVE_SERVING，六个独立引擎实际执行与请求/step/KV/恢复账本 |
| What was measured | 同一复用 cohort0 的三臂×两顺序 block；完整吞吐、最长 ITL、平均及逐请求完成时间、真实首次动作和成本路径 |
| What was not measured | 新文档泛化、持续到达、质量分数、显著性、非劣保证；B 对新执行 native/headroom 的直接比较 |
| Strongest baseline | 同机新执行 A=least_progress、B=most_output；历史 completion_headroom/native 仅作前序依据，不混入本次配对 |
| Oracle/headroom status | 没有新增 Oracle；单次动作不足不证明每个后续动作必要，也不构成全策略上界 |
| Claim ceiling | 同一复用工作负载、两个顺序 block 的实测权衡与尾段分解 |
| Failure category | 首次动作使末两请求进度更不均衡，双请求阶段缩短被单请求阶段增长大幅抵消；不是无效动作或整个问题 NO-GO |
| Resurrection condition | 新证据表明该简化动作在明确运行域有完整请求收益；不以扫描“前2/3/4次交换”代替新假说 |
| One next smallest experiment | 新文档 cohort 上做 native、native A/A、completion_headroom、持续 most_output 四臂×两反向 block；同一实际 KV/长度/到达参数，直接测强基线与同条件漂移，尚未准备或运行 |

## 执行与资格

执行记录：[execution02_weste_23478/execution.json](execution02_weste_23478/execution.json)。命令：[RUN_WESTE_COMMAND.sh](RUN_WESTE_COMMAND.sh)。原包 SHA256：
`fd1342a6aea8f2cc886a0a6957d4059dcc2d3ffad1797b0428fb23d8be162b4c`。

GPU `GPU-0a66cc34-b091-2000-ba7b-e576b6d3d7d6`，RTX5090；driver595.71.05、Torch2.11.0+cu130、vLLM0.26.0、Transformers5.15.1。固定 OLMoE revision 的三个 shard 内容哈希一致。全部实际 KV 为16,089,350,144字节、7671 usable blocks；prompt3072/output1024，32请求，50ms到达，cap32，token budget1024。

执行顺序为 block0 A/C/B、block1 B/C/A。A 始终 least_progress，B 始终 most_output，C 仅首个实际成功的 forced swap 用 most_output，此后用 least_progress。六项首个 forced swap 均在 step836；C 从第二个成功交换 step866 起切回 least_progress。A/C 均8次 forced+2次 natural，B均9+2；held request steps均0。六项通过输入/身份/参数/实际KV、原生事件与动作计数及首次切换检查；同一请求的输出 token 序列在六项间完全一致。这不替代任务质量评测。

旧 `execution` 是 westc:53036 上的 STAGED、0项；本次未执行其中任何 cell。其 Qwen3 下载任务未被修改或终止。新六项没有混用旧主机的性能结果。

## 同 block 完整请求比较

Δ 为 action 相对 baseline；平均完成时间越低越好。两列依次为 block0 / block1。

| 比较 | 吞吐 Δ | 平均完成时间 Δ | 最大 ITL Δ |
|---|---:|---:|---:|
| C 相对 A | +0.166% / −0.425% | −0.032% / +0.644% | +1.479 / +0.918 ms |
| C 相对 B | −1.332% / −1.513% | −1.065% / −0.850% | +28.553 / +30.427 ms |
| B 相对 A | +1.518% / +1.105% | +1.044% / +1.507% | −27.075 / −29.510 ms |

A 最大 ITL 为1.009063/1.009009s，C为1.010542/1.009927s，B为0.981989/0.979500s。C 相对 A 的墙钟变化为−37.665/+97.055ms；不能据此宣布零效果、非劣或稳定变差。B 相对 A 平均完成变慢，27/29条请求完成延迟增加；C 相对 B 有28/26条请求完成得更早，但吞吐更低。逐请求数只按实际正负计数，不是显著性分类。

同角色跨 block 吞吐变化 A +0.223%、C −0.368%、B −0.185%，是运行漂移观察，不是噪声底或置信界。本实验只有一份复用文档集合、两顺序 block；不能把192请求或六引擎当成独立工作负载重复。

六项 reference SLO 都是32/32通过，因而其 goodput 等于吞吐；TTFT5s/平均TPOT0.2s不约束最大ITL，也不是新业务SLO。完整六项数值见[主分析表](analysis02_weste_23478/report.md)和[analysis.json](analysis02_weste_23478/analysis.json)。

## 为什么只改第一次不够

两 block 的下述步数、入口输出和完成排名完全一致。末两请求均为 `0007877/0008125`（省略 `memory-train-article-` 前缀）。

| 实际路径 | A | C | B |
|---|---:|---:|---:|
| 进入双请求阶段时，7877/8125 已输出 | 937 / 925 | 966 / 921 | 1016 / 1005 |
| 该阶段剩余输出 | 87 / 99 | 58 / 103 | 8 / 19 |
| 纯 decode width2 调用 | 87 | 58 | 8 |
| 后续 width1 调用 | 12 | 45 | 11 |
| 纯 decode width4 调用 | 46 | 46 | 7 |
| 全部纯 decode 调用 | 1119 | 1123 | 1015 |
| 重算位置 | 38561 | 38752 | 43471 |
| 全部引擎调用 | 1257 | 1261 | 1162 |

A/C 都在 step1158 进入相同末两请求阶段。C 使7877多输出29个，却使8125少输出4个；双请求阶段少29次，8125独跑多33次，最终调用数反而多4。width4阶段没有压缩，另有少量宽批重排，不能说“只有width1/2改变”。

C 对 A 的 width2 实际 engine host 时间减少159.713/161.553ms，width1却增加146.077/145.453ms，抵消91.46%/90.03%；两桶合计只减少13.636/16.100ms。这些互斥实际桶可以相加核账，但不是独立因果节省或反事实硬界。完整墙钟还包含其他批宽、恢复、调度与调用外时间。

早四请求3733/3820/3941/4015在 A 的完成排名为1/2/3/4；C为11/1/2/3；B为23/9/25/13。C 相对 B，这四条实际完成提前约0.693–1.326s；相对 A，3733仍延后0.656/0.796s。降低 B 的完成代价这一部分确有实际路径变化，但没有同时保住 B 的吞吐和最大ITL结果。

证据：[路径表](diagnostics/measured02_weste_23478/report.md)、[逐请求/阶段数据](diagnostics/measured02_weste_23478/diagnostics.json)。

## 成本模型修正与下一步

在本次末两请求阶段，无新到达、无后续抢占，每个decode调用各产生一个新token，且两请求均生成至1024。令入口剩余输出为r≤s，则双请求调用数为r，单请求调用数为s−r；尾段调用总数为s。本次六项均满足这条计数关系。若以对应批宽的每调用成本c2、c1近似，尾段成本为：

`T_tail ≈ r·c2 + (s−r)·c1`。

因此只看双请求阶段长度或剩余token总和会漏掉进度不均衡造成的单请求尾部。它是已观测末段的条件模型；未来哪两请求最后完成仍由策略状态演进决定，不能把这里的事后身份当作在线输入或跨策略固定trace。

当前停止“首次交换足够”这条简化假说，保留持续 most_output 作为强简单策略。唯一下一项是上表的新文档四臂对照，先回答持续策略能否相对 native/headroom 保住完整服务量，同时用同条件 A/A 呈现漂移；本次未执行该八项。此次回答是：**只改变第一次交换不足以实现本轮预期的吞吐与完成时间组合；问题本身仍未被否定。**

完整性复核：fresh GPT-5.6-Sol ultra 返回 PASS、P0=0/P1=0、独立指标差异0；同家族 provisional，未升级科学裁决。范围与未验证项见 [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md)。


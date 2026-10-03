# 固定脉冲的受控重复：原有小幅正结果未复现

2026-09-06。接续 [原报告](REPORT.md)，保留其全部原始结果。
本次唯一问题：原 A/B/B/A 中不稳定的 pulse/hold6 差异，在相同配置下能否复现？
最弱环节是时序与 SLO 门槛附近的重复稳定性。本次没有增加 predictor、调整动作时刻或 SLO。

**Verdict：`MEASUREMENT_ONLY / FIXED_PULSE_GAIN_NOT_REPRODUCED / U_C_ACTION_INCREMENT_UNVERIFIED`。**
新增 16 个 GPU cells 全部完成，共 256 次请求执行，仍然只是原来的 16 条输入；原 pilot 目录
累计为 80 个 cells。其它文本 cohort 的结果在兄弟目录独立保留，不重复计入这里。

复用原 A/B/B/A 四个进程、steady/bursty、OFF/ON、TTFT=5 s、mean TPOT=0.2 s。
A 为保持6并执行 sham 回调；B 在1.5 s升8、3 s降6。模型、源码、输入与预热规则相同。
本次等待已存在的 GPU 文本试验队列结束、进程列表为空后启动；16个 cell 前后进程检查通过。
这只验证边界采样时的 GPU 进程隔离，不证明整个宿主机或全时段隔离。

| Pulse 相对 hold6 的 goodput | 原执行：配对0 / 配对1 | 本次受控重复：配对0 / 配对1 |
|---|---:|---:|
| steady OFF | +4.42% / +4.48% | −14.13% / −33.29% |
| bursty OFF | −9.90% / −2.87% | −24.25% / −19.73% |
| steady ON | +2.00% / +5.41% | −16.99% / −32.50% |
| bursty ON | −7.84% / +73.15% | −55.92% / −9.11% |

本次8组配对全部为负。原 steady OFF 约4.4%的增益没有稳定复现；原 bursty ON 的73.15%
不能用作正结果：该配对的 hold6 只有8/16请求联合达标，而前一个 hold6为14/16；两次pulse
均为13/16。门槛附近的对照退化显著放大了百分比。保留这些真实测量，不把它们标成
`INVALID_EXPERIMENT`，也不猜测温度、CPU调度或某个CUDA内核是已证实原因。

从真正的 `applied_s`（循环首、prefill前）重建 active、请求输出前缀、KV长度和等待集合：
原执行7/8对离散前沿相同，本次6/8对相同。前沿不相同的样本仍保留；相同也不证明KV tensor、
等待年龄、近期延迟或设备状态相同。不能拿可能落在prefill内部的 `requested_s` 代替。
两次执行的全部32个动作cells均从请求时间重建了全active集合，没有发现decode裁掉活跃请求。

本次升档从应用到达到8耗时0.438–0.872 s，降档自然排空到6耗时0.316–0.741 s；
包含循环边界等待后分别为0.568–0.889 s与0.409–0.786 s。
原执行4个ON pulse的动作前U/C窗口相同；本次不同窗口又伴随active=5/6、queue=11/10变化。
因此尚无普通状态相近但专家结构不同的独立动作样本，不能识别U/C是否改变边际收益。

| 交付字段 | 结论 |
|---|---|
| Evidence type / claim ceiling | RTX5090、固定OLMoE、custom runtime的有限请求级受控重复；无native或稳定容量结论 |
| What was measured | 完整请求goodput、联合SLO、真实非抢占升降档延迟、离散动作前沿及重复稳定性 |
| What was not measured | U/C相对batch/KV/queue/近期延迟的增量，信号存活到动作生效，硬件拥塞、native serving |
| Strongest baseline | 同输入同SLO的hold6；原静态bracket已说明应先比较强简单档位 |
| Oracle/headroom | 未测动态Oracle；此固定pulse不提供Oracle上界 |
| Failure category | 当前固定墙钟pulse收益不稳定；只覆盖该动作和运行域，不是family/system NO-GO |
| Continue / reopen | 在时序可控、普通状态可比的条件下取得不同专家结构及稳定动作响应后，再讨论U/C策略 |
| One next smallest experiment | 使用已保留的原cohort和offset32文本，在完成第6个decode step的共同前缀，对照保持6与只升到8；分别独立执行未来状态，核对batch/KV/queue/近期延迟可比性，再问此前U/C是否区分动作响应。只测这一处升档，不搜索回落时刻、不训练predictor |

执行命令见 [run-actions-repeat.sh](run-actions-repeat.sh)，原始数据见 [actions-repeat/](actions-repeat/)，
日志见 [actions-repeat.log](actions-repeat.log)。重算命令：

```bash
.venv/bin/python refine-logs/expert_saturation/outputs/admission_capacity/20260906_gpu_pilot_r01/analyze_controlled_repeat.py
```

分析器从raw重算32个cells的请求指标，并核对与原metrics完全一致；两轮workload与6个执行文件
的记录hash相同。输出 [controlled_repeat.json](controlled_repeat.json) 只创建一次，不覆盖旧报告。

**直接回答：这个固定pulse没有显示可重复的净收益；目前支持的是并发控制存在请求级权衡，
不支持U/C已能改善动作选择，也不支持CCF B方法链已经闭合。**

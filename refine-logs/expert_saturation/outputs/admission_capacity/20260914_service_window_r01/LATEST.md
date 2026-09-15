<!-- cohort3-window-frontier-update -->
> **最新增补：cohort3八格已全部完成，当前短恢复缺口被强native/most覆盖。** [新八格与全阈值结论](../20260914_service_window_holdout_r01/REPORT.md)：native/most的1–2输出后再抢占段均0，residual仍5段；两轮每个有非零完成率的间隔断点区间，residual均未超过native/most已测完整策略的较优Q(g)。停止当前residual组件的窗口扩展，主问题仍OPEN。下文保留此前六格阶段记录，其中cohort3的CPU准备/GPU UNRUN已过期；不把这个新结果改写成窗口家族NO-GO。
>
> [恢复前v2资源证书](../20260914_service_window_model_r01/pre_resume_window_certificate_v2.md)已补齐，receipt仅验证不作pre-action输入。[host预算边界](../20260914_host_budget_r01/REPORT.md)查明共享容器90GiB、无可写cgroup委派；独立进程树同预算确认仍未满足。新增曲线/host helper合计11测试通过，原raw与各报告保持。

# 服务窗口当前结论：预算分离有效，短服务仍未消失

2026-09-14；接续本目录 [20:03 UTC 分析快照](REPORT.md)。原快照和审计不改写。本增补使用随后回传的真实六格，不再将它们记为等待执行。

**Verdict：OPEN / NATIVE_SERVING / MEASUREMENT_ONLY。** 将恢复 KV 保护与执行预算分开，改善了原保护实现的效率；相对同组 fit-scan 形成最长停顿更短、完整服务稍慢的实测交换。首输出之后仍存在昂贵的一 token 服务段，窗口假说尚未完成真实动作排序与性能验证。

## 最新六格与剩余缺口

原长任务会话统一执行、回传 `fit / guard_all / guard_residual` 反序六格，全部 COMPLETE，192/192 请求、每格 32,768 输出。实际 6,656 usable KV blocks，APC off、native recompute、无 host KV offload；host 全进程峰值尚未测量。本会话只读原件，不另起 GPU driver。

[原执行者完整资格与性能分析](/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_restore_token_reservation_r01/analysis/analysis.json)已核对冻结源、实际 plan 与资源、所有义务、所有请求。本目录 [指标引用及 raw 散列](reservation_metrics.json)保存其对应来源；[新增生命周期复算](lifecycle_reservation/analysis.json)用同一分账代码重新连接实际计算、返回输出及失效。

| 策略 | wall，两次 / s | 全局 max ITL，两次 / s | 每格重算位置 | 每格零输出再抢占 | 每格 1–2 输出短段 / 新输出 / 重算位置 |
|---|---|---|---:|---:|---:|
| fit | 27.902 / 27.602 | 4.920 / 4.823 | 75,742 | 2 | 11 / 14 / 37,851 |
| guard_all | 29.163 / 28.900 | 4.293 / 4.157 | 96,955 | 0 | 4 / 4 / 13,767 |
| guard_residual | 27.947 / 28.354 | 4.339 / 4.356 | 89,448 | 0 | 5 / 5 / 17,505 |

residual 相对 guard_all：吞吐 +4.35%/+1.93%，平均完成 −4.71%/−3.53%，32/32 请求完成均改善；每对 28 个请求 max ITL 改善、4 个变差，全局最大 ITL 则 +1.07%/+4.77%。重算 −7,507 位置，不把这些位置直接换算为可扣墙钟。

residual 相对 fit：全局最大 ITL −11.82%/−9.70%，吞吐 −0.16%/−2.65%，平均完成 +0.75%/+2.61%；两对全部 32 请求完成更晚，各自 max ITL 改善/受损为 26/6 和 1/31。不能只呈现全局尾值，不能根据结果事后选 3% 效率预算宣布通过，也不因指标存在交换自动判失败。跨臂输出序列并非全同：residual/fit 每对 28/32 相同，质量未测。

最小预算干预真实生效：每个 residual 有 54 个实际调度动作改变，保护优先级之后累计 1,999 个 ready 候选 request-step 中实际选中 1,998 个；不是 1,999 个独立请求或全部服务保证。每个 residual 25 次义务均以新输出解除，0 中断。其重算分账为 `89,448 = 0 首输出前丢失 + 64,024 已输出后丢弃 + 25,424 服务至完成`。其中 5 个短段仅换来 5 个新 token，17,505 个恢复位置在下一 residency 均实际重执行。

因此共批执行税已得到一个简单修正，但它没有消除短服务后的重复恢复。应把 residual 纳入后续强简单底座，不将它包装为新的保护机制。

## 最小服务窗口模型

[185 行 CPU 核心](../../../experiments/admission_capacity/service_window_model.py)、[推导与实例](../20260914_service_window_model_r01/REPORT.md)已完成；与生命周期代码合计 10 个定向测试通过。模型是有限动作资格器，没有接入每步调度或使用全轨迹模拟器。

状态包含真实有效 prefix/位置、已持有/空闲 KV、host 使用和预算、待恢复历史、最后新输出年龄及 action-specific 恢复队列/依赖路径。额外恢复税 C 与首输出整段延迟 r 分开；C 缺测时不宣布摊销。并行阶段只沿实际依赖路径计时，且恢复阶段的临时资源与后续 KV 增长均需可容纳。

预先给定每新输出允许摊销的税 α 时，`L=max(1,ceil(C_remaining/α))`。候选窗口必须同时满足 `n≥L`、`n≤U_KV`、`n≤U_age`。输出年龄只被新 token 重置。n 是未知 EOS 下可提供的机会数，终止即释放；继续保留仅比较从现在起能避免的未来成本，历史账单不是理由。

支持实例（合成，非 GPU 跑数）：剩余额外税 4ms、α=1ms/token，所以 L=4；共同 batching 允许 peer 在恢复期间继续输出，4-token 窗口满足 KV/等待约束；仅 1 token 不满足该摊销预算。反例：可用 KV 降低后 U_KV=3<L=4，当前动作不可行，但不能推出整个负载无解。真实 step406 只证明 `142+3≤148` 块、`994+30=1024` token 的一次共同执行可行，不能拿合成毫秒替它作性能预报。

## 近邻与唯一下一步

[近邻核对](neighbors/REPORT.md)已覆盖 LTR、Andes、UniBoost/MemGuard、TokenFlow 和安装版 native offload。几何有效服务保护、恢复成本与其它请求损害比较、传输排队都已有工作。完整论文系统的同预算实跑尚未完成，不能据组件接入现象指控原论文失败。

默认 native prompt-offload 已完成，`offload_prompt_only=True`，两对 wall +23.55%/+5.17%，尚无完整服务收益；不是 full-decode offload 或轮转兼容的结论。16GiB host 配置不等于实际 host 峰值已测。此处不重跑该基线、不增加 host 资源解释算法增量。

新增 [真实短窗口证书](../20260914_service_window_model_r01/real_short_window_certificate.md)选择最新 residual 的**首个**一 token 后再抢占事件，没有换样本：3640 在 405 恢复、408 首输出、409 再抢占。409 before 的 free=0，target 自身再输出 3 个只需现有块；但让 31 个 resident 全部共同再输出 1/2/3 个，累计需新增 3/5/6 块，每步 31 token 的计算预算反而足够。第一步跨块的是 799、2820、3345。原动作驱逐 target；保护它时不能再信用它释放的 205 块。

因此“延长到 4 token 且所有 peers 照常执行”在这个状态不可行。同前态固定 FCFS fit 子集算术则允许三步分别执行 28/26/25 个请求（均包含 target），不新增块或 victim；六个 peer 分别在第 1/2/3 步起被推迟，窗口内额外等待上限为 3/2/1·t_batch。原本排队的 3571 也增加最多 3·t_batch 的窗口内等待；不能只计 resident。t_batch 尚未为这条候选路径实测，窗口结束后不承诺立即输出。这证明需要安排后续共批子集及其代价，不证明该安排改善性能；局部 all31 不可行不等于全部窗口或整个负载无解。

**唯一下一项 GPU 沿共享主线执行：新 cohort3 的 native / most / fit / residual 及反序八格，CPU 准备中、GPU UNRUN。**排除此前 128 个文档，不另起窗口组。它区分当前实测交换是否在独立文档上保留，以及既有 most 轮转是否已经覆盖这个交换。若 most/native 已覆盖，则保留强简单策略，停止为同一现象增加窗口模型；若 residual 在强基线之后仍留下稳定、有成本解释的边界，才研究窗口动作。

条件接续的窗口实验应针对上述可见 KV/age 状态，明确共同执行子集或固定规则下的 victim，比较一个预先声明的小输出窗口与简单规则；不能直接延长保护并假设 peers 不受影响。需要同时记录被推迟请求、新输出量、再恢复成本和完整服务指标。新文档仍来自同一分布，不替代异构上下文、未知 EOS、长期持续到达和第二模型；不得因这八格完成就宣称跨运行域成立。

## 本轮边界

Evidence type：旧 8 格和新回传 6 格的只读 NATIVE_SERVING 分账；合成与单步 STRUCTURAL 模型。前 8 格有 fresh same-family/provisional 独立复算，后 6 格的执行资格引用原执行者、生命周期由本轮相同计费函数重新计算；不把前 8 格审计追认到新六格。审计后仅增加 campaign/格数 CLI 参数以接收六格，原审计 source hash 保留；逆向去掉这几行参数修改后精确匹配审计散列，计费函数没有变化。

Strongest baseline：当前同组 fit/residual；旧 native/most/least 与完整近邻仍应进入独立数据上的强对照。Oracle/headroom：无完整 Oracle，真实窗口排序仍 UNVERIFIED。未测：真实 host 全峰值、全历史 offload、新窗口策略、未知 EOS/异构/长期持续到达、第二模型及质量。

当前直接结论是：**昂贵恢复确实可以兑现首输出，简单共批修正可以改善其执行代价；仍没有证据证明，多保护几个输出就一定改善完整服务的延迟—效率边界。**

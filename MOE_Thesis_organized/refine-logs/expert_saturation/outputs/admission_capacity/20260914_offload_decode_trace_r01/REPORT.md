# Native offload 共同 decode 活动定位

Verdict: MEASUREMENT_ONLY / NATIVE_PROFILER_DIAGNOSTIC。两臂 COMPLETE；在冻结的相同32调用中，额外时间主要落在已记录GPU活动之外，尚未定位为特定 connector 函数，也不构成优化收益。

## 实测

固定旧d6/OLMoE/BF16/单5090/native vLLM0.26，offload off与默认prompt-only on16GiB各一次。500..531调度逐请求签名均匹配冻结expected_window，32/32完整输出序列一致。每臂32个CPU调用标记；另有32个同名GPU注释，分析器按user_annotation区分。每臂9920 kernel +354 memcpy，10274/10274 GPU活动均能在本trace找到CUDA API correlation，记录流均13；这不证明捕获了进程外活动或给出请求级因果归属。

| 固定32调用，ms | off | on |
|---|---:|---:|
| 首CPU scope开始至末scope结束 | 648.874 | 868.054 |
| CPU scopes并集 | 628.315 | 841.210 |
| GPU活动并集 | 511.701 | 511.551 |
| kernel并集 | 511.526 | 511.376 |
| memcpy并集 | 0.174 | 0.175 |
| CUDA API并集 | 514.953 | 520.873 |
| CPU算子并集 | 23.705 | 50.121 |

CPU、GPU、API/算子彼此重叠，表中行不可相加。窗口减GPU活动并集为137.174/356.504ms，增加219.330ms；仅表示本trace未记录GPU活动的时间，不能直接称硬件空闲或可移除成本。attention kernel总时长266.052/265.760ms，fused_moe_kernel213.831/213.910ms，未显示kernel计算变慢能解释219.180ms窗口增加。

cudaEventSynchronize总时长497.388/477.200ms，说明API时间大量与GPU计算重叠，不能计为额外CPU税。cudaGraphLaunch11.220/29.341ms；CPU算子类别计数相同而耗时广泛增加。这支持继续检查主机执行/派发环境，不能单凭它判定某个offload函数是根因。

## 边界与判断

证据层级：原生runtime中的profiler诊断。只跑off/on各一次，无反序；CPU频率、竞争、profiler差异影响没有隔离。逻辑工作/输出相同不代表KV物理布局相同。完整运行带profiler，不作mean completion或吞吐GO；也未测试轮转+connector兼容、full-decode offload、第二workload或多卡。

最强对照为相同固定资源/workload的原生offload关闭臂。Oracle/headroom未由本实验建立；此前复制微基准和减少重算均不能替代完整请求净收益。当前失败类别仍是默认offload实现的成本/运行环境待分离，不是KV保留策略家族失败。

唯一下一最小实验：同一32调用、反序on/off复测，保持封存配置，同时在两格记录CPU频率/负载与进程CPU时间；检验主机侧差异是否随offload状态而非顺序复现。若差异反转或消失，先归为运行漂移，不实现猜测性优化；若复现，再对已定位主机路径做一个同状态消融。

归档SHA256：4d6c0214dcb203945bce3fd64b859886c606555e95ba8828fe525657c0a19083。原件只读保留；analysis.json、trace_qualification.json为派生分析。本次为执行者定向核验，未宣称新增独立审计。

复算：python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_bounded_decode_trace.py --results refine-logs/expert_saturation/outputs/admission_capacity/20260914_offload_decode_trace_r01/readback/results --expected refine-logs/expert_saturation/outputs/admission_capacity/20260914_offload_decode_trace_r01/expected_window.json --output /tmp/offload-decode-analysis.json

## CPU时间边界补充（原trace只读分析）

32个CPU scope互斥分为首次CUDA API前/首次API至唯一cudaEventSynchronize返回/同步返回至scope末。off为40.027/552.913/35.374ms，on为187.880/605.646/47.685ms，逐调用精确闭合。scope额外212.895ms中147.852ms（69.45%）位于首次CUDA API之前。该定位缩小到调用早段，不自动等价于scheduler或connector函数；两臂顺序/环境混淆仍在。代码analyze_decode_host_phases.py，派生host_phases.json，原件未修改。

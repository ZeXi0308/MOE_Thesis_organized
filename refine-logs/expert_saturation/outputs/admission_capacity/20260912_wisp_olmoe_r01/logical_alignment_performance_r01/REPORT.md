# Logical alignment：单cohort请求阶段正信号

**Verdict：MEASUREMENT_ONLY / REQUEST_PHASE_SIGNAL。** 在本组固定资源和16文档上，Y相对强F与原X的capture时间均在正反序下降，最大ITL也下降，达到冻结的探索继续条件。全进程Y/X费用仍翻转，不能宣称稳定净收益或方法GO。下一只换独立文档验证，不改机制/预算。

F为private20×16+stage64的fullstage；X为private21×16+shared48的原physical384 oneshot；Y在相同X资源/拷贝/LRU上，按logical64分桶后映射到物理槽。Y使用已有vLLM原语并完整计入包装/映射成本，不将其包装成调度或MoE特有的新机制。

## 原始六格

| 格 | capture s | 均完成 s | TTFT均值 s | 最大ITL s | 整个子进程 s | 编译准备 s |
|---|---:|---:|---:|---:|---:|---:|
| 0_F | 8.176383 | 5.474759 | 0.820231 | 0.190423 | 60.730738 | 6.842109 |
| 1_X | 8.271075 | 5.560276 | 0.834699 | 0.212861 | 59.046425 | 5.767553 |
| 2_Y | 8.006667 | 5.326670 | 0.792387 | 0.186193 | 60.040813 | 5.442077 |
| 3_Y | 8.033705 | 5.339523 | 0.800831 | 0.190086 | 58.164760 | 5.440195 |
| 4_X | 8.231974 | 5.520317 | 0.837149 | 0.200036 | 60.666077 | 5.708491 |
| 5_F | 8.172975 | 5.481042 | 0.825004 | 0.198490 | 60.034148 | 6.824224 |

## 固定方向比较

百分比为target/baseline−1，未翻分母或挑有利格；capture和延迟负值表示下降。

| 比较 | block | capture | 均完成 | 最大ITL | 全进程 | H2D | D2D |
|---|---|---:|---:|---:|---:|---:|---:|
| 2_Y / 0_F | forward | -2.076% | -2.705% | -2.221% | -1.136% | -1.833% | -54.460% |
| 2_Y / 1_X | forward | -3.197% | -4.201% | -12.528% | +1.684% | +0.519% | -0.339% |
| 1_X / 0_F | forward | +1.158% | +1.562% | +11.783% | -2.773% | -2.340% | -54.305% |
| 3_Y / 5_F | reverse | -1.704% | -2.582% | -4.234% | -3.114% | -1.833% | -54.460% |
| 3_Y / 4_X | reverse | -2.409% | -3.275% | -4.974% | -4.123% | +0.519% | -0.339% |
| 4_X / 5_F | reverse | +0.722% | +0.717% | +0.779% | +1.053% | -2.340% | -54.305% |

两引擎均值之比：Y/F capture −1.8899%、均完成−2.6434%、最大ITL均值−3.2485%、全进程−2.1193%；Y/X分别−2.8036%、−3.7399%、−8.8685%、−1.2588%。这些均值没有替代上表Y/X全进程+1.684/−4.123%的翻转。最大ITL聚合为两次engine最大值的均值，不是pooled请求最大值。

Y相对X的H2D反而增加0.5187%，D2D只减少0.3386%。相对F的大幅D2D减少并不都是本次logical动作所得：原X已有−54.3049%，本组原X仍比F慢+1.1581/+0.7219%。只根据字节下降不能解释完整收益，也不把嵌套apply/load时间相加为墙钟saving。

## 重复与证据范围

| 同arm后/前 | capture变化 | 均完成变化 | 最大ITL变化 | 全进程变化 | 相同输出 / route |
|---|---:|---:|---:|---:|---|
| 5_F / 0_F | -0.042% | +0.115% | +4.236% | -1.147% | 16/16 / True |
| 4_X / 1_X | -0.473% | -0.719% | -6.025% | +2.743% | 16/16 / True |
| 3_Y / 2_Y | +0.338% | +0.241% | +2.091% | -3.125% | 16/16 / True |

每arm两次输出与route均相同；跨armY/F只有13/16输出相同、Y/X14/16，route均不同。各arm独立推进到达、batch、KV和生成轨迹，未共享future或按相同输出过滤。两次有序engine、一个文档cohort不是总体噪声底，不以观测到的小repeat差构造显著性或3倍阈值。最大ITL改善尤其仍只是这组描述结果。

96请求/3072生成token全部完成；15,264实际scheduled positions。每引擎896个measurement层调用，共5376；含初始化与warmup共6048。所有18个GPU边界检查通过。没有抢占、重算或已经输出请求的decode缺席，故本组不支持长上下文恢复或防饥饿主张。

Y两个引擎各32 initialization、80 warmup、896 measurement，每个原始MoE调用均恰一次fused_experts及其alignment helper（内部两次GEMM不计为一次CUDA launch），原trace身份有序SHA、rows和参数逐phase一致；F/X开关关闭。无资格参考双算/负控/前缀/数值readback，新增aggregate接口记录成本留在Y。每格320编译准备完成，物理local384/global64与原X local/global384区分；pager/KV状态保持，另从原始host probe直接计数，六格measurement阶段所有被观察compiler/load事件均为0；这仍不保证整个runtime稳态。F各26个、X/Y各22个预编译key。

## 封存与边界

[冻结协议](protocol.json)；[完整复算](analysis_rebuilt.json)；[原件](attempt01/results/execution.json)；[回读核对](readback_verification.json)。[限定完整性审计](EXPERIMENT_AUDIT.md) PASS，P0/P1=0；fresh same-family requested-route，结论为 provisional。

输入包276801B/SHA `513da51fb7276e05f28e75eda66a7661aece0bb95cd281b12e938578e4591961`，40输入；回读21980850B/SHA `d82c23e224e574267f485173d9972106d581248dc5fe576a223bc924fa1c9fd9`，1471成员/1470载荷。全部hash/size通过，本地与远端封存分析逐字段相同，issues=[]。

remote `/root/autodl-tmp/logical-alignment-performance-20260914-root-r01`，controller55793，finished1789332537.156036。1789332592.376现场PID不存在、GPU空、共同锁可取，整组已释放。后续SSH空闲断连只影响回传，未重新运行或覆盖任何GPU数据。

证据层级：原生eager pager的有限到达完整请求测量，单OLMoE BF16/RTX5090，人工384专家槽/1GiB KV预算，P128/O32，0.25s到达，token160/prefill32；实际模型可装入整卡，不是超显存部署证明。最强性能基线F，原X为动作对照。Oracle/稳态容量/SLO/任务质量/EP/跨模型未测。失败类别：尚无稳定全进程净收益证据；不否定paging家族。

唯一下一实验：保持source、instrumentation、预算、目标和成本不变，按来源顺序选择独立B文档eligible113..128，逆置整体臂位置Y/X/F/F/X/Y。检查方向是否跨cohort保持；任何翻转和不利指标照实保留，不扩大当前主张。

独立验证补记：[第二cohort](../logical_alignment_validation_r01/REPORT.md) 的Y/X方向未保持（+1.7684/−0.0648%），全进程两块增加；P原数值保留，停止稳定增量主张，不继续cohort扫描。

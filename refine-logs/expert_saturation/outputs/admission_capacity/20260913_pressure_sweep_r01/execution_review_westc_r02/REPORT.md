# 固定 KV 压力扫描：轮转改善停顿，但高压完成代价扩大

2026-09-14。**16/16 格执行结束、512/512 次请求完成、524288 个新输出 token 对齐。14 格 COMPLETE；2 格 d0 轮转 raw 为 INVALID_NO_ACTION，runner 最终为 INCOMPLETE，原标记保留。NATIVE_SERVING（原生vLLM进程内；冻结runner ceiling为NATIVE_INPROCESS_CAPACITY_QUALIFICATION） / MEASUREMENT_ONLY；研究问题 OPEN。**

GPU RTX 5090（70fa1c0a-77d4-c14a-9daf-7e685874eef9），vLLM 0.26.0 / torch 2.11.0；OLMoE BF16。32条固定3072输入/1024输出、50ms到达，显式KV池8192/7671/7168/6656可用块；原最久等待恢复、least_progress victim及冷却/保护参数不变。两个反向区组复用同一32篇输入，不是512个独立样本。

唯一问题：固定轮转的长停顿—服务量权衡是否跨KV压力保持；并检验先前冻结的native压力预测。执行前更正见 [PRE_EXECUTION_ADDENDUM](PRE_EXECUTION_ADDENDUM.md)。原始预测未改写。

## 完整请求主表

每项按 block0 / block1 排列；正吞吐变化为更快，正平均完成变化为更慢。仅同block、同实际资源配对。

| 点 | native最大ITL(s) | rotate最大ITL(s) | 吞吐变化(%) | 平均完成变化(%) | 全调用 native→rotate |
|---|---:|---:|---:|---:|---:|
| d2 | 4.763 / 4.630 | 1.055 / 1.057 | 1.706 / -0.905 | 0.075 / 2.480 | 1348→1257 |
| d4 | 9.701 / 9.618 | 2.147 / 1.850 | -0.031 / 0.778 | 4.235 / 3.488 | 1606→1426 |
| d6 | 14.197 / 14.399 | 2.816 / 2.805 | -0.955 / 2.849 | 8.905 / 4.486 | 1869→1581 |

长停顿优势在这三个压力点及两个区组均保持，且绝对差随压力扩大；轮转本身的最大ITL也从约1.06秒升到约2.81秒，不构成固定停顿上界。吞吐在每个压力点均随区组变号，不能称非劣或可重复吞吐提升。d4平均完成慢3.49%–4.23%，d6慢4.49%–8.90%。

d4两区组分别32/32、31/32请求完成变慢；d6分别32/32、30/32变慢。d6平均增加1.841/0.945秒。因而这不只是均值被一条极端请求拉高；大量请求承担了避免长输出停顿的代价。最长停顿与完成时间是不同目标。

## 冻结预测检验

误差以下按(实测−预测)/预测计算；采用实测作分母也不改变M1/M2的通过/触发结论。旧pure调用指标只用于检验原数值预测，已撤回其物理浪费解释。

| 预测 | 冻结值 | 两block实测 | 误差 | 判断 |
|---|---:|---:|---:|---|
| d4 onset | 466 | 458 | -1.72% | M1未触发（20%） |
| d4 native pure调用 | 1732 | 1492 | -13.86% | M2未触发（30%） |
| d6 onset | 213 | 202 | -5.16% | M1未触发（20%） |
| d6 native pure调用 | 2543 | 1748 | -31.26% | M2触发（30%） |

**d0实际可行，触发M4；d6的线性步数外推触发M2。** d4/d6 onset分别458/202，两策略和两个区组一致；支持共同首次干预前轨迹上的校准块增长预测，不证明策略无关定律。native pure1492→1748，M3非单调条件未触发。

d0不可行预测的错误已定位：本机vLLM `config/cache.py:171–178`说明，显式kv_cache_memory_bytes时忽略gpu_memory_utilization。将0.9用作这一路径的硬总显存门槛不成立；并非实验调整资源凑出了可行结果。实际8193 total / 8192 usable已由引擎资格确认。

## 成本模型修正

所有调用和返回token一一对齐，计时互斥；pure计数包含最后调用，旧预测的pure计数另外排除最后调用。

| block0点 | 重算类调用 native→rotate | 其中新输出 native→rotate | pure类engine秒 native→rotate | 重算类engine秒 native→rotate |
|---|---:|---:|---:|---:|
| d2 | 8→40 | 233→1185 | 20.980→19.783 | 0.237→1.212 |
| d4 | 15→100 | 404→2831 | 22.082→19.473 | 0.460→3.149 |
| d6 | 22→156 | 545→4226 | 23.377→19.770 | 0.631→4.318 |

d6 pure类时间少约3.608秒，同时重算类时间多约3.687秒；重算类还产生4226个新输出，不能把它全部称为重算税。全调用少288步，但各调用的工作和批宽不同，不推出墙钟或平均完成改善。这里是各策略实测轨迹的分解，不是保持未来状态不变的因果扣减。

## d0负控与运行边界

四次d0均1121个全调用、旧pure1022、零抢占。轮转未执行强制动作，因此仅作为负控，未放进机制胜负主表。

两组d0配对的全部scheduler请求顺序均一致，且各32/32输出序列相同，见 [zero_action_check.json](zero_action_check.json)。在这些零动作对照中仍有墙钟差，不把它归因于输出工作量变化，也不将有限样本差值升级成噪声上界。

| block | native/rotate wall(s) | rotate相对native吞吐变化 | native/rotate最大ITL(s) |
|---|---:|---:|---:|
| 0 | 23.1019/22.8481 | +1.111% | 0.1028/0.0966 |
| 1 | 22.8094/22.9706 | -0.702% | 0.1182/0.0892 |

这是零动作观察差，不是总体噪声界；不能移用它证明d6非劣。每格初始GPU边界为空、源码/输入/模型一致、实际KV匹配；不宣称连续物理独占监控。日志中的16条JIT提示均早于估计的正式测量开始（1.11–1.92秒；日志秒级分辨率），不能用这些提示解释测量内长调用，也不据日志缺失证明没有编译。

## 裁决与唯一下一步

- Verdict：固定least_progress轮转的暂停—完成时间权衡在d2/d4/d6存在；不能定位为普遍吞吐优化。
- Evidence：NATIVE_SERVING（原生vLLM进程内；冻结runner ceiling为NATIVE_INPROCESS_CAPACITY_QUALIFICATION） / MEASUREMENT_ONLY；16格原始结果完整保留。
- 已否定：d0内存不可行估计；单点缺席→额外步线性预测在d6的冻结精度要求。
- 未测：在线候选动作排序、动态/异构终止长度、质量、第二模型、多卡、一般停顿保证。
- 强基线：本扫描只比较native；headroom/most_output已有d2证据，高压d6的直接对照仍缺。
- Oracle/headroom：本实验不是Oracle，不能由调用数差给可回收服务量上界。
- 失败类别：成本模型外推与目标权衡；不是问题family死亡。
- 继续条件：保持长停顿目标，显式计入完成代价；新机制须超过同压最强简单策略。
- 唯一下一实验：在d6加入既有headroom与most_output，与native/原least_progress轮转做两个顺序区组的同底座直接对照，先判断简单策略是否已有更好权衡，不训练完整缺席预测器。此后续尚未准备/运行，按共享GPU队列另行执行；本次GPU已交接，不自动续跑。

研究问题的直接答案：**主动恢复确实能抑制固定KV压力下的长输出停顿；压力越高，固定轮转给多数请求带来的完成代价越明显。当前证据支持可优化的权衡问题，不支持已实现无损服务量提升。**

## 接续与复算

原始归档 `results.tar.gz` SHA256：`910046956d3152788aadccaa5c21043a48dc45e89e187055c6dbcac4bbe30d6c`，已与远端一致。原始目录 [readback/results](readback/results/)，配对与调用数据 [analysis.json](analysis.json)，源码/资源检查 [comparison_checks.json](comparison_checks.json)，JIT边界 [jit_window_check.json](jit_window_check.json)。

复算命令（从仓库根目录；输出新文件，工具拒绝覆盖）：
```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_pressure_review.py --results refine-logs/expert_saturation/outputs/admission_capacity/20260913_pressure_sweep_r01/execution_review_westc_r02/readback/results --output /tmp/pressure-review-recheck.json
```

## 有界结果审阅

Fresh gpt-5.6-sol/ultra，同模型族 provisional：Overall WARN，未发现影响主表与M1/M2/M4结论的P0/P1。范围限制为同一工作负载两区组、高压强简单基线缺失、仅原生vLLM进程内而非外部服务部署验证。策略实现的未来信息合法性未在本次结果审阅中重审；d0额外顺序/输出一致性由执行方计算，fresh审阅未另行复算。见EXPERIMENT_AUDIT.md/json。未扩大为第二轮审计。

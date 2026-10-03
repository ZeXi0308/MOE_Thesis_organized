# 中间预算512：尾部极值改善，完整请求仍有代价

2026-09-08。本线程独立问题，承接上一轮已预选的唯一中间静态点。

**回答：512没有同时改善主要完整请求指标。** 相对同引擎1024，两个反序block的
ITL p99均降低约29%–30%，但平均TTFT增加6%–8%，完成墙钟增加约2%，平均TPOT和
ITL p95也增加。512缓和了上一轮256的代价，仍是一项配置权衡；按冻结规则停止继续
扫静态预算，不由某个较好的百分位宣布整体GO。

## 实际执行及回传

固定单RTX5090、OLMoE/BF16/vLLM0.26.0、cap8、编译token容量1024、显存预算0.70。
沿用相同16篇实际输入token：128/2048交替，固定128输出，50ms到达，seed20260905。
每个新引擎先做相同mixed512、mixed1024两个warmup，测量顺序正反对调。

**4次测量＋4次预热全部完成，两个进程退出码均为0；64次正式请求执行、64次预热
请求执行，每条128输出。** 独立输入仍只有16篇，不能把重复执行数当独立样本数。

所有raw、输入、执行代码、配置、时间戳、step记录、stdout/stderr、环境、命令与退出码
均已回传 `readback/results/{forward,reverse}/`。两个归档各28文件，逐文件字节核对、
全部JSON读取通过；远端原件保留。传输记录见 `TRANSFER-{forward,reverse}.json`。

```
forward 613897 bytes  8b4befc0e66db21ddb7c9cb0bf43c3cf20856f4e4bb592e4299e004b0d56825f
reverse 617523 bytes  7326efbdae94d9aeae2e44ec8f743b8aa7e66b3f577b5f4df6eb1164c648bc7a
```

## 同引擎结果：1024 → 512

| 全体请求指标 | forward | reverse |
|---|---:|---:|
| TTFT均值 | 425.862→452.961ms（+6.36%） | 425.161→458.339ms（+7.80%） |
| 平均TPOT的请求均值 | 8.551→8.629ms（+0.90%） | 8.564→8.655ms（+1.07%） |
| ITL p95 | 13.630→14.108ms（+3.51%） | 13.080→14.076ms（+7.61%） |
| ITL p99 | 20.713→14.520ms（−29.90%） | 20.594→14.523ms（−29.48%） |
| 整批完成墙钟 | 2.526→2.579s（+2.12%） | 2.529→2.589s（+2.38%） |

长请求ITL p99也下降约29.6%–29.8%，但p95增加约7.7%–8.7%；短请求p95在两次运行
中一降一升。不能把“p99变好”写成“所有token间隔变好”。分位数是实际间隔的描述性
合并分布，不把相邻steps或2032个间隔当独立统计样本。约1%的TPOT差异仍接近运行波动，
不做统计稳健性主张；TTFT、p99和完整墙钟均分别保留两次原值。

参考SLO仍是5s/0.2s，不据结果重新设置阈值，也不以参考SLO全达标宣布goodput收益。
主比较只在每个新引擎内部完成。旧256与本轮512不构成同引擎直接配对，
因此“代价缓和”仅是两轮各自相对1024的描述性比较。

## 动作与时间边界

每个1024 cell均有16步实际超过512；512有32步触顶，动作确实暴露。
每格既有decode的2032次推进检查全部通过，skip/preemption/KV adjustment均为0。
混合prefill/decode步骤均由27增加至41。512减少单次较大停顿，但增加中等停顿及
混合步骤次数；这是观测到的分布变化，不是kernel时间或可免费回收成本的证明。

同host钟TTFT分解为：

```
arrival → engine add返回 → 首次schedule入口 → 首token receipt
```

| 全体TTFT均值组成，ms | forward1024 | forward512 | reverse1024 | reverse512 |
|---|---:|---:|---:|---:|
| arrival到add返回 | 4.351 | 5.451 | 3.972 | 5.670 |
| add返回到首次schedule | 384.372 | 402.209 | 384.575 | 406.847 |
| 首次schedule到首token | 37.139 | 45.301 | 36.613 | 45.821 |

TTFT的增加同时体现在首次调度前的等待区间与首次调度后的首token区间。
这里的中间桶是host观察边界，不等同于native scheduler独占排队时间；不能把384ms
直接当成某个新队列策略可省掉的Oracle。

进一步只按实际已提交、尚未首次调度的请求重建观察到的等待集合：每格均有3个
新prefill接纳时刻，最早到达者为长prompt，同时有短prompt已经在等；等待集合大小
分别为7、5、3。只用事件当时已有的arrival、add返回和prompt长度，没有引用未来route。
这证明存在可区分的候选顺序，尚未证明重排接口、收益或公平性。

## 校验与结论边界

复用上一轮执行包的runtime和输入；最小修改见 `midpoint.patch`。独立算术复核从raw
直接计算，未调用本分析器，304项主指标及配对差异全部一致，最大绝对差异0。
首次分析器标签替换误将sha256名称改为sha512，入口立即报错，未生成结果；已恢复身份
字段并重算，详见 `ANALYSIS_CORRECTION.md`。GPU代码、raw和输入均未受影响。

上一轮shared runtime的fresh审查为WARN/provisional、P0=0/P1=0；这里没有冒称又完成
一轮完整语义审查。本轮检查聚焦实际参数生效、完整请求和精确重算，不扩张审计。

各策略自由推进其KV/batch/输出，跨预算只有3/16和4/16条完整输出序列相同。
不主张任务质量、token一致性、same-state反事实或精确数值conformance。
warmup独立保存，不进入正式episode分母；正式墙钟包括提交、排队、执行与采集。
数据仅覆盖固定自然文本、一个到达模式、一个模型和原生in-process runtime。
这些普通静态预算结果是MoE研究的简单基线，尚无专家信号残差或独立方法新颖性。

## 裁决与唯一下一步

| 字段 | 结论 |
|---|---|
| Verdict | MEASUREMENT_ONLY；512未同时改善主要完整请求指标，结束当前静态预算扫描 |
| Evidence type | REQUEST_LEVEL / NATIVE_IN_PROCESS_GPU_INTERVENTION |
| What was measured | 4测量＋4预热；真实512/1024预算、完整TTFT/TPOT/ITL/墙钟、step与等待机会 |
| What was not measured | 新排队策略、任务质量、Oracle、专家残差、HTTP生产服务、多模型、多卡 |
| Strongest baseline | 同引擎1024静态预算；不是全局最优或动态Oracle |
| Oracle/headroom | 未测；3个顺序选择事件不等于存在净收益 |
| Claim ceiling | 该域静态分块预算将极尾间隔、较常见间隔及首token/完成成本重新分配 |
| Failure category | 目标之间存在已测权衡；不是动作非法或整个prefill调度家族NO-GO |
| Resurrection condition | 明确的逐token间隔约束允许付出该TTFT成本，或新机制能避免已测等待传播 |
| One next experiment | 固定cap8和预算1024，对尚未首次调度的请求比较FCFS与短prompt优先；已在decode的请求持续推进，分别计入短/长请求代价；新策略UNRUN |

先验证最强简单队列动作是否还有完整请求余量，再谈专家感知机制。**本轮问题已经回答：
中间预算512也没有消除权衡，继续微调静态chunk不是当前最有信息增益的下一步。**

主数据为 `analysis/summary.json`，可读全表为 `analysis/TABLE.md`。
三个小分析脚本分别保留主指标、互斥时间桶、TTFT边界；顺序机会脚本为
`queue_order_opportunity.py`，原始结果均不修改。

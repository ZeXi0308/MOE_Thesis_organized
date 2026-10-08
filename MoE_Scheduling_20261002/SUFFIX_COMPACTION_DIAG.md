# 跨层后缀压缩诊断：实际缩小执行，完整收益尚待同组比较

2026-10-02。**`n2fixed0c` 已真实缩小后续层 query 与 LM-head 执行行，并在发生 KV 抢占/恢复的完整请求中自然结束。** 这是单格执行诊断，不是加速结论；正在运行的八格主实验尚未纳入本报告。

证据为 [完整分析](execution_suffix_compact_diag_41307_r02/results_suffix_compact_diag_r02/suffix_metrics.json)、[原始 compaction trace](execution_suffix_compact_diag_41307_r02/results_suffix_compact_diag_r02/00_n2fixed0c/compaction_trace.json) 与 [suffix trace](execution_suffix_compact_diag_41307_r02/results_suffix_compact_diag_r02/00_n2fixed0c/suffix_trace.json)。原始记录未修改。

## 实际执行与边界

配置沿用 OLMoE、人工 expert24、cap16、Q2048、GPU/CPU KV 各1GiB；N2提案、第0层固定撤销全部 eligible 草稿，greedy自然EOS/max512。真实 torch/vLLM 测试已通过14/14；本表来自随后一次完整请求运行。

| 观测 | 实测 |
|---|---:|
| 测量 forward / 实际跨层压缩 | 201 / 195 |
| 其余 no-cut / 不支持路径 fallback / 失败 | 6 / 0 / 0 |
| 被裁 request-step / 不同请求 / 草稿行 | 797 / 16 / 1594 |
| 195次压缩内，15个后层 query row-layer 位置 | 182400 → 158490（−23910，−13.11%） |
| 195次对应 LM-head 原始 / 实际执行行 | 3090 → 1496（−1594，−51.59%） |
| 决定层实际 after-load 符合局部冷并集账 | 195/195 |
| q/seq/context、query offsets、合法原前缀核对 | 各195/195 |
| 后层输入行及 pager 输入行对应核对 | 各195/195 |
| 返回原 hidden 形状、LM-head prefix位置核对 | 各195/195 |

第0层 attention 仍按原始 query 形状执行；该层 MoE 作合法活行 gather，之后15层真正缩短 query/位置/slot映射，最后恢复 runner 所需 hidden/logits 外形。这与 r01 仅减少 MoE 行不同。上表行数统计限定在实际压缩的195次调用，不能直接换算成全部 episode FLOPs 或时间节省。

q/seq检查验证已记录 plan 的 `seq_len − query_len = computed_start`、请求连续前缀和实际层输入行记录；它不是独立 GPU metadata 回读，也不证明浮点输出等价。原请求次序、KV块表/物理槽身份和原生拒绝回退接口保留；这里未新建 KV 池或按预测输出长度分配容量。

## 完整成本、质量与连续推进

| Capture / served / drained（s） | 实际 token/s | Mean flow / TTFT（s） | 最大 host gap（s） | 输出 / 正确 / 自然stop |
|---:|---:|---:|---:|---:|
| 18.2959 / 18.2948 / 18.2983 | 82.1933 | 12.1432 / 2.5467 | 0.8246 | 1504 / 9题 / 16请求 |

无长度截断；全部输出 receipt 为单token。Native sampler在stop过滤前提交1504，最终返回1504，接受草稿数为0，与fixed0动作一致。完整 engine wall为18.272491s，drained余量25.844ms保留；末尾native drain调用0次。初始化、warmup与导出仍按原捕获边界排除。

本次有2次原生抢占，真实KV load **178,257,920 B**、store **1,639,972,864 B**；末态无pending transfer/store。这证明本输入中压缩后请求能跨多轮及这些恢复事件继续运行到自然结束，不能外推为所有KV状态均正确。

专家全episode实际copy为**643.314GB**。决定层各次相同入口状态的局部冷字节差合计**18.484GB**，after-load核对通过；该局部差不是与另一策略的全episode实测节省。没有使用后层死行路由构造未执行的反事实。

## 子计时包含性

| Host窗口 | 总计（s） | Median / P95 / Max（ms） |
|---|---:|---:|
| Compaction setup（195次） | 0.037345 | 0.18571 / 0.22232 / 0.39517 |
| 最后hidden scatter（195次） | 0.005850 | 0.02943 / 0.03628 / 0.04481 |
| Logits adapter（195次） | 0.027404 | 0.14025 / 0.16405 / 0.21129 |
| 完整inner forward（201次） | 17.836219 | 93.39015 / 149.05532 / 222.64503 |

此外decision总计约0.0465s，sample重排钩子0.012530s。Inner forward已经包含各decoder层、setup和hidden scatter；logits adapter包含缩短后的LM-head调用及其gather/scatter，位于inner forward之后、engine.step之内。Host窗口记录提交和可能的等待，不是独立GPU kernel耗时；不把它们与engine wall、drained或CUDA加载跨度再相加，也不从总时间扣除。

## 与 r01 fixed0 只比较内容

本格为1504 token/9题正确，r01 fixed0为1436/9；仅**4/16完整token序列相同、10/16提取答案相同**。正确数相同掩盖了两题相反变化：source0037从答案5（错）变2（对），source0038从10（对）变13.33（错）。全部差异均保留。

这是跨组单次结果，不用于归因压缩加速。计算形状变化可能影响BF16 greedy，但尚未通过同前缀logits/KV对照排除接口或状态差异，不能称“已证明只有形状敏感”。本诊断确认可执行的压缩与连续自然完成；质量保持和超过运行波动的完整收益，留给同场强基线与主实验判断。


## 后续独立 attention 数值检查

`compaction_attention_gpu_r03.json` 已在真实 vLLM 0.26 / FlashAttention2 / HND / BF16 上通过2 seeds×3动作共6个case，覆盖5请求混合prefill/decode、非连续物理blocks/slots。测试使用与本报告相同hash的compaction/runtime源：完整执行与压缩后的保留prefix在这些case中最大绝对差为0；压缩对FP32 dense reference最大绝对误差0.00726843，逐元素满足atol=rtol=0.01。整个KV cache（包括未写区域）均逐位符合预期，错误旧seq_lens与错误独立slot写入的负对照均被检出。它排查这些有界状态的attention/slot接口错误，不证明完整OLMoE的greedy位等价，也不改变本组性能负结论。r02在数值case之前因独立harness未显式设置KV layout而失败，原日志保留；r03只修测试配置，服务代码未改。

# Qwen3 BF16 原生加载与请求资格化 r03

**RUNTIME_COMPLETE / NUMERICAL_DIAGNOSTIC_DIFFERENCE / MEASUREMENT_ONLY。** vLLM0.26、RTX5090、Qwen3-30B-A3B BF16 的真实超显存运行路径已完成四请求；最后一层的数值诊断差异仍待定向解释，尚未开始性能对照。

冻结模型 revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`，权重 61,064,245,248 B（56.87 GiB），GPU 31.84 GiB。16片完整SHA校验、18,867源张量及435目标参数闭合；匿名镜像仅改变下载入口。逐片消费后清理本次临时分片，未删除其他模型。r01默认字段错误、r02网络失败均保留。

[独立分析](analysis_v2.json)确认4/4请求、32个正式输出、284计算位置、624次expert-layer调用；另有1条暖机/2输出。48层请求与位置关联通过，已有ready decode每步推进。固定expert cap48、token64、maxseq4、prefill32、BF16/expert/LRU/nested。

配置KV预算512 MiB，实际341×1,572,864 = **536,346,624 B（511.5 MiB）**，余524,288 B不足一个完整block；1个null block，340个空闲block。原分析器错误地要求实际值等于预算，失败保存于[analysis_attempt01.json](analysis_attempt01.json)；[v2](../analyze_qualification_v2.py)从模型和block大小推导精确分配，错误容量回归被拒，raw未改。

48次same-precall全权重参考均finite；47/48通过原有`rtol=.01, atol=.01`诊断，只有layer47未通过：maxabs0.25、relative L2 0.00221848。全层最大relative L2为0.00273967。实际按不相交expert组分别计算并用BF16求和，参考一次计算全部expert；差异来源尚未定向确认。原诊断值与阈值保留，不从finite或小L2推出质量等价，也不把一次allclose差异扩大为模型或研究问题NO-GO。

[资源采样](hardware_summary.json)：11,174条0.5秒周期记录，GPU总占用峰值26.1888 GiB，cgroup总占用峰值88.3931 GiB，未记录OOM增量或其他GPU PID。总内存包含运行前文件缓存及内核计费分配，不是模型独占RAM要求；离散采样不是持续隔离证明。worker85558已退出，parent5624.998秒，其中原生日志model loading5498.824秒。

本轮参考复制另有54 GiB张量payload；正式阶段pager payload另计。参考计算、首次JIT和复制均污染计时，因此不报告本轮TTFT、ITL或吞吐为性能结果。参考沿用实际native top-k，验证的是局部执行，不是任务质量、独立router或完整resident引擎等价。

强基线是同一层、同一precall输入/top-k的全权重kernel；调度Oracle未测。当前失败类别是分析器KV取整错误已修、数值差异待定位。下一唯一动作是核查最后一层分组执行与全权重参考的差异来源，再决定已准备的同预算static32/16 ABBA；不调阈值制造通过，不以两静态点结束总研究目标。

回读archive SHA `e94c772e7ea980f33681365c19088d2d076599197f06fe2e820a22cb6e1cccfc`，5,321,223 B，61文件。所有运行原始结果在`readback_r01`，重分析另写文件。

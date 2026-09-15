# 原生完整增量保存已资格化，短恢复机制暂缓

**Verdict：NATIVE_FULL_INCREMENTAL_SCOPE_QUALIFIED / MEASUREMENT_ONLY。** 64请求完整，3,866个真实保存作业和36个加载作业均接受且完成；保存覆盖非当前prepare victim和decode KV，原生方法未被selected limiter覆盖。本格没有0或1—2输出后再抢占，故不能继续只用前一格的两次异常支持窗口/headroom方法。原生完整保存进入必须比较的强基线；完整服务优劣等待共同轻量计时。

唯一root controller20050/shell20051于1789413926.865—1789414018.886前台执行，exit0。接受包SHA256 `71ee97f3b1928758f883eda3a702f292fce0538bd21ae6c1e992a6cc52725902`；完整归档SHA256 `f79e4c6051683c1b920552c96421a52ce13a08678532edfc2f1f9d457dab0c7a`，22,526,879B，73文件全部回读核验，raw SHA256 `015787d83949548568d93fa3e1666c03cf110ba8f00bc5d9b0ae2761a55882a9`。1789414105.591归档完成，两GPU空、共同锁可取且明确释放。

## 保存语义与真实资源

只改变保存范围。保留原始`_calc_num_offloadable_tokens`和`offload_prompt_only=False`，由native builder依据当时scheduled/已结束请求、完整chunk、已有缓存及分配结果生成增量store；worker沿原生延期提交/flush路径复制。原生本步合法新完成块不截回selected的prepare前prefix。完整增量保存不保证所有历史一直驻留，也不使用实际未来EOS。注册范围证据不是逐字节KV数值一致性证明。

和D相同：64完整文章、334—3011输入、0.2s到达、EOS允许/cap1024，GPU0实际4,096usable＋1null块／8,592,031,744B；host实际16GiB／8,192块。current most_output、30/20/30守卫、prepare→commit、首新输出保护、token budget1024及队列取消规则均不变。没有新victim、窗口、计算份额或更宽松的full专用调度。

| 原件证据 | 结果 |
|---|---:|
| 完成请求 / 实际输出 | 64/64 / 59,581 token |
| EOS / 长度上限停止 | 6 / 58 |
| 接受且完成store / load | 3,866 / 36 |
| store / load真实字节 | 21,667,774,464 / 12,043,943,936 |
| 当前selected prepare之外的完成store | 3,864 |
| 包含decode位置的完成store | 3,723 |
| prepare本步超出旧prefix范围的合法job | 2 |
| 已知finished请求的store / flush事件 | 5 / 9 |
| 强制轮转 / 全部恢复段 | 25 / 36 |
| 确认重新计算位置 | 266 |
| 恢复后再抢占 / 最后完成段 | 16 / 20 |
| 0输出或1—2输出后再抢占 | 0 / 0 |

全部36段都有实际首新输出；再抢占段最少输出4个。原件含10,332个不同的注册key及对应GPU source块，native作业全部完成。host末态8,192有效块、16GiB已满；因此“完整”指保存资格范围，不能称所有曾保存历史同时驻留。未单独导出精确逐次淘汰计数，不将总store减容量直接当作该计数。

末态无pending store/load/ack，post-request drain为0调用、9.567微秒。RSS20,044,177,408B、HWM23,378,276,352B；这些与16GiB KV重叠，父cgroup184GiB仍不是独立进程树硬预算。初始化约20.995s、应用预热0.849s、全controller92.02s，与请求测量及后续序列化/退出分开。

## 解释与边界

仅作本格描述：episode48.830571s，平均完成26.976799s，最大engine-return gap3.971221s，平均TTFT6.866966s，输出1,220.157751 token/s。L→统一host-call入口中位1.765869s，入口→首新输出0.063728s。更多保存作业产生更多诊断工作，且墙钟到达会改变运行轨迹，**禁止与D的诊断时间计算性能增量**。

相对于D的资格记录，本格确实出现不同成本结构：重算位置15,319→266，store从25作业／7.076GB变为3,866作业／21.668GB，host有效状态从6.590GiB变为16GiB。它们是两次各自真实执行的诊断计数，不是固定轨迹的反事实或完整效率胜负。D的两次1—2输出短段未在本格出现，但不能把一个不同观测负担的episode升级为机制已消除或整体改善；仍有16段更长服务后再抢占及长等待。

首次prepare8.907950s，共5次在最后到达12.6s之前；六个EOS仍全部在首次prepare前结束。两格输出量不同17个token，其中两个早停请求长度变化，质量未评估；不强制匹配实际未来输出，也不自动判整个实验无效。有限到达、上限停止主导、自然EOS允许、长期持续服务和质量各是不同证据。

**研究决定：** 暂停窗口/headroom实现，先比较完整保存与selected保存的完整服务成本。二者都是合法同资源候选，不能提前认定谁最强。当前唯一未决问题是：完整保存减少恢复计算后，新增作业/搬运/host占用是否值得，哪个底座给后续恢复节奏留下更好的停顿—效率取舍？

**唯一下一实验：** 复用D/E资格，64请求/格、同输入资源及current调度，轻量selected/full/full/selected四格；不再跑诊断，使用同一EOS语义和稀疏抢占观测，计入必要成本并记录drain。确认最强底座后才比较恢复启动节奏。模型会话可以反驳共同增长信号的决策价值；不能靠两次事后异常预定一个保护机制。

**Evidence type：** NATIVE_SERVING进程内诊断。**Strongest baseline：** 原生完整增量保存已获路径资格，服务排名未定；旧prompt-only和完整近邻系统不能混称。**Oracle/headroom：** 无秒级Oracle或全局上界。**Claim ceiling：** 单模型单episode的native-full范围及资源/恢复测量。**Failure category：** 无执行失败；原短恢复论据在此格不稳定，完整效率未决。**Reopen condition：** 共同轻量计时及强底座仍支持可避免的状态依赖代价，才进入最小调度干预；否则停止对应机制。

复算入口：[冻结README](README.md)、[主分析](execution_weste_26862/analysis.json)。本报告复用唯一原件；资格计数不是新的性能重复，也没有MoE特有、客户端SLO或论文方法GO主张。

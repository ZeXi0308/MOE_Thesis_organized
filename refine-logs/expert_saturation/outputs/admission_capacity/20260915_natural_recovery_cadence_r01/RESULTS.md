# Natural recovery cadence: complete-request result

**Verdict: MEASUREMENT_ONLY；eager可升级为本运行域的停顿优先简单参照。** 相同selected保存下，取消全局冷却使两配对最大生成gap分别下降63.43%和53.12%，实际输出吞吐变化+1.066%/−0.706%，平均完成−1.473%/+1.443%。这是真实请求返回停顿的改善，不只是内部保存/加载指标；同时，一部分请求承担了更多间隔和完成代价。尚未建立业务SLO、一般运行域保证、充分验证的方法或独立新颖性，不增加控制器或参数扫描。

七格448/448请求执行完成：64条组合诊断、384条轻量计时，均复用同64篇文章，不能称448篇独立文章。固定顺序为诊断eager→native/current/eager/eager/current/native，接受包SHA256 `f1a3bda18a22f7f500024c632cbb6a14b81da391dddd017fd43d5291fe727027`。唯一执行、回读及主分析为`/root/prepare_start_contrast`；没有重复、追加或替换运行。证据层级为 **NATIVE_SERVING_INPROCESS_HOST_MEASUREMENT / REQUEST_LEVEL**，不是生产网络服务。

current/eager只改变全局冷却20/0；selected/save-on、victim、target排序、30absence/30residency/.90progress/8absence guards、准备/提交/取消、保护与原生计算份额均固定。native_full_native跳过额外轮转安装，保留完整原生增量保存和原生调度，属于同时含保存范围及调度差异的系统参考，不能把其差异归因冷却。

## 组合资格与完整服务

诊断完成54次实际轮转、54个native store和58个load完成，58个已加载段随后交付新输出。54个READY commit与实际轮转计数一致，其中43个相邻成功commit间隔＜20；例如step819→823间隔4。故此次干预确实改变了实际动作节奏。资格为NATIVE_SAVED_RECOVERY_QUALIFIED，errors为空，随后六格按冻结清单执行。该计数不是收益分数；诊断不加入性能配对，不与D/C旧诊断墙钟比较。

| 轻量格 | 输出tokens / length-stop | capture wall(s) | 输出tokens/s | 请求/s | 平均完成(s) | 平均TTFT(s) | 最大gap(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| block0-native_full_native | 59579 / 58-6 | 36.32744 | 1640.06 | 1.76175 | 18.23363 | 3.96392 | 9.01743 |
| block0-current | 59576 / 58-6 | 35.65827 | 1670.75 | 1.79482 | 18.34344 | 4.33262 | 4.06512 |
| block0-eager | 59576 / 58-6 | 35.28223 | 1688.56 | 1.81394 | 18.07326 | 4.22632 | 1.48655 |
| block1-eager | 60538 / 59-5 | 35.97984 | 1682.55 | 1.77877 | 18.82505 | 4.49660 | 1.34098 |
| block1-current | 60538 / 59-5 | 35.72567 | 1694.52 | 1.79143 | 18.55733 | 4.38240 | 2.86046 |
| block1-native_full_native | 58572 / 57-7 | 34.60380 | 1692.65 | 1.84951 | 16.99650 | 3.70116 | 7.51498 |

| eager相对current | 配对0 | 配对1 |
|---|---:|---:|
| 最大gap | −2.57858s / −63.43% | −1.51948s / −53.12% |
| 输出/请求吞吐 | +1.066% | −0.706% |
| 平均完成 | −0.27017s / −1.473% | +0.26772s / +1.443% |
| 平均TTFT | −0.10631s / −2.454% | +0.11421s / +2.606% |
| 输出token数量 | 相同 | 相同 |

两配对的64/64请求分别具有相同输出长度和stop类型，排除了通过缩短生成长度直接造成本配对停顿改善的解释。但输出token序列完全一致的只有42/64和39/64；不能宣称位级输出或任务质量等价，也不能视各策略后续状态为相同。当前结果保留各自独立执行轨迹，不使用固定trace反事实。

## 逐请求影响及系统参考

每请求最大生成间隔的等权分布如下，单位秒；分位数对64个请求排序后以`(n−1)q`线性插值。它描述既定停顿分布，不是新增SLO。

| 轻量格 | median | p90 | p95 | max |
|---|---:|---:|---:|---:|
| block0-native_full_native | 0.02608 | 0.39702 | 3.36390 | 9.01743 |
| block0-current | 0.03291 | 1.38241 | 3.20645 | 4.06512 |
| block0-eager | 0.03173 | 1.15422 | 1.24220 | 1.48655 |
| block1-eager | 0.03742 | 0.99435 | 1.11842 | 1.34098 |
| block1-current | 0.03718 | 1.72532 | 2.70032 | 2.86046 |
| block1-native_full_native | 0.02576 | 0.11536 | 0.98958 | 7.51498 |

eager/current的逐请求gap改善/恶化数量为27/37、28/36；配对差值中位数分别+1.950ms、+0.242ms，最大单请求恶化+1.45364s、+1.01898s。最大及上尾gap显著下降，并不等于多数请求都改善。完成时间改善/恶化数量为46/18和13/51；TTFT分别46/18和23/41。第二配对中的平均代价及peer等待符合必须保留完整服务成本的预期，不从第一配对的同向改善外推普适支配。

原生完整保存、无额外轮转参考在两格分别有9.01743/7.51498s最大gap，但其median/p90较低，且平均完成和TTFT有明显优势的格仍保留。因此系统参考展示了集中长停顿与更广泛等待之间的取舍；它没有被当作已知缺陷版本排除，也不能被描述为在所有指标上较弱。

| 原生参考相对其他臂 | 最大gap | 输出吞吐 | 平均完成 | 平均TTFT |
|---|---:|---:|---:|---:|
| block0: native−current | +121.82% | −1.84% | −0.60% | −8.51% |
| block0: native−eager | +506.60% | −2.87% | +0.89% | −6.21% |
| block1: native−current | +162.72% | −0.11% | −8.41% | −15.54% |
| block1: native−eager | +460.41% | +0.60% | −9.71% | −17.69% |

第一原生格比current/eager多3输出tokens，第二原生格少1966输出（−3.248%），后者有7个stop而current/eager各5个。原生对每个对应臂均有62/64条相同输出长度、38/64条相同输出序列。第二原生格更短墙钟不能被称为同等生成量的加速。原生相对eager的gap改善/恶化请求数两格均为48/16，强调了本域选择eager是停顿上尾优先，而非全分布或全指标支配。

## 稀疏恢复证据和EOS边界

六轻量格的实际成功抢占数依次18/27/50/60/29/9；current/eager主动轮转为16/41/50/19，原生参考的轮转字段为NOT_APPLICABLE。1–2新输出后再次抢占的段依次0/1/1/0/1/1，零新输出后再次抢占均为0。eager增加了切换，却降低最大输出停顿；不能仅以切换次数或短服务段数量判断完整效益，亦无依据在本轮自动增加保护窗口。

每格最大gap都跨越该请求的一次实际抢占至其后首个新输出：原生两格的文章5122分别7.36892→16.38635和8.03256→15.54754s；current最坏为文章0913的13.19386→17.25898、文章0748的11.60127→14.46173；eager为文章0944的12.37974→13.86629、文章0406的11.06567→12.40664s。ID仅用于离线定位，不进入策略。各臂抢占轨迹不同，不能把这些段当同一动作前状态的逐段反事实或可删除时间上界。

本轮分别确认了真实更密的commit、原生保存/加载完成、新输出返回及完整请求停顿变化。**轻量对照没有测内部重算下降或精确恢复启动提前**；诊断S仅为host提交/engine-call边界，未与current形成新的成对S计时。故支持的是动作节奏和已交付输出停顿的因果对照，不把缺少的启动/重算归因补成已测事实。

六格所有EOS/stop请求自身都没有成功抢占记录；诊断的8个stop也无自身抢占、无`target_terminal`事件。六格在全组首次抢占前完成的stop数依次6/6/6/5/5/6。唯一更晚的stop是block1-native文章6914，在20.01437s终止、20个输出，但自身未被抢占。这仍未覆盖被恢复请求的自然EOS或保护期间terminal释放；允许EOS和在某些请求恢复期间出现其他请求EOS不能替代该边界。

## 完整成本与归档

七格实测GPU KV均8,592,031,744bytes，即4096usable16token blocks加1null、2MiB/block；host唯一KV分配均17,179,869,184bytes/8192blocks。六计时格重置后有效host KV均为0，结束及drain后有效KV如下；请求结束的scheduler store/load/ack及pending store entries均为0。

| 轻量格 | 结束有效host KV blocks / GiB | post-request drain(μs) | 进程VmHWM(GiB) |
|---|---:|---:|---:|
| block0-native_full_native | 8192 / 16.00000 | 8.606 | 18.33617 |
| block0-current | 1656 / 3.23438 | 8.487 | 18.35672 |
| block0-eager | 3289 / 6.42383 | 8.009 | 18.33728 |
| block1-eager | 4027 / 7.86523 | 7.821 | 18.34378 |
| block1-current | 2413 / 4.71289 | 9.927 | 18.34182 |
| block1-native_full_native | 8192 / 16.00000 | 7.299 | 18.33580 |

eager有效缓存占用增加但仍在相同16GiB分配内；边界有效字节不等于额外分配，也不构成其过程中峰值。full结束达到该缓存容量上限。VmHWM包含初始化/warmup历史；共享父cgroup `memory.peak`不存在，保留UNKNOWN，父上限197,568,495,616bytes、swap上限0不是进程独立硬限制。KV/RSS/cgroup视图重叠，不能相加。

capture+drain六格依次36.32744/35.65828/35.28224/35.97985/35.72568/34.60381s，输出率1640.0548/1670.7480/1688.5550/1682.5528/1694.5233/1692.6462tokens/s。该加总不重复计内部传输，排除其间host快照/序列化。很短的post-request drain不说明保存免费：最终store/control若在capture路径中执行，其成本已被计入。计时格详细job、加载/重算数量均NOT_MEASURED；不会从诊断回填或写成0。

每格66个相同warmup请求在448测量预算外，原件及初始化、reset、drain、关闭时刻均保留。六计时进程初始化20.601–21.202s、应用warmup0.844–0.877s、完整进程墙钟64.516–66.730s。诊断另花初始化20.794s、warmup0.854s、进程墙钟76.976s，VmHWM21.603GiB；该详尽观测开销不混入六轻量对照。

本轮新增证据改变的是简单参照选择和干预价值：同域eager的最大/上尾停顿改善两次出现，主要效率指标付出较小且有方向变化的代价；原生系统参考证明长停顿问题不依赖已知有缺陷的自定义baseline。结论限于一个模型、一个受控0.2s到达点、两个对应块，未测动作Oracle、恢复期EOS、低/更高压力转移、其他offload/APC域或生产SLO。下一步应由root选少量运行域边界验证，当前不追加控制器或GPU运行。

原件：[唯一回读](execution_weste_26862/readback/)、[冻结主分析](execution_weste_26862/analysis.json)、[归档核验与释放](execution_weste_26862/readback-verification.json)。唯一归档SHA256 `25410e861876fc2a5fe22da7ebf981a297c8c2213c6d2b894a436a301d4feb8b`，27,281,847bytes；241文件及30接受payload全部核SHA。前台controller23893/shell23894运行于1789417112.8619413至1789417603.7314255；双GPU空闲且共同flock于2026-09-14T20:28:14.282763+00:00释放，本地回读后执行方明确释放整个窗口。无后台任务，接受包与raw未改。

从仓库根目录复算，指定尚不存在的输出：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_recovery_cadence_r01/analyze_cadence.py \
  --results refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_recovery_cadence_r01/execution_weste_26862/readback/results \
  --output /private/tmp/natural-recovery-cadence-reanalysis.json
```

逐请求差值在`performance_comparisons[].request_deltas`，系统参考在`system_reference_comparisons`；上述分布由各格`requests.requests[].max_engine_return_gap_s`按已写公式得出。最大区间取相邻`token_times_s`最大差，与同请求`preemption_events.method_entered_s`对齐。EOS按其自身request_id核对成功抢占，均是已测轨迹的描述，不是新的在线特征。

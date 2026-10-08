# 自然文章持续到达：保存恢复资格与执行持续性边界

**Verdict：NATIVE_SAVED_RECOVERY_QUALIFIED / MEASUREMENT_ONLY。** 单格64请求全部完成，真实完成25次store和31次load；6次轮转准备发生在最后一个请求到达之前。动态到达中的保存恢复路径成立。两次恢复仅产生2/1个新输出后再次抢占，定向源证据指向peer KV增长，不能解释成加载失败。没有性能比较或方法GO。

主执行和主分析均为long-task root。原controller18977、shell18978于1789412553.090—1789412635.117前台执行，exit0；单一接受包SHA256 `3603de480217fd574f7839af46c863ae6b070534dac933d9bcd02e9f17cd8b00`。70文件唯一归档SHA256 `674854a097e4c8427b54f278878f8b92b9329ad338da4799ecce4e16bd4fc6c6`，20,876,821B；原raw SHA256 `a791df9511ca8f1479c3f38ef8c613eef88e9e8afcbb1d60aa48058ce50e391b`。全部已核验并回读，终态两GPU空、共同锁可取，1789412686.140归档完成且明确释放整机。无失败、舍弃请求或追加压力点。

## 合同与实际测量

复用原64篇完整文章和次序，输入334—3011 token；每0.2s到达一条，最后到达12.6s；EOS允许，1024仅为声明输出上限。GPU0 UUID4015…d0e5，OLMoE固定revision、vLLM0.26原生同步进程内执行，4096 usable块＋1 null块实际8,592,031,744B GPU KV，native pinned host KV实际17,179,869,184B／8192块。原0.5s低压力四格沿原合同保留，无额外轮转；其no-host-offload结果不作本格配对，也不能据跨合同差异把动作归因于到达率这一项。

当前most_output和longest_absence、30/20/30守卫、首新输出保护、token budget1024、save-on均固定。open适配仅允许请求进入/结束，保留mixed-prefill和skipped队列资格限制；没有调整victim、窗口或份额。只在prepare保存选中victim已物化的完整16-token块，沿native增量保存、flush、lookup/load和fallback推进。每个策略状态由真实执行产生；没有使用实际未来EOS或事后选定step。

| 实际结果 | 单格诊断 |
|---|---:|
| 请求完成 / 输出token | 64/64 / 59,564 |
| EOS停止 / 声明上限停止 | 6 / 58 |
| 强制轮转 / 全部原生抢占事件 | 25 / 37 |
| 完成store / load作业 | 25 / 31 |
| 实际store / load字节 | 7,075,790,848 / 11,085,545,472 |
| 恢复段 / 涉及请求 | 37 / 22 |
| 有新输出后再抢占 / 最后完成 | 15 / 22 |
| 其中0输出 / 1—2输出再抢占 | 0 / 2 |
| 确认重新计算位置 | 15,319 |

所有37段都观察到恢复启动和随后新输出，不能仅由`completed_recoveries`推断这一点。统一host engine-call入口的L→S中位1.600199s，S→首新输出中位0.055879s；累计恢复间隔59.070705 request-s跨请求重叠，不是墙钟或可回收上界。

完整诊断episode为39.925626s，平均完成21.664046s，最大engine-return生成间隔3.531065s，平均TTFT5.586604s，输出1491.873932 token/s。它们包含诊断成本，仅作描述，**不得与旧轻量计时计算性能增量**。全部输出chunk为单token，无插值。初始化约21.13s、应用预热约0.856s、测量及后续序列化/退出分别留痕；全controller约82.03s。请求结束后无pending store/load/ack，drain为0额外调用、8.881微秒。

host末态3374有效块／7,075,790,848B，未耗尽16GiB分配；末态RSS20,036,452,352B、进程HWM23,363,751,936B。父cgroup上限197,568,495,616B、swap0，非独立进程树硬预算；RSS、KV与cgroup charge重叠，不相加，缺失的memory.peak保留未知。

## 新证据改变了什么

首次prepare发生在8.109950s，最后在21.427335s，共6次早于最后到达，不能再将动作空间限定为封闭cohort。资格记录将混合prefill、pending队列和资源条件分开。absence≥30时，201个cooldown观察点具有固定most-victim完整历史容量，另115个已经直接可容纳；这些不是独立样本或立即可执行Oracle，特别是直接可容纳不证明cooldown阻止了native调度。它们保留了启动选择问题，未证明全部等待均可删除。

两次短段见[定向源定位](short_service_source/SOURCE_NOTE.md)：0406加载2464历史位置、重算17位置，产生2个新输出；0748加载3792、重算7，产生1个新输出。随后step903/1161均由native路径抢占，非新的forced plan。目标自身增量为0，free为0，peer分别需要6/5个新块，释放目标156/238块后下一快照free为150/233。**状态已恢复、当步历史可容纳，仍不足以保证首输出之后继续执行。** 这两例支持考察共同资源增长与执行分配；既有host历史可以复用，不能把全部历史计算称为损失，也不能从已付成本直接推出延长保护。

这两例也与原先固定的停顿目标有关：0748先等待3.531065s（本格最长间隔），得到1个输出后又暂停2.154805s；0406得到2个输出后再次暂停2.849399s。前者最长间隔发生在短服务之前，不能把之后的再次抢占称为该最大间隔的原因；这些是完整时间线中的诊断事实，不是改善幅度或可删除成本上界。仅出现两例尚不足以建立方法的平均收益。

重要限制：六个stop均确认最终token为模型EOS50279，但它们在1.37—6.39s结束，全部早于首次prepare。全部22个发生恢复的请求最终仍达到1024上限。因此，本格验证异构输入、流入期间动作和真实EOS允许，却**没有验证恢复途中自然EOS结束**。语义质量未评估，单episode不提供显著性或广泛迁移证据。

## 研究决定

**Strongest baseline：** 当前同预算selected-save staged most_output；旧完整服务current/eager两个简单点继续保留。原生prompt-only基线已测但不是完整decode KV保存；同资源native完整增量保存和完整近邻实现仍缺，不能把本适配器的限制归因给原生能力或已有论文。

**Oracle/headroom：** 只有完整历史资源必要条件与实际路径，无秒级性能Oracle或全局上界。**Claim ceiling：** NATIVE_SERVING进程内单格诊断；generic LLM serving on OLMoE，无MoE专属性。**Failure category：** 本格无执行失败，观察到首输出后peer增长导致的再次抢占；总体效率代价仍未在本域做对照。**Resurrection condition：** 更强同资源保存基线后仍存在可在线判断、完整服务有代价的短恢复，才进入最小执行分配干预；若强基线消除或充分吸收，停止对应机制。

**唯一下一项实验：** 同一自然输入、预算和current调度下，先资格化native完整增量保存，补齐强基线；不继续扫cooldown或增加窗口。研究投入收窄为“恢复目标与peer随后几步的KV增长怎样共同获得资源”，先要求这项缺口在正确强基线后仍存在，才实现调度增量。下一版本须另行接受，不改变本组身份。

主分析[analysis.json](execution_weste_26862/analysis.json)复算命令见[冻结README](README.md)。域解释与资源汇总为[domain_summary.json](execution_weste_26862/domain_summary.json)，由`python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_saved_recovery_gate_r01/summarize_domain.py`生成，拒绝覆盖输出。两个定向分析复用同一原件，不增加实验数。

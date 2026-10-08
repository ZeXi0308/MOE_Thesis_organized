# APC 开启后的固定轮转：暂停改善与完成时间代价

Verdict：`MEASUREMENT_ONLY`，研究问题仍 `OPEN`。**APC兼容most_output确实缩短了最长暂停，但本次两组均推迟平均完成，且吞吐一正一负；没有实现同时保住这三项指标。** 当前固定规则不支持方法GO，不把这一规则的权衡扩写为整个恢复调度问题NO-GO。

Evidence type：`NATIVE_SERVING`，原生vLLM进程内完整请求。RTX5090，OLMoE-1B-7B BF16，vLLM0.26.0；同cohort2的32请求×3072输入/1024输出、50ms到达、token budget1024、实际KV16,089,350,144 bytes /7671可用块。APC全部开启、每格独立引擎，公共warmup后冷reset。native/most/most/native四格全部COMPLETE，128请求/131072输出；原始结果、全部重复和完整代价保留，无GPU失败重跑。

| block | 策略 | wall(s) | 请求/s | 平均完成(s) | 最大ITL(s) | 真重复位置 | 调用数 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | native_apc | 23.883872 | 1.339816 | 21.429087 | 4.760487 | 6677 | 1348 |
| 0 | most_apc | 23.486378 | 1.362492 | 21.983279 | 1.047418 | 36272 | 1159 |
| 1 | most_apc | 23.819228 | 1.343452 | 22.312623 | 1.041838 | 36272 | 1159 |
| 1 | native_apc | 23.739650 | 1.347956 | 21.292560 | 4.672636 | 6677 | 1348 |

most/native两block吞吐 +1.692445% / −0.334092%，平均完成 +2.586171% / +4.790702%；31/32与32/32请求完成更慢。最大ITL下降3.713069/3.630799秒，但两block各7/32请求的自身maxITL变差。全部逐请求变化在 [analysis.json](analysis/analysis.json)。两臂本组输出序列32/32相同，只能用于身份/结果一致性诊断，不能替代任务质量或自然EOS验证。

What was measured：每次most执行9次主动交换与2次自然抢占；18份交换前后资格汇总及9次实际free差额闭合。每次1029个提案从当步before-state和既往真实事件重放吻合；这是已执行策略的因果输入核对，不是未选动作的性能Oracle。真实原生_preempt_request、缓存完整块资格和worker恢复路径均实际执行，策略没有手动touch缓存、清缓存或改token历史。

native两次真正重复执行6677位置、成功自历史缓存复用1008；most两次重复36272位置、复用7200，故相对本组native仍多重复29595位置。新prefill98304、新decode32736不变；总调用1348→1159。混合重算调用还推进其它请求，token位置与调用数不能直接换成GPU时间或节省量。首次victim0017069在native到step1030恢复，在most到step836恢复；主动交换也把其它请求变成victim，其停顿与完成损失全部计入。

两most的实际执行/决策/输出指纹相同，wall仍差0.332850秒。互斥时间账本为wall = scheduler inclusive + engine excluding scheduler + outside engine：两most依次为1.029985+22.023387+0.433007、1.064364+22.250177+0.504686秒。decision分别0.126548/0.135836秒，已含于scheduler；新增资格和观察成本没有扣除。相关重复不能作为总体噪声界、显著性或非劣证据，不删较慢样本。

What was not measured：两most的held_request_steps均为0，未证明资源保护分支在本域必要或有效；离线只能核对资格汇总/实际free差额，不能重放每个物理块的实时refcount。首块互异+冷缓存且live检查无共享限定了本域，shared-system-prompt或partial-hit/CoW仍不支持。没有新holdout、异构长度、第二模型、外部HTTP客户端、自然EOS/任务质量、业务SLO、完整动作Oracle或最近邻系统复现。

Strongest baseline：本组是同资源、同APC-on、同捕获的原生默认缓存基线，不能拿旧APC-off轮转跨组拼接排名。已有APC-off的least/headroom/most强简单对照只作背景；APC-on headroom及LTR/Andes/UniBoost/TokenFlow-style同runtime对照未执行，不能声称完整baseline ladder已覆盖。

Oracle/headroom status：本组未测未选动作的完整未来，现有before-state重放不是Oracle。Claim ceiling：在该单cohort/固定资源域，轮转动作在APC开启后仍能减少秒级最长暂停，但伴随平均完成代价；当前只有可复述的测量权衡。Failure category：目标与victim损失的权衡，非动作无效、缓存失效或实验运行失败。Resurrection condition：同问题的新动作若能在完整受害者损失与执行税后改善该权衡，可以继续；仅改名、换seed或扫cooldown不构成新证据。

One next smallest experiment：共享d6强基线已经显示更大的victim/完成分布差异，下一步复用其计划的同一实际前态 least/most/不交换三分支，沿各分支真实推进到完整请求完成，核对一次选择的受益者与受害者代价；先关闭动作收益排序这一最弱链，不直接再写预测器。该分支尚未执行，d6关闭APC的结果不能代替本域APC-on验证；与A会话协调单一执行者，不重复实现相同controller。

复算命令 [ANALYZE_COMMAND.sh](ANALYZE_COMMAND.sh)。冻结输入SHA256 `eb40e068a70cc185587159dfaf1b13839dc49be7c46f0df0c1896ba46d0adf3d`；整组 [execution.json](execution/execution.json) 与 [group-execution.json](execution/group_readback/group-execution.json) 保留共享flock、各格GPU边界和终态。独立有限审阅在audit目录，未提前宣称通过。既有safe-cap元数据的247仅为cap29余量；分析显式记录cap32整段声明余量7671−8192=−521，不消费该误名字段。

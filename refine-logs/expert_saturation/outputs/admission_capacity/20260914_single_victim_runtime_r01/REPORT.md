# 单次 victim 替换：恢复资源与后续有效输出

当前状态：`STAGED / GPU_UNRUN`。本轮只检验一个已定位的因果问题，不是新的在线策略。

研究问题：固定实际 KV 池和重算后端下，同一次恢复选择多释放少量 KV 的 victim，能否避免后续重算尚未产出新 token 就再次被抢占，并改善完整停顿—服务效率权衡？

来源为 `20260914_funding_filter_comparison_r01` 原六格、7,702 步结构验证和全部 27 个独立分叉。默认 step891 victim0020484 释放211块，候选0017453释放215块；两者目标0020902首输出均894。模型预测后者总重算89761→86902、末步1451→1448、最大输出间隔90→84步，原因候选之一是step1029可用52块足够49块，而默认48块不足。候选后续状态已经独立演进，不能把全部变化归结成孤立4块的效果。这些是模型结果，不是 GPU 毫秒或吞吐预测。

候选按完整27分叉结果后选出，保留探索性质；同一文档集不是 fresh holdout，两个反序block不是独立工作负载重复。先验证结构排序是否能兑现，再做独立数据确认。

## 唯一干预与同资源对照

顺序固定 `most_output / least_feasible / single_victim / single_victim / least_feasible / most_output`。三个角色均运行同一个事件观察器；只有single_victim打开一次替换。输入、到达、2560/3072交替上下文、1024强制输出、cap32、每步1024工作量、实际6656块、OLMoE/vLLM与原native recompute、KV保护结束条件全部沿用。

在预定step891，先执行原selector，按规范化source ID检查当前running/waiting、computed/output/owned、free/need与selector历史是否符合冻结前态；只在匹配且新victim仍满足原guard和funding约束时替换一个victim。之后全部恢复原least_feasible。合同输入只来自原before891；新执行不读取原future。事件身份是人为指定的探索干预，不包装成可部署在线选择器。

若前态不匹配或未触发，正常保留全部请求与执行终态，另标诊断`UNMATCHED`，不筛除、不偷偷换事件或重跑。是否真实强制抢占了新victim，须由原native事件与观察器双向核对；仅记录proposal不算动作生效。

## 计费与结论

主目标保持每请求最大ITL分布与完整episode吞吐的权衡。附报TTFT、平均TPOT、完成时间、受损/改善请求、重算位置、实际恢复等待及观测/调度开销。全部到达、失败、未完成请求都保留；混合重算调用不等于纯浪费，计费沿用原互斥工作桶与完整墙钟分母。参考SLO全部通过时goodput等于吞吐，不算另一个正信号。

先核对同block filtered/event 在动作前的身份、结构前缀和已经返回的token前缀，再比较独立未来。未比较GPU浮点KV张量逐字节一致性。最有信息量的结果是：替换合法实际生效，但模型关于1029重抢占/首新输出的排序是否失配；若结构兑现而完整性能不改善，则归为计算/调度成本或损失转移，不再用步数推时延。

most_output仍是同组强简单参照。当前完整Oracle不存在；27候选只是单事件结构分叉，不覆盖其它动作、墙钟成本或未知EOS。未测自由EOS、突发、第二模型、持续到达泛化及公开近邻完整系统。首次输出保护、老化、时间片和成本比较本身不主张原创。

## 接续与资源

已有授权host为westc:53036，目标GPU记录为bd5e9bb9。1789398884.010797现场Qwen6954/6967均存活，共享锁busy；本轮没有GPU driver。按共享协调顺序等待B整组、已登记A repeated-staged完整释放，不能利用模型加载的空闲间隙。CPU包完成后可暂存；启动前仍核验实际UUID、源码、进程与整组锁。

实际接续：原生factory入口、事件替换/关闭、重复调用、错误前态、不可资助victim和ID歧义的CPU检查已通过，27项selector测试及既有native allocator fixtures也通过。`CPU_CHECKS.json`绑定已测源码和`event_contract.json`；合同覆盖29个有序running、3个waiting、32请求完整计数/owned块及selector历史。snapshot的`APPLIED/applied_count`只表示proposal替换，必须再由真实native抢占回执确认。

新包SHA256 `21d162ffc855ea74ac17c912ae517505b3e54a18786de02cac6a76a8e9eab9ce`，已暂存到`/root/autodl-tmp/moe-single-victim-runtime-20260914-r01`。`execution/staging_verification.json`确认22文件全部匹配、无launch-once/results、GPU仍为Qwen worker6967且锁不可取。旧funding包未改；本包base selector相对旧版仅说明注释变化，Python AST逐字相同，新native适配器只增加可选tracker factory。暂存不等于运行，没有GPU性能结果。

唯一下一实验就是这个同资源单事件六格。正结果仅支持该事件可达分支与当前运行域的权衡；后续必须形成可见状态下的选择规则并在新独立数据上确认。若简单most已覆盖净收益，则保留其强基线地位，不包装新方法。

分析入口已完成：`analyzer_cpu_checks.json`记录四条旧most/filtered实际轨迹的observer-off重放等价、动作前token前缀比较与错误目标进度拒绝。它复用原请求指标和成本/内存检查，只增加事件与真实native动作核对；没有生成新arm结果。独立调用写出的`analysis_r01.json`为六格全部UNRUN、0比较，并保留完整9对比较合同。以上是CPU资格，不是新实验审计或性能接受。当前阻塞是前序GPU整组尚未释放，下一仍为原暂存六格，不追加调参或重复审计。


资源阻塞接续：连续三轮结束时的现场观察为1789400726.2165527、1789400808.6543057、1789400951.291732，原Qwen6954/6967仍存活，后两次命令身份一致且共同锁busy。本组六格原目录无launch/results。当前无能替代真实对照的必要CPU工作，目标标记`BLOCKED_RESOURCE_BUSY`停止自动空转；科学问题仍OPEN，实验仍STAGED/GPU_UNRUN。原授权、原包和B→A→本组顺序保留；待前序整组终态后按COMMANDS重新现场核验，接原目录首次执行。详见execution/resource_wait.json。

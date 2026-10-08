# 请求计时与调度诊断分离：CPU入口已验证

状态：**CPU_PREPARED / GPU_UNRUN**。HEAD仍为de64dae5，共享dirty修改保留，无提交或新GPU启动。本次新增问题是：已有完整请求性能路径是否仍逐步扫描请求与KV，使小幅结构变化的性能判定受诊断成本影响？没有声称插桩已经造成当前时间漂移。

现有staged-store包的`native_capture.capture_episode`始终包装`scheduler.schedule`并记录逐请求状态；`capture_with_memory`另在每步前后扫描KV。旧logical-alignment的performance模式仅关闭CPU诊断与runtime observer，保留其他日志，而且属于另一运行域。因此没有可直接复用的、已完成GPU资格的无调度诊断入口。

## 实际改动与验证

新增`experiments/admission_capacity/request_measurement.py::measure_episode(engine, workload, config, regime, arrival_scale, run_id, max_seconds=120)`，135行。保持外部策略hook及其安全检查、日志与执行成本，不安装或移除scheduler、memory或worker observer。调用者须提供尚未安装诊断wrapper的引擎；`DIAGNOSTICS_DISABLED`描述本入口，不会自动清理其他代码安装的插桩。现有A重复保存冻结包和runner没有被修改。

保留全部计划请求、实际提交、引擎返回的增量token、完成/失败及总调用数。输出时间取同步`engine.step`返回后的第一个host时间点，不是客户端网络接收。无新token的执行和重复累计输出不新增token时间。engine调用抛错仍计入attempt count，成功返回单列；未测的抢占、重算、恢复数为null，不伪造零成本。`admission_s-arrival_s`仅为提交延迟，不冒称原生队列等待。

要求async=False、stream_interval=1、n=1、温度0、累计输出、关闭detokenize。默认保留原固定长度协议；显式`ignore_eos=False,min_tokens=0`允许实际stop结束，不读取未来EOS。返回数据可交给现有`summarize_episode_requests`，异常前未到达的计划请求保留并从实际到达分母区分。时间上限是调用间检查，不是强制中止正在执行的GPU调用。达到上下文上限而早于所配输出数量的length终止仍会判不完整，调用者须像当前已测负载一样满足声明prompt+output不超模型上下文；此入口尚不是任意流量的通用服务驱动。

独立worktree运行：

```bash
cd /private/tmp/moe-window-measurement-20260914/refine-logs/expert_saturation/experiments/admission_capacity
python3 -m unittest -v test_request_measurement
```

三项CPU行为测试通过：

1. 空返回/内部工作、重复累计token不重置输出等待；人为延迟host读取output字段也不后移已取得的返回时间。
2. worker异常和runtime limit均保留部分输出、已到达未完成请求与未来请求；完整服务goodput不会把失败请求算成功。
3. 自然EOS可结束，外部策略hook身份保持且其耗时进入完成时间；未观测诊断字段是null，未产生伪造memory/scheduler空轨迹。

测试文件首次执行因缺少括号报SyntaxError，修正后3/3通过；最终仅补充计时/诊断语义后再跑3/3通过。测试使用fake engine，不是native API或GPU资格。主树与独立worktree的metrics.py逐字节相同，集成前确认新文件没有不同内容，未覆盖并行修改。

文件SHA256：request_measurement `e7ea0eab9184a7fd74ae132cb913ac7e573ba0fc4c55f91b54399c709b766baa`；test `980d3e78fd0d9fefd839394d9b63a7a837bd74fd9bfcfd76eaaa8684f017da97`。源代码和测试已同步主工作区及root独立worktree。

## 对当前研究结论的影响

最强完整简单策略仍为同组most/least延迟—效率包络。funding过滤修正了18次同一episode拒绝，却新增了每轮一次零输出和一次两输出再抢占；不能用这两个反例声称most也有同样残差。before1026证书表明全部27peer继续执行时第四批需243块而仅242块，固定22peer前缀可在242块内得到首输出，但让另外5peer各少4次输出机会。模型的资源可行性已经有证据，净收益与在线动作排序仍未验证。

重复保存模型即时most为1315调用，假设加载1/2/4步的save-on为1308/1310/1312；加载8步为1316。模型没有物理保存和同步墙钟税，也未实测host峰值。减少3–7次调用是结构敏感性，不能替代完整成本收益；每次调用更便宜仍是待测路径。

下一GPU组保持A原方重复staged-store off/on执行资格对照，两臂相同16GiB native缓存与实际GPU KV预算；不替换已暂存21文件冻结包。其诊断先回答重复保存/加载/miss fallback是否实际发生并兑现有效输出，n=1不判性能GO。资格成立后，若推进时间比较，本入口可用于单独的同配置性能测量，并保留即时most强基线及全部失败；不得把旧插桩时间直接与新轻量时间配对。实际host峰值与引擎原生接口资格尚待补齐。没有恢复后短服务残差或净服务增量时，停止扩大保存/窗口机制，不继续扫阈值。

23:17只读现场确认Qwen r04 monitor6954和worker6967仍存活、命令指向原r04目录；进程存活不证明数值或性能资格已通过。B持有完整组队位，A已暂存下一组，root没有driver或后台候卡器。

本轮结论：**测量入口准备完成，窗口方法仍OPEN**。它为完整成本判断提供可用代码，没有新增性能或论文主张，也没有把当前局部可行证书升级为全局Oracle。

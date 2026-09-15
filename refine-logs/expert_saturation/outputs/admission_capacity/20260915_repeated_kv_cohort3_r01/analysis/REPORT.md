# cohort3：保存强基线的文档迁移验证

Verdict：在相同固定长度/压力域的新文档集合上，两对交错执行继续支持save-on作为强基线。证据为NATIVE_SERVING同步进程内请求测量；不是新调度方法、统计显著性、质量保证或跨运行域结论。限定完整性复核待返回。

本线职责为资源状态模型与本已登记的迁移验证；保存语义、原四格、统一启动时机主实验由其原负责人负责，本报告不重复扩展这些任务。

四格controller10974，1789410020.6213577—1789410240.8598757，exit0，128/128请求完成。次序block0-off/on/block1-on/off固定，所有输出长度均1024。两对均32/32请求完成更早，但每对只有26/32完整输出文本相同；质量和逐层KV一致性未测。真实batch/token轨迹各自演进，没有共享未来trace。

| 对 | 指标 | off | on | on相对off |
|---|---|---:|---:|---:|
| 0 | 平均完成 / s | 22.704639 | 21.222815 | -6.53% |
| 0 | 输出吞吐 / token/s | 1328.383170 | 1416.483989 | +6.63% |
| 0 | 最大输出间隔 / s | 2.796451 | 2.243055 | -19.79% |
| 0 | 平均TTFT / s | 0.351896 | 0.352166 | +0.08% |
| 1 | 平均完成 / s | 22.553030 | 21.338719 | -5.38% |
| 1 | 输出吞吐 / token/s | 1336.943138 | 1409.362372 | +5.42% |
| 1 | 最大输出间隔 / s | 2.772463 | 2.269051 | -18.16% |
| 1 | 平均TTFT / s | 0.351954 | 0.356542 | +1.30% |

平均TTFT变化为+0.270/+4.587ms；仅报告观测差，不扣除、不直接归因于保存。该工作负载首输出发生在轮转保存之前；本轻量组不补详细调用/重算/传输诊断来包装完整因果分解。

资源和执行：同GPU0 UUID4015…d0e5、6656 usable GPU KV块、配置13,960,740,864B，实际host唯一KV storage四格均17,179,869,184B。adapter四格均DRAINED、各38次主动轮转；on末态5961有效host块，off为0，全部scheduler store/load/worker acknowledgements为0。原运行时及策略20文件逐字复用。抢占/重算量在轻量raw为unknown，不填0，不借旧诊断代替。

输入：复用此前已固定的cohort3，32篇文档ID、内容SHA与prompt-token SHA均与原保存组无交集；到达、3072/1024长度、模型revision相同。该集合曾用于其它调度比较，因此只称固定机制的文档迁移验证，不称从未接触的语料或独立总体。未按保存结果选择文档；没有此次参数校准。

排除的解释：收益不只局限于原32篇文档，也不是仅增加on的KV分配上限；不能排除一般运行漂移或外推自然EOS、异构长度、持续到达/第二压力点。最强同底座对照是staged most_output save-off；未新增原生全量保存/其它prior-art比较。没有新的Oracle或全局上界。

改变的决策：保存强基线获得第二文档集合支持，本线停止继续同域文档或seed扫描。统一主会话下一动作为save-on下cooldown20/0的启动代价对照；本线只提供状态模型接口，不另造控制器、不自行增加GPU任务。之后应由主会话选少量异构/EOS/持续到达边界。

原件：../execution_weste_26862/readback，唯一archive SHA ddb371dd691b1e8d0a2b61a516fc24991b937c22485c8dd3051dbeec44c49722；执行包b91181ec…9255、23项逐一一致。GPU计算进程清空且共同锁可取后已明确交接，未结束他人进程。

复算（仓库根目录，写新的输出路径）：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_repeated_kv_cohort3_r01/analysis/analyze.py --results refine-logs/expert_saturation/outputs/admission_capacity/20260915_repeated_kv_cohort3_r01/execution_weste_26862/readback/results --output /private/tmp/cohort3-new-analysis.json
```

分析入口复用service组；未设计的diag-off/on显示UNRUN，不进入四个performance comparisons。不得将不存在的diagnostic结果与其它cohort拼接。结果与计算源码同目录交付。


复核后补充：限定审阅已返回WARN（same-family/provisional），无P0/P1；原文“待返回”保留为历史状态。见 [复核](../EXPERIMENT_AUDIT.md) 与 [元数据说明](../ADDENDUM.md)。

# A：prepare之后仍需驱逐原victim吗？

**发现一个可执行的决策差异，而非只提高模型拟合：** 在D/E全部25＋25个实际commit_check上，固定当前态规则各改变1次动作。到commit时，target已经可由空闲块资助；原适配器仍按准备好的计划额外驱逐victim。候选为仅在现有全部READY资格通过后，若free≥target.remaining_blocks，保留victim、沿既有目标队列提升/恢复保护执行；否则完全沿用原swap。

| 既有诊断 | commit | free / target需 | 原victim持有 | 保留全部peer下一decode后target余量 |
|---|---:|---:|---:|---:|
| selected D | 859 | 193 / 95 | 233块 | 97块 |
| native full E | 1399 | 277 / 106 | 184块 | 171块 |

两例分别26/22个running，全部pure decode，没有skipped load；按当前块边界全部peer的一步增长1/0块，容量仍足够target完整历史。没有用真实未来EOS、实际完成顺序、request身份特判或未来host命中。D的额外资金来自prepare期间另一个peer被native抢占，E来自一个当时已1023/1024输出peer结束；规则无需预测这两种原因，只使用commit现在的实际free。

既有prepare_margin分析仅发现变化来源；本轮新增判断是**这会使计划中的victim驱逐变为非容量必需动作**。这不是声称原策略错误或跳过驱逐必有净收益：保留victim提高后续并发/KV增长，可能把抢占推给别人，必须看完整服务。当前准备阶段已经支付的保存工作不能事后扣掉。

## 已完成与未完成

已完成14行纯CPU决策组件commit_disposition.py，以及全部原commit前态检查（每组25，全部保留，零遗漏）。任何非READY原因仍返回取消；不改变原身份、ownership、纯decode、skipped、保存资格检查。实际GPU动作未执行，未改变任何共享adapter或已接受四格包。

现有native adapter即使free已足够，READY分支仍无条件_preempt_request(victim)。集成时必须新增独立direct_resume阶段，而非只注释这一行：建立target保护与output_start、同样提升target等待顺序；仅swap分支增加applied_rotations/forced通知/等待victim flush；direct_resume不可伪造preempt或rotation计数。沿用既有recovery_guard和native metadata提交，不释放victim源块、不手动取消旧store。schedule末尾两种提交阶段均清plan。保持prepare时已消耗的cooldown，先避免捆绑第二个机制。

这是基线完善候选，不主张新颖调度算法。当前最强比较必须是相同保存scope、同其他策略、仅开关commit重检；不能拿old selected的总收益与full拼排名。

## 唯一建议实验

先完成已接受selected/full轻量四格，选定主方采用的保存底座。随后只做上述default-off开关的最小因果资格：从同一明确commit前状态分叉，或同输入真实独立演进，确认direct恢复成功、没有victim preempt、guard未被绕过、输出/peer进展及后续抢占全保留。若资格成立再做完整请求对照，比较生成停顿、平均完成、吞吐和新增peer代价；若直接容量分支不再出现就保留no-action，不加压找动作。

本线暂不打包或启动GPU；不更改C执行方在途身份。问题是“重新检查当前资金能否避免不必要的额外驱逐而改善服务取舍”，不是预测EOS或永久保护目标。

复算（仓库根目录，输出新路径）：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/commit_recheck/check_commit_disposition.py --selective refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/execution_weste_26862/readback/results/diagnostic-native-full/selective-store.json --output /private/tmp/commit-recheck-full-new.json
```

D换原selected组diagnostic-current/selective-store.json。数据为selected.json/full.json，代码同目录，输出拒绝覆盖。证据层级STRUCTURAL/current-state candidate；无新GPU、无方法GO、无性能或quality收益。两episode多次分析不增加样本数。

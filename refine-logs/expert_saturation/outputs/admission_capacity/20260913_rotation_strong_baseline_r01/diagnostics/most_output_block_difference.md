两次 most_output 的 wall 差主要定位到同一执行路径上的 step 839；内部停顿原因仍未验证。

问题：固定输入与配置下，为什么 block 0 为 24.313487 s / max-ITL 1.519854 s，block 1 为 23.342345 s / 1.020476 s？证据层级为 NATIVE_SERVING_INPROCESS_HOST_CAPTURE，结论保持 MEASUREMENT_ONLY。

- 1,162 步的实际 scheduled work、规范化完整决策、engine 逻辑输出均相同，32/32 最终输出相同；11 条恢复路径也相同。两次均为 9 次 forced + 2 次 natural，98,304 新 prefill、43,471 recompute、32,736 新 decode，held 为 0。
- 首个状态差异是 step 60 的 waiting before/after：8/7 对 7/6。request 0016451 的计划 arrival 为 1.35 s，而该步开始于 1.355976 / 1.349902 s；实际提交为 1.355611 / 1.374383 s。只有步骤 60、62、64、66、68 的 waiting 计数不同，没有改变执行选择。

| 成本，秒 | block 0 | block 1 | block 0 − block 1 |
|---|---:|---:|---:|
| 完整 wall | 24.313487 | 23.342345 | 0.971142 |
| scheduler inclusive | 0.880433 | 0.872652 | 0.007781 |
| engine excluding scheduler | 22.992491 | 22.053836 | 0.938654 |
| outside engine calls | 0.440563 | 0.415856 | 0.024707 |
| decision（已含于 scheduler） | 0.082142 | 0.090376 | −0.008234 |

engine excluding scheduler 的差由互斥桶组成：new prefill +0.006202 s，recompute-bearing +0.752460 s，pure new decode +0.179992 s。重算桶内仍包含同时进行的有效 decode 与主机开销，不是纯重算或 GPU 时间。

step 839 是首个 forced recovery 的末次调用：width 31、798 recompute +31 decode，总量 829，两次逻辑内容相同。该调用为 0.766318 / 0.031701 s，差 0.734617 s；scheduler 为 0.000946 / 0.000681 s，差仅 0.000265 s。其余可见停顿之一是 step 1140（width 3 的纯 decode），0.117480 / 0.007363 s，差 0.110118 s。

block 0 最长间隔属于 request 0017069：step 806 被自然抢占，836 开始恢复，839 首次返回新 token。1.519854 s = boundary 0.000524 + wait 0.662780 + recovery span 0.856550。同一事件在 block 1 为 0.781381 s = 0.000496 +0.659717 +0.121168。两者重算位置都是 3,780；step 839 的时间差进入了这条请求的 max-ITL。block 1 的全局最大值转为 request 0015839、step 996→1036→1040 的 1.020476 s；这是最大值对应事件改变，没有新的 action 路径。

真正的单请求 completion tail 是最后 11 步 1151–1161，last request 均为 0017069：0.051914 / 0.052276 s。它没有解释 wall 增量。width=1 的全程汇总有 14 次调用，其中还含开头 3 次 prefill，不能全部称为尾部；对应 engine 为 0.118321 / 0.118065 s。

日志核对：block 0 的 JIT warning 为 09-13 23:36:27，measured 为 23:36:28.671393–23:36:52.984880；block 1 的 warning 为 09-14 00:02:41，measured 为 00:02:42.232871–00:03:05.575215（+08:00）。显式 warning 都在测量前。step 839 的 block 0 wall 区间是 23:36:46.202789–23:36:46.969106，没有对应的测量期 JIT/compile/tactic 通知。日志未显示冷 action shape 的直接证据，也不能凭日志沉默排除未报告或被抑制的编译事件。当前只能定位 engine 内、scheduler 外的停顿，无法区分首次形状成本与其他 runtime 停顿。

唯一下一步：One additional controlled four-engine group with the same frozen cohort2/runtime/common warmups: native -> most_output, then most_output -> native. Start only after the queued F/X and A-review work and keep the group uninterrupted. Preserve caches, every slow call, all outputs and canonical metrics. Align the candidate stall by the final call of the first successful forced recovery; use step 839 only when the executed trajectory matches. Observe whether that call recurs and whether the native-relative throughput direction repeats. This repeat is not guaranteed to identify the internal stall cause; direct compile/cache or CPU/GPU phase evidence would still be needed to name JIT.

两次完整结果均保留；不删除或扣除慢调用，不替换 canonical，不推导纯 GPU tax 或未来动作反事实。依据为已有 analysis.json 的 cells / recovery_accounting / same_role_repeats，再核对两个 most_output 的 raw、decisions、当前 warmups 与 stdout/stderr。完整时间锚和字段比较见同名 JSON。

可复用 CPU 诊断脚本：`most_output_block_difference.py`（108 行），只读输入、默认输出 JSON 到 stdout；`--output` 使用独占创建，拒绝覆盖。运行命令的 argv、工作目录、环境与全部直接输入 SHA256 已记入 JSON。脚本已对本次两个 label 执行，复现逐步动作/输出一致与 step 839 最大差值。新运行用 `--run-dir` 和 `--labels` 指定，按首个 successful forced recovery 对齐；未匹配的逐步路径不生成同 step 的差值排行。

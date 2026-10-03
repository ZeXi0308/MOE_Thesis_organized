# H1：原生直接恢复动作已发生，收益尚待对照

## 本轮回答

固定已见 H128 输入上的 `commit_recheck=on` 确实改变了两个合法原生动作：提交时保留计划 victim，直接接纳原先 PREEMPTED 的目标进入异步加载。对应加载完成后均有新输出，目标最终完成。269 次检查中只有 2 次 direct，其余 267 次因空闲 KV 块不足沿用原 victim 路径；这一比率只描述本条诊断轨迹。

这是 `NATIVE_SERVING / OBSERVED_DIRECT_ACTION_CHAIN` 资格结果。尚未测量 off/on 净收益，不能说减少了 peer 停顿、节省传输、目标恢复更快或形成新算法。

## 实际证据

| 原生 step | 目标文章 | 计划 victim 文章 | 调度前空闲/目标完整历史块 | running 数 | load job | 加载完成→首次后续输出 |
|---|---|---|---:|---:|---:|---:|
| 1269 | 0010411 | 0011296 | 225 / 159 | 22 | 234 | 21.313 ms |
| 4862 | 0019990 | 0020233 | 213 / 190 | 21 | 658 | 18.668 ms |

两次原生接纳均为 `ASYNC_LOAD_ADMITTED`，当步调度 token 为 0；因此接纳、加载完成和新输出分别验证。两个 target 的 native reservation 都为 0，计划 victim 未出现在当步抢占列表。两目标最终均以 `length` 结束，不能据此声称被恢复请求经历自然 EOS。时间是同一诊断中的 host 观测时间，不能与轻量性能格排名，也不是 KV 传输的纯设备时延。

### 空闲块来自哪里

[相邻原生调度与资源快照复算](A_H1_OBSERVED_RESOURCE_TRANSITIONS_20261001.json)进一步定位：两次 prepare 前空闲均为 0。step1268 的原生调度抢占文章0008173（原持229块），其余 running 合计增长4块，下一次提交前得到225空闲块；step4861 同样抢占文章0018480（原持217块），其余增长4块，得到213空闲块。原生抢占列表与退出 running 的请求一致，held/free 差分闭合。旧资源分析的 `preceding_forced_commit_events` 使用了不存在的 `commit` 事件名，该字段不作为证据；后续按实际 `commit_recheck` 分支复算。

[全部269次提交复算](A_H1_READY_COMMIT_TRANSITIONS_R01_20261001.json)发现：只有上述2次在prepare步骤发生了其它请求的原生抢占，目标资源余量从−159/−190变为+66/+23块；其它267次提交仍缺1–235块（中位58），均继续抢占计划victim。当前轨迹没有更多被H1漏掉的完整可容纳机会。

因此 H1 保留计划 victim 的机会出现在**前一步已经原生抢占了另一个请求之后**。不能将两次 direct 描述为无抢占恢复，也不能把原生腾出的资源记成 H1 的额外容量。两个计划 victim 的 prepare 已分别生成178、6个新 store block（job233、657），这些保存成本已经发生；后续不驱逐不等于撤销这些成本。净 peer 代价仍须用完整配对轨迹核验。

全 cohort 128/128 完成，121 length、7 stop；267 次强制轮转，0 个失败/未完成。控制器耗时 346.286 s，cell exit 0，归档 VERIFIED，退出 GPU compute 列表为空。沿用单 RTX5090、固定 OLMoE revision、vLLM0.26、4096 usable GPU KV 块、16 GiB host KV 和 90 GiB cgroup。该 cgroup 限额不替代 host 过程峰值或独立 tensor 相等性证据。

## 可重建来源

- 会话：`moe-a-h1-guard-qual-session-r02-20260930/`；原件不修改。
- [严格审计](A_H1_GUARD_QUALIFICATION_AUDIT_R02_20261001.json)：SHA `7a3b323673ac7b42d666f758de2109ad21ed625df9957bed8cfd149a75437f08`。
- [审计器](audit_h1_guard_qualification_r02.py)：SHA `4dee6402dda5cf9d585fa5dbaade823857b08d16e2181bea87c6467f0ae9d3ad`；6 项定向 CPU 检查通过。
- 计划 SHA `48c6fe98eddfca76faec95676184b3ca24a9cf7ce81a24567a825eaddf509a81`；25 文件包 manifest SHA `4231a687db066f113fcc14676f91e8e5825be05b76ef0834112e053726aa13aa`。
- 原始 raw 为 1,236,406,449 B，SHA `b67958938547c4ff3515b49ccbed8f6777a8cf90aecc1fcdcbbf37d899b7c2b0`；27 个归档文件全部按原字节核验。完整取回压缩包 SHA `17a46e3448437bb9c352322c2bec788afde110e63ee82cc68da250a3e334446d`。
- 本地使用 ijson 3.5.1 流式读取，保留 128 requests、5981 scheduler steps、5981 engine steps。5981 memory trace 和124478 output events 仅计数，未用计数代替其中内容的检查；动作链依赖完整保留的请求/调度记录及原生 offload 回执。未展开的数组仍参与完整 raw 哈希。

## 改变的判断与下一实验

H1 已从条件 CPU 动作推进到实际原生动作；当前最弱环节变成完整请求效用。防御性的 reservation 门禁两次都观测为 0，没有非零拒绝，不能包装成已测故障修复收益。H1 本身仍是简单的“提交时足够资源则直接恢复”规则，独立创新性未成立。

[已冻结性能协议](H1_PERFORMANCE_PAIRED_PROTOCOL_R01_20261001.json)规定同一输入、同一后端和 adapter，两个相邻配对块依次 off/on、on/off，各独立使用公共锁。每对都报告全128请求、实际输出、TTFT、mean flow、请求等权 gap 分布、逐请求得失和固定 goodput 前沿。沿用 on/off 输出率≥97%、mean flow≤105%，并要求最大gap更低；这是开发判据。两块均保留，不因第一块有利才补第二块，不追加参数或 seed 抢救。

已见 H128 不能作盲确认。两个独立策略轨迹不提供同前态逐动作反事实；自然生成输出变化也不允许等工作量加速或质量不变声明。正式性能结果另见[配对报告](A_H1_PERFORMANCE_PAIRED_RESULT_R01_20261001.md)。若负结果覆盖当前简单机制，仅关闭此代码与运行域，不否定整个恢复时机问题。

历史启动断点（现已恢复并完成两对性能运行）：两份固定性能包和控制器已完成CPU检查（controller4项、analyzer5项）；待上传包3,423,599B，SHA `7be9cd85d342da548cad37cd314ee8f9d65cafd76b79a0375b4153847a34a24d`。新GPU格为0。最新TCP握手到 `36.103.198.204:45495` 返回 `Connection refused`，当前机器/锁状态未知；[恢复清单](A_H1_PERFORMANCE_LAUNCH_CHECKPOINT_R01_20261001.json)保留准确版本和下一命令。

# 首次驱逐与持续改序：六项原生诊断

2026-09-13。**BLOCKED_APPROVAL / GPU_UNRUN**；六项包已准备，未上传、GPU执行0项。自动审批未接受新六项范围，拒绝原文和具体载荷见[UPLOAD_REVIEW_RECORD.json](UPLOAD_REVIEW_RECORD.json)。现有GPU最近一次只读检查为空闲，当前阻塞是审批而非GPU。

唯一问题：仅首个实际成功强制交换选择已输出最多者，之后回到最少进度，是否足以压缩相同收尾请求的窄批尾段，并减少持续改序带来的早完成请求延迟？这是一项同问题下的因果诊断，不是新方法GO。

三臂least_progress / most_output / first_most_then_least，复用已见cohort0的32篇文本；block0 A,C,B，block1 B,C,A，共6格192次正式请求。所有输入、固定KV、预热、30/20/30和保护/资金规则与上一轮一致，各臂实际状态独立演进。

原选择器25项CPU检查与4项原生方法/分配夹具通过；首选模式只由此前已成功forced交换数推进。分析器12项CPU检查通过，逐step校验模式与此前实际强制交换数，覆盖UNRUN、部分6、错配置/切换拒绝和旧A/B raw适配；见[检查记录](preparation/analyzer_cpu_checks.json)。这些不含新C的GPU动作或性能。完整协议见[DECISIONS.md](preparation/source/DECISIONS.md)，顺序见[campaign.json](preparation/source/campaign.json)。

包SHA256：`fd1342a6aea8f2cc886a0a6957d4059dcc2d3ffad1797b0428fb23d8be162b4c`。现有端点connect.weste.seetacloud.com:23478；计划远端新目录`/root/autodl-tmp/moe-rotation-first-swap-20260913-r01`。每格初始化/正式测量前检查占用；遇失败保留退出，不杀其它任务或自动重跑。全部六格合格后才比较完整吞吐、max-ITL、平均完成、逐请求代价及真实工作/恢复账本；保持无显著性/非劣/质量/方法GO边界。

| 固定报告项 | 当前状态 |
|---|---|
| Verdict | BLOCKED_APPROVAL / GPU_UNRUN；主问题OPEN |
| Evidence type | CPU选择与原生方法夹具，GPU尚未运行 |
| What was measured | 首次机会只在实际成功交换后消耗的状态推进 |
| What was not measured | 新C完整请求性能、收尾/受损请求/重算变化 |
| Strongest baseline | 同底座A少完成代价、B高完整吞吐；本次都重新执行 |
| Oracle/headroom status | 无全动作Oracle或暂停硬界 |
| Claim ceiling | 一组已见文本上的探索性因果消融 |
| Failure category | 自动审批未接受新六项范围；非科学失败 |
| Resurrection condition | 原问题未判死；本轮无事后阈值搜索 |
| One next smallest experiment | 本六项具体范围获审批接受后，执行冻结顺序并回传；不重选阈值/文本 |

复算入口：`analyze_rotation_first_swap.py --run-dir <本bundle/execution> --output-dir <新目录>`。六格全部合格前不形成数值比较。原始八项及其未完成审阅状态保留；本轮未以新代码回写旧raw或旧结论。

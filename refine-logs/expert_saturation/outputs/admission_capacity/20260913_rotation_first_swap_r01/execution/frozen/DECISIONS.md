# 首次驱逐选择与持续改序的因果诊断

PREPARED_UNRUN；六项顺序与解释规则在新GPU结果前冻结。
问题：一次不同的驱逐选择是否已足以压缩收尾，同时避免持续改序让早完成请求反复等待？
A least_progress：每次合格交换选择计算进度比例最小者。
B most_output：每次选择当前原生输出token数最多者。
C first_most_then_least：仅第一次实际成功强制交换使用B，之后使用A。
“首次”由已成功执行的强制交换数决定，不由绝对step、未来轨迹或事后指定request决定。
noop、资金不足的提案、自然抢占、原生恢复均不消耗首次机会。运行失败保留，不靠重启重置策略。

## 不变量与范围

OLMoE BF16 / vLLM0.26 / 单RTX5090；实际KV16,089,350,144 bytes / 7671 usable blocks。
32请求、3072输入/1024输出、50ms到达、cap32、token budget1024。
最长缺席恢复、等待30/交换冷却20/恢复驻留30、近完成保护0.90、最多8次缺席、完整历史资金和首新输出前保护不变。
每格独立引擎、相同预热、策略独立演进请求/KV/batch/输出。计数来自request.num_output_tokens，不含重算，不等于消费或质量。
仅复用既有cohort0的32篇已见文本；一cohort两顺序block六执行，不是六个独立workload或新holdout。
旧八项仅提供假说，不能替代本轮A/B/C完整请求比较；本轮不重跑native/headroom/safe29，不冒称完整公平或prior-art基线。

## 六项固定顺序

block0 A,C,B；block1 B,C,A。label及具体参数以campaign.json为准。
全六格完成且输入、资源、请求、计时与模式转换均合格后，逐block比较A→C、B→C、A→B，再保留三对同角色跨block差值。
记录每step的effective_victim_order与此前实际强制交换数；C首次成功交换为B，随后全为A，未应用提案不能推进计数。
若C无实际交换，记INVALID_NO_ACTION；若模式转换不符，记INVALID_EVIDENCE，不拿不生效实验判机制失败。

## 测量与判断

主目标仍为max-ITL与完整服务量；同时报告TTFT、平均完成、每请求损益、所有完成/失败、输出差异。
核对原末两请求的实际服务进度、完成次序、width1/2/4收尾、实际重算及互斥host成本。
wall = scheduler-inclusive + engine-non-schedule + outside-engine；decision是scheduler子集。重算调用也可包含其它请求的新decode。
若C保持B类收尾变化且减少对早完成请求代价，支持首次选择贡献；若无此结构或净代价仍在，首次选择不足，按实际后续链解释。
结构变化不等于显著性能收益；近零/变号保持未确认，不设事后GO门槛，不把两次运行差当噪声界，不从策略差中减A/A。
此轮不追加阈值/次数搜索；不宣称暂停硬界、质量、native非劣、公平保证、新颖性或方法GO。

## 执行与保留

现有connect.weste.seetacloud.com:23478；初始化与测量前检查GPU占用，失败即保留ABORT，不杀其它任务，不自动重跑。
每格归档回传核验后才推进下一格；所有结果保留，原始结果不覆盖，修正另写addendum。

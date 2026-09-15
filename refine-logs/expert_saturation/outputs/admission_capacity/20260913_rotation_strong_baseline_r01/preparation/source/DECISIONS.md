# 持续驱逐排序：新文档上的强基线对照

PREPARED_UNRUN；输入、比较、八项顺序在本轮GPU数据前冻结。
问题：同实际KV预算，持续most_output是否相对native与fast completion_headroom改善最长输出暂停—完整服务量权衡？
沿用已测持续排序，不新增controller；首次交换方案已显示局部width2减少可被width1增加抵消。

## 固定资源、输入和动作

OLMoE BF16 / vLLM0.26 / 单RTX5090；实际KV16,089,350,144 bytes / 7671 usable blocks。
32请求、3072输入/1024输出、50ms steady到达、cap32、token budget1024。
新增cohort2取固定WikiText shard中原96文档之后的下一32篇足长完整源文章；执行其3072-token前缀。
原96文档逐字及token重现，并排除文档hash、输入前缀hash和源行区间重叠。凭据见fresh_inputs_receipt.json。
这是同语料、同长度/到达域的新文档检验，不是独立总体或新运行域。
四角色：native；native_aa（完全相同策略）；fast headroom；持续most_output轮转。
轮转保持等待30/交换冷却20/恢复驻留30/近完成保护0.90/最多8次缺席，以及完整历史资金、首新输出保护和最长缺席恢复。
most_output只按当时原生已生成输出数选择合格受害者；非未来信息，不计重算，不代表客户端消费。
每格独立引擎、共同预热、策略独立演进请求/KV/batch/输出；不共享未来轨迹。

## 八项顺序与分析

block0：native, headroom, most_output, native_aa。
block1：native_aa, most_output, headroom, native。
全八格完成、同输入/资源/计时合格、headroom与most实际动作合格后，逐block比较native→headroom、native→most、headroom→most。
两对native→native_aa、四对同角色跨block作为运行漂移保留。native始终为预先指定主基线，不以A/A替代、不减A/A、不把两重复差写成噪声界。
不能把8引擎或256次请求执行当8独立workload；同一32文档的请求差值也不是独立策略重复。

## 目标、成本与停止规则

主目标仍为最大ITL与完整吞吐，同时报告TTFT、平均完成、每请求损益、失败/完成、重算、恢复及输出变化。
wall = scheduler-inclusive + engine-non-schedule + outside-engine；decision是scheduler子集。
含重算的调用可能同时推进其它请求的新decode，不把整段算为纯GPU重算税。收尾宽度变化是观察到的成本，不能跨策略直接相加为因果saving。
若持续排序相对headroom仍有一致收益，保留其强简单策略资格；若有代价或变号，按完整成本报告，不改变主指标或删格。
若无实际干预，标INVALID_NO_ACTION并区分运行域与实现失败；若输入/资源/账本不合格，标INVALID_EVIDENCE。
本轮不追加参数/文档搜索；不以任意百分比判GO，不称硬暂停界、吞吐非劣、生产p99、质量等价、公平保证或已证明新颖性。
参考TTFT5s/平均TPOT0.2s不约束最大ITL；全通过时goodput退化为吞吐，不能据此证明长暂停SLO改善。
Oracle/相邻系统直接对照/第二模型/动态到达范围仍未测；结果仅NATIVE_SERVING、MEASUREMENT_ONLY。

## 执行与保留

现有connect.weste.seetacloud.com:23478。初始化及正式测量前检查GPU占用；忙碌或查询失败ABORT，不终止其它任务。
每格归档回传核验后推进下一格；所有真实尝试与失败保留，不自动重跑、替换canonical或覆盖raw，修正写addendum。

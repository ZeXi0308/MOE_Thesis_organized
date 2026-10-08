# C 线投稿场所候选

访问日期：**2026-10-07**。这里只核对场所匹配与当前公开规则，不代表已有足够贡献、论文已就绪或可以承诺赶上某个截止日期；未投稿、未联系编辑或会议。

## 等级依据与选择

等级依据为 CCF [第七版正式目录发布页](https://www.ccf.org.cn/Academic_Evaluation/By_category/)及其 [官方 PDF](https://www.ccf.org.cn/ccf/contentcore/resource/download?ID=112CF3BF7E1140ACEB271ADAED12A67ADFABB8FF099E40C2759502A85C8A281F)，2026-03-31 发布、04-09 勘误更新。本次从发布页标注“第七版……（正式版）”的附件链接下载全文核对：共 72 页，首页标“（2026年）”，PDF 创建/修改日期为 2026-04-09，与勘误日期相符。等级不采用 CORE 或第三方列表。

- **首选会议系列：ISPASS**，软件工程/系统软件/程序设计语言方向 C 类会议，第 8 项，PDF 第 30 页。
- **备选期刊：The Journal of Supercomputing（TJSC）**，计算机体系结构/并行与分布计算/存储系统方向 C 类期刊，第 9 项，PDF 第 4 页。

CCF 对会议目录的口径限正式发表的 Full/Regular 长文，不能用 poster、short paper、workshop 或仅 artifact 替代。初查的 ICPE 虽有相合的[官方 scope](https://icpe2027.spec.org/call-for-contributions/)，但上述完整官方 PDF 文本中未见 ICPE 或 Performance Engineering 条目；Cluster Computing 的唯一匹配为第 6 页 B 类 IEEE CLUSTER 会议，不能当作同名期刊的等级，故两者不进入最终候选。这一结论针对本次官方附件，而非仅凭网络搜索无结果。ICPADS 是 C 类，但[2026 年新稿截止](https://www.cloud-conf.net/icpads2026/index.html)已于 08-15 结束，2027 官方 CFP 本次未查到，暂不作为可执行窗口。

## 1. ISPASS：主题首选，下一届窗口待公布

官方最新可核实来源：[ISPASS 2026 CFP](https://ispass.org/ispass2026/cfp.php)、[2026 作者指南](https://ispass.org/ispass2026/submission.php)。**ISPASS 2027 官方 CFP、截止日与届次专属规则：截至访问日未核实公布。** 不把第三方预测的 2026 年 12 月日期当作正式截止。

| 项目 | 已核实内容 |
|---|---|
| 匹配范围 | 最新 CFP 涵盖性能分析、瓶颈识别、GPU/加速器、内存系统、深度学习工作负载，以及系统代码调优和优化。因此真实原生 runtime 的准入干预、资源压力因果分析与完整服务性能是合理的主题匹配判断；这不是录用判断。 |
| 2026-10-07 后截止 | **尚无已核实的下一届正式截止。** 2026 届摘要为 2025-12-08、全文为 2025-12-15，均已过期，不适用于新稿。 |
| 篇幅 | **2027 未公布。** 仅作准备参考：2026 投稿为 IEEE 双栏 10 pt，正文最多 9 页；图、附录等非参考文献内容计入这 9 页，参考文献可另加页。不要混用 camera-ready 篇幅。 |
| 匿名 | **2027 未公布。** 2026 为双盲：正文去除姓名及身份线索，自引使用第三人称；暴露身份的工具/代码链接不能直接放入匿名稿。 |
| AI 政策 | **2027 会议专属条款未公布；2026 CFP/作者指南未找到单列 AI 条款。** IEEE 已有统一披露规则，见下段；下一届若有更具体要求，以正式公告为准。 |

[IEEE 官方 AI 作者指引](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/)要求在致谢说明文中 AI 生成内容，包括工具、涉及部分及使用程度；仅编辑/语法改进一般不在该强制范围内，但建议披露。C 的实验设计、代码和分析若由 AI 参与，不能笼统描述为只润色语言。匿名稿如何放置披露应待下一届规则明确，不能因此泄露作者身份或省略实际研究参与。

## 2. The Journal of Supercomputing：常规期刊备选

官方来源：[Aims and scope](https://link.springer.com/journal/11227/aims-and-scope)、[Submission guidelines](https://link.springer.com/journal/11227/submission-guidelines)。选择常规研究稿，不绑定特刊，也不把特刊截止视为期刊统一截止。

| 项目 | 已核实内容 |
|---|---|
| 匹配范围 | 官方范围包括高性能计算的体系结构与系统、程序、性能度量和方法，接收有深入技术与相关工作对比的理论和实践研究。单卡 LLM runtime 准入研究在这些主题内有合理匹配可能，但需解释其高性能系统价值，不能只展示单一配置调参。 |
| 2026-10-07 后截止 | 官网当前提供常规在线投稿入口，作者指南**未规定统一截稿日**，按常规期刊投稿通道准备；这不是一个已公布的未来日期，也不是对送审或出版时间的保证。 |
| 篇幅 / 格式 | 当前指南**未公布统一正文页数上限**，不据此声称无限篇幅。推荐 Springer Nature LaTeX 模板，提交可编辑源码及 PDF；也接受 Word。摘要 100–150 词、4–6 个关键词。 |
| 匿名 | 指南要求 title page 包含作者姓名、单位、通讯信息，**没有要求投稿稿件去除作者身份**；本次页面未明确单盲/双盲的完整评审模型名称，因此只确认这一实际作者侧要求。不能把其他 Springer 期刊的单盲说明移植过来。 |
| AI 政策 | 遵守 Springer Nature 当前政策：AI 不能作为作者，人工作者承担学术判断与责任；实质研究、写作和图表中的 AI 使用须透明说明并核实结果。见下段官方政策入口。 |

AI 要求依据为 Springer Nature [研究者 AI 指引](https://group.springernature.com/gp/group/ai/ai-guidance-for-researchers-editors-reviewers)，并应在准备时核对[研究使用政策](https://www.springernature.com/gp/policies/editorial-policies/using-ai-in-research)及[稿件准备政策](https://www.springernature.com/gp/policies/editorial-policies/ai-manuscript-preparation)。官方页面对纯 copy-editing 披露的措辞存在差异，本文不推定统一豁免；本线涉及实质研究代码/实验/分析，也不属于仅语法修改。披露应准确说明工具及实际参与，人工作者须独立核实核心论断。作者指南还要求原创研究提供数据可用性声明，可复用本线真实代码与原始数据说明。

## 当前使用方式

按 ISPASS 的性能分析读者定位组织后续证据，等 2027 正式 CFP 后再冻结篇幅、匿名版本与时间表；期刊备选不依赖会议年度窗口。两者均先要求本线证明真实准入动作的完整服务收益及强简单基线后的增量。投稿场所选择不能将当前探针、零动作负结果或未完成确认实验升级为已成立的方法贡献。

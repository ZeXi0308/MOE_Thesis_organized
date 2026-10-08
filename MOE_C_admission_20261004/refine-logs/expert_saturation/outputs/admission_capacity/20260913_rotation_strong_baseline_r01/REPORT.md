# 新文档上的持续轮转强基线对照

2026-09-13，`PREPARED_UNRUN`。尚未获得八项性能结果。

唯一问题：固定实际 KV 预算，持续 `most_output` 对原生与 fast `completion_headroom` 是否仍改善最长输出停顿和完整服务量？上一轮首次交换不能保住持续改序吞吐结果，详见[六项实测补充](../20260913_rotation_first_swap_r01/RESULTS_WESTE_ADDENDUM.md)。本轮只检验已有规则的输入迁移与直接强基线，不扫描参数。

## 固定设计

| 项目 | 值 |
|---|---|
| runtime / 模型 | vLLM 0.26.0，OLMoE-1B-7B-0924 BF16，固定 revision |
| 资源 | 单 RTX 5090；KV 16,089,350,144 字节，7671 usable blocks |
| 请求 | 新32篇文档的3072-token前缀，各生成1024 token，50ms steady到达 |
| 调度 | cap32，token budget1024，各角色独立引擎及相同预热 |
| block0 | native → headroom → most_output → native_aa |
| block1 | native_aa → most_output → headroom → native |
| 主配对 | 每block native→headroom、native→most、headroom→most，共6对 |
| 漂移记录 | 2对native A/A，4对同角色重复；不替换主baseline、不减漂移、不当噪声界 |

从固定 WikiText shard 的源行12128至17106选择下一32篇足长文章，保留完整源文章并执行固定前缀。此前96篇已逐文本及token重现，新旧128篇的文档hash、输入hash与源行区间互斥；这仍是同一语料/长度/到达域，不是独立总体。见[输入收据](inputs/inputs_report.json)。workload SHA256：`feff45f7b6dd3cfe209f7af9ed47b82f31371aedbb5109f8054315fa73b2a75d`。

包 SHA256：`7cbd2f0e595327ddcba427a3d3599f46594e3c67ca96b45b4ec6f1e1a591d9fe`。采集、metrics、headroom、预热保持已测H底座；轮转模块更新为已测过持续most_output的当前版，加入新cohort与victim_order接线。native/headroom不进入轮转安装路径。完整顺序和解释规则见[冻结协议](preparation/source/DECISIONS.md)。

## 已完成的本地工作

输入来源重现、资源/角色编排检查通过；分析器12组CPU检查通过，含原样读取实际输入收据的正例。无GPU输入时八格均为UNRUN，全部数值配对关闭。见[检查结果](preparation_checks/analyzer_cpu_checks_final.json)和[未运行分析](preparation_checks/unrun_analysis/analysis.json)。准备期Python3.9兼容与token哈希序列化问题均已修正，见[修复记录](preparation_checks/development_fixes.json)；没有生成或修改测量raw。另一个agent有界只读复核输入、角色和冻结来源，未发现P0/P1，见[准备复核](preparation_checks/bounded_review.json)；这不是GPU结果审计。

三个入口合计387行，复用既有输入解析、运行与成本分析。新代码只表达旧20格/6格工具无法表达的新32篇排除规则与四角色8格编排，不新增runtime或controller。

## 执行与判断

目标为已授权现有 `connect.weste.seetacloud.com:23478`。使用[暂存命令](STAGE_COMMAND.sh)上传校验；GPU空闲后使用[接续命令](RESUME_COMMAND.sh)，每格回传校验后再执行下一格。模型已在该机，无需下载。GPU占用或查询失败立即退出，不终止其它任务。实际状态以后续[STATUS.json](STATUS.json)及执行记录为准。

完整八格且同输入/资源/计时/动作合格后才做配对。主指标是完整吞吐和最大ITL，同时呈现TTFT、平均完成、逐请求损益与全部失败。互斥host成本保留，含重算调用不当纯GPU重算税。若结果变号或转移等待，按代价解释；无动作与坏测量分别标记，均不据此判死整个问题。

证据上限为本运行域 `NATIVE_SERVING / MEASUREMENT_ONLY`；当前无新GPU证据。强基线为native与fast headroom；Oracle、最近邻系统直接对照、质量、第二模型与动态到达尚未测。参考SLO全通过时goodput等于吞吐，不证明长暂停SLO改善。下一步只有执行本冻结八项，不追加参数搜索。

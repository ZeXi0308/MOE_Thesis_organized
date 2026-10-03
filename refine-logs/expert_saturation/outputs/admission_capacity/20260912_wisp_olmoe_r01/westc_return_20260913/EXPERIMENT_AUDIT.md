# F/X 资格与性能限定审计

日期：2026-09-14。审阅者：GPT-5.6-Sol ultra，`/root/fullstage_qualification_integrity`；资格阶段fresh，性能阶段复用同一审阅者，same-family/provisional。总体 **WARN**；本次原始数据完整性 **PASS**。按用户AGENTS §6.3不生成`.aris`。

| 检查 | 资格 | 性能 | 证据与影响 |
|---|---|---|---|
| A 来源 | PASS | WARN | 性能外部WiSP hash未由旧analyzer拒绝校验；实际4份均匹配冻结值 |
| B 分母/会计 | PASS | PASS | `(X-F)/F`；H2D、stage D2D、writeback D2D独立复算一致；时间跨度不相加 |
| C 文件/终态 | WARN | PASS | 原资格协议只含F，执行前HOST_ADDENDUM已增加X；性能四格rc0，547文件重哈希通过 |
| D 执行路径 | PASS | WARN | 性能唯一WARN同A；请求/step/position/16层关联与禁用参考均通过 |
| E 范围 | PASS | PASS | 两文档block、每engine八相关episode；5/16不利maxITL配对保留 |
| F 类型 | synthetic_proxy_differential_conformance | native_runtime_descriptive_no_gt | 层参考不是质量GT；性能无质量GT |

资格P0/P1均为0，支持测试输入上的生命周期一致性。资格中的额外参考、JIT、显存与控制器暂停均不作性能依据。原F-only协议与新主机F/X安排分别见`../full_stage_qualification/protocol.json:4,47`、`HOST_ADDENDUM.json:52`、`run_new_host_qualification.py:11`。

性能P0=0、P1=1：外部依赖自动检查遗漏。`performance/source/wisp_v026_adapter.py:66,397`导入并记录外部WiSP；`performance/analysis_source/analyze_layer_budget.py:10`冻结期望值，但其131行检查位于未调用main中；`performance/analysis_source/analyze_shared_pool_execution.py:8,90`仅复用helper，未验证`upstream_sha256`。四份实际pager_summary逐份匹配`5572d4f05593a5a9fc4adaa14421cc596886a435b6f5f51f476a1dae6e521857`，路径、hash与文件hash见[机器记录](EXPERIMENT_AUDIT.json)。本次来源实际一致；旧分析不追改，新finite-arrival包在运行前和分析中直接比较该hash。

32次测量/96请求/1280输出、160次预热/416请求/5184输出、62592全部层调用/12288测量层调用、76GPU边界检查全部重算；547 manifest文件无缺失或hash错误。32份实际资源记录均为384唯一expert槽/4,831,838,208B及1GiB KV。16/16输出与route相同，16/16最终cache与复制轨迹不同；所有参考关闭。

REPORT性能新增段的数字与限定通过：capture/cycle/旧边界ITL/旧完成/新TTFT/新完成16对均下降；旧/新maxITL各11降5升，最坏+10.698%/+25.243%。同engine相关重复不提供16个独立实验，GPU边界PASS不证明连续独占。

允许结论：本次RTX5090、OLMoE、人工专家池、冻结事件注入输入上，X/F两block均值capture、实验cycle、process与D2D下降。不能外推显著性、非劣效、持续服务摊销、质量、SLO-goodput、部署或方法GO。原始artifact与冻结checker均未修改；不追加广泛审计。

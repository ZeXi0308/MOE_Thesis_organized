# 最新状态

2026-09-14：`STAGED / GPU_UNRUN`，没有测量raw或新性能结果。四格包SHA1d0e5f050d620c4a84ebac6af4eb76d7fde109ce84317a74f24d4085b852cce6；暂存回执见execution/execution.json，源与预热/输入校验已完成。前序B第二cohort已释放，继续按GPU_COORDINATION排context-victim整组后。远端查询曾挂起后返回，后续限30s；不据查询超时重新发起实验。

已完成：64篇自然完整文章、持续到达与EOS合同；open adapter；EOS输出/终止区分；共享90GiB父预算观测；支持0/1输出及零抢占的分析器。CPU46个相关定向/兼容测试通过；不把它们当作GPU资格或科学收益。

新科学问题和执行前判读固定在PROTOCOL.md。旧cohort3八格的强基线结论见../20260914_service_window_holdout_r01/REPORT.md：1–2输出短恢复不再出现在native/most，residual未超过两者Q(g)较优边界；当前无方法GO。

01:31 UTC接续：前序context-victim执行方本地receipt为UNKNOWN_REMOTE，更新1789349258.417784，错误为SSH interrupted。远端可能仍有已启动组，不能作为已释放或自动重启依据；本组保持STAGED/GPU_UNRUN，无run driver。先确认前序组终态和现场占用，再按原序执行。

接续更新：前序context-victim已COMPLETE/exit0且原PID退出，队列阻碍解除。现为STAGED/GPU_UNRUN_AUTH_UNAVAILABLE；本组没有GPU attempt。旧控制连接已失效；自动审批拒绝所选askpass助手，原因是它被识别为从无关附件读取类似凭据值。本次重连未执行，不继续使用此来源。需恢复该主机正确认证或提供已授权ControlPath。认证不可用期间释放执行顺序给后续已暂存A保真单格，不持GPU窗口。详见execution/connection_status.json。

CPU证据接续：context-victim六格原件已由执行方完整回读，本方生命周期分析全臂/全repeat零输出及1–2输出再丢弃均0，完整结果见../20260914_service_window_context_r01/REPORT.md。它覆盖2560/3072异构但固定输出1024，不能代替本64篇持续到达/EOS四格。本组仍STAGED/GPU_UNRUN_AUTH_UNAVAILABLE，无新连接授权或GPU attempt。

授权执行接续：B会话通过用户直接提供凭据建立的连接，核验原SHA与现场后首次接续原四格；controller72155、shell72156、runner72167已由A现场确认存活，execution.json为RUNNING_ASSISTED，原STAGED快照保留。原执行方不重启/复制实验；统一原目录回读后使用已准备分析器。旧认证失败记录保留，不再是本组GPU_UNRUN的当前状态；尚无测量结论。


完成接续：原四格COMPLETE/exit0，各64/64；原归档SHA e485246755e0c11db330ce191002375a4b16f573220be9cd89105a58714296bd已回读，首次下载断线仅重传同包。当前科学状态MEASUREMENT_ONLY，见RESULTS.md，前述UNRUN/RUNNING为历史。自然抢占2/5/2/0、强制轮转全0，9恢复段均至完成、各得996–1024新输出，无再丢弃；3段在首输出前、6段在生成中。四格各6请求提前stop且末尾实际返回EOS50279，58请求仍达1024上限；返回59576 ID含6 EOS，输出质量未测。实际4096usable块、共享父90GiB观察确认，独立host硬预算未实现。停止本档加保护/调压力参数；下一复用已暂存funding-filter六格，排既有Qwen完整生命周期后，本方仅CPU分析不占GPU。

一次fresh Sol限定复核完成：integrity PASS / scope WARN，P0/P1=0，same-family/provisional；见EXPERIMENT_AUDIT.md。当前迭代收尾，不增加审计或重跑。

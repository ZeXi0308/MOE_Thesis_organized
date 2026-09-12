# 准入研究简记

2026-09-08；探索记录，不替代 sealed verdict 或 `docs/current/README.md`。

中心问题：哪些可执行的 MoE serving 接纳动作，能在普通 batch/KV/queue/延迟基线之外
改善完整请求的 SLO-goodput？当前还没有专家感知方法收益证据。

稳定事实：共同原生引擎中的短请求静态 cap8 优于 cap6，已有独立文本复测；
普通 ITL 反馈和一次非抢占降档没有稳定超过强静态点。数据在
`outputs/admission_capacity/20260906_native_{fixed_engine,fresh_cohort,knee,single_action}_r01/`。

当前主攻：同次原生引擎中比较旧 `[8,12,16,32]` 与捕获点对齐 `[8,16,24,32]`
反馈档位。Sep8 历史数据重建提示阶梯成本，但不能证明档位是反馈失败的原因。
对旧报告的归桶错误、单次padding常量和误差上界措辞已作追加纠正。

最近调整依据：旧待测方案缺同次旧规则基线、预热不一致，并混用了实际width平台与
episode总体中位数。因此执行前缩成同引擎配对、共同预热、完整反序重复。
没有新GPU结果，没有据旧相关性升级科学状态。

唯一下一实验：[32-episode 同次配对](../../outputs/admission_capacity/20260908_capture_ladder_paired_r01/DECISIONS.md)。
代码、输入、重算器和执行包已准备；GPU `UNRUN`，远端凭据使用待自动审批要求的明确授权。
若只有普通规则改善而不胜同次静态点，保留工程结论；若变号，先受控重复。
自然长上下文KV压力继续保留UNRUN；不同时启动另一条实验链。

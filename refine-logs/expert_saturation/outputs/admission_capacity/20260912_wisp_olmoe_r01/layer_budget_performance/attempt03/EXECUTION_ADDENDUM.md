# 执行中断与接续规则（2026-09-13）

原冻结U/M/G/G/M/U、每格8次none/early、资源、文档与指标保持不变。第一次尝试在0_uniform初始化期间因新外部GPU进程OOM，零raw。第二次尝试完整完成0_uniform和1_selected；2_min_groups初始化期间又因外部GPU进程OOM，零raw。所有原始尝试和日志保留。

在读取这两格性能指标之前确定：每个预定label采用首次完整完成、原冻结源码/资源/输入相符的cell；只补尚未执行请求的2_min_groups、3_min_groups、4_selected、5_uniform。若以后有部分请求后失败，保留整格并另行判断有效性，不静默重跑或替换。不得根据快慢选择重复或重新优化容量。

两个方向block存在实际中断和时间漂移风险；记录各进程原始起止、全部GPU边界、预热和完整周期。它不是连续独占或无干扰的六格实验。最终若集齐六格，组合execution元数据只是可追溯索引，原attempt execution.json不改；统计继续限于描述，不以共享文档和ordinal配对制造独立样本。

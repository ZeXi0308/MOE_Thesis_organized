# 保存前缀状态模型：异步恢复边界

Verdict: PARTIAL_STRUCTURAL_CALIBRATION。CPU状态转移模型，非GPU新实验、非墙钟预测/方法GO。

沿用已有固定长度闭集模型，补两个可选输入：native_preemptions（指定候选原生抢占，不提前处理waiting）和saved_prefixes（加载先占完整前缀KV块，保持waiting，ready后再恢复计算）。默认无此输入时，原least/most/defer全部预测与封存结果逐字段相同。

输入仍取旧封存step329前态；逐请求computed/output/preemptions/block数量、running顺序和free块数与新四格核对一致。victim由该状态running中最少output选出，保存前缀为computed向下对齐16；未向模型传入动作后的真实调度、输出或资源轨迹。

首先发现：原模型本来就同时推进重算与其他请求decode，不需要另造混合服务模型。无保存时，仅补入同329原生抢占动作，两次实测的329–1868全部1540步调度、KV空闲块、输出数量逐步一致。

保存时引入异步加载：330先占206块并把computed恢复为3296，仍在waiting；按首次实测校准ready=issue+2，332执行恢复。两次实测329–1040的712步逐步一致，但1041均首次不一致：实际第二次1040加载在1041即可恢复，固定两步模型要到1042。最终last_step仍恰好预测1866，不能因总数相同掩盖中途状态错误。

| 模型对照 | 两次实际末步 | 预测末步 | 首次不一致 |
|---|---:|---:|---|
| 无保存 | 1868 | 1868 | 无（从329验证） |
| 保存、固定两步load | 1866 | 1866 | 1041，调度/空闲块/输出均不一致 |

这解释了上轮会计为何需要保留完整batch进度，但尚未构成可用于任意动作的预测器。load_delay_steps=2是首事件事后校准，不是物理延迟规律或事前冻结预测；没有用第二次事件的实际ready步去逐点拟合。

本轮否定对象仅为“native异步load固定等待两步”的近似；保存/恢复动作与研究问题未被判死。每步耗时会变化，搬运提交与完成查询也有具体时序，完成延迟不能天然用固定步数表示。上述可能原因尚未由本轮定位，不把它们写成已证根因。

唯一下一CPU工作：从封存native worker/connector源码定位load提交、完成查询、scheduler可见ready三个边界，判断应使用可观测完成状态还是经时钟校准的完成条件。先修这一个状态转移，不追加GPU重复、不设计第三个预测器、不调第二次load专属常数。

实现：experiments/admission_capacity/recovery_progress_model.py；可重算验证：validate_saved_prefix_model.py（输出目录须不存在；现有结果不可覆盖）。validation.json保留全部四格检查，model_source.py封存本次版本。执行者定向验证，非fresh独立审阅。资源预算仍计满加载已占块，未模拟host竞争、真实EOS、其他模型或自由生成质量。

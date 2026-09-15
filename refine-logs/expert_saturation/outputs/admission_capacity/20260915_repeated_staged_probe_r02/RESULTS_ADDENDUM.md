# 原两格资格结果：COMPLETE

本补记更新旧 STATUS 的准备期状态，不修改原包、原配置或原报告。原 controller 2344 于 1789407478.333702—1789407713.820840 在 weste:26862 GPU0 完成，off/on 均32/32请求、32768输出、44次原生抢占（其中38次为调度器主动轮转）。完整原件及分析在 [execution_weste_26862](execution_weste_26862)。

资格结论为 REPEATED_EXECUTION_QUALIFIED：on 实际完成38 store、43 load；保存12,501,123,072B，加载20,984,102,912B；off无搬运。恢复重执行163835→3731，实际调用1316→1312。单对观察值为吞吐+5.10%、平均完成−5.03%、全局最大gap−17.58%，仅原固定 off→on 顺序的描述，不能作为独立ABBA结果或稳定性能归因。所有混合调用时间均为包含其它decode的区间，不是纯重算税。

26/32输出序列相同，质量与KV数值保真未验证。已有限定复核确认原件和主表可复现；不追加该组复核。原始 STATUS/config 中的 UNRUN 是准备期记录。

新的诊断＋轻量ABBA六格已经单独完成，完整服务判断使用其 [主报告](../20260915_repeated_kv_service_r01/analysis/REPORT.md)，本组不混入新组配对。当前研究选择由 [统一清单](../../../CURRENT_EXPERIMENT.json) 和 [论文论证](../../../PAPER_ARGUMENT.md) 管理。

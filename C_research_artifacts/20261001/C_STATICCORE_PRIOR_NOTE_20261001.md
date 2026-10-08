# StaticCore 近邻：Where Should Requests Wait?

**来源边界。** 作者仓库当前 HEAD 为 [`08b18ad46a82e594c2de1274321254b3ff2d6977`](https://github.com/manishraj1/StaticCore/tree/08b18ad46a82e594c2de1274321254b3ff2d6977)（2026-07-12）。本说明读了该提交的 [README](https://github.com/manishraj1/StaticCore/blob/08b18ad46a82e594c2de1274321254b3ff2d6977/README.md)、[补丁说明](https://github.com/manishraj1/StaticCore/blob/08b18ad46a82e594c2de1274321254b3ff2d6977/StaticCore/src/patch/README.md)、[补丁](https://github.com/manishraj1/StaticCore/blob/08b18ad46a82e594c2de1274321254b3ff2d6977/StaticCore/src/patch/osl_reserve_v4.py)和 [复现说明](https://github.com/manishraj1/StaticCore/blob/08b18ad46a82e594c2de1274321254b3ff2d6977/StaticCore/REPRODUCE.md)。README 指向的 `paper/draft_v0.1.md` **不在当前树中**；它在匿名化提交 `f7a0738` 被删除。下述论文细节来自上一个含论文的提交 [`06840b0` 的作者草稿](https://github.com/manishraj1/StaticCore/blob/06840b0ce99b93be8453f0f0041fb35b9f9c11d2/StaticCore/paper/draft_v0.1.md)（Git blob `ad7536d124ff6508c56320a4c045cde136aa1957`）。作者标为 *Draft v0.1*，README 说预印本引文待补；这里不称其经过同行评审。

| 机制与实验 | 作者稿的准确范围 |
|---|---|
| 原生 full ISL | vLLM V1 0.24.0 的 `scheduler_reserve_full_isl` 默认开启，准入时把尚未算完的完整输入长度纳入块需求；不预留整个输出。 |
| Watermark | 留出固定比例的空闲 KV 块；作者测 0.05、0.15。它与请求自己的 `max_tokens` 无关。 |
| OSL 补丁 | 在 `KVCacheManager.allocate_slots` 原 full-ISL 路径，把单请求需求改为 `min(P + int(α·M), max_model_len)`；`α=0` 为原状，`α=0.5/1` 为分档输出上限预留。补丁不改原生调度/抢占，也**不维护所有 resident 请求的合计上界**，作者草稿明确说不能据此保证无驱逐。 |
| 运行域 | Qwen2.5-1.5B-Instruct、`max_model_len=1024`、单消费级 GPU、320 条一次提交、APC 关闭；温度 0.8、`max_tokens=650`、`ignore_eos=True`，每请求强制同长输出，合计 208,000 输出 token。把 KV 池设为 150/200/300 块制造深度超额需求。 |
| 完整请求指标 | 全部 320 条计入总 token/墙钟吞吐、原生抢占计数、逐请求首 token 延迟和 decode 时长的分位数。成批提交时首 token 延迟主要反映排队位置，不能直接当开放到达的服务 TTFT；没有 C 的联合完成请求 goodput。 |

**作者稿的结论与局限。** 在该短上下文、强制等长域，准入越保守，尾部首 token 等待越长、decode 尾部越短；预留过多使同时运行批次变薄。0.05 水位在较宽松池中约减少 41% 抢占而三 seed 确认的吞吐比约 0.991，不能报告为可重复吞吐收益；`α=0.5` 的 OSL 与 0.05 水位在相近保守度下未见可检测的吞吐差异。作者还明确限制为一个小模型、约 1k 上下文、全批到达、强制等长输出及有会话漂移的消费级 GPU；自然 EOS、异质输出长度和长上下文的效果未由这些结果确定。

**与 C 已测事实的交集。** C 的 [full-bound FIFO](C_NATIVE_BOUND_MATCHED_RESULT_20261001.md)已经是强简单基线：它按 `ceil((P+M)/16)` 对所有 resident 和待准入队头做合计容量约束，零抢占；两组开发配对的 20/4 goodput 提高 12.1%/17.7%，但平均完成 flow 增加 4.20%/3.96%，实际输出吞吐下降 9.20%/6.90%，多数请求 TTFT 变差。StaticCore 加强了“预留会迁移等待并可能削薄批次”作为已知解释；不能把一般等待位置迁移或已知 `max_tokens` 预留包装为 C 的新方法。C 的 [条件退休包络](C_NATIVE_RETIREMENT_FRESH_RESULT_20261001.md)则用当前 decode 计数与已知 cap 约束未来峰值，新准入仍保留 full bound；它不是 StaticCore 的单请求 `α·M` 补丁，后者也未测试自然 EOS 的退休机会。两者模型、vLLM 版本、上下文、内存压力和指标不同，StaticCore 的排序数字不能移植到 C，更不能替代 C 的完整近邻比较。GSM8K Instruct 的 native128 实测峰值仅 932/4096 块且零抢占，仍按已登记决定停止该输入的容量方向。

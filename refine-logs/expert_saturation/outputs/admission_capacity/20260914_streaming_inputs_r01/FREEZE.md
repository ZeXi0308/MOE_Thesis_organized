# 64篇完整自然文章与单档压力配置冻结

Status：`FROZEN_INPUTS_GPU_UNRUN`。没有GPU、生成或EOS结果。`candidate/`原件保持逐字节不变；`frozen/`保留完全相同的64个source rows、完整原文和token IDs，只冻结到达与声明的输出/资源上限。

| 项目 | 冻结值 |
|---|---|
| 模型/Tokenizer | allenai/OLMoE-1B-7B-0924；revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 |
| 输入 | 64个唯一完整文章；排除旧160个identity/hash/重叠原始row；不重复文档 |
| 实际prompt | 334–3011，mean1659.90625，median1542.5；总106234 tokens |
| config.prompt_tokens | 3011，仅表示上界；实际长度逐请求保留 |
| Arrival | source order，第i个在i×0.5s到达，覆盖0–31.5s；不由完成时刻触发 |
| 生成 | max_tokens1024、min_tokens0、ignore_eosFalse；真正EOS与输出长度未知 |
| KV单档 | usable4096块×2097152 bytes=8GiB；另加1个null block |
| 传给engine的KV存储 | (4096+1)×2097152=8592031744 bytes；实际分配由runtime核验 |
| 并发/计算预算 | max_num_seqs32、max_num_batched_tokens1024、max_model_len4096 |
| 比较 | native与most_output；执行顺序、重复、host预算由主protocol负责 |

为何不跑原6656块档：任意32个候选文档的最大声明KV上界Σceil((P+1024)/16)=6551<6656，余105块；连续32窗口范围5052–5873。原13GiB usable档保留为低压力边界，不用该档检验恢复增量。

4096块是在执行前一次固定的压力档，不保证一定发生抢占或出现EOS：每请求实际生成可能提前结束，active set由实际到达、完成和独立策略演进决定。若该档没有恢复或没有EOS，仍原样报告；不根据action频率修改文档、长度筛选、arrival或预算。

CPU source-order候选只扫描row[1,8173)，128篇闭合原始文章中64篇由旧identity/row排除，另64篇全部落入预先给定256..3072范围。该source interval全部文章自然token长度334–17756、median3053.5；候选的短长度域是已声明的选择边界，不是整个WikiText总体。

输出时间只沿用引擎返回边界，候选不包含任何将来token/EOS时间。现有native_capture的固定length断言需要主会话EOS适配后才能执行；本冻结不代替runner验证。

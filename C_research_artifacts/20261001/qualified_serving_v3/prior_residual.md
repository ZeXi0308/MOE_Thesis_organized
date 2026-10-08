# 共享物理前缀与 hard-cap 准入：局部残差核查

本轮只读查新；未运行 GPU，未实现控制器。依据最新 `C_BBH_APC_RESULT_20261001.md`、`C_RESEARCH_STATE_20260930.md`、直接近邻记录及保存的作者 AE 源码，并重新读取必要官方源码。旧 C 退休峰值公式的新颖性撤回保持不变。

**结论：尚未找到已经具备自然机会证据、足以支持新控制器的残差。** 可以保留一个非常窄的待证伪假说：在 APC 已启用、多个前缀群交错到达且前缀群生命周期不同的健康任务中，把当前前缀命中当成永久“零新增容量”，会遗漏新消费者延后共享块最后释放时刻的效应；精确处理该依赖可能在不破坏 decode 推进承诺的条件下，优于简单的“共享块始终计一次”保守准入。它至多是一个系统耦合/执行契约假说；集合去重或 `max(owner retirement)` 本身不是新算法，也尚未排除更广泛现有系统覆盖。

## 已有机制覆盖到哪里

| 层次 | 已有支持与本轮核对 | 不应声称 |
|---|---|---|
| 当前物理共享 | vLLM APC 查找已计算块；活动块被再次命中时增加引用，最后一个引用消失才回到 free queue；命中 `ref_cnt=0` 的缓存块需要重新消耗空闲容量。 | 不应说原生 APC 不理解多 owner，或请求一结束就错误释放共享块。 |
| 当前空闲与缓存可驱逐空间 | vLLM 区分 active blocks 与可驱逐 cached blocks。LightLLM 当前 router 也从占用中扣除 radix tree 中无引用缓存。 | 不应把所有保留 hash 的缓存块都当不可用容量，或把 KV 池活跃量降低称为 CUDA 分配减少。 |
| 静态共享前缀 + 时间峰值 | 作者 Past-Future AE 的 request tuple **已减去 `prompt_cache_len`**，全局另扣 `prompt_cache_used_tokens`。其 manager 在 splitfuse 模式初始化固定 `prompt_cache_strs`。 | 不能概括为“Past-Future 完全没有共享前缀会计”。仅看 queue 的逐请求求和会漏掉 tuple 转换。 |
| 动态共享块未来生命周期 | 已查 AE 路径采用静态缓存单独收费；已保存的官方 LightLLM queue 使用逐请求 `(has_tokens, remaining_steps)` 峰值。上述判据中未找到“物理块→当前 owners→最后 owner 退休时刻”的未来事件表达。 | 这只证明所查路径未表达该对象，不能证明所有 LightLLM 分支或所有研究都没有。 |
| 当前分配合法性与未来保证 | vLLM `allocate_slots` 有缓存命中、lookahead、watermark、reserved blocks 等检查；`full_sequence_must_fit` 读取的是当前 `request.num_tokens`，不是整批各请求的未来输出 cap 日历。 | 不能把该 flag 的名称直接解释成完整 hard-cap 退休保证；也不能忽略它已有的准入检查。 |

官方来源：vLLM 0.26.0 的 [BlockPool touch/free](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/block_pool.py#L647-L681)、[每类型分配增量](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/single_type_kv_cache_manager.py#L125-L204)、[KVCacheManager](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/kv_cache_manager.py#L258-L426)。这些路径表明原生 allocator 已正确处理当前共享；本轮未重新检查远程安装包与线上 tag 的逐文件一致性。

作者 AE 固定 commit `d93ff69c07d4097b0a6a4ee67ea355e8780e3095`：[request tuple](https://github.com/WuSiYu/lightllm-ae/blob/d93ff69c07d4097b0a6a4ee67ea355e8780e3095/lightllm-server-pastfuture/lightllm/server/io_struct.py)、[queue](https://github.com/WuSiYu/lightllm-ae/blob/d93ff69c07d4097b0a6a4ee67ea355e8780e3095/lightllm-server-pastfuture/lightllm/server/router/req_queue.py)、[manager](https://github.com/WuSiYu/lightllm-ae/blob/d93ff69c07d4097b0a6a4ee67ea355e8780e3095/lightllm-server-pastfuture/lightllm/server/router/manager.py)。本轮直接读取已有带 SHA 的本地原件；部分 raw URL 再抓取返回 cache miss，未换来源填补。原文峰值机制仍见 [Past-Future §3.3](https://arxiv.org/html/2507.10150#S3.SS3)。

官方 LightLLM：已有固定 queue 记录为 commit `6c9e03b14ed3e0da8d4c867258be7a90e46d941d`；补看[当前 request](https://github.com/ModelTC/LightLLM/blob/main/lightllm/server/core/objs/req.py)、[router](https://github.com/ModelTC/LightLLM/blob/main/lightllm/server/router/manager.py#L379-L387)、[BaseQueue](https://github.com/ModelTC/LightLLM/blob/main/lightllm/server/router/req_queue/base_queue.py)。新读的 main 未取得固定 commit，只能标为当前可访问代码。其 request 还给分页、overlap/MTP 与停止传播预留余量，故“考虑退出延迟”本身也不是可直接申报的新贡献。

## 唯一条件假说及最小反例

以下为独立推导，不声称文献未曾提出。限定 full attention、不可变的已计算完整前缀块、无外部 pin/offload/异步/speculative，所有获得有限退休时间的 owner 都有可兑现的一 token/轮进度。对共享物理块 `b`：

`active_b(k) = 1[至少一个 owner 在 k 轮仍持有 b]`。

当只有当前 owners、无新增 attach 时，该块最晚释放时刻由 `max_i r_i` 决定，不能随最早 owner 退出而释放。`Σ_i blocks_i(k)` 会多算共享块；从这项总和中始终减去“当前重复引用数”又可能在 owners 退出后少算。正确做法是按时刻计算物理并集，或者保守地把共享块始终计一次。

最小风险状态：一组 S 个完整块由 A/B 共享，已有承诺按 A 在第1轮、B 在第2轮完成，计划从第3轮复用 S 块。新请求 C 命中这些块，当前增加的物理前缀块为零，但 C 要持有到第10轮。如果准入只给 C 计私有增量，且保留原来的“第3轮可复用 S 块”证书，就重复承诺了同一容量。原生 `touch/free` 会正确保留这些块；失效的是外加未来证书，最终只能表现为后续分配失败/抢占/进度停顿，而不是原生误释放。

因此候选的动作若存在，应是：**在 native attach 之前，连同新请求对已共享块生命周期的延长一起复核未来承诺**；只复核当前 allocation 还不够。候选在 prefill 阶段没有可靠推进上界时，不能凭 hard cap 给它假定退休轮次；要将其共享块保守保留，直到明确 P→D 边界。简单地给新请求保留完整上界也能吸收此风险，必须作为强简单参照。

潜在贡献只能是：证明上述依赖在自然服务中频繁导致可观测的准入保守损失或生成停顿，并以低开销的原生执行契约避免它。**单个构造反例、集合并集公式、加 refcount 字段均不够。** 本轮没有证明现实系统正在使用这个错误折扣，也不把反例当成已发现的现有产品 bug。

## 最便宜的证伪探针

优先级低于当前健康生成资格，不单独申请新 GPU campaign。先用新资格格已有输入几何、APC 开启下的完整结果检查：若请求全部可在保守上界内容纳、无内存造成的等待/抢占，则此格对准入残差没有动作空间，**不从本格构造新控制器**。不要通过关闭 APC、重挑错误长输出或缩紧到不自然的单点预算制造现象。

仅当健康任务出现实际容量压力，下一次已授权的小格才追加只读 observer（16–32 请求足够做结构资格）。在同一 pre-schedule 边界保存：resident 的 `P/O/M/C/phase`、实际调度 token 数、FIFO 队头、原生块表与 refcount、实际可命中 block IDs、free/reclaimable 与 native reservations；每次 free/attach 保存 owner 变化。CPU shadow 同状态比较：

- **S：共享块始终计一次**，私有块沿同一 hard-cap/推进条件增长；不对共享块预支未来释放。这是最强必须先过的简单参照，不等于逐请求全量相加。
- **O：物理并集与 last-owner 释放**，新候选完整上界且所有 attach 对已有共享块的生命周期影响入账；没有进度保证的 owner 不给予有限退休时间。
- 原生 APC 的实际准入/等待/抢占记录作为现实需求证据；原生当前 admission 与 O 不能仅凭离线 shadow 比出收益。

只问一个问题：**是否存在 S 拒绝而 O 通过、且同 native legality 和进度条件下可执行的真实 FIFO 队头？** 统计不同请求/前缀群数和等待持续时间，不把重复 schedule 视为独立样本。若该差异为零，或差异只来自常规当前 prefix 去重，或现成 S 已覆盖所有合法动作，则该 last-owner 扩展在此运行域停止。若有差异，也只授予小原生对照资格；还需 native APC 与 S/O 从共同输入独立演化，证明完整请求收益和成本，才能谈系统贡献。

当前 BBH APC 日志 **不足以执行此完整 shadow**：518 个 schedule snapshots 主要是汇总量及最多4个共享 refcount 样本；只有首次两 owner witness，未保存每步完整 owner 图和 `P/O/M/phase`。lookup/allocation 事件不能在缺少完整 free/phase 事件时被假装成完整反事实状态。不补造这些数据，也不为该低正确率旧任务重跑一遍。

## 当前投入裁决

APC on 的 BBH 开发批已把峰值降到1022/4096、抢占归零，并且正确率仅1/32；它反对在该实例投入新的 sharing-aware 准入机制。**现在值得投入的是健康服务资格与真实容量压力的恢复，不是共享控制器。** 上述 last-owner 假说保留为一个条件结构探针；截至本轮，没有足够证据把它升级为唯一 Primary 或“可投稿 CCFC 新方法”。

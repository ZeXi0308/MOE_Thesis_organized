# B/C 备线的有界可执行性判定

2026-10-02。范围：当前 `suffix_src`、本地 WiSP vendor、缓存 vLLM scheduler 与 v0.26.0 官方源码；只读判定，没有改代码、运行 GPU 或新增实验组。

**结论：B 不适合作为当前同步 runner 的立即备线。C 存在真实物理重分配原语，但只有全请求排空后的版本，且不能直接接到当前 v0.26 + CPU KV 路径；它不是 episode 内的现成容量交换接口。** 不应为了保住 B/C 而先造异步执行引擎或活页迁移器。**目前没有证据足够、可直接替换 A 的新备线。** 先完成已在推进的一次跨层压缩实际对照，再决定是否转向。

| 候选 | 当前实际接口 | 可以立即成立的主张 | 判定 |
|---|---|---|---|
| B：零主动等待的同层双微批合并 | 每步一个模型调用、每层一个完整行集合；没有独立层就绪队列 | 当前 expert-major 已联合同层全部现有行，冷专家各加载一次 | 缺少候选要求的两个独立 ready 微批；停止这个版本 |
| C：专家/KV 物理交换 | vendor 有 `resize_cap`、`resize_kv_blocks`，后者要求空 KV 池 | 全请求完成后，销毁/重建 KV storage 的旧原语确实存在 | 可移植成 burst 边界工程基线；不是活请求阶段小改动，也不是新的机制 |

## B：同步执行中没有被遗漏的第二个 ready 微批

`suffix_src/run_native_pager.py:339` 明确使用 `async_scheduling=False`、`enforce_eager=True`、单 GPU；adapter 在 `wisp_v026_adapter.py:163` 排除 DBO、EP/EPLB、TP/PP/DP 多卡。`native_capture.py:300` 到 `engine.step()` 返回前不会再提交另一模型步。`suffix_runtime.py` 也排除 `use_ubatching`。因此：

1. 一次层调用中的 prefill、decode、验证行已经在同一张 `x`/路由表内。它们是多个请求，但不是两个独立层执行任务。
2. `wisp_expert_groups.py:198` 取得这次调用的全部真实 routes；`partition_experts` 与最终断言保证全部当前行的冷专家并集各加载一次。把已有行人为拆成两组，再比较“分别执行”与“合并执行”，只是在制造一个弱基线。
3. 同一请求下一轮 token 必须等本轮完整模型及采样完成，无法提前形成第二个当前层输入。不同请求也由一次原生调度集合推进；未调度请求没有已算好的该层 hidden/router 可供零等待合并。
4. 层内 CUDA copy/compute 的异步 stream 并不产生独立已就绪微批。把下一批推进到同一层，需要保存逐批 hidden/residual、attention 元数据、KV/采样状态，以及可暂停/恢复的层级执行控制。那已经是新的执行器，不是“加一个短队列”。

这不说明层级异步在所有系统中不可行，只说明它不存在于本次已运行底座中。当前 B 的最小实验前提不成立，不应上传一个人为拆批后再合批的“正信号”原型。

最近邻也不支持以“跨批同专家摊销/ready 层队列”本身声称创新：

- **ExpertFlow §3.3** 已把相邻两批合成 token 集合，按预测的跨层专家路线重分成两批，并配套 KV Merge/Reindex 和双批流水。B 的“实际当前层路线、只合并已就绪工作”与其预测/重分组不同，但单凭这两个限制还不足以构成方法贡献。[原文](https://arxiv.org/html/2410.17954v2#S3.SS3)
- **AMoE §3.2、§3.4** 已维护带 RequestID/LayerID 的层队列、完整 top-k 依赖合并，以及 GPU 空闲时重组就绪 token 执行。它在多 GPU EP 域，不能直接当单 GPU 专家卸载收益证据；但“ready 后排队并合批”不是新的抽象。[原文](https://arxiv.org/html/2505.08944v2#S3.SS2)

这两项已足够判定当前 B 的接口与主张问题；没有继续扩查 HybriMoE/QLLM，也没有把应用域不同等同于机制新颖。

## C：已有物理原语，但条件比“两个 engine step 之间”严格

本地 `vendor/WiSP/src/wisp/integrations/vllm/fused_moe.py:439` 的 `WispMoEState.resize_cap(new_cap)` 真正创建新 scratch、复制保留槽并替换旧 tensor；`resize_layer_caps` 遍历注册层。本 adapter 的 postload 仍调用 upstream postload，所以这些 state 并非完全不可达；实际 forward 也读 `state.scratch_*` 和 `state.cap_experts`。

`vendor/WiSP/src/wisp/dynamic/kv_resize.py:125` 的 `resize_kv_blocks` 也不是只改逻辑 quota：它删除 attention/runner 的旧 KV 引用，调整 tensor size，重新分配并绑定 KV，然后重建 block pool。因此“vendor 完全没有物理交换代码”是不准确的。

但这段代码**不是保留活 KV 的增长/收缩器**：

- `_assert_drained` 要求除 null block 外全部 free；`serve_hook.py` 只在 `scheduler.has_requests()` 从真变假时调用。全请求排空与一次同步 forward 结束不是同一条件。当前完整 16 请求 episode 中，只有结束后才自然满足该边界；当时再分配不能改善刚结束 episode 的停顿。
- 它销毁所有 KV storage，而非迁移活页并保留 request block table。若要在 mixed→decode 或恢复压力出现时交换，必须增加活页存续/迁移与元数据协调，超出“现成小接口”。
- 当前 adapter 设 `WISP_DYNAMIC=0`，`install()` 拒绝更换 pager 配置。即使手动调用 underlying state，运行配置、容量记录和校验也需要与真实变化一致，不能默默绕开固定预算日志。

与当前版本还有三个具体不兼容点：

| 位置 | 源码证据 | 对当前实验的含义 |
|---|---|---|
| kernel block-size 准备 | vendor 调 `mr._prepare_kernel_block_sizes(cfg)`；v0.26.0 runner 使用独立函数 `prepare_kernel_block_sizes(cfg, self.attn_groups)`，没有这个成员方法 | 旧函数直接调用会失败，需要明确移植 |
| CPU KV connector 的 storage 引用 | vendor 只调用 `initialize_kv_cache_tensors`；官方完整初始化还要 `register_kv_caches`/`register_cross_layers_kv_cache`。offload worker 将 GPU storage 包成 canonical views | 清空 attention/runner 列表并不保证旧 GPU storage 已释放；旧 worker 也不会自动指向新 buffer，不能宣称真实等预算交换 |
| 失败与临时峰值 | KV resize 修改 pool/config 后没有真正 rollback；controller 捕获异常只改日志。专家 resize 在旧 scratch 仍存在时先分配新 scratch | 失败可能留下不一致状态；等最终字节不等于等峰值。cap24→23 时，单层会先额外申请约 276 MiB 新 scratch，不能忽略此峰值 |

前两项官方依据：[v0.26.0 GPUModelRunner 的 KV 初始化](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/worker/gpu_model_runner.py#L6896)、[OffloadingConnectorWorker 的 KV 注册](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py#L54)。第三项可直接检查本地 vendor 函数；其 docstring 声称失败会恢复，但函数体并没有该恢复路径。

OLMoE 的换算仍成立：每层每专家 12 MiB，16 层统一改变一个槽为 192 MiB，等于当前 2 MiB KV block 的 96 块。它仅说明终态容量守恒，不说明上述动态操作可运行或有收益。

**若未来确实需要 C 的最小移植**，限定为两个自然完成 burst 之间、所有 offload job 已完成，先更正 v0.26 API、重注册 connector/相关 cache config、计入重建时间与真实峰值，再测下一 burst。无需先建设新内存管理器，但这只复用 vendor 已有 drained-resize 思路；当前不建议为它改变工作负载来制造频繁空池。活请求内 C 暂不实施。

## 是否存在可直接替换的新备线

**本次没有找到。** 有一个更短、可实现的想法是“有待处理 prompt 时供给 N1，纯生成尾段不给草稿”；原生 `spec_token_ids` 支持在 schedule 前清空，无需新执行器或内存管理器。但它只是验证前阶段自适应供给，既有 SpecMoEOff/EVICT 已覆盖相邻范式，不能因为接口短就把它升格为新机制。

新六格虽然显示 N1 的 pure-decode wall +1.71%、专家字节 +3.57%，mixed 平均时间却下降，仍不足以支持上述策略：自然 EOS 内容/输出量不同，首个 AR 的 prefill/mixed 波动明显，N1 全程吞吐的两配对方向也相反。不能跨臂拼接阶段时间来推算一个尚未运行策略的收益。

因此不建议现在追加阶段控制器或启动新组，也不把旧重排、分组流水、尾块预留、静态 Q2048、已被否定的层间 cache 配额重分配换名重提。本次有用的新信息就是关闭 B 的虚构接口前提、确认 C 的真实但不适用的边界。root 已在推进的跨层压缩对照仍是当前最短的实际证据链；本报告不预判它的结果。

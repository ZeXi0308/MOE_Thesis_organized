# F：研究主张与冻结判决协议

2026-10-08。预算：首轮 1–2 工作日规模；最多 2 轮候选，最多 2 种预定混合负载。CPU 阶段先筛选；只有问题、独立价值、稳定收益全部成立才申请共享 GPU 锁做单卡闭环。

## 半页主张（先验假设，尚未成立）

具体问题：vLLM 的同一个异步输出服务把普通文本与带 top-20 logprobs 的逐 token 响应交给相同 CPU；一轮引擎批次中的 detokenization、logprob 对象构造和 JSON 序列化可能造成请求间的队头等待。拟用原生 OutputProcessor、原生 tokenizer、原生 OpenAIServingChat streaming generator 和 Pydantic serializer，由既有单卡时间记录驱动 engine 输出替身，在 HTTP 客户端测量轻请求交付。

最近邻缺口：现有按输出条数 chunk/yield 与 RFC #57479 的扁平对象、Rust 分块/有界通道已覆盖降低对象开销和合作式让出；它们不自动等于按实际 CPU 成本对异质请求分配服务。必须先验证此缺口是否有可测价值，不能把分块或公平轮转本身写成创新。

候选原则：只有在单条输出存在足够大的不可抢占成本时，才值得把其构造/序列化变成保留状态的有界工作单元，并按已消耗 CPU 时间实施跨请求公平、限制单请求积压。请求内顺序、完整 token/logprobs/格式不变。若原生 top-20 单 token 已足够小，优先检验简单 chunk/yield，停止复杂方案。

完整服务理由：可恢复处理减少共享 CPU 被重响应连续占有的时长，降低轻请求从引擎产出到客户端可见的等待；重请求获得保证的 CPU 服务，积压不转嫁为不完成。主张要求吞吐、重请求、CPU/内存共同过关。

证伪条件：实际产出速率不足以形成 CPU 瓶颈；瓶颈实为网络/慢读/GPU；固定配额或原生优化足够；候选相对最强简单方法轻请求尾延迟改善不足 20%，或吞吐下降超过 5%，或重请求饥饿/显著退化；复杂度没有独立方法价值。不得为找正收益调整请求内容、加大 logprobs、压缩时间或扩大负载扫描。

## 测量与执行约定

- 主指标：轻请求逐 token 的客户端可见交付延迟 P99（原始 trace 的 token ready 时间 → SSE JSON 完整到达且客户端解析后）。另报客户端 inter-token gap P99，避免混淆交付等待与生成间隔。
- 副指标：全部请求完成后的 request/s 与 token/s；重请求交付 P99 和完成延迟；全部完成/最大排空等待；server/client CPU seconds 与峰值 RSS。
- 重请求固定为 API top_logprobs=20 的 streaming；候选收益不得通过减少 metadata、token 或改变 SSE 切分获取。若针对性剖析未发现此路径瓶颈，本轮不改换超大响应。
- 两种预定负载：既有单卡低/中压与高压 trace 中各选择一条未缩放的原生固定策略记录；不重复/压缩产出时间、不增大并发。各 trace 固定 request-id 哈希选 25% 为重请求，其余轻请求；正常文本与 logprobs 响应 token 内容同源。具体 trace 与 CPU 核绑定在首项正式测量前写入 frozen_workloads.json。
- 原始 trace 若没有真实输出 token IDs：优先使用同模型既有真实文本输出；不得假称合成 logprobs 数值为真实 GPU 输出。模拟 top-20 的数值/候选 token 分布及缺少 GPU logprob 成本均须披露，这只形成 CPU 筛选结果。
- A 当前安装运行时，保留原生 chunk/yield 默认；B 在相同 CPU 核数上采用与已定位瓶颈匹配的固定配额 yield；C 最多两次候选修改。第一轮的初步剖析不构成正向论文证据。
- 正式运行使用 A/B/C 与 C/B/A 的配对反序重复；少量重复，必须报告所有运行，不挑最好一次。若问题不存在，停止，不为填满表格强行构造 candidate。
- 模型不加载到 GPU。CPU 回放数据不声称是 GPU 端到端服务结果。GPU 锁路径：/root/autodl-tmp/moe-research-gpu.lock。

## 首次定位后的 C1 冻结（未查看修正 GC 后结果）

一次 profile 把主要工作定位到 LogprobsProcessor 的 token 解码/字典对象与 chat logprobs 对象；serializer 非主因。先验证最小候选 C1：按真实 thread CPU 累积到 100 μs 后让出，保存原生 batch cursor，在下一 EngineCoreOutput 恢复；仍用原生单 token 对象和序列化。阈值固定，不扫描。API generator 和原生 collector 完全不变，FIFO与完整内容保持。它还不实现复杂 serializer 状态机，也**不自动构成论文方法贡献**：若单 token 原子边界已经足够，或此成本配额不胜固定配额，即停止，不把额外状态机堆上去。当前原生 collector 仅一个合并槽，token积压可变长；C1不声称实现严格字节/CPU积压界限。

回放修正：真实 API lifespan 自带 freeze_gc_heap；初版遗漏后产生大 GC 停顿。diagnostic_high 和 pilot_high 保留但排除性能判决；所有有效 A/B/C 都恢复相同的原生 freeze_gc_heap。该接线修复不属于候选迭代或方法收益。

正式测量前第二个计时接线修正：利用已经存在的 consumed_ns/client_visible_ns 分解发现，pilot_gc_high 的轻请求 P99主要位于 collector→client（约82–114 ms），而 engine→collector约17–18 ms。客户端为离线校验保留每条嵌套解析对象，制造了不必要的大量长期存活GC对象。正式客户端仍逐条真实JSON解析后立刻记可见时间，但只保留原始完整SSE字节用于计时后校验，不保留嵌套对象。旧pilot_gc_high同样仅保留为无效接线诊断，不作为性能收益。正式组保留B1和B8两种固定配额，避免从受客户端影响的pilot事后挑基线；序列 A/B1/B8/C/C/B8/B1/A，两负载各2次，按最强B比较。不会新增负载或阈值。

## 主时钟口径澄清与双口径裁决

初始文字写的是原始trace计划ready时间。首次诊断之前的实现（run_replay.py）已同时记录计划ready与producer实际发送前时间，并将后者用作 `light_delivery_p99_ms`，前者用作 `light_planned_delivery_p99_ms`；原因是独立producer仍受OS调度影响，需要区分实际输出后等待和未按期产出的延误。这一实现先于任何性能结果，但协议文字未及时同步，现明确保留此偏差。最终不能仅依赖实现口径：**原协议计划时钟与实际发送时钟并列裁决**。高压两者均未达到20%门槛；中压时序污染也不会因切换起点被消除，仍标记不可做完整配对推断。

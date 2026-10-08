# CPU KV 恢复的尾部空间预留：最小直接实验

2026-10-02。状态：六臂全部COMPLETE（22937/controller71905）。预留动作成立，但完整成本与质量不支持继续；结果见`KV_RESERVE_RESULTS.md`。下文为保留实验协议。

## 已确认的因果

单格只读诊断 `results_kv_restore_diag_r01/00_swap16/restore_wait_diagnostic.json` 对齐通过。请求0044在step51异步恢复704个已计算token，占44块；同一步的新prefill耗尽剩余容量。step52数据已到达，恢复计算剩余3个token需要第45块，却因free=0分配失败；step53–55仍失败，step57才成功新增一块。不能把这一段等待归为DMA慢。诊断输出1416token，与原生恢复r02一致；完整路径/输出逐项核对另存诊断简报。

## 模型与动作

设已知可恢复前缀为P，恢复后首次推进需要的已知尾部为t，块长B。所需尾部容量为

`Δb = ceil((P+t)/B) − ceil(P/B)`。

只满足前缀容量会产生“KV已驻留，但请求不可执行”的状态。当前P=704、t=3、B=16，Δb=1。最小动作在原生异步恢复分配时一并占用这1块，总GPU块数不增加。实现借用原生lookahead容量字段，但不增加已计算长度、不增加CPU加载长度、不改变模型token、采样或真实输出契约。仅在原生CPU connector、单组full attention、无speculative、PREEMPTED、外部命中且前缀块对齐、已知尾部一块可容纳等条件成立时启用。其它情况走原生路径。

它只保证该次已知尾部的容量条件，不保证立即调度、不保证未来输出长度和全程SLO。预留也会阻挡新请求或把抢占转移给同伴，因此必须按完整请求与全部传输计费。它是最简单的容量修复候选，尚非独立论文创新。

## 直接对照

- 同22937 GPU（45495已有其他组在运行），同一公共运行代码，整组共享物理锁；三臂正反序：S16/H10/P16/P16/H10/S16。
- S16：原生CPU KV1GiB，cap16；P16：同S16加上述单块预留；H10：无CPU KV、cap10的已跑强尾延迟基线。
- 固定expert24、GPU KV1GiB/512块、Q2048、engine最大4096、HND、eager；总GPU KV与专家资源相同，CPU资源差异显式报告。
- 每臂fresh engine；旧训练warmup16请求、budget512、固定32输出；measurement原source32–47完整prompt、自然EOS、最大512输出、16同时到达。每请求原有独立cache salt保留，排除跨请求前缀复用。
- 无host诊断probe，策略只保留有限动作/计数receipt；计入wrapper/预留引起的全部运行代价。所有六格与失败保留，不能挑最好一次。所有性能结论只在本组三臂之间比较，不跨22937/45495直接比较时间。

## 可证伪预测和裁决

首先核对原生S路径是否重现，并验证P实际获得额外尾块但CPU加载前缀未增加。若策略未触发或原生API不兼容，记实现/适用性未验证，不称性能NO-GO。

若预留触发，预期已观察的0044 ready后缺块等待减少；同时检查所有victim的缺席、恢复跨度、后续输出和新抢占，防止只改善指定请求。报告完整drained wall、实际token/s、flow/TTFT、每请求最大gap分布、EOS/答案/完整输出差异、实际保存/加载和专家搬运、GPU峰值与CPU pinned内存。输出若改变，只称实际episode对比，不称等工作量加速。

只有请求级收益在两次方向一致、质量无已观测回退、完整成本与强基线具有有用取舍，才进入新请求集与真实容量不足运行域。若只修复局部等待而整体无收益，停止当前单块方案；不立即堆新预测器。即使本组成功，也还缺独立方法新颖性、泛化与规模证据。

## 已有机制与贡献边界

本组启动后的四次定向核查发现，vLLM官方[PR #44560](https://github.com/vllm-project/vllm/pull/44560/files)已修复多个异步KV加载互相挤占已知尾部空间的问题，并引入`_inflight_prefill_reserved_blocks`。因此“异步恢复需要预留继续执行的空间”是已有机制。所读[当前scheduler](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/sched/scheduler.py)只在`load_kv_async`分支计算该reserve；根据源码与本格观测推断，普通新prefill仍可能消费恢复者的尾部空间。未实测最新upstream的完整路径，不能声称最新版本仍有同一故障。

[Full-ISL admission PR #37307](https://github.com/vllm-project/vllm/pull/37307)检查整个已知序列当下能否装入，不能等同于持续拥有全部尾块；当前诊断中的`full_sequence_must_fit=True`仍出现了上述阻塞。P16是对这一个竞争方向进行物理预留的最小修补。即使阳性，单凭此动作的新颖性仍弱，不能据此宣称已完成CCF论文方法。

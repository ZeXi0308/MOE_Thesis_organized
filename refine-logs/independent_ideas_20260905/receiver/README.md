# Receiver：exposed return 的结构条件界限准备

当前为 **代码准备，真实自然 EP trace 与 8 GPU Gate 均 UNRUN**。没有 Receiver
Controller、codec 或新的调度动作。4 个 CPU fixture 明确为 synthetic，只验证 DAG 会计。
科学权威仍是 `docs/ideas/receiver_aware/README.md` 与
`cpr_ranklane/EP_Return_Path_8xA100存在性Gate.md` §6；本目录不修改其裁决。

旧 `run_multirank_rr_census.py` 可复用 message/request/layer/token 绑定、消息守恒、
NVTX 和 topology 字段，但 barrier、随机专家和 rank-local clock 不满足自然 serving。
旧 `analyze_multi_moe_inference.py` 可参考全层 census 与完整分母，单卡 MoE duration
不能转成 EP return fraction。这两个入口都未提供可直接复用的因果 DAG replay。

## 最小输入接口

`analyze_exposed_return.py --trace trace.json --output-dir <全新目录>` 输出 analysis.json
与 report.md。`NOT_IDENTIFIED` 时退出码 2，不计算任何删除界限；识别到的结构返回 0，
但 `scientific_go` 始终 false。本目录没有可引用的真实 8 GPU cell 结果。

JSON 使用 `schema=ep-return-explicit-dag-v1`，时间单位统一为 us：

- `provenance=natural_ep_serving`、`clock_domain=unified_nsys_cupti`；环境包含
  `gpu_count=8`、`optimized_backend=true`、具体 `transport`，以及
  `gpu_model/backend_name/backend_version/hardware_backend_compatibility_verified`。
  compatibility 标志是外部预检义务，不是分析器检测过硬件的证明。
- `coverage` 必须明确声明 `causal_dependencies/resource_serialization/exogenous_releases/
  request_completions/all_moe_layers/message_identity` 全部覆盖。这些声明只是 exporter
  的输入义务；分析器不能独立证明其完整性，也不能证明发生过真实采集。
- `nodes`：唯一 `id`，`start_us/end_us`，`request_ids`，`resources`，`return_path`，
  `phase`。return service 的 phase 限于 `return_a2a/receiver_unpack/receiver_combine`，
  并提供 `message_ids`。共享 service 可关联多个 request，不重复计时。
- 每个 node 必须提供语义 `release={time_us,kind,evidence,...}`。`request_arrival`
  要绑定 request ID 且数值等于其到达时间；`external_event` 必须有外部事件来源，
  明确 `action_independent=true`。**禁止用 observed start 当 release**；资源阻塞、
  前序 completion、CPU launch 依赖应表示为边或 service，不能钉死为外生时间。
- `dependency_edges` 提供 `source/target/evidence`，需要真正的 event wait、数据依赖、
  collective/消息匹配或 runtime join 证据。仅 kernel chronology 不足以推出因果边。
- `resource_sequences` 提供 `resource_id/node_ids/evidence`，明确同一实际串行资源
  的顺序，覆盖每个 node 的全部 `resources`。工具据这些显式序列加入 serialization
  edges。不能把所有同 GPU kernel 强行串行，也不能遗漏 stream/link/CPU queue 约束。
- `requests` 提供 `request_id/arrival_us/completion_node/completion_us` 以及逐 token
  的 `token_index/completion_node/completion_us`。每个 token 必须有到最终完成节点的
  因果路径、相邻 token 的推进依赖，以及到达约束。输入要求这一批请求全部完成；
  无完结映射的截断 trace 返回 NOT_IDENTIFIED，不静默丢失请求。
- 可选 `negative_control_node_ids` 必须是在运行前指定的非 return 节点。工具仅报告
  删除后的 completion shift 是否为零，不把选中节点自动认证为非关键路径。

**真实 Nsight/CUPTI + serving identity 到上述接口的 exporter 尚未接通。** 要获取
原始依赖、all-layer census、message 守恒、资源等待以及 release 来源，仍需在实际
backend 打点；不能把旧 rank-local half trace 自动转换后声称问题已测。

## 唯一实验与会计

三臂是同一完整输入 DAG 的结构分析：A 保留全部 service；B 将明确的 return service
duration 置零；C 将预先指定的 noncritical non-return service 置零作为负控。
所有 release、非目标 service、因果边和实际资源顺序保留。

`start(v)=max(exogenous_release(v), finish(all predecessors))`，然后加 service duration。
首先要求 A 在统一时钟量化容差内重建每个 node 和 request/token 的观测时刻；任何未解释
gap 都是 NOT_IDENTIFIED，不用临时增加 release 修平。B 的请求完成差除以该请求从
arrival 到 completion 的完整 latency；**不相加 return spans，不跨层累加 saving**。

结果严格称 **fixed-order structural conditional bound**。baseline reconstruction
只说明相容，不证明因果完备；固定资源顺序和 duration 也未模拟动作改变 contention、
route、batch、collective progress 或后续调度。因此它不是真实系统 Oracle、可部署收益
或科学 GO。输入缺边时既不能据此判正，也不能把所得零值用于判死。

最强基线先冻结**硬件 × 优化 backend 版本兼容性**。不能默认 8×A100 支持当前
[DeepEP main/V2](https://github.com/deepseek-ai/DeepEP)；使用 A100 时需核实兼容的
backend/commit，或换到该 backend 官方支持的 GPU，并记录 CUDA、Torch、NCCL
版本及 transport。DAG 工具本身不依赖 DeepEP，真实自然 trace 入口仍未实现。
通过预检后，采用 8 张独立 GPU 上合法最佳 EP/TP、既有 overlap/fusion/zero-copy 开启的
unmodified optimized backend。自然 continuous arrivals、decode-heavy 与 mixed workload，
全部 MoE 层和最终 request completion 必须闭合。不得用关闭 overlap、人工 barrier、
单卡或 synthetic trace 替代。单节点结果不能外推 RDMA。当前复活条件是取得这种真实
trace，并实测 return 删除在完整请求中的作用；之后才讨论协议 §6 的模型/负载重复。

```bash
.venv/bin/python -m unittest discover \
  -s refine-logs/independent_ideas_20260905/receiver -p 'test_*.py' -v
```

# RIC-Clean-v2 Service LUT Phase 2 冻结补充协议

状态：**FROZEN REVISION 2 FOR SERVICE-LUT IMPLEMENTATION / NO SCIENTIFIC RESULT**  
冻结日期：2026-07-23

本文件只细化 clean-v2 calibration service LUT。它不修改已冻结的
calibration data/route、N1 gate 或 sealed 状态；实现、Phase-4 签字与产物必须
同时绑定 base protocol、config、Calibration Route addendum 和本文件的 SHA-256。
本阶段只生成 N1 的 calibration input，`scientific_result=false`。

## 1. 固定输入、输出与消费边界

- 每模型只消费已签字的
  `clean_v2/routes/calibration/{olmoe,llmjp}`；必须验证 route trace、parity、
  placement、capture metadata、embedded producer signoff 的 self/file SHA 闭包。
- 路由输入仍是固定 calibration manifest 的 64 requests × 128 tokens。
  禁止读 sealed，禁止用 logits 重算结果取代 route artifact 中的 native tuple。
- 模型、tokenizer、model-tree SHA、runtime identity 与 RTX 5090 必须与 route
  capture 一致。开始前及结束后不得有外部 GPU compute process。
- 每模型输出固定为 `clean_v2/service_lut/calibration/<model>`。formal API
  必须在模型加载前使用每模型 `O_EXCL` one-shot ledger 预留；失败
  fail-closed，禁止删 ledger 追结果重跑。
- 不接受可替换 input/output/model/warmup/trial 的 formal CLI。产物只能原子
  rename 发布，已存在则 BLOCKED。

## 2. 测量对象与 route-derived 选择

### 2.1 主表覆盖

- 每模型的 layer set 必须精确等于 route parity 中的
  `selected_replay_layers`，且数量等于 config 的
  `selected_layer_count_per_model=4`。
- 主 expert service surface 为全部
  `(model_key, selected_replay_layer, expert_id)`：OLMoE 为 `4×64`，LLM-jp 为
  `4×32`。不得用四个选中 expert 的 max/mean 代替。
- 每个 `(model,layer,expert)` 的 row-1 输入必须来自该 expert 在固定
  route trace 中实际收到的 token activation。候选 contribution 按完整冻结
  identity
  `(request_id,forward_id,batch_id,phase,decode_step,layer_id,token_id,token_block_id,topk_slot,expert_id,sender_rank,receiver_rank,epoch)`
  做 canonical JSON，取 `sha256(identity)` 最小者；该规则不读 timing/energy。
- producer 重放对应 request 以取 activation 时，必须先通过与 Route
  addendum 相同的 raw-logit/native ordered-tuple/output parity，并证明选中
  `(token,slot,expert)` 与 route row 精确相同。不得改选另一个更快的
  activation。
- 任一 `(layer,expert)` 没有实际 routed activation 时直接
  `BLOCKED_ROUTE_SPECIFIC_SERVICE_COVERAGE`；禁止合成输入、跨 expert/layer
  借样或跳过后改分母。

### 2.2 被测组件

GPU 张量在进入计时区间前全部预分配且已在 device 上；计时区间内
禁止 H2D、文件 I/O、Python 选样或重新 tokenization。测量以持久的单条
CUDA stream 顺序执行。

1. `expert_execution_route_specific_row1`：选中 expert 对其自身 row-1
   activation 的原生 forward，按 `(model,layer,expert)` 记录。它只用于
   contribution `expert_ready_offset_us`，不声称是真实 batched expert
   serving latency。
2. `sender_pack_route_specific_row1`：对该 expert 实际 weighted row-1 output
   调用冻结 pack primitive，按 `(model,layer,expert)` 记录。
3. `receiver_unpack_route_specific_row1`：对上一步的实际 packed tensor
   调用冻结 unpack primitive，按 `(model,layer,expert)` 记录。
4. `canonical_combine_once_per_join`：对实际 top-k siblings 按冻结 slot
   顺序做 canonical reduction，每个 `(model,layer)` 从全部完整 join 中按
   `sha256(full_join_identity)` 排序取前 32 个，每 join 各测 30 次。N1
   使用该 layer 全部 `32×30` measured trials 的总体 median，且每
   join 只计费一次。
5. `shared_cut_analytic_l2_proxy`：不做网络实测。每 contribution 的
   bytes 为实测 weighted output tensor payload + config 的 16-byte descriptor +
   16-byte alignment padding；按 200 Gb/s 主口径计算，100/400 Gb/s 仅作
   sensitivity。必须标 `ANALYTIC_NETWORK_L2_PROXY_NOT_RDMA`。

可以另测 rows `{1,2,4,8,16,32,64}` 的 expert/pack/unpack batching surface，但只能
标 `BATCHING_SENSITIVITY_NOT_N1_PRIMARY`。四个 outcome-blind expert 的 conservative max
同样只是 sensitivity，不能回填主表。

## 3. 时间与能耗口径

### 3.1 GPU 时间主口径

- 每个 GPU 测量点固定 10 次 warmup + 30 次 measured trials；两类都逐次
  写入 raw 表，用 `phase={warmup,measured}` 分开。warmup 不进 summary。
- 每次同一 invocation 同时记录：
  - `cuda_event_us`：同 stream start/end CUDA event，end event synchronize；
  - `wall_sync_us`：从 start event record 前到 end synchronize 后的
    `perf_counter_ns` wall time。
- N1 主 LUT 值只取 30 个 `cuda_event_us` 的 median。`wall_sync_us`、
  CUDA-event p95/max 是运行时与 jitter sensitivity，不得代替主值。
- 每次 trial 必须 `0 < cuda_event_us <= wall_sync_us`，且每点
  `p95(cuda_event_us) / median(cuda_event_us) <= 2.0`。否则该模型整个 LUT
  `BLOCKED_UNSTABLE_SERVICE_POINT`，不得丢弃 outlier 或只重测该点。

### 3.2 Host 与 energy 辅助口径

- 若后续 LUT 包含 host codec/control primitive，其主口径是 10 warmups + 30
  measured `perf_counter_ns`；空 harness 单列 raw，禁止负数 clamp。host
  control 不进入本轮 N1 的四段 data-path service tuple。
- Energy 只是辅助诊断，不进 N1 LUT、GO gate 或 Energy-SLO claim。对每个
  `(model,layer)` 取 `sha256(model_revision||layer_id||expert_id)` 最小的 expert
  为 canonical expert 点，combine 点取本协议排名第一的 join；在预分配
  输入上固定
  16,384 次 back-to-back operation，同时测等时长 idle window。优先记录
  NVML total-energy counter 前后差；不支持时只可标
  `ENERGY_UNAVAILABLE_NOT_A_LUT_FAILURE`，不得伪造数值。
- 若 energy 可用，raw 必须保存 active/idle duration、counter 前后值、
  GPU clocks/power limits/temperature；只报
  `(active_j-idle_w*active_s)/16384` sensitivity。净能量非有限或 `<=0`
  时标 `ENERGY_INVALID_AUXILIARY`，但不改变 timing LUT 的 valid/invalid。
- H2D 可作独立 sensitivity，source 必须精确为 `H2D_NOT_RDMA`；它不是
  sender-pack、shared-cut 或 receiver-unpack，不进 N1 service tuple，不得与
  analytic network 相加后称为 RDMA/NCCL 实测。

## 4. Raw trial、summary 与会计

`service_lut_raw.csv` 每行至少包含：

`model_key, model_revision, layer_id, expert_id, component, route_identity_sha256,
route_tuple_sha256, input_tensor_sha256, input_shape, input_dtype, output_tensor_descriptor_sha256,
rows, phase, trial_index, cuda_event_us, wall_sync_us, energy_status, source,
gpu_uuid, stream_id, producer_source_sha256` 。

`service_lut.csv` 必须由 raw 重算，不接受 producer 自报布尔值。每主键保存
measured count、median/p95/max CUDA time、median wall time、稳定性比率、payload
descriptor 与 source tag。必须验证：

- 每个主 `(model,layer,expert,component)` 恰好 10 warmup + 30 measured；
- 主 expert/pack/unpack 键集恰好等于 route-derived 的全 Cartesian surface；
- 每个 layer 恰好 32 个 combine joins，每 join 恰好 30 measured；
- payload bytes = tensor elements × element size，transport bytes = payload + descriptor +
  frozen padding；
- raw/summary/metadata/route/placement/parity/data/model/runtime/review/signoff 的 self/file
  SHA 全部闭合，数值禁止 NaN/Inf。

任一行缺失、重复、键越界、source 混用或会计不等时整个模型的 LUT
invalid；禁止局部删行、补样、插值或用另一模型的值代替。

## 5. N1 唯一允许的 lookup

N1 对每个 route contribution 使用下列确定映射：

- `expert_ready_offset_us = median(expert_execution_route_specific_row1[model,layer,expert])`；
- `sender_pack_us = median(sender_pack_route_specific_row1[model,layer,expert])`；
- `shared_cut_us = aligned_transport_bytes * 8 / (200 * 1000)`；
- `receiver_unpack_us = median(receiver_unpack_route_specific_row1[model,layer,expert])`；
- `canonical_combine_us = pooled median(canonical_combine_once_per_join[model,layer])`，
  一个 join 仅在 top-k 全部到达后计费一次。

N1 必须把上述五项、payload/transport bytes 与完整 LUT fingerprint 写进
immutable task/service fingerprint。Matched worlds、S/B/R0 及全部 policy arms 必须
逐 contribution 使用完全相同的 lookup 结果。禁止 policy-dependent service、
按 outcome 选 median/p95、临时插值或将 Energy/H2D 加入主口径。

100/400 Gb/s、CUDA-event p95 与 batching surface 只能在主 N1 完成后做
sensitivity；必须复用同一 route、arrival、placement 与 task identity，只替换协议
明确允许的 service 口径。

## 6. Valid / invalid 门禁

Service LUT 没有科学 `GO`；最高状态是
`VALID_CALIBRATION_INPUT_FOR_N1`。只有以下全部成立才可写该状态：

1. data/route/model/runtime/review/signoff 闭包与固定路径全通过；
2. 两模型各自完整的 4-layer × all-expert 主表和 combine 表存在；
3. route-specific activation/native tuple/parity 全精确；
4. raw counts、Cartesian keys、字节账本、时间门和 summary replay 全通过；
5. 无外部 GPU process、无路径/模型树漂移，产物原子发布。

任一主口径失败时状态必须是 `BLOCKED_<reason>`，不得输出部分
LUT 供 N1 消费。Energy 不可用或辅助 energy invalid 只降低 energy
evidence，不得伪装成 timing LUT 失败，也不得换成 Energy-SLO 结论。

## 7. Phase 4 实现验收用例

1. 删除一个 `(layer,expert)` route activation，必须 coverage BLOCKED；
2. 交换两个 expert 的 activation 或 tuple hash，必须 identity/parity BLOCKED；
3. 重复一行并删除另一行保持总行数，必须 Cartesian census BLOCKED；
4. 将 H2D source 标成 RDMA 或加入 shared-cut，必须 accounting BLOCKED；
5. 篡改一个 raw trial 后 summary replay 必须 hash/value BLOCKED；
6. 注入 NaN/Inf、零/负时间或 `p95/median>2`，必须 unstable BLOCKED；
7. 第二次 formal API 调用必须在模型加载前被 ledger BLOCKED；
8. 签字在测量期间改变，必须在原子发布前 BLOCKED；
9. N1 fixture 对同一 contribution 在 B/R0 中读到不同 service fingerprint，
   必须 BLOCKED；
10. energy/H2D 值不得出现在 N1 primary service tuple。

## 8. Revision 2 规范闭包（SUPERSEDES 本文较早的歧义措辞）

本节是本文件的最终规范。若第 1--7 节与本节冲突，以本节为准；实现与
Phase-4 签字必须绑定本文件 Revision 2 的完整文件 SHA，而不能绑定先前版本。

### 8.1 唯一 primitive 与 source tag

所有 GPU primitive 都在 `torch.inference_mode()`、模型 `eval()`、BF16 原生
hidden dtype、固定持久 CUDA stream 上运行。输入与显式 staging buffer 必须在计时前
位于 device；primitive 自身按下列定义产生 output 或写入 output buffer，其原生
allocation/copy 成本不得另行剔除。

- expert input 是 native MoE block **forward pre-hook 的第一个 hidden-state tensor**，
  reshape 为 `[128, hidden]` 后取 route row 的 `token_position`，形成 `[1, hidden]`；
  它是 weighting 前的 expert input。expert output 是
  `first_tensor(moe.experts[expert_id](input))`，不得调用另一 expert 或 fused surrogate。
- weighted contribution 是 expert output 乘 route artifact 的 effective BF16
  `route_weight`；禁止使用 `route_weight_fp32_precast` 替代。
- `sender_pack_route_specific_row1`：计时前创建固定
  `reverse_index=torch.arange(rows-1,-1,-1,device=device,dtype=torch.int64)`；计时调用
  `torch.index_select(weighted_contribution,0,reverse_index)`，返回值必须新分配且不得与
  input alias。row-1 时 permutation 虽为 identity，仍必须执行该 `index_select`。
- `receiver_unpack_route_specific_row1`：计时前创建与 packed tensor 同 shape/dtype 的
  persistent `unpacked` buffer；计时调用
  `unpacked.index_copy_(0,reverse_index,packed)`。每次 invocation 覆盖全部 rows，禁止
  clone/contiguous/no-op 替代。
- `canonical_combine_once_per_join`：siblings 固定为
  `[1,top_k,hidden]` BF16 weighted contributions，按 `topk_slot=0..top_k-1` 排列；计时前
  创建 `[1,hidden]` persistent accumulator，计时先
  `accumulator.copy_(siblings[:,0,:])`，再按 slot 递增执行
  `torch.add(accumulator,siblings[:,slot,:],out=accumulator)`。禁止 tree reduction、
  FP32 accumulation 或重排。

前三类 GPU component 的 `source` 精确为 config allowlist 中的
`measured_5090_cuda`。analytic cut 的 `source=analytic_network`，并另存
`evidence_boundary=ANALYTIC_NETWORK_L2_PROXY_NOT_RDMA`。可选 H2D 的
`source=measured_5090_h2d_not_rdma`，另存 `evidence_boundary=H2D_NOT_RDMA`。
禁止把 evidence boundary 字符串塞进 `source` 绕过 config allowlist。

### 8.2 Re-forward 与 activation/route identity

模型重放严格使用与 route capture 相同的 tokenizer、`batch_size=1`、恰 128 tokens、
`use_cache=False`、`return_dict=True`、BF16、`eval()` 与 `torch.inference_mode()`。
每个 request 恰重放一次并同时捕获所有 selected layers；禁止为测得更快 input 重放
多次择优。

重放必须再次观察 native `aten.topk`，并对该 request 的全部 selected layers 验证：
raw gate-logit identity、ordered expert tuple、effective BF16 weight tuple及 native output
reconstruction parity。对每个选中 contribution，hook tensor 的 flattened
`token_position` 行、artifact `topk_slot`、native expert/weight必须逐项相等；重复捕获若
发生，其 input tensor SHA 必须完全一致，否则 BLOCKED。

`route_identity_sha256` 使用第 2.1 节 13 字段、字段名与值构成的 strict canonical JSON
object（UTF-8、key lexicographic sort、无空白、禁止 NaN/duplicate key）后取 SHA-256。
consumer 必须从 route row 独立重算 expert placement、sender rank、receiver rank、
token identity、epoch 与该 hash，不能相信 producer 自报。

`full_join_identity` 精确为：
`(model_key,model_revision,data_manifest_sha256,placement_manifest_sha256,request_id,
forward_id,batch_id,phase,decode_step,layer_id,token_id,token_block_id,receiver_rank,epoch)`。
编码与 route identity 相同。一个完整 join 必须恰有 `top_k` 个互异 slot `0..top_k-1`，
每个 sibling 的 expert、sender、weight 与 signed route row 相等。siblings 由同一次 native
re-forward 的 pre-hook hidden row分别调用各自 native expert，再乘各自 effective BF16
weight构造；禁止拼接来自不同 forward 的 sibling。

### 8.3 统计、执行顺序与 raw census

median 对偶数 `n` 定义为排序后第 `n/2` 与 `n/2+1` 个值的算术均值。p95 定义为
nearest-rank order statistic：排序后第 `ceil(0.95*n)` 个值；30 measured trials 因而取
第 29 个值。max 为排序后最后一个值。稳定性 gate 只用 measured trials，但 warmup 与
measured 的每次 invocation 均须满足有限、`0 < cuda_event_us <= wall_sync_us`；不设
额外浮点容差。

测点禁止按 expert 连续跑完。对每个 component 和 phase，先按
`sha256(canonical_json({model_revision,layer_id,expert_id_or_minus1,
join_identity_sha256_or_empty,component}))` 排序全部 point；每个 round 轮转起点
`round_index % point_count`，依轮转顺序让每个 point 恰执行一次。先完成 10 个 warmup
round，再完成 30 个 measured round；raw 保存全局单调 `execution_ordinal`。combine 的
每个 join 同样恰有 10 warmup + 30 measured，raw 行的 `expert_id=-1` 且必须有非空
`join_identity_sha256`；expert/pack/unpack 行的 `join_identity_sha256` 为空且
`expert_id>=0`。

主 raw census 精确为：
`3 * selected_layer_expert_cells * 40 + selected_layers * 32 * 40` 每模型。
combine summary 同时保存每个 join 的 summary row，及每 layer 一个
`component=canonical_combine_pooled_layer` row；pooled 主值只从该 layer 的
`32*30=960` measured CUDA samples计算，不混入 warmup或 per-join summary 值。

### 8.4 Tensor、descriptor 与 CSV 可重放定义

所有 tensor SHA 必须直接调用 reviewed `native_route_core.tensor_sha256`；其 preimage 为
原 shape、dtype、stride 的 ASCII 表示，各字段以 NUL 分隔，随后连接 contiguous CPU
`uint8` raw bytes。不得另写第二套 tensor hash。

raw 不只存 descriptor hash，还必须存可独立重算的实体字段：
`tensor_numel,tensor_element_size_bytes,payload_bytes,descriptor_bytes,
alignment_boundary_bytes,alignment_padding_bytes,transport_bytes`。其中
`descriptor_bytes=16`、`alignment_boundary_bytes=16`，其余严格按 config 公式重算；
descriptor hash 是上述字段 strict canonical JSON 的 SHA-256。

CSV 固定为 UTF-8、LF、`csv.QUOTE_MINIMAL`、固定协议列序；整数用 base-10，浮点用
Python `format(value,'.17g')`，空 optional field编码为空字符串，结构字段用本协议 strict
canonical JSON 紧凑编码。CSV 不自哈希；post-run metadata 单向绑定 raw/summary file
SHA，validator 必须重新解析 raw、重算 summary 与 descriptor 后才接受 metadata。

### 8.5 N1 会计、energy 与失败恢复

本文件明确 **SUPERSEDES** N1 addendum 中“每 contribution 四段 service / receiver-apply”
的旧措辞。expert execution 只形成 ready offset；每 contribution 只经过
`pack -> shared-cut -> unpack` 三个服务段；top-k 全到后每 join 只 enqueue并执行一次
canonical combine。因此 full drain 恒等式是 base protocol 的 `3N+J`，任何
per-contribution receiver-apply 或 expert-service 二次计费均 BLOCKED。

Energy 必须写独立 `energy_auxiliary.jsonl`，不得混入主 raw census。只有 NVML
total-energy counter feature probe 成功且 unit/resolution 可记录时才执行；每 layer 的
active block 与紧邻 idle window 的次序由 layer hash 奇偶 counterbalance，固定 16,384 次
Python-loop同 stream launch并在窗口边界 synchronize。`idle_w=idle_delta_j/idle_duration_s`，
`net_j_per_op=(active_delta_j-idle_w*active_duration_s)/16384`；counter wrap、非单调、
分辨率不足或净值非正均标 auxiliary invalid。禁止用瞬时 power 或 TDP 推算。

ledger 一旦建立永久保留。任何 formal 失败都另写不可覆盖的 BLOCKED attempt receipt；
协议/代码修复只能使用新的 addendum revision、output namespace、ledger namespace与全新
Phase-4 签字，不能删除/复用旧 ledger。两模型组成一个 LUT release：不得把不同协议
revision或不同 reviewed-source manifest 的 per-model产物拼接；一个模型失败时整个
release不可供 N1 消费。

config 的
`formal_artifact_compaction.full_four_stage_action_trace_required_in_acceptance_fixture`
字段名是 **SUPERSEDED historical label only**，不得解释为每 contribution 的第四段
`receiver-apply`。在不修改已被 data/route 产物绑定的 config bytes 的前提下，其 Revision 2
唯一语义是：acceptance fixture 必须保留四类 component trace，即 per-contribution
pack/shared-cut/unpack 与 once-per-join combine；完成数仍严格为 config 同节的
`3*num_contributions+num_joins`。

### 8.6 固定 schema、签字 DAG 与双模型 release

formal primary release 不生成 batching/H2D sensitivity 行；这些敏感性必须由未来单独
协议与 artifact 承载，不能改变本轮主 raw census。主 raw CSV 的**完整固定列序**为：

`model_key,model_revision,layer_id,expert_id,component,route_identity_sha256,join_identity_sha256,route_tuple_sha256,input_tensor_sha256,input_shape,input_dtype,output_tensor_descriptor_sha256,tensor_numel,tensor_element_size_bytes,payload_bytes,descriptor_bytes,alignment_boundary_bytes,alignment_padding_bytes,transport_bytes,rows,phase,trial_index,execution_ordinal,cuda_event_us,wall_sync_us,energy_status,source,evidence_boundary,gpu_uuid,stream_id,producer_source_sha256`。

主 summary CSV 的**完整固定列序**为：

`model_key,model_revision,layer_id,expert_id,component,route_identity_sha256,join_identity_sha256,rows,measured_count,median_cuda_event_us,p95_cuda_event_us,max_cuda_event_us,median_wall_sync_us,stability_ratio,payload_bytes,descriptor_bytes,alignment_boundary_bytes,alignment_padding_bytes,transport_bytes,output_tensor_descriptor_sha256,source,evidence_boundary,producer_source_sha256`。

不存在的 optional identity 使用空字符串；combine 的 `expert_id=-1`。同一 summary group
的 descriptor/bytes/source 必须唯一；pooled combine row 的 join identity 为空，但其 32
个 join 的 descriptor/bytes必须完全相同，否则 BLOCKED。`energy_status` 在主 timing raw
固定为 `AUXILIARY_SEPARATE`。

全局执行 component 顺序固定为
`expert_execution_route_specific_row1 -> sender_pack_route_specific_row1 ->
receiver_unpack_route_specific_row1 -> canonical_combine_once_per_join`；每 component 内
按 §8.3 完成 warmup rounds 再 measured rounds。`execution_ordinal` 从 0 开始跨全部
component/phase严格递增。协议不要求人为 cooldown；每个 round 边界必须记录 GPU
clock、temperature、power limit 到 `execution_environment.jsonl`，这些字段只作热漂移
诊断，不允许据此删 trial。

正式 provenance 是下列不可逆 DAG：

1. **Pre-run Phase-4 signoff**：只绑定 base/config/Route addendum/本 Revision 2 文件
   SHA，data manifest self/file 与 data signoff，route metadata/trace/parity/placement/
   embedded route signoff self/file，model tree与 expected runtime，reviewed-source manifest
   self/file，review/test report，固定 input/output/ledger/receipt 路径以及本节 measurement
   contract/schema/census。它必须在测量前生成，**不得**绑定未来 raw/summary 的实际 SHA。
   固定 review evidence 路径为 `clean_v2/review/RIC_Clean_v2_ServiceLUT_CodeReview.md`、
   `clean_v2/review/RIC_Clean_v2_ServiceLUT_TestReport.json`、
   `clean_v2/review/reviewed_source_manifest_service_lut.json`；per-model signoff 固定为
   `clean_v2/review/signoff_service_lut_<model_key>.json`，release-finalizer signoff 固定为
   `clean_v2/review/signoff_service_lut_release.json`。
2. **Per-model reservation**：固定为
   `clean_v2/state/service_lut_calibration_<model_key>_consumption.json`，在 model load 前用
   `O_EXCL|O_NOFOLLOW` 创建，反向绑定 pre-run signoff self/file、data/route/model/source
   与 output namespace。实际 ledger self/file SHA 只由 post-run metadata 绑定。
3. **Post-run metadata/automatic receipt**：每模型 atomic publish 前，重新验证 signoff、
   reviewed source、route、data与model tree未漂移；metadata 绑定 signoff self/file、ledger
   self/file、raw/summary/environment/energy-auxiliary file SHA、actual runtime/GPU、source
   closure并 self-hash。automatic receipt 不是人工 Phase-4 签字，禁止事后补签。
4. **BLOCKED receipt**：固定为
   `clean_v2/service_lut/attempts/<model_key>.blocked.json`，在 reservation 后任一可捕获失败
   时以 `O_EXCL` 写入，反向绑定 ledger、pre-run signoff、异常类别与已发布文件清单；不得
   覆盖。进程被强杀而来不及写 receipt 时，永久 ledger 本身仍使 attempt fail-closed。
5. **N1 consumer**：独立解析 raw、重算 census/identity/descriptor/summary/stability，验证
   metadata self/file、raw/summary SHA 与本 Revision 2 SHA；不得相信 automatic receipt
   的 valid 布尔，也不得把它冒充 Phase-4 签字。

双模型 release 只能由无参数、已审查的 fixed-path finalizer 创建。它必须先以
`O_EXCL|O_NOFOLLOW` 创建
`clean_v2/state/service_lut_calibration_release_finalization.json`，独立执行上述 N1-consumer
级验证，再原子发布不可覆盖的
`clean_v2/service_lut/calibration/release_manifest.json`。release manifest schema 固定为
`ric-clean-v2-service-lut-release-v1`，`status=VALID_CALIBRATION_INPUT_FOR_N1`、
`scientific_result=false`，并绑定两个模型各自 metadata self/file、raw/summary file SHA、
相同 Revision 2/config/base/Route-addendum/reviewed-source manifest self/file SHA，以及
finalizer source/review/signoff SHA；最后 self-hash。缺任一模型、revision/source不相同或
任一 consumer validation失败时不得发布 manifest。N1 必须以该 manifest 为唯一 LUT
入口，禁止直接消费单模型目录。

expected runtime 的 exact union 为：route metadata 已记录的 Python resolved executable、
transformers、PyTorch、CUDA、driver、GPU name/UUID与model tree，加上 config/data manifest
已冻结的 lexical Python environment、datasets、tokenizers版本；producer 必须逐项验证并
在 post-run metadata重录。route 未记录的字段由已签字 config/data manifest补证，禁止
producer自行推断“等价版本”。

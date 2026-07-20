# GPU 第三轮有效性实验：Receiver Progressive Transport 与 Codec Break-even

日期：2026-07-20  
GPU：NVIDIA GeForce RTX 5090 32 GB  
目标：在投入多节点 RDMA 原型前，先验证 receiver-credit progressive transport 的表示质量和单卡 codec 必要条件。

## 结论

**INT4/MXFP4 base + residual progressive transport 判定 NO-GO。**

原因不是一个，而是两个相互独立的失败：

1. 在相同 metadata-aware wire bytes 下，progressive representation 在两个模型上都没有优于直接 FP8/4-bit 混合；多数配置显著更差。
2. 即使忽略质量劣势，真实 packed Triton codec 的额外依赖与 kernel launch 开销，使 progressive 只在低于约 1–59 Gbps 的链路上可能击败统一 8-bit；现代 100–800 Gbps EP 网络不在该区间。

这不等于 Receiver-aware 整体判死。保留的方向应改为：

> receiver credit + topology-domain price + homogeneous one-shot codec lane

也就是在整条 peer/lane 上一次性选择 FP8 或低比特格式，而不是先发 base、再发 residual，也不在小 tile 内逐 vector 混合两种 codec。

## 1. 实验设计

### 1.1 表示质量 gate

模型和数据：

- OLMoE-1B-7B，E64K8；
- LLM-jp optimal-sparsity，E32K16；
- WikiText-103 train，固定 seed 20260720；
- offset 160，24 篇新文档；
- 每篇 128 token；
- full reference logits 只生成一次。

等 wire 对照：

- `progressive_50`：所有 routed pair 发送一条 INT4/MXFP4 base；gate 较高的 50% pair 再发送同格式 residual。
- `direct_equal_wire`：部分 pair 直接发送 FP8，其余发送一条 INT4/MXFP4；低比特比例按 scale metadata 精确匹配 progressive wire bytes。
- `qerr/reserr`：使用 sender output quantization error 的昂贵上界，检验失败是否仅来自 gate selector。

wire bytes 包括：

- FP8：每 128 elements 一个 scale；
- INT4：每 vector 一个 FP32 scale；
- MXFP4：每 32 elements 一个 scale；
- progressive refined vector 的第二份 scale。

### 1.2 Codec gate

新 Triton kernel 实际生成 packed buffer：

- symmetric INT8 pack/unpack；
- symmetric INT4 两个 nibble/byte；
- INT4 base pack/unpack；
- residual pack 和 receiver-side in-place add。

扫描：

- hidden size：512、2048；
- routed vectors/tile：32、128、512；
- 50% refinement；
- 100/200/400/800 Gbps 理论 wire time；
- pinned H2D 实测；
- warmup 20、重复 200 次。

`progressive_50` 与 `direct_mixed_50` 都平均使用 6 payload bits/element。实验排除了 selection compaction、RDMA descriptor 和 credit feedback，因而是对 progressive 有利的乐观下界。

## 2. 表示质量结果

### 2.1 OLMoE

INT4：

- direct equal-wire：mean token KL = 0.021753；
- progressive：KL = 0.076739；
- `direct − progressive` KL = -0.054986；
- 95% paired bootstrap CI [-0.060798, -0.049753]；
- progressive 更优概率为 0。

即 progressive KL 是 direct 的约 3.53 倍。

MXFP4：

- direct equal-wire：KL = 0.005345；
- progressive：KL = 0.005746；
- 差值 -0.000400，95% CI [-0.000922, 0.000103]。

该差异未显著，但 progressive 仍没有正优势。使用 `qerr/reserr` 上界 selector 后也没有得到优势。

### 2.2 LLM-jp

INT4：

- direct equal-wire：KL = 0.010440；
- progressive：KL = 0.016798；
- 差值 -0.006358；
- 95% CI [-0.006828, -0.005849]；
- progressive 更优概率为 0。

MXFP4：

- direct equal-wire：KL = 0.008265；
- progressive：KL = 0.013031；
- 差值 -0.004766；
- 95% CI [-0.005155, -0.004382]；
- progressive 更优概率为 0。

昂贵的 output-aware selector 也未挽救 residual representation：

- INT4 residual upper bound 比 direct upper bound 多 0.008209 KL；
- MXFP4 residual upper bound多 0.005432 KL；
- 两个 CI 均严格小于 0。

因此失败不能归因于 gate selector 太弱。

## 3. Codec 结果

### 3.1 Correctness

所有 Triton kernel 均完成真实 byte packing。随机 BF16 输入上：

- INT8 reconstruction MSE 约 5.4e-5–7.3e-5；
- uniform INT4 MSE 约 0.0169–0.0226；
- direct mixed 与 progressive MSE 都约 0.0085–0.0114。

微观 MSE 接近并未转化为模型级 KL 等价，说明不能用随机向量 codec MSE 替代端到端模型质量。

### 3.2 Progressive 相对统一 INT8

progressive 增加约 35–37 μs pack+unpack 开销，几乎不随 tile size 缩小。原因是第二次 quantization、依赖 base 的 residual kernel 和 receiver add。

最大 tile（512 vectors，hidden=2048）：

- uniform INT8 codec + 200 Gbps wire：75.87 μs；
- progressive：100.70 μs；
- progressive 慢 24.83 μs；
- pinned H2D serial 路径慢 30.59 μs；
- progressive break-even 仅为 59.22 Gbps。

其余 tile 的 break-even：

- 32×512：0.90 Gbps；
- 32×2048：3.53 Gbps；
- 128×512：3.60 Gbps；
- 128×2048：14.65 Gbps；
- 512×512：14.32 Gbps。

因此在 100/200/400/800 Gbps 下均不能回本。

### 3.3 Direct per-vector mixed codec 也不是理想实现

`direct_mixed_50` 需要分别启动 INT8 和 INT4 codec。相对 uniform INT8，它增加约 41 μs codec 开销；最大 tile 的 break-even 约 50.75 Gbps。

这意味着可部署实现不应在小 tile 内动态拆成两种 per-vector codec。更合理的是预先形成 homogeneous peer/lane：

- 整个 lane 使用 FP8；或
- 整个 lane 使用一种 4-bit codec。

### 3.4 Homogeneous INT4 的系统机会与质量矛盾

homogeneous INT4 kernel 没有额外 mixed/residual launch。最大 tile、hidden=2048、200 Gbps 模型下：

- uniform INT8：75.87 μs；
- uniform INT4：54.61 μs；
- INT4 快 21.26 μs。

但全局 uniform INT4 的模型质量不可接受：

- OLMoE mean token KL = 0.257494；
- LLM-jp mean token KL = 0.196984。

所以剩余研究问题不是“INT4 是否更快”，而是：

> 能否只让经过真实瓶颈 topology domain 的少量、质量安全 homogeneous lane 使用 INT4，并由 quality debt 限制重复伤害？

## 4. 对 Receiver-aware 方向的修正

### 删除

- base + residual progressive transport；
- 逐 routed-vector mixed codec；
- HHI regime classifier 作为核心；
- 在单卡/线性带宽模拟中声称 topology 收益。

### 保留

1. **Receiver credit**：下一 microbatch 的 NIC/ring/combine backlog 与 deadline slack。
2. **Topology-domain price**：区分 endpoint、NIC/rail、ToR shared cut 和 receiver GPU backlog。
3. **Homogeneous one-shot lane**：在 descriptor 构造前整条 peer/lane 选择 FP8 或 4-bit。
4. **Quality debt**：只允许低风险请求承担低比特 lane，并限制连续降级。
5. **Direct-benefit decision**：预测降级一个 lane 对真实 combine/TPOT critical path 的收益，不做 regime 二分类。

## 5. 下一实验的硬门槛

必须在真实多节点环境完成：

1. DeepEP、UCCL-EP flow control、FAST、UCCL+FAST 下是否仍存在 residual receiver/shared-cut queue。
2. homogeneous 4-bit lane 的真实 GPUDirect RDMA pack/transport/unpack。
3. endpoint hotspot、rail imbalance、ToR oversubscription、receiver combine backlog 四种瓶颈的区分。
4. topology-domain price 对比 per-receiver queue threshold，在相同低比特 bytes 和质量预算下的 P99 TPOT。

停止条件：

- UCCL-EP/FAST 已消除可观 residual queue；
- homogeneous low-bit lane 不能改善至少 5% P99 TPOT；
- topology-domain price 相对简单 receiver queue threshold 小于 2%；
- matched quality 下任务 accuracy 或 worst-request CVaR 不可接受。

## 6. 复现文件

- 质量 gate：`experiments/idea_a_mac/run_receiver_progressive_quality_gate.py`
- Triton codec：`experiments/idea_a_mac/run_receiver_codec_break_even_gpu.py`
- OLMoE 输出：`experiments/idea_a_mac/outputs/receiver_progressive_quality_olmoe_2026-07-20/`
- LLM-jp 输出：`experiments/idea_a_mac/outputs/receiver_progressive_quality_llmjp_2026-07-20/`
- codec 输出：`experiments/idea_a_mac/outputs/receiver_codec_break_even_2026-07-20/`

## 证据边界

本轮提供的是单 GPU 上的 representation 与 codec 必要条件，不是 RDMA/topology 充分条件。Pinned H2D 不能替代 GPUDirect RDMA；理论 link time 不能替代真实 incast、ECN/CNP、QP、ToR queue 和 FAST/UCCL 调度。Topology-aware 的论文 claim 必须等待真实多节点结果。

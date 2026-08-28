# Multi-rank timed trace：执行入口与硬件门

## 当前状态

`BLOCKED_BY_PHYSICAL_RANK_COUNT`

当前远端仅暴露一张 RTX 5090（`torch.cuda.device_count()==1`）。同卡启动多个
process rank 只能测 CUDA context/stream 争用，不能形成 GPU-to-GPU ingress、多个
sender NIC/QP 或 receiver incast，因此不得作为 timed multi-rank evidence。

## 最小有效硬件

- **功能 dry run**：2 GPU，同一节点；验证 identity、send/recv、timestamp 与守恒。
- **incast existence gate**：至少 4 GPU；3 个独立 sender rank 同时发往 1 个 receiver。
- **首选**：8 GPU 单节点，使 clean-v2 的 EP8 sender identity 无需折叠。
- 若跨节点，必须另报机内与跨机结果，不能混合；记录 GPU/NIC topology、NCCL
  transport 与实际接口。

## 冻结 trace 单元

每条 contribution 保存：

`run_id, policy, wave_id, request_id, layer_id, token_position, topk_slot,
expert_id, sender_rank, receiver_rank, payload_bytes, expert_start_ns,
expert_ready_ns, credit_recv_ns, send_start_ns, send_end_ns, recv_visible_ns,
unpack_start_ns, unpack_end_ns, join_close_ns, stream_id, message_id`。

所有时间必须来自同一节点可比较的 Nsight Systems/CUPTI timeline；rank-local CUDA
event elapsed time只用于持续时间交叉检查，不得直接拼成跨 rank 绝对时间轴。

## 第一轮执行

1. 从两个模型各抽 64 个 native receiver-waves，选择规则只依赖 route hash。
2. 各 sender 执行真实 shape 的 BF16 expert GEMM，随后发送真实 hidden-size payload。
3. receiver buffer credit `B={1,2,4,8}`；每个 cell 20 warmup + 100 measured waves。
4. 首轮只跑 `RR-credit`，建立自然 overlap、transport 与 receiver busy-period trace；
   不先运行 join-aware policy，避免先看候选收益再定义 baseline。
5. 只有自然 busy period 中 `>=2 joins && >=2 senders` 的比例达到 10%，才运行
   `oldest-ready / EDF / request-FCFS / Join-Deficit Credit`。

## 必须守恒

- 每个 route identity 恰好一次 expert-ready、一次 send、一次 receive、一次 unpack；
- send/receive message ID 一一对应，bytes 完全相等；
- `expert_ready <= send_start <= send_end`；
- `send_start <= recv_visible <= unpack_start <= unpack_end <= join_close`；
- join 只在全部有效 top-k siblings unpack 完成后关闭；
- 所有 policy 使用同一 route wave、权重、输入、payload 与 warmup/measured 划分；
- 若 NCCL silently 回退到 socket/SHM，单独标记并阻止 RDMA/NVLink claim。

## 进入调度机制实验的门槛

- 自然 temporal incast 比例 `>=10%`；
- receiver busy period P95 大于单 contribution service P95 的 2 倍；
- 至少 20% busy periods 存在两个以上合法首-credit动作；
- 以上条件在 OLMoE 与 LLM-jp 均成立。

任一失败即 `NO_GO_PHYSICAL_INCAST_HEADROOM`，不靠注入 sleep、人工 queue depth 或
选择性同步请求进行救援。


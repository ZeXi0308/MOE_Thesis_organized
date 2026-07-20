# Idea A Stage-0 资源与可行性门

## 判定

**本机只允许继续质量实验、trace/protocol 开发和 buffer-layout correctness；不允许产出 native FP4 kernel、all-to-all、RDMA、TTFT/TBT/P99 结论。**

真实系统 Stage D/E 当前被外部 CUDA 多 GPU 资源阻塞，不是被代码问题阻塞。

## 本机审计结果

| 项目 | 结果 | 对实验的含义 |
|---|---|---|
| 机器 | MacBook Pro `Mac17,9`，Apple M5 Pro | 非 CUDA 平台 |
| CPU/GPU | 18 CPU cores，20-core integrated Apple GPU | 无 NVIDIA Tensor Core / NCCL 路径 |
| 内存 | 48 GB unified memory | 可运行 OLMoE article-level offline inference |
| 磁盘 | 约 765 GiB 可用 | 足够保存 trace/CSV |
| OS | macOS 26.5.1 | 不能复现目标 Linux CUDA/RDMA serving stack |
| Python/PyTorch | Python 3.9.6，Torch 2.8.0 | 当前质量脚本可运行 |
| CUDA | `torch.cuda.is_available() = False`，device count 0 | 不能做 GPU kernel/multi-GPU benchmark |
| MPS | built=True，available=False | 当前 venv 不能把推理/量化迁移到 Apple GPU |
| NVIDIA toolchain | 无 `nvidia-smi`、无 `nvcc` | 不能编译/测 CUDA FP4 kernel |
| NCCL | pkg-config 无 NCCL | 不能做 NCCL collective |
| RDMA | `ibv_devinfo` 存在，但 `No IB devices found` | 不能测 IB/RoCE receiver queue/incast |
| Serving/runtime | 无 vLLM、SGLang、DeepEP、FlashInfer、Triton、CuPy | 本机没有目标 runtime 集成环境 |
| FP8 dtype | CPU float32→`float8_e4m3fn` cast 可运行 | 只说明 dtype software path 存在，不等于 native communication kernel |
| FP4 dtype | 有 `float4_e2m1fn_x2` 名称，但 CPU copy kernel未实现 | 不能把 dtype presence 当 FP4 hardware support |

## 当前允许执行

- article/document-level quality、paired bootstrap、routing drift；
- MXFP4/NVFP4 fake quant 与 scale-aware logical-byte accounting；
- frozen/dynamic route trace schema、intervention replay 协议；
- CPU reference pack/unpack 与 bit-layout correctness；
- 未校准 simulator 的定性敏感性分析，且必须标为 proxy。

## 当前禁止声称

- native FP4/FP8 operator speedup；
- GPU pack/unpack、NCCL/DeepEP all-to-all 收益；
- receiver-aware queue/P99 收益；
- TTFT、TBT、TPOT、goodput 改善；
- 任何由 `bytes/BW` 直接换算出的端到端 latency。

## 解除资源门的最低条件

### D0 单机 operator gate

- 至少 1 台 Linux CUDA 服务器、4～8 张支持目标 FP8/FP4 路径的 GPU；
- 可用 CUDA、NCCL、Triton/CUDA extension 与 profiler；
- 能实现并测 uniform FP8、fixed R-layout、dynamic gate 三个 matching packed-buffer baselines。

### D3/E 跨节点系统 gate

- 至少 2 节点 × 4 GPU；
- 真实 IB/RoCE/RDMA，记录 NIC/GPU topology；
- 可修改的 vLLM/SGLang + DeepEP/NCCL 路径；
- 足够运行每配置至少 10,000 requests、5 个独立 runs 的 GPU-hours；P99 主张建议 30,000～100,000 requests。

## 资源到位后的第一条命令链

```text
uniform BF16/FP8 operator baseline
  -> fixed R-layout packed kernel
  -> calibrated gate-selector packed kernel
  -> held-out shapes 校准/验证 simulator
  -> R-layout serving Core matrix
```

在 R-layout kernel 打赢 uniform FP8 且其开销显著低于 dynamic gate selector 之前，不启动 receiver-aware、Graceful 或 QTree 工程实现。

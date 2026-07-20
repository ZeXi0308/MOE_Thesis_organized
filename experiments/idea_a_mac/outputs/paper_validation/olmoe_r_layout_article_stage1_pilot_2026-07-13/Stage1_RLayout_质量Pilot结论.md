# Stage-1 R-layout Article-Level 质量 Pilot

## 结论

**有条件 Go，不能据此宣称 fixed-rank 已经优于动态 gate。**

在修正 BF16 accumulation order、使用独立 validation/test articles、冻结 calibration threshold 后，R-layout 的质量信号仍成立：它显著优于 head/interleaved 同字节反例。但在相同 logical-byte budget 下，calibrated gate threshold 的 KL 显著低于 fixed-rank R-layout。因而论文的待验证问题已经收敛为：

> fixed-rank 能否用定长、规整的 FP8-head/MXFP4-tail packed kernel，换回约 `0.000655` absolute KL（约 `11.2%` total-KL）的质量劣势，并在真实 operator/serving 中形成净收益。

如果真实 kernel 没有明显低于动态 gate selector 的 pack/schedule 开销，fixed-rank 不构成 Pareto 主方法，应切换为 gate selector。

## 实验设置

- 模型：`allenai/OLMoE-1B-7B-0924`，16 层，top-8，hidden size 2048；
- calibration：WikiText-2 validation，16 个完整 articles；
- untouched test：WikiText-2 test，16 个完整 articles；
- sequence length：256；test 有效 next-token 数：4,080；
- 格式：uniform FP8 + selective MXFP4 fake quant；
- 统计：article-paired bootstrap 5,000 次；
- exactness：patched full 与原模型 logits `max/mean abs diff = 0/0`；
- 32 个 calibration/test article hashes 全部唯一。

本实验没有 native FP4 kernel、all-to-all、RDMA、TTFT、TBT 或 P99。

## 结果

| 策略 | metadata-aware logical saving vs BF16 | mean token KL | PPL delta vs full |
|---|---:|---:|---:|
| uniform FP8 | 49.61% | 0.003206 | +0.0275 |
| fixed rank tail-4 MXFP4 | 61.52% | 0.005849 | +0.0246 |
| gate threshold MXFP4 | 61.59% | 0.005194 | +0.0053 |
| cumulative tail-mass MXFP4 | 61.53% | 0.005502 | +0.0250 |
| contribution tail oracle | 61.52% | 0.005421 | +0.0411 |
| head-4 MXFP4 control | 61.52% | 0.031741 | +0.1709 |
| interleaved-4 MXFP4 control | 61.52% | 0.029148 | +0.1446 |

`metadata-aware logical saving` 只计 payload 与 scale bytes，不含 alignment、message header、pack/unpack 或 collective 固定成本。

## Paired 检验

| candidate vs reference | paired KL delta | 95% CI | paired PPL delta | 95% CI |
|---|---:|---:|---:|---:|
| uniform FP8 vs full | +0.003206 | [0.002884, 0.003566] | +0.0275 | [0.0086, 0.0499] |
| R-layout vs uniform FP8 | +0.002644 | [0.002102, 0.003386] | -0.0030 | [-0.0373, 0.0299] |
| gate vs R-layout | **-0.000655** | **[-0.001132, -0.000304]** | -0.0193 | [-0.0502, 0.0141] |
| tail-mass vs R-layout | **-0.000347** | **[-0.000740, -0.000020]** | +0.0005 | [-0.0347, 0.0338] |
| oracle vs R-layout | -0.000428 | [-0.000990, +0.000018] | +0.0165 | [-0.0170, 0.0493] |
| head control vs R-layout | +0.025891 | [0.021748, 0.030442] | +0.1464 | [0.0843, 0.2099] |
| interleaved control vs R-layout | +0.023298 | [0.018272, 0.029375] | +0.1200 | [0.0389, 0.2069] |

逐 article 符号也一致：gate 在 16 篇中有 14 篇 KL 低于 R-layout；head 和 interleaved controls 在 16/16 篇都更差。

## 严格解释

1. **rank criticality 成立**：同字节压 head/interleaved 的损失约为 tail 的 5 倍，并且逐 article 方向一致。
2. **fixed rank 不是质量 Pareto winner**：gate 在几乎相同 logical bytes 下将 total KL 降低约 11.2%，paired CI 排除 0。
3. **系统假设仍可检验**：fixed rank 的唯一合理优势是定长 buffer、静态 offset、较少 mask/prefix-sum/metadata 和更容易融合的 kernel；这些尚未测量。
4. **PPL 证据弱于 KL**：除 uniform FP8 vs full 和两个强反例外，策略间 PPL CI 多数跨 0，不应声称 PPL 显著提升或下降。
5. **样本仍是 pilot**：16 articles/4,080 tokens 不足以进入论文主表。WikiText test 共只有 61 个合格 articles，且本 pilot 已查看前 16 个；下一轮使用 32 个 validation calibration articles 与尚未查看的 test offset 16 后 45 个 articles。更大的样本量和跨域证据必须从第二 corpus 补，不复用 pilot articles。

## Go / No-Go

- Stage-1 pilot：**Go 到 formal quality rerun**；
- R-layout kernel：**尚未 Go**，需 formal 质量结果与硬件资源门；
- receiver-aware / Graceful / QTree：**不启动工程实现**，等待 R-layout kernel/serving 与真实瓶颈归因。

原始证据：`signal_comparison.csv`、`paired_comparisons.csv`、`sample_metrics.csv`、`config.json`、`data_manifest.json`。

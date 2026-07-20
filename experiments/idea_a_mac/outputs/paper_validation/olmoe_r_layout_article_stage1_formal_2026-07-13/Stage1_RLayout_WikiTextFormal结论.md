# Stage-1 R-layout WikiText Article-Level Formal 结论

## 硬结论

**rank criticality 通过；fixed-rank 质量 Pareto 不通过；R-layout 作为系统候选有条件保留。**

在 45 个未参与 pilot、未参与 calibration 的 WikiText test articles 上，fixed-rank R-layout 明显优于相同 logical bytes 的 head/interleaved 反例，但被 calibrated gate、tail-mass 和不可部署 contribution oracle 小幅而稳定地支配。fixed-rank 的论文价值因此不能来自“质量更优”，只能来自尚待实测的系统规整性：定长 buffer、静态 offset、少 mask/prefix-sum/metadata 和更容易融合的 pack/unpack。

## 实验协议

- 模型：`allenai/OLMoE-1B-7B-0924`，top-8，16 个 MoE layers；
- calibration：WikiText validation articles `[0:32]`；
- formal test：WikiText test articles `[16:61]`，共 45 篇；
- pilot 使用 test `[0:16]`，与 formal test article hash overlap 为 0；
- sequence length 256，formal test 有效 next-token 数 11,475；
- patched full 与原模型 logits exact equal，max/mean absolute diff 均为 0；
- calibration/test 内及彼此之间 article hashes 全部唯一；
- article-paired bootstrap 5,000 次；
- threshold 在看 test 前冻结：gate `0.044922`，tail-mass `0.157227`。

本实验仍是 fake-quant + logical-byte 质量实验，没有 native FP4、真实 all-to-all、RDMA、TTFT、TBT 或 P99。

## Formal 结果

| 策略 | metadata-aware logical saving vs BF16 | mean token KL | PPL delta vs full |
|---|---:|---:|---:|
| uniform FP8 | 49.61% | 0.003359 | +0.0134 |
| fixed rank tail-4 MXFP4 | 61.52% | 0.006042 | +0.0197 |
| gate threshold MXFP4 | 61.63% | 0.005635 | +0.0221 |
| cumulative tail-mass MXFP4 | 61.55% | 0.005616 | +0.0295 |
| contribution-tail oracle | 61.52% | 0.005559 | +0.0169 |
| head-4 MXFP4 control | 61.52% | 0.035938 | +0.2805 |
| interleaved-4 MXFP4 control | 61.52% | 0.027707 | +0.1799 |

## Paired comparisons

| candidate vs reference | paired KL delta | 95% CI | 胜/负 article | paired PPL delta | 95% CI |
|---|---:|---:|---:|---:|---:|
| uniform FP8 vs full | +0.003359 | [0.003046, 0.003757] | — | +0.0134 | [0.0023, 0.0245] |
| R-layout vs uniform FP8 | +0.002683 | [0.002416, 0.003029] | — | +0.0063 | [-0.0152, 0.0274] |
| gate vs R-layout | **-0.000406** | **[-0.000695, -0.000113]** | 33/12 | +0.0023 | [-0.0170, 0.0218] |
| tail-mass vs R-layout | **-0.000426** | **[-0.000617, -0.000245]** | 34/11 | +0.0097 | [-0.0062, 0.0282] |
| oracle vs R-layout | **-0.000482** | **[-0.000640, -0.000320]** | 40/5 | -0.0028 | [-0.0205, 0.0148] |
| head vs R-layout | +0.029896 | [0.026065, 0.033999] | 0/45 | +0.2608 | [0.2049, 0.3211] |
| interleaved vs R-layout | +0.021665 | [0.019436, 0.024035] | 0/45 | +0.1602 | [0.1147, 0.2067] |

`胜/负 article` 对前三个 selector 表示 candidate KL 低于/高于 R-layout；对 controls，0/45 表示没有任何 article 优于 R-layout。

## 与 pilot 的关系

- pilot gate 相对 R-layout：`-0.000655`，formal：`-0.000406`；方向复现，差距缩小；
- pilot gate 在 14/16 篇更好，formal 在 33/45 篇更好；
- pilot/formal 中 head 与 interleaved 都在所有文章上更差；
- calibration 从 16 扩到 32 后，gate threshold `0.044678 → 0.044922`，tail-mass `0.156616 → 0.157227`，较稳定。

## 研究决策

1. **保留核心现象**：低 rank tail 是安全低比特候选；同字节压 head/interleaved 的损失分别是 tail 的 `5.95×/4.59×`。
2. **撤回质量最优叙述**：fixed-rank 在相同 logical-byte budget 下被动态 selector 稳定支配约 `6.7%～8.0%` total KL。
3. **有条件进入 kernel gate**：绝对 KL 差约 `4×10^-4`，足够小，可以验证定长 R-layout 是否换来真实 operator 收益；但没有 kernel 证据前，不能称其为 Pareto 方法。
4. **PPL 不作主差异结论**：R-layout、gate、tail-mass、oracle 间 paired PPL CI 均跨 0。
5. **尚非完整论文质量闭环**：仍需第二 corpus/domain 和至少一个第二 MoE 模型；不得把同一 article 的多窗口当独立样本。

## 下一 gate

- 先完成 Stage-0 硬件/runtime 能力与资源审计；
- 有 CUDA/FP4/多 GPU 资源才运行 uniform FP8 vs fixed R-layout vs dynamic gate packed-kernel microbenchmark；
- primary system question：fixed layout 的开销优势是否足以补偿 `0.000406` KL；
- kernel 未赢 uniform FP8 或未显著低于 dynamic selector overhead，则停止 R-layout 系统主张并转向 gate selector。

原始证据：`signal_comparison.csv`、`paired_comparisons.csv`、`sample_metrics.csv`、`config.json`、`data_manifest.json`。

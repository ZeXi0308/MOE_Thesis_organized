# CPR-MoE RankLane 快验分析器

这个目录实现一个**不冒充多卡实验**的失败即停门槛：复用已有 RTX 5090 的跨模型 rank-quality 与 codec 证据，在 codec 成本为零的最乐观条件下，计算 RankLane 相对 uniform FP8 的精确 Amdahl 端到端收益上界。

它回答的是：当原始 BF16 EP 回传路径占端到端时延不超过 20% 时，固定 RankLane 执行器有没有可能达到 5% 端到端改善。它不回答回传路径是否真实暴露、receiver ordering 是否存在，也不产生 NCCL/RDMA、decode TPOT 或 P99 结论。

## 复跑

从仓库根目录执行：

```bash
python3 -m unittest docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick/test_run_gate.py -v

python3 docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick/run_gate.py \
  --repo-root . \
  --config docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick/configs/quick_validate.json \
  --output-dir docs/ideas/receiver_aware/cpr_ranklane/outputs/cpr_moe_ranklane_quick_gate_2026-07-25
```

输出目录必须不存在；程序拒绝覆盖已有证据包。`source_manifest.json` 固化输入哈希，`decision.json` 是机器可读裁决，`matrix.csv` 是暴露占比扫描，`report.md` 是人读摘要。

## 公式

将 BF16 端到端时间归一化为 1，原始回传路径占比为 `p`，字节节省率为 `s`：

```text
T(s) = 1 - p*s
gain(candidate vs baseline) = p*(s_candidate-s_baseline)/(1-p*s_baseline)
required_p(q) = q / ((s_candidate-s_baseline)+q*s_baseline)
```

该公式假设通信时间随字节数线性下降，且 codec、launch、排队与元数据成本全部为零，所以是对候选最有利的上界，不是生产性能预测。

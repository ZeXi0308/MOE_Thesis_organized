# 当前前态中的轮转资助资格

新增问题：先排序再检查资源是否漏掉可执行交换？

Verdict: `NO_OBSERVED_RANK_FIRST_FUNDING_MISS`；CPU 只读诊断，无新 GPU 执行。

| Cell | 前态核对 | 交换提案/实际释放核对 | 合法/可资助候选 | 漏动作 | 最小余量块 |
|---|---:|---:|---:|---:|---:|
| cohort2-block0-most_output | 1162 | 9/9 | 173/173 | 0 | 11 |
| cohort2-block1-most_output | 1162 | 9/9 | 173/173 | 0 | 11 |
| cohort0-block0-least_progress | 1257 | 8/8 | 160/160 | 0 | 8 |

表中候选为 request × 已观察提案前态，不是独立请求或实验重复；完整枚举、身份、块数与输入 SHA256 见 `analysis.json`。

块守恒：free + sum(owned) = usable；恢复需求 `ceil((prompt + observed_output)/block_size) - target_owned`。候选保留进度、驱逐次数和最短驻留约束；仅检查 `free + victim_owned >= need`。

实际选中者由原生抢占前后 pool 差额核对；未选候选只得到结构上的可资助性，没有执行其未来状态。合成负控检验“第一名不足、另一名足够”会被检测；缺块字段或被选者不在候选集中会拒绝分析。

当前三个 cell 均关闭 prefix caching；结果不覆盖共享缓存块、自然 EOS、异构终止或其它压力点。未测性能/质量，没有 Oracle 或方法 GO。

唯一下一步：继续原冻结同路径运行时重复；本域若无遗漏，不以调整资助排序新增 GPU 实验。

重跑必须使用新的输出目录：

```text
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_rotation_candidate_feasibility.py --cell refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_strong_baseline_r01/execution02_westc_53036/gpu_results/cohort2-block0-most_output --cell refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_strong_baseline_r01/execution02_westc_53036/gpu_results/cohort2-block1-most_output --cell refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_victim_order_r01/execution/gpu_results/cohort0-block0-least_progress --output-dir refine-logs/expert_saturation/outputs/admission_capacity/20260914_rotation_candidate_feasibility_r01
```

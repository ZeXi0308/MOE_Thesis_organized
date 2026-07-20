# GPU 首轮有效性实验结果：Receiver Causal Audit 与 Expert Prefetch Validity Audit

> 日期：2026-07-20  
> GPU：NVIDIA GeForce RTX 5090，32 GB  
> PyTorch：2.8.0+cu128  
> 原则：先修复会改变结论的效度错误，再决定是否投入端到端原型。

---

# 1. 本轮实际运行了什么

## 1.1 Receiver-aware v3 因果修正版

新脚本：

`experiments/idea_a_mac/run_receiver_aware_v3_causal_audit.py`

修复：

1. 检测器观察 warm-up 后，策略只作用于 warm-up 之后，不能回溯修改历史动作；
2. warm-up 统一使用 random；
3. adaptive 在所有 test regime 中只使用一份由 hotspot calibration 构建的 structural profile；
4. 扫描 `detect_frac={0.1,0.2,0.3,0.5}`；
5. 报告 whole-scenario saving、post-warm-up saving、genie regret；
6. 对 adaptive 与 `always_causal_prev_step` 做分 regime 配对 bootstrap。

规模：

- OLMoE E64K8；
- LLM-jp E32K16；
- 每模型 24 hotspot + 24 balanced scenario；
- 每个 random baseline 20 次；
- 50/50 mixed-regime test。

## 1.2 Expert Prefetch GPU 有效性审计

新脚本：

`experiments/idea_a_mac/run_expert_prefetch_gpu_validity_audit.py`

修复：

1. cache key 改为 `(layer_id, expert_id)`；
2. 同时评估 top-1 和真实 full top-k；
3. cache 跨 decode-sized chunk 持续存在；
4. 分别评估 legacy global cache 与 per-layer partitioned cache；
5. 加入 reactive LRU、static frequency pinning、frequency prefetch、random、transition 和 one-layer oracle；
6. 使用 GPU 实测 H2D；
7. 使用只计算实际活跃 expert 的 sparse padded batched BMM proxy，而不是所有 token×所有 expert 的 dense BMM；
8. 实测 compute 与 H2D 并发争用；
9. safe overlap budget 允许为 0，不强制至少预取一个。

规模：

- OLMoE：cache capacity 8/16/32，requested budget 8；
- LLM-jp：cache capacity 6/12/24，requested budget 6；
- batch tokens=32；
- calibration 12 docs，test 28 docs；
- 3584 layer-batches/model/rank-mode；
- GPU microbenchmark 20 repeats。

---

# 2. Receiver-aware：原 v3 正结果明显收缩

## 2.1 因果修正后的 pooled saving

| model | detect_frac | detection accuracy | adaptive | always causal | adaptive − causal |
|---|---:|---:|---:|---:|---:|
| OLMoE | 0.1 | 58.3% | 15.400% | 16.430% | -1.031pp |
| OLMoE | 0.2 | 93.8% | 16.717% | 16.430% | +0.286pp |
| OLMoE | 0.3 | 93.8% | 16.068% | 16.430% | -0.363pp |
| OLMoE | 0.5 | 100% | 14.799% | 16.430% | -1.632pp |
| LLM-jp | 0.1 | 58.3% | 14.301% | 15.944% | -1.643pp |
| LLM-jp | 0.2 | 93.8% | 16.662% | 15.944% | +0.717pp |
| LLM-jp | 0.3 | 95.8% | 16.271% | 15.944% | +0.327pp |
| LLM-jp | 0.5 | 100% | 14.751% | 15.944% | -1.194pp |

原文在 `detect_frac=0.3` 报告：

- OLMoE adaptive=0.1706；
- LLM-jp adaptive=0.1729。

因果修正后变为：

- OLMoE 0.1607；
- LLM-jp 0.1627。

下降来自两个真实部署成本：

1. warm-up 已经发生，不能回溯享受检测后的策略；
2. structural profile 不再根据 test 的 ground-truth origin mode 偷换。

## 2.2 配对 bootstrap：最佳窗口也没有稳健胜过 causal

adaptive − `always_causal_prev_step`，stratified bootstrap 95% CI：

| model | detect_frac | mean difference | 95% CI |
|---|---:|---:|---:|
| OLMoE | 0.1 | -1.031pp | [-1.690, -0.388]pp |
| OLMoE | 0.2 | +0.286pp | [-0.400, +0.885]pp |
| OLMoE | 0.3 | -0.363pp | [-1.003, +0.229]pp |
| OLMoE | 0.5 | -1.632pp | [-2.028, -1.247]pp |
| LLM-jp | 0.1 | -1.643pp | [-2.410, -0.836]pp |
| LLM-jp | 0.2 | +0.717pp | [-0.008, +1.329]pp |
| LLM-jp | 0.3 | +0.327pp | [-0.221, +0.817]pp |
| LLM-jp | 0.5 | -1.194pp | [-1.531, -0.859]pp |

结论：

- `detect_frac=0.2` 是当前最优窗口；
- 但两个模型的 95% CI 都包含 0；
- 0.5 虽然达到 100% 分类准确率，却因为 warm-up tax 显著更差；
- 优化 detection accuracy 不是正确目标，应直接最小化 cumulative regret 或预测动作收益。

## 2.3 修正后的 claim

不能再声称：

> HHI detector 构成了优于固定策略的部署级 adaptive controller。

当前只能声称：

> 在 synthetic mixed-regime 回放中，约 20% warm-up 可在点估计上接近 genie regime switch，但相对最强固定 causal baseline 的增益小且未达到统计稳健；更长观测窗口虽然提高分类准确率，却因不可恢复的 warm-up 成本降低总收益。

而且这仍是带宽回放，不是真实 RDMA queue 实验。

---

# 3. Expert Prefetch：full top-k 暴露了工作集饱和

## 3.1 真实 full top-k working set

| model | rank mode | mean unique experts/layer-batch | P95 | expert-pool coverage | all-expert layer-batches |
|---|---|---:|---:|---:|---:|
| OLMoE E64K8 | top-1 | 15.73 | 22 | 24.6% | 0% |
| OLMoE E64K8 | full top-8 | 49.83 | 59 | 77.9% | 0% |
| LLM-jp E32K16 | top-1 | 12.75 | 17 | 39.9% | 0% |
| LLM-jp E32K16 | full top-16 | 31.90 | 32 | 99.7% | 91.2% |

这是本轮最重要的系统边界：

- top-1 predictor 的高 hit rate 并不代表完整 MoE working set 可被小预算覆盖；
- LLM-jp batch=32 时，绝大多数层已经需要所有 expert；
- 在这种情况下，transition、frequency、random 的 prefetch precision 都接近 100%，因为“随便预取谁都会被需要”；
- 因此命中率不能再证明 predictor 提供了系统价值。

## 3.2 GPU 硬件测量

| metric | OLMoE | LLM-jp |
|---|---:|---:|
| one expert bytes | 12.58 MB | 3.15 MB |
| isolated H2D | 240.54 μs | 73.34 μs |
| sparse padded compute median | 417.95 μs | 82.19 μs |
| one-copy concurrent total | 436.19 μs | 144.82 μs |
| two-copy concurrent total | 539.07 μs | 201.20 μs |
| safe budget（≤5% compute extension） | 1 | 0 |

关键观察：

1. OLMoE 一个 copy 基本可被掩盖，但第二个开始暴露约 121 μs；
2. LLM-jp 即使只复制一个 expert，并发总时间也从 82.19 μs 增到 144.82 μs；
3. LLM-jp 的 safe overlap budget 是 0，而旧脚本通过 `max(1, ...)` 强制为 1；
4. H2D 与 compute 不能按独立常数简单取 `max()`：并发时 compute stream 和 copy stream 都出现 slowdown；
5. 新 sparse padded proxy 仍有 1.8×–5.9× padding，不等于生产 grouped-GEMM；更快 kernel 只会进一步缩短可用窗口。

## 3.3 正确 per-layer cache 下的 full-top-k 结果

### OLMoE，safe budget=1

| capacity/layer | transition | frequency | random | oracle |
|---:|---:|---:|---:|---:|
| 8 | -0.241% | -0.377% | -0.616% | -0.079% |
| 16 | -0.231% | -0.436% | -0.654% | +0.037% |
| 32 | -0.488% | -0.836% | -1.367% | +0.256% |

即使提前知道下一层真实 working set，oracle 最高也只有 +0.256%。transition 在全部容量下都比 reactive LRU 更差。

requested budget=8 时：

- transition：-12.9% 到 -26.3%；
- oracle：-11.9% 到 -21.5%。

说明固定大预算的符号反转依然存在，而且 full top-k/cache pollution 后更严重。

### LLM-jp，safe budget=0

不存在满足“并发时间不超过 compute 5%”的可安全预取预算。

若仍强制 budget=6：

| capacity/layer | transition | frequency | random | oracle | static pinning |
|---:|---:|---:|---:|---:|---:|
| 6 | -16.33% | -16.34% | -16.38% | -16.32% | +0.11% |
| 12 | -20.98% | -20.99% | -21.07% | -20.96% | +0.27% |
| 24 | -48.42% | -48.47% | -48.68% | -48.20% | +1.19% |

transition、frequency、random、oracle 几乎重合，因为下一层基本需要全部 expert。复杂 predictor 没有实际选择空间。

## 3.4 legacy global cache 为什么仍会显示正收益

在正确 `(layer, expert)` key、但总 cache 只有 6–32 个对象的 global 模式下：

- OLMoE full-top-k、safe budget=1：
  - frequency +1.519%；
  - transition +1.603%；
  - oracle +1.680%。
- LLM-jp full-top-k、强制 budget=6：
  - frequency +3.549%；
  - transition +3.558%；
  - random +3.515%；
  - oracle +3.564%。

这不是 predictor 的成功：

- OLMoE transition 相比 frequency 只多 0.083pp；
- LLM-jp 只多 0.009pp；
- global cache 在层切换后几乎没有同层复用，prefetch 实际退化成“提前流水化任意必需 expert”；
- LLM-jp 中 random 也几乎一样好。

因此应把“是否值得 pipeline expert load”和“是否需要 routing predictor”分开。当前数据支持前者可能存在，但不支持后者。

## 3.5 Prefetch go/no-go 结论

当前方向按原定义应判为：

**NO-GO：Routing-predictability-driven Expert Prefetch 独立论文。**

原因不是路由信号不存在，而是：

1. full top-k 后 working set 过宽；
2. 正确 per-layer cache 下 safe-budget oracle 上限接近 0；
3. LLM-jp 没有安全 overlap window；
4. predictor 相比 frequency 的增量不足 0.1pp；
5. existing related work 已覆盖该机制空间。

可保留的新问题：

> 当 full top-k 使 working set 饱和时，是否应完全放弃 prediction，改为 deterministic load pipeline、expert block splitting、降低 top-k 或 GPU-side replication？

这是新的系统问题，但不再是当前 transition-table prefetch 论文。

---

# 4. 本轮没有完成的实验

## 4.1 真实 Receiver-aware RDMA

远端只有一张 RTX 5090，没有多 GPU、NIC/RDMA topology，因此无法合法验证：

- receiver incast；
- ECN/CNP；
- QP queue；
- DeepEP/UCCL-EP；
- bytes reduction 到 P99 queueing 的映射。

不能用单 GPU copy 冒充 RDMA 证据。

## 4.2 Quality Isolation

本轮没有继续消耗 GPU 做旧 Part C proxy，因为它的主要问题是数据协议：

- test-set proxy selection；
- second degraded forward；
- 静态 32-doc quota；
- 没有 streaming debt。

下一步应先实现严格 split 和 causal pre-decision proxy，再运行 GPU；直接重跑旧 forward 不会回答有效性问题。

---

# 5. 原始产物

## 脚本

- `experiments/idea_a_mac/run_receiver_aware_v3_causal_audit.py`
- `experiments/idea_a_mac/run_expert_prefetch_gpu_validity_audit.py`

## Receiver outputs

- `experiments/idea_a_mac/outputs/receiver_aware_v3_causal_olmoe_2026-07-20/`
- `experiments/idea_a_mac/outputs/receiver_aware_v3_causal_llmjp_2026-07-20/`

每个目录包含：

- `*_causal_raw.csv`
- `*_causal_by_regime.csv`
- `*_causal_pooled.csv`
- `*_metadata.csv`

## Prefetch outputs

- `experiments/idea_a_mac/outputs/expert_prefetch_validity_olmoe_2026-07-20/`
- `experiments/idea_a_mac/outputs/expert_prefetch_validity_llmjp_2026-07-20/`

每个目录包含：

- `hardware.json`
- `working_set_summary.csv`
- `simulation_results.csv`
- `best_policy_by_setting.csv`

---

# 6. 论文路线更新

1. **Receiver-aware**：从“已完成 adaptive controller”降级为“staleness/warm-up characterization”。只有获得真实多节点 RDMA 后才继续系统 claim。
2. **Expert Prefetch**：停止 transition-driven 独立论文；现有 positive system result 被推翻。
3. **Quality Isolation**：成为下一条唯一值得做 validity-first GPU 实验的候选，但必须先修严格 split、zero-extra-forward proxy 和流式 debt。
4. **合并论文**：是否推进 Receiver + Quality，取决于能否取得多 GPU/RDMA 环境并实现真实 mixed-precision EP data path。


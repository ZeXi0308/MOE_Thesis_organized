# GPU 第二轮有效性实验：Quality Isolation

日期：2026-07-20  
硬件：NVIDIA GeForce RTX 5090 32 GB  
模型：OLMoE-1B-7B（E64K8）、LLM-jp optimal-sparsity（E32K16）

> **第四轮事后更新**：本文的正结果只针对“同一 teacher-forced 文档上的 prefill
> 特征与 fixed-tail KL”。后续真实 KV-cache、approximation 仅在 decode 启用的严格实验
> 未能复现 prefill→future-decode 风险预测：四种 action 的 test Spearman CI 均跨 0，
> 旧冻结 proxy 的迁移 CI 也全部跨 0。本文不能再被引用为 decode controller 的证据。
> 详见同目录 `GPU第四轮有效性实验结果_Prefill到DecodeFragility_2026-07-20.md`。

## 结论

Quality Isolation **没有被判死，但成立范围显著收窄**。

- **成立的弱正结论**：在 LLM-jp 上，正常 prefill 已产生的 router/gate 统计和 NLL 可以预测后续 fixed-tail INT4 的文档级 KL 风险；冻结 proxy 后在 64 篇全新文档上复现，Spearman ρ=0.517，95% bootstrap CI [0.315, 0.673]，双侧 permutation p<1e-4。
- **不成立的强结论**：当前证据不支持到达时预测、首个 prefill chunk 决策、跨模型通用 fragility、worst-request 精确识别或完整在线 quality-debt fairness。
- **模型依赖明显**：同一协议在 OLMoE 上未得到稳定 proxy；跨模型迁移失败。
- **论文状态（已被第四轮收紧）**：这里只能表述为 model-specific 的 same-input
  characterization；不能再声称可用于后续 decode/chunk。prefill-only quality-debt
  controller 暂停，除非新的 one-step online predictor 通过 sealed test。

## 为什么要重做 P0

原 P0 有三项会直接夸大结果的问题：

1. 在测试集上选择相关性最高的 proxy，构成 selection leakage。
2. proxy 使用另一种 degradation 的测试时 KL，需要额外完整 forward，不能部署。
3. 用文档数而不是实际 token/byte 成本匹配 full-precision quota。

本轮实验取消了这三项假设。degradation KL 只用于离线标签，不作为在线特征。

## 实验协议

### 数据与划分

- 语料：此前未用于 Quality Isolation 的 WikiText-103 train 文档。
- 每个模型先采集 96 篇文档，固定随机种子 20260720。
- 严格顺序划分：48 train、16 validation、32 sealed test。
- 所有 feature group 和 ridge α 只按 validation Spearman 选择。
- 选择完成后，以 train+validation 的 64 篇文档重新拟合，再打开 test。
- LLM-jp 弱正结果出现后，冻结 `post_prefill_all + ridge α=1`，再采集 offset 96–159 的 64 篇未接触文档做 test-only replication；不重新选择特征或超参数。
- 序列长度固定为 128 token，避免长度/成本成为主要混淆变量。

### 降级标签

- OLMoE：top-8 中后 4 ranks 使用 simulated INT4，前 4 ranks 使用 FP8。
- LLM-jp：top-16 中后 8 ranks 使用 simulated INT4，前 8 ranks 使用 FP8。
- 每篇文档运行一次 full reference 和一次 fixed-tail degradation，标签为 mean token KL。
- 当前仍是 fake-quant expert-output perturbation，不等于真实 mixed-precision communication data path。

### 候选信号

- `length_only`：固定长度控制基线。
- `arrival_lexical`：字符数、token 数、unique-token ratio、token-ID entropy、相邻重复率；不需要模型 forward。
- `early_router_plus_lexical`：正常 full forward 前 25% token 的 gate weight、top-1/top-2 margin、tail mass、routing entropy、expert concentration 等；不增加 forward，但只能用于后续 chunk/decode。
- `full_router_plus_lexical`：完整 prefill router 统计；只能用于后续 decode。
- `prefill_nll_only` / `prefill_nll_plus_lexical`：检验结果是否只是 PPL/NLL 的替代。
- `post_prefill_all`：lexical + full router + full NLL。

## 首轮严格划分结果

### OLMoE

validation 选择 `post_prefill_all`，sealed test 上：

- Spearman ρ=0.286，95% CI [-0.105, 0.615]，未得到统计支持。
- worst-decile AUROC=0.518，recall@10%=0。
- 10% 和 25% token quota 下，P95 KL 降幅均为 0%；oracle 分别可降 9.57% 和 29.61%。
- `prefill_nll_only` 在 test 上偶然达到 ρ=0.422，但 validation ρ=-0.271，说明方向和切分不稳定，不能事后替换为主 proxy。

判断：OLMoE 当前 proxy NO-GO。

### LLM-jp

validation 同样选择 `post_prefill_all`，sealed test 上：

- Spearman ρ=0.453，95% CI [0.156, 0.662]。
- worst-decile AUROC=0.750，但 recall@10%=0.25；仍漏掉 3/4 个最脆弱请求。
- 10%、25%、50% token quota 下，P95 KL 分别下降 3.71%、3.71%、12.38%。
- 对应 random 平均下降 0.90%、2.49%、5.13%，oracle 上限为 6.94%、10.93%、18.25%。
- `full_router_plus_lexical` 的 test ρ=0.466，而 `prefill_nll_only` 仅为 0.196 且 CI 跨 0；正信号不是单纯由 NLL 解释。

判断：存在可复验价值，但第一轮 test 只有 32 篇，必须做冻结确认。

## LLM-jp 冻结确认实验

训练仅使用首轮的 48 train + 16 validation；首轮 32 test 不参与拟合。冻结 `post_prefill_all` 和 α=1 后，在 64 篇新文档上：

- Spearman ρ=0.517，95% CI [0.315, 0.673]。
- 双侧 permutation p=9.999e-5。
- worst-decile AUROC=0.727，recall@10%=0.286，即只找回 7 个最差请求中的 2 个。
- 10%、25%、50% token quota 下，P95 KL 分别下降 6.23%、12.19%、24.09%。
- random 平均下降 0.79%、4.01%、11.87%。
- oracle 上限为 17.61%、26.45%、34.34%。

控制组：

- `full_router_plus_lexical`：ρ=0.468，CI [0.232, 0.651]。
- `prefill_nll_only`：ρ=0.181，CI [-0.073, 0.415]。
- `arrival_lexical`：ρ=0.260，CI [0.022, 0.476]。

因此，LLM-jp 上确有 post-prefill 风险排序信号，且 router 统计提供了 NLL 之外的信息；但 tail-risk recall 仍远低于可直接承担保护决策的水平。

## 跨模型检验

- OLMoE 训练的 `post_prefill_all` 迁移到 LLM-jp：ρ=0.142，95% CI [-0.244, 0.500]。
- LLM-jp 训练的同一 proxy 迁移到 OLMoE：ρ=0.219，95% CI [-0.142, 0.542]。
- 两个模型在同一批 sealed test 文档上的真实 degradation harm 相关性仅为 ρ=0.331。

这反驳了“prompt 有一个跨模型稳定 fragility 标签”的简单解释。风险更接近 `request × model × action`，甚至还可能依赖 layer、token phase 和 congestion action。

## 当前 Go / No-Go

### GO

- LLM-jp 内部、post-prefill 后、面向后续 decode/chunk 的 risk ranking。
- 将 router/gate 统计作为 action-specific harm estimator 的输入。
- 继续验证不同 quantization action、不同领域和真实生成 workload。

### NO-GO

- 用该 proxy 决定请求的第一个 prefill chunk。
- 声称 arrival-time、零历史信息即可隔离脆弱请求。
- 声称跨模型通用的 per-request fragility。
- 仅凭相关性结果声称实现了 quality fairness。
- 把保护请求的 KL 直接设为 0 后当作真实端到端收益。

## 下一轮必须完成

1. **Action generalization**：冻结 predictor，测试不同 tail count、MXFP4/NVFP4、drop/quantize 等动作；若排序不能迁移，controller 必须显式建模 action。
2. **真实生成时序**：以 prefill 统计预测 decode token 的实际 harm，而不是同一 teacher-forced prompt 的文档级 KL。
3. **Streaming debt**：实现 request/tenant 级 debt 更新、衰减、上限和 anti-reset；与 round-robin、DRR/VTC-like、公平随机和 oracle 比较。
4. **系统闭环**：真实 mixed-precision EP communication、controller 开销、TPOT/P99、吞吐和质量共同测量。
5. **任务质量**：在代码、数学、对话和长上下文任务上测 accuracy/EM/pass@k，而不只测 KL。

## 复现文件

- 采集与严格划分：`experiments/idea_a_mac/run_quality_isolation_proxy_gpu_strict.py`
- ablation 与跨模型迁移：`experiments/idea_a_mac/analyze_quality_isolation_proxy_strict.py`
- 冻结确认：`experiments/idea_a_mac/evaluate_quality_isolation_frozen_replication.py`
- OLMoE 输出：`experiments/idea_a_mac/outputs/quality_isolation_proxy_strict_olmoe_2026-07-20/`
- LLM-jp 首轮输出：`experiments/idea_a_mac/outputs/quality_isolation_proxy_strict_llmjp_2026-07-20/`
- LLM-jp 冻结确认输出：`experiments/idea_a_mac/outputs/quality_isolation_proxy_frozen_replication_llmjp_2026-07-20/`
- 跨模型输出：`experiments/idea_a_mac/outputs/quality_isolation_cross_model_transfer_2026-07-20.csv`

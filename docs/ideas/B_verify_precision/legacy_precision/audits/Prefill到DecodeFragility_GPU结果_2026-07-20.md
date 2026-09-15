# GPU 第四轮有效性实验：Prefill 信号能否预测未来 Decode Fragility

日期：2026-07-20  
GPU：NVIDIA GeForce RTX 5090 32 GB  
模型：LLM-jp optimal-sparsity E32K16  
结论：**当前 prefill-only Quality Isolation proxy 对真实未来 decode harm 判定 NO-GO。**

## 1. 为什么必须做这轮实验

第二轮 Quality Isolation 在同一 teacher-forced 文档上得到过冻结正结果：

- `post_prefill_all` 对 fixed-tail INT4 文档 KL 的 Spearman ρ=0.517；
- 95% CI [0.315, 0.673]。

但该标签和特征来自同一段 prompt forward。它只能证明“同一输入上的 router/NLL 与同一输入的降级 KL 相关”，不能证明：

> prefill 信号能够预测后续 autoregressive decode 的质量风险。

如果这个迁移不成立，prefill proxy 就不能用于 decode-only brownout 或 quality debt。

## 2. 严格实验协议

数据：

- WikiText-103 train；
- 固定 seed 20260720；
- offset 184；
- 48 篇此前未使用文档；
- 每篇 64-token full-precision prompt；
- 后续 16 token 做 teacher-forced autoregressive decode。

划分：

- 24 train；
- 8 validation；
- 16 sealed test。

因果时序：

1. prompt prefill 始终使用 full policy，生成真实 KV cache；
2. 记录 prefill lexical、router/gate 和 NLL 特征；
3. full decode trajectory 作为 reference；
4. 每个 action 从一份新的 full-prefill KV cache 开始；
5. approximate action 只在 decode token 启用；
6. 所有 trajectory 输入相同 ground-truth continuation token；
7. 标签为逐 decode step logit KL。

因此，本实验没有：

- 用 approximate prefill 污染 KV cache；
- 把同一 prompt 的 degradation KL 当作未来标签；
- 使用额外 degradation 的 KL 作为 proxy；
- 在 test 上选 feature group 或 ridge alpha。

## 3. Decode 动作

1. `fp8top8_rest_int4`：top-16 中前 8 rank 使用 FP8，后 8 rank 使用 INT4。
2. `rankk_drop_renorm`：删除最低 rank 一个 expert contribution，并重新归一化 gate。
3. `keep12_drop_renorm`：保留 top-12，删除后 4 个，即减少 25% assignments。
4. `keep8_drop_renorm`：保留 top-8，删除后 8 个，即减少 50% assignments。

当前 drop policy 在 Hugging Face expert loop 完成 expert compute 后才 mask output。因此本轮只测质量标签，不声称真实 latency saving。

## 4. Action harm 规模

48 篇全部文档的 mean decode KL：

- fixed-tail INT4：0.00343；
- drop 最低一个 expert：0.00439；
- drop 25%：0.02953；
- drop 50%：0.16671。

drop 50% 的平均 KL 约为 drop 25% 的 5.65 倍，已不属于温和 brownout。

sealed test 上：

- fixed-tail INT4：mean 0.00293，P95 0.00393；
- drop 最低一个：mean 0.00347，P95 0.00501；
- drop 25%：mean 0.02810，P95 0.03821；
- drop 50%：mean 0.15901，P95 0.25004。

## 5. 严格 proxy 结果

每个 action 单独在 validation 上选择 feature group 和 ridge alpha，随后用 train+validation refit。

### Fixed-tail INT4

- selected group：`post_prefill_all`；
- validation ρ=0.119；
- sealed test ρ=-0.103；
- 95% CI [-0.518, 0.373]；
- worst-decile AUROC=0.286；
- recall@10%=0。

### Drop 最低一个 expert

- selected group：arrival lexical；
- validation ρ=0.262；
- test ρ=-0.397；
- 95% CI [-0.835, 0.152]；
- recall@10%=0。

### Drop 25%

- selected group：arrival lexical；
- validation ρ=0.190；
- test ρ=-0.076；
- 95% CI [-0.486, 0.412]；
- recall@10%=0。

### Drop 50%

- selected group：arrival lexical；
- validation ρ=0.595；
- test ρ=0.044；
- 95% CI [-0.545, 0.583]；
- worst-decile AUROC=0.750；
- recall@10%=0.5。

validation 的 0.595 完全没有在 sealed test 复现，是小 validation split 上典型的 selection noise。

## 6. 冻结旧 proxy 的迁移

为了排除“本轮重新训练样本太少”的解释，将第二轮已经冻结的：

- source label：same-prompt fixed-tail INT4；
- feature group：`post_prefill_all`；
- ridge α=1；
- fit data：原 48 train + 16 validation；

原样应用到本轮 16 个 decode test 文档。

结果：

- future fixed-tail INT4：ρ=0.350，CI [-0.224, 0.756]，recall@10%=0；
- drop 最低一个：ρ=0.244，CI [-0.351, 0.819]，recall=0；
- drop 25%：ρ=0.029，CI [-0.510, 0.579]；
- drop 50%：ρ=-0.053，CI [-0.568, 0.503]。

所有置信区间均跨 0，也没有可靠找回最差请求。

因此第二轮正结果不能升级成“prefill 预测未来 decode”的证据。

## 7. Action-specific fragility 是否存在

sealed test 上真实 harm 的跨 action Spearman：

- fixed-tail INT4 vs drop 最低一个：0.412；
- fixed-tail INT4 vs drop 25%：0.341；
- fixed-tail INT4 vs drop 50%：0.129；
- drop 最低一个 vs drop 25%：0.482；
- drop 最低一个 vs drop 50%：0.400；
- drop 25% vs drop 50%：0.865。

这说明：

1. 相近的 drop action 共享部分 request fragility；
2. communication quantization 与 aggressive expert drop 的风险排序明显不同；
3. “每个请求有一个固定 fragility 标签”不成立；
4. 当前问题确实是 `request × action × decode state`。

## 8. 研究决策

### 判定 NO-GO

- 使用静态 prefill router/NLL 特征预测整段未来 decode risk；
- 直接把第二轮 same-prompt ρ=0.517 用作 decode controller 依据；
- 当前版本的 prefill-only quality-debt guarded brownout；
- 用单一 fragility score 控制 quantization 与 expert drop。

### 仍可尝试

1. **One-step online predictor**：用当前 decode step 正常产生的 router margin、logit entropy、hidden norm 预测下一个 step 的 action harm。
2. **Action-conditional predictor**：不同动作使用不同 head 和 uncertainty calibration。
3. **Predictor-free fairness**：consecutive-K、round-robin、tenant debt 只保证降级次数公平，不声称知道谁更脆弱。
4. **温和 actuator**：只 drop 最低一个 expert；drop 25%/50% 质量代价过大。

上述方向仍需新的 sealed test。当前数据不支持直接进入系统实现。

## 9. 复现文件

- 主实验：`experiments/idea_a_mac/run_decode_fragility_strict_gpu.py`
- 冻结迁移：`experiments/idea_a_mac/analyze_frozen_prefill_proxy_to_decode.py`
- 新 policy：`keepN_drop[_renorm]`，位于 `experiments/idea_a_mac/policies.py`
- 输出：`experiments/idea_a_mac/outputs/decode_fragility_strict_llmjp_2026-07-20/`

## 10. 证据边界

- test 只有 16 篇，置信区间较宽；但四个重新训练 proxy 和旧冻结 proxy 均未通过，不能以扩大样本为由继续宣称正结果。
- teacher forcing 保证 trajectory 可比，但不等于 free-running generation 的任务质量。
- drop 尚未在 fused MoE kernel 前裁剪 assignment，因此没有 latency 结果。
- 当前只运行 LLM-jp；由于它是第二轮表现最好的模型，在该模型上失败已足以暂停 prefill-only claim。

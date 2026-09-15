# 2026-07-20（续）：receiver-aware v3 自适应控制器 + Quality Isolation / 路由预取协同性检验

> **【事后更正】** 本文档"一、receiver-aware v3"部分的adaptive controller结果存在因果评估错误（把检测cutoff之后才选定的策略回溯应用到warm-up期），已被同日晚些时候的`run_receiver_aware_v3_causal_audit.py`推翻——修正后adaptive相对最强固定基线(`always_causal_prev_step`)的优势在4个detect_frac×2模型组合下95%CI全部跨0，不构成统计显著的系统改进。详见`GPU首轮有效性实验结果_receiver_causal_ExpertPrefetch_2026-07-20.md`和`MoE_Approach_Registry_2026-07-19.md`。"二、协同性检验"部分（Quality Isolation与路由预取信号无重叠）不受影响，结论仍然有效。

## 一、receiver-aware v3：因果合法的拥塞regime检测器 + 自适应控制器

**背景**：v2 已经精确诊断出问题——"结构性拥塞"(hotspot)下静态离线画像(calib_static)几乎已经是最优解，"瞬时拥塞"(balanced)下现实可行的在线信号(causal_prev_step)反而比随机更差。但v2从未构造过一个能在**部署时**（不知道ground-truth regime标签）区分这两种情况的控制器——它只是"因为知道标签"分别报告了两组数字。v3补上这一环。

**方法**（`run_receiver_aware_v3_adaptive.py`）：
1. **检测器离线校准**（一次性，只用calibration场景，不接触test set）：分别在"hotspot"和"balanced"两种calibration场景上，测量receiver负载的Herfindahl集中度指数（load在各receiver间的集中程度：1/ep_size=完全均匀，1=全部集中在一个receiver），取两者中点作为判决门槛。
2. **在线检测**（因果合法）：每个test场景只用前30%的global step（一个短暖启动窗口，不看后续），计算同一个集中度统计量，与门槛比较。
3. **策略切换**：判定为"结构性" → 用`calib_static`（该regime下v2证实的近似最优解）；判定为"瞬时" → 退化为`random`（避免causal信号在该regime下的负面效应）。

**关键设计要点**：这个检测器不依赖"知道自己在哪个regime"这种作弊信息，只依赖一个短暖启动窗口里可以真实测量的统计量——这是它相对v2最重要的系统性推进：v2停留在"诊断问题"，v3给出"部署时可执行的解法"。

**结果**（真实OLMoE + LLM-jp路由数据，24 scenario seeds，50%hotspot+50%balanced混合测试集，控制器不知道当前处于哪个regime）：

| model | 检测准确率(balanced/hotspot) | adaptive | always_calib_static | always_causal | always_random |
|---|---|---|---|---|---|
| olmoe | 100.0% / 87.5% | **0.1706** | 0.1521 | 0.1643 | 0.1232 |
| llmjp | 100.0% / 91.7% | **0.1729** | 0.1399 | 0.1594 | 0.1225 |

**adaptive在两个模型上都超过全部三个"固定策略"基线**（包括之前v2认为"整体最强"的causal_prev_step），且是在**不知道当前regime**的混合测试集上取得的。按regime拆分看：

| model | regime | adaptive | calib_static | causal | random |
|---|---|---|---|---|---|
| olmoe | balanced | 0.1225 | 0.0733 | 0.0993 | 0.1224 |
| olmoe | hotspot | 0.2187 | 0.2309 | 0.2293 | 0.1239 |
| llmjp | balanced | 0.1204 | 0.0464 | 0.0865 | 0.1205 |
| llmjp | hotspot | 0.2254 | 0.2334 | 0.2334 | 0.1246 |

在balanced下，adaptive几乎精确落在random的水平（正确避开了causal的负面效应，比causal多省2.3-3.4个百分点）；在hotspot下，adaptive因为87.5%-91.7%的检测准确率而略低于"事后知道标签"的calib_static（-1.2~-0.8个百分点），但仍远超causal/random。

**结论**：这是一个真实、可部署、跨两个模型一致的**正向系统改进**——不需要ground-truth标签，只用一个短暖启动窗口的负载集中度统计量，就能自动选对v2诊断出的"正确策略"，把receiver-aware从"两个互相矛盾的固定策略、不知道该用哪个"升级为"一个能自我调节的控制器"。局限：检测准确率不是100%（尤其hotspot方向有8-13%的误判），仍是纯带宽分析回放，真实RDMA队列/incast环境下的集中度统计量测量误差可能更大。

---

## 二、Per-request Quality Isolation 与 路由预取信号：目前无重叠

**问题**：两者是否共享同一份"文档难度"诊断，能否合并成一个统一信号？

**检验方法**（`run_quality_routing_synergy_check.py`，纯本地Mac分析）：用与Routing-Predictability P0**完全相同**的calibration/test路由数据（`olmoe_r_layout_article_stage1_formal_2026-07-13`，wikitext2_docs:test offset16 n=45），在**同一批45篇文档**上补跑一次`fixed_tail4`降级的KL（`run_layer_budget_experiment.py --dataset-split test --test-offset 16 --test-samples 45`），确保sample_id直接对应同一篇文档。然后计算：
- 每篇文档的路由可预测性（用calibration学到的top1转移表，在该文档上的真实命中率）
- 每篇文档在`fixed_tail4`降级下的`mean_token_kl`
- 两者的Spearman相关系数

**结果**：
```
Spearman(top1_hit_rate, mean_token_kl)      = -0.177  (p=0.245，不显著)
Spearman(mean_routing_entropy, mean_token_kl) = -0.021  (p=0.891，不显著)
```

**结论**：**目前没有重叠**——文档级"路由可预测性"（决定专家预取该不该激进）和文档级"质量降级风险"（决定该不该优先保护）在统计上互相独立（n=45，相关系数不显著）。这意味着两个系统**不能共享同一个诊断信号**，必须分别计算：Quality Isolation应该继续使用它已经验证过的信号（另一个已算过的降级机制的KL，Spearman 0.75-0.98，见上一轮报告），而不能用路由预取的命中率/熵作为廉价替代。这是一个诚实的负结果，但避免了"强行合并两个本来独立的机制"这个潜在的设计陷阱。

局限：n=45文档量适中但不算大，且只测了OLMoE一个模型；"路由可预测性"和"KL风险"没有相关性也可能是因为二者本身作用在不同的因果层面（前者关于专家身份序列的规律性，后者关于量化误差对下游分布的影响幅度）——这本身也是一个有意思的系统性观察：**MoE系统里"容易预测的"和"容易被降级破坏的"是两个正交的维度**，不能用一个信号偷懒覆盖两个决策。

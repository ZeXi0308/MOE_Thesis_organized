# Additive-KL 建模审计：能否用轻量校正救回可加性假设（2026-07-20）

> **SUPERSEDED（2026-07-21）**：本文中的 `locked_additive` **3.77×** 及“残差流非线性导致 3–5×”叙事作废。根因是 `sum(单层 KL(full,·))` 重复累加共享 FP8 baseline。正确增量公式与新判决见 [`../../errata/判死结论勘误_2026-07-21.md`](../../errata/判死结论勘误_2026-07-21.md) 与 `experiments/idea_a_mac/outputs/additive_kl_audit_corrected_2026-07-21/`。下文仅作历史记录，禁止继续引用 3.77×。

> 背景：additive-KL MILP 早在2026-07-14就因"单层sensitivity不可加和预测多层端到端KL"被[Observed]判死，但从未拆解过不可加性的具体来源，也从未测试过"轻量级校正项能否救回一个足够便宜的近似可加模型"（比重新做端到端MILP搜索便宜得多）。本实验补上这个建模侧的空白。

## 实验设计

用OLMoE-1B-7B、MXFP4尾部精度（与本项目PLTB/layer_budget系列实验同一设置，结果可直接比较），在calibration split(n=16)上测6个独立层(0,3,6,9,12,15)各自的**单层隔离扰动**KL(routing锁定vs自由两种)，学习一个全局漂移比例；在sealed test split(n=32)上测同样6层**单独扰动**(用于构造加性预测)以及**联合同时扰动**(真实联合效果，作为ground truth)，三个预测器对比：

- `naive_additive`：直接对各单层free-routing KL求和（MILP原本的失败假设）
- `locked_additive`：对各单层**锁定路由**的KL求和（排除跨层路由漂移这一交互源，纯数值误差是否可加）
- `locked_additive_plus_global_drift`：用locked_additive乘以(1+校准集学到的全局漂移比例)做校正（最廉价的修复尝试）

## 结果（n=32 sealed test，bootstrap CI，与真实联合效果对比）

| predictor | 预测/真实比值 | 95% CI | 平均相对误差 |
|---|---:|---|---:|
| naive_additive | **4.63×** | [4.48, 4.80] | 366% |
| locked_additive | **3.77×** | [3.67, 3.87] | 276% |
| locked_additive_plus_global_drift | **2.51×** | [2.37, 2.69] | 160% |

三者全部严重高估真实联合KL，置信区间全部远离1.0（不是噪声）。全局漂移校正把误差从4.63倍降到2.51倍，仍然远远不够用（160%平均相对误差，实用场景完全不可接受）。

## 关键发现：不可加性的主因不是路由漂移，是数值误差本身的非线性交互

`locked_additive`（已经排除了路由漂移这个交互源，路由被强制锁定为全精度模型的选择）依然高估3.77倍——这意味着**即使没有任何路由变化，多层同时量化产生的真实损伤也远小于各层独立损伤之和**。这排除了"额外做一次路由漂移建模就能修复"这个假设，指向一个更根本的机制：量化误差在残差流(residual stream)里存在**饱和或部分抵消效应**，不是简单累加——比如后面层的计算可能会部分"吸收"或"纠正"前面层引入的量化噪声，使得联合损伤明显小于线性叠加的预期。

## 结论：确认原判决，且排除了"轻量校正修复"这个后续假设

**additive-KL建模路线彻底无法挽救**，不只是"MILP不该用"，是"任何试图用单层信息线性/可加地预测多层组合效果"的思路在这个系统里都会系统性高估3-5倍——不是校正系数没调好，是加性框架本身与真实的误差交互机制矛盾。这与本项目已确立的"元模式六"（简单静态信号已逼近oracle上限）是不同性质的教训：这次不是信号不够强，是**函数形式假设(加性)本身错误**，值得作为独立的元模式记录，提醒未来任何"用单点/单层profiling外推组合效果"的候选都要先做类似的联合vs独立对照检验，而不是想着"用一个校正系数补救"。

## 证据与复现

- 实验脚本：`experiments/idea_a_mac/run_additive_kl_modeling_audit.py`
- 完整结果：`experiments/idea_a_mac/outputs/additive_kl_audit_2026-07-20/`（n=16 calib / n=32 test，GPU RTX 5090实测）
- 复用基础设施：`capture_moe.py`的`lock_routing`/`routing_cache`机制（原为`run_drift_attribution.py`的单策略drift分解设计，本实验首次将其用于多层联合场景的可加性检验）

# Receiver-Aware v2：系统化跨模型时序重跑的最终结论

> 日期：2026-07-20。范围：`MoE_全部候选选题完整时间线_2026-07-19.md` 里的 receiver-aware(v1)（2026-07-13前提出，此前状态"降级"）。本文档取代此前 `run_receiver_isolation_experiment.py` 单模型/单场景重跑给出的初步结论。

## 一句话结论

**receiver-aware 的价值完全取决于拥塞是"结构性"还是"瞬时性"：结构性拥塞下纯离线校准就能拿到几乎全部收益（不需要在线信号）；瞬时性拥塞下需要真正的在线信号，但现实中唯一可得的在线信号（上一时刻已实现负载）本身就不够，效果反转为负。它不是一个能直接部署的通用机制，而是一个条件成立的机制——条件是"热点位置是否可预测"，这个条件本身在真实系统里几乎总能被更简单的静态放置策略满足，削弱了"receiver-aware"作为独立创新点的价值。**

## 实验设计（相对v1的四项系统化扩展）

1. **跨模型**：OLMoE(E64K8) + LLM-jp(E32K16)，路由数据来自 WikiText-103（`capture_routes_v2.py`，40篇/模型，offset=0，此语料库此前从未被本项目任何实验触碰过，天然是干净的sealed数据，不需要额外排除注册表）。
2. **跨文档**：job池来自40篇真实文档独立采样（`--num-jobs 16`，24个scenario seed各自独立采样文档子集+到达错位），而非v1的单一route文件复制。
3. **时序动态**：job以错位到达（stagger∈[0, num_layers*0.5]）模拟continuous batching下不同请求处于不同layer进度，全局拥塞状态随时间真实变化，不是静态快照。
4. **信息陈旧度分级**（这是最关键的扩展）：
   - `oracle_same_step`：用当前时间步的真实负载评分（v1的假设，一种"同时刻先知"，是信息可用性的上界）；
   - `causal_prev_step`：只用上一时间步已实现的负载评分（真实在线调度器能拿到的信息）；
   - `calib_static`：用独立校准场景算出的固定离线画像评分（零在线信号，下界）。

性能优化：把逐token的字节分配从"每次重新扫全表"改为"预计算全FP8基准计数+对被选中候选行做稀疏增量修正"，单次实验耗时从最初卡住数小时（原始实现，24 seeds需要重新处理全部候选表22次/cell）降到16.5分钟（24 seeds×2 placements×2 origin_modes×3 budget_fractions，两个模型）。

## 核心结果（24 seeds，`frac_seeds_hot_beats_random_ci`几乎全部=1.0，即24个seed里全部落在random 95% CI之外，不是噪音）

| origin_mode | oracle_same_step | causal_prev_step | calib_static | staleness cost (oracle−causal) |
|---|---:|---:|---:|---:|
| **balanced**（瞬时/随机拥塞） | **+2.3%~+6.1%**（正） | **-1.1%~-2.8%**（负！） | -1.7%~-7.6%（更负） | +4.2%~+8.0%（跨2模型2placement高度一致） |
| **hotspot**（结构性/持续拥塞） | +5.8%~+11.1% | +5.6%~+11.0% | +5.7%~+10.9%（几乎与oracle重合） | 仅+0.1%~+0.3%（几乎为0） |

**P99 job暴露量代理指标同样支持这个结论**：hotspot下hot策略的P99显著低于random（如OLMoE hotspot frac=0.5: hot P99=7359us vs random median=8436us，降低约13%）；balanced下causal策略的P99反而普遍高于或接近random中位数（如OLMoE balanced frac=0.5: causal P99=2831us vs random median=2734us，是变差的）。

## 解读

1. **v1重跑的"复活"结论是真实存在的，但只在一种特定场景下成立，且这个场景恰好是"信息陈旧度不重要"的场景。** v1用的是`oracle_same_step`假设（同时刻先知），在hotspot类场景下这个假设几乎不产生额外价值（因为hotspot的位置本身是结构性、持续的，静态画像已经知道），而在balanced类场景下这个假设制造了一个不存在的优势——现实中拿不到"同一时刻"的完整负载信息，只能用滞后一步的因果信息，一旦换成这个更现实的约束，balanced场景下的优势直接反转为劣势。
2. **"结构性拥塞下静态画像即可、不需要在线感知"这个发现本身削弱了receiver-aware的必要性**：如果热点是结构性的（比如某几个receiver因为流量模式持续过载），最直接的解法是调整专家/请求的静态放置（这正是MassCover-EP、R-layout等已经在做、且已经被证明效应量不够的方向），而不是引入一套在线负载感知机制；一套复杂的在线机制如果只在"静态方法也能解决"的场景下有效，在"真正需要在线感知"的场景下反而失效，价值定位就很尴尬。
3. **这跟TokenRace-EP的教训是同一类，但更早被发现（不需要真实GPU就能发现）**：机制在一个过于乐观的信息可用性假设下看起来有效，一旦换成真实系统能提供的信息粒度，效果消失甚至反转。这进一步印证了Approach Registry里"元模式五"的普适性——不仅是硬件开销会被仿真低估，调度决策所需的信息时效性也同样容易被过度乐观地建模。

## 最终判定

**receiver-aware(v1) 从"降级"改判为"CONDITIONAL_STRUCTURAL_ONLY"**：不是无效，而是有效范围被精确限定在"拥塞位置结构性、可被静态画像捕捉"的场景，这个场景下静态放置策略是更简单的替代方案；在真正需要在线感知的瞬时拥塞场景下，现实可得的信息粒度（causal_prev_step）不足以支撑正收益。不建议作为独立候选继续投入，除非能找到一个"拥塞位置结构性拥塞、但现有静态放置方法无法覆盖"的具体子场景（目前没有找到）。

## 证据与复现

- 路由数据：`experiments/idea_a_mac/outputs/receiver_aware_v2/olmoe_routes.csv`、`llmjp_routes.csv`（GPU远程实例上，40篇WikiText-103文档，`capture_routes_v2.py`采集）
- 完整结果：`outputs/receiver_aware_v2_2026-07-20/{receiver_aware_v2_raw.csv, receiver_aware_v2_summary.csv, receiver_aware_v2_staleness_cost.csv, report.md}`
- 实验脚本：`experiments/idea_a_mac/run_receiver_aware_v2_systematic.py`（含分析式增量字节计算优化）、`capture_routes_v2.py`
- 语料库扩展：`prompts.py` 新增 `wikitext103_docs` dataset选项（复用WikiText-2的文章解析逻辑，切换到`wikitext-103-raw-v1`config，池子从~121篇扩大到数千篇）

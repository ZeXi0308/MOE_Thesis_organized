# 请求粒度动作吸收不了小额 KV 赤字：一个在准入时刻可判定的结构性边界

2026-09-12。分支 `agent/publish-current-moe-code`，HEAD `7059fc9`（工作区有未提交的 expert-union 改动，全部保留）。
**本轮 GPU 新执行数为 0。** 全部结论来自 4 个已封存 native vLLM cell 的重算，原始 `raw.json` 未修改。
本文不改 `docs/current/README.md`，不改任何旧裁决，不 push。

---

## 0. 本轮定位

上一轮（[`20260912_kv_budget_r01`](../20260912_kv_budget_r01/RESEARCH_REPORT.md)）测到：在 3072/1024 长上下文域，
KV 预算 0.90→0.95 消除了 4.59 s 的秒级暂停，吞吐 +3.99%/+4.70%。结论写成 `MEASUREMENT_ONLY`，
并明确指出"不能写成**同显存预算**的策略收益"。

那正是缺口所在。本轮问的是同一问题的下一环，且不需要新 GPU：

> 同一显存预算下，是否存在一个合法调度动作，能拿到加显存所拿到的东西？
> 如果不能，**是信息不够、动作窗口关闭、还是动作粒度不匹配？**

这三个原因导向完全不同的下一步，必须分开。

---

## 1. 结论

在本运行域，三个原因里**只有第三个成立**，而且可以精确量化。

| 候选原因 | 判据 | 结果 |
|---|---|---|
| 信息不够 | 能否在事前预测抢占的发生与时刻 | **否证**。步 119–519 的观测即可预测步 806 的抢占，误差 1 步 |
| 动作窗口关闭 | 准入动作窗口 vs 约束生效时刻 | **成立但非根因**。窗口在步 94 关闭，约束在步 807 生效，滞后 713 步 |
| 动作粒度不匹配 | 赤字规模 vs 最小动作单位的代价 | **成立，是根因**。6.4% 的显存赤字，两种请求粒度动作的代价都是 17%–4400% |

一句话：**这不是一个预测问题，也不是一个时机问题，是一个粒度问题。**
赤字 521 块，而请求粒度动作的最小单位是 237–256 块，且其代价不按赤字比例计费，
而是按"一个请求的整段生命周期"计费。

---

## 2. 三条可复算的定量结果

代码：[`kv_deficit_model.py`](../../../experiments/admission_capacity/kv_deficit_model.py)（纯原语，36 项测试）、
[`analyze_kv_deficit.py`](../../../experiments/admission_capacity/analyze_kv_deficit.py)。
块大小 **不假设**，由两个受害者的 `(computed_tokens, block_counts)` 唯一反解为 16；不唯一或不自洽都抛错。

### 2.1 抢占在准入时刻已被声明边界决定（4/4）

共存需求只用声明量：`N x ceil((P+O_max)/B)`。

| cell | usable | 需求 | 余量 | 预测抢占 | 实测抢占 |
|---|---:|---:|---:|:--:|:--:|
| repeat0-budget90 | 7671 | 8192 | **−521** | 是 | 2 |
| repeat1-budget90 | 7671 | 8192 | **−521** | 是 | 2 |
| repeat0-budget95 | 8425 | 8192 | +233 | 否 | 0 |
| repeat1-budget95 | 8473 | 8192 | +281 | 否 | 0 |

`static_test_agreement = 4/4`。这里没有任何遥测、任何动态、任何拟合——只有四则运算。

更一般的形式是一个**竞赛判据**（`preemption_race`）：池子耗尽早于首个自然释放才会抢占。
它包含上表，并能正确处理"需求超池但先排空"的情形。两个操作数都可在准入时刻算出：
`first_release_step(prefill_steps=3, max_output=1024) = 1027`，实测首次完成在步 **1025**。

### 2.2 耗尽轨迹是线性的，前向预测误差 1 步

在**固定因果窗 [119, 519]** 上拟合（窗口在末次准入后打开、在任何抢占与任何完成之前关闭，
因此是外推不是内插；若窗口会看到抢占，分析器直接抛错拒绝出结果）：

| 量 | 值（4 个 cell 全同） |
|---|---|
| 拟合斜率 | **−2.00000 块/步** |
| 名义斜率 `width/B = 32/16` | −2.0 |
| 401 点最大残差 | **1.31 块** |

外推结果：

| cell | 预测耗尽步 | 首次释放步 | 预测 | 实测首次抢占 |
|---|---:|---:|:--:|---:|
| budget90 ×2 | **807** | 1025 | 抢占 | **806** |
| repeat0-budget95 | 1184 | 1025 | 不抢占 | 无 |
| repeat1-budget95 | 1208 | 1025 | 不抢占 | 无 |

在步 519 就能说出步 806 会发生什么，误差 1 步。**信息不是瓶颈。**

### 2.3 停顿律：受害者等的是"下一个自然完成"，不是重算

四个受害者（两个 cell × 2）的等待全部由一条规则给出——
被抢占者按恢复次序等待第 `rank` 个自然完成：

| cell | 受害者 | 抢占步 | rank | 预测等待 | 实测等待 | 绝对误差 |
|---|---|---:|:--:|---:|---:|---:|
| r0-b90 | article-0003571 | 928 | 0 | 1.912275 s | 1.913225 s | **0.95 ms** |
| r0-b90 | article-0003640 | 806 | 1 | 4.472705 s | 4.473617 s | **0.91 ms** |
| r1-b90 | article-0003571 | 928 | 0 | 1.894909 s | 1.895854 s | **0.94 ms** |
| r1-b90 | article-0003640 | 806 | 1 | 4.441975 s | 4.442875 s | **0.90 ms** |

4/4 全部 < 1 ms。误差方向一致，来源是抢占时间戳取的是受害者最后一次收据（早于引擎决策亚毫秒量级）；
测试因此用绝对毫秒容差，不声称相对精确。为防止偶然拟合，另设负控：
给 `article-0003640` 用错误的 rank 0，误差跳到 > 100 ms。

互斥分桶的赤字账本：

| cell | 总停顿 | 等待 | 重算 | **等待占比** | 最长停顿 | 释放块 |
|---|---:|---:|---:|---:|---:|---:|
| repeat0-budget90 | 6.6240 s | 6.3868 s | 0.2372 s | **96.42%** | 4.5906 s | 482 |
| repeat1-budget90 | 6.5767 s | 6.3387 s | 0.2379 s | **96.38%** | 4.5597 s | 482 |

**96.4% 的停顿是容量等待，3.6% 才是重算。** 这条直接改变机制方向：
凡是优化重算代价的做法（swap 替代 recompute、前缀缓存、部分重算加速）在本域最多触及 3.6%。

桥接算术也闭合：耗尽步 807 → 首次释放步 1025 共 218 步，需 `218 x 2 = 436` 块；
单个受害者产出 237 块，故最少 `ceil(436/237) = 2` 个受害者，**实测正好 2 个**。
原生策略在"抢占个数"这一维上已经是最省的，并且是惰性的——
第二个受害者在步 928 才被抢，因此只付 1.91 s 而不是 4.47 s。**原生基线不弱。**

---

## 3. 为什么请求粒度动作在这里必然昂贵

### 3.1 两个端点的实测代价（同引擎参数，仅准入上限不同）

来自 [`20260908_native_preemption_r01`](../20260908_native_preemption_r01/REPORT.md) 的封存 cell，
`gpu_memory_utilization=0.9` 等全部 engine args 逐项相同：

| 策略 | 最长 ITL | 最长 TTFT | 整批墙钟 | 抢占 |
|---|---:|---:|---:|:--:|
| cap32 原生（抢占吸收） r0 / r1 | 4.473 s / 4.512 s | 0.768 s / 0.803 s | 23.161 s / 23.311 s | 2 / 2 |
| cap29 可行性保守（by construction） r0 / r1 | **0.108 s / 0.108 s** | **17.911 s / 18.117 s** | 27.095 s / 27.293 s | 0 / 0 |

cap29 正是模型给出的 `feasible_concurrency = floor(7671/256) = 29`。它**确实兑现了承诺**：
抢占归零，最长 ITL 降低 41 倍。代价是把等待整体搬到首 token 之前——最长 TTFT 涨 23 倍，墙钟 +17.0%/+17.1%。

这与 [`20260911_aimability_r01`](../20260911_aimability_r01/REPORT.md) 在短文本域发现的
"cap 把 TTFT 失败换成 TPOT 失败"是同一个守恒签名，但这次发生在**已判别的运行域**里，并且有了结构性解释。

### 3.2 赤字只有 6.4%，代价却是 17%

块大小 2 MiB（`14.984375 GiB / 7672`）。

| 量 | 值 |
|---|---|
| 赤字 | 521 块 = **1.018 GiB** |
| 占终端需求 | 521/8192 = **6.36%** |
| 占可用池 | 521/7671 = **6.79%** |
| 加显存方案实际补了 | 754 / 802 块 = 1.473 / 1.566 GiB |

于是三点权衡是：

| 方案 | 最长 ITL | 最长 TTFT | 墙钟 | 显存 |
|---|---:|---:|---:|---|
| cap32 + 抢占 | 4.47–4.51 s | 0.77–0.80 s | 23.16–23.45 s | 0.90 |
| cap29 保守准入 | 0.108 s | 17.91–18.12 s | 27.10–27.29 s | 0.90 |
| cap32 + 多给 6.4% KV | 0.083–0.099 s | 0.72–0.83 s | 22.28–22.55 s | 0.95 |

**第三行在每一项上都优于前两行。** 也就是说：同预算下的两个请求粒度动作互不支配，
而且都被"多给 6.4% 显存"支配。这就是同预算收益缺口的确切形状。

### 3.3 错峰准入为什么救不了

有人会问：不要整槽等待，只把几个请求**延后一点**入场行不行？

不行，而且原因可以写成一行。请求一旦 prefill 完成就立刻持有 `ceil(3072/16) = 192` 块，
占满载 256 块的 **75%**。准入时刻的延后只能削减 decode 增长那部分，即每请求至多
`decode_blocks = 64` 块。要覆盖 521 块赤字：

```
k x min(d/B, 64) >= 521   =>   k >= ceil(521/64) = 9
```

**至少要延后 9/32 个请求，每个至少 928 步**（输出长度本身才 1024 步）。
这已经不是"错峰"，是另一种形式的整段序列化。

由此得到一条可外推的结构规律，且模型中已参数化：

> 准入时刻动作的全部动态范围等于 footprint 的 **decode 份额**。
> `prompt >> output` 时该份额趋于 0，准入控制退化为纯并发削减。
> 本域 prefill 份额 0.75，所以准入控制几乎没有杠杆。

模型中的反向对照（短 prompt 长输出，128/4096）`prefill_share < 0.05`，
`min_deferred_requests` 显著更小——**同一公式预测准入错峰在那个域重新变得有效**，
这是一条可证伪的预测，本轮未执行。

---

## 4. 这如何收窄并解释既有负结果

| 既有结果 | 本轮给出的结构性原因 |
|---|---|
| 四个反馈式机制 goodput 符号不稳 | 反馈观测到池子告急时，准入窗口已关闭 713 步；反馈控制的是一个**当时已不存在的动作** |
| "降 cap 时 waiting 已为空" | 闭合 cohort 的准入窗口在步 94 关闭，这是必然而非偶发 |
| cap29 吞吐 −17% | 不是实现问题：6.4% 赤字 × 请求粒度 = 一个整槽生命周期 |
| 加 KV 有效但"不是同预算收益" | 确认。同预算下不存在等效的请求粒度动作 |

**被收窄而不是被推翻的**：这些负结果依然成立，但它们的适用范围现在只到
"请求粒度动作 + prefill 主导 footprint"这一层，不能外推为"KV 压力域没有调度收益"。

---

## 5. 由此定义的主问题与成功判据

本轮不换题，而是把主问题收窄到一句可证伪的话：

> **在固定显存预算下，能否用一个亚请求粒度的 KV 动作，以 O(δ) 的代价吸收 δ 比例的显存赤字，
> 而不是当前两个端点的 O(一个请求整段生命周期)？**

成功判据在执行前冻结：对 δ = 6.4% 的赤字，机制必须同时满足

1. 最长 ITL 相对 cap32 原生显著下降（目标量级：4.5 s → 亚秒）；
2. 最长 TTFT 不出现 cap29 那种量级的转移（保持 < 1 s）；
3. 墙钟不劣于 cap32 原生（23.2–23.5 s）；
4. 显存预算与 cap32 原生逐字节相同（`gpu_memory_utilization=0.9`）。

四项有一项不满足即为该 formulation 的负结果。注意 0.95 那一行是**参照上界**而不是竞争对手：
它多花了 1.47 GiB，不在同预算比较里。

### 机制方向（尚未实现，不作为已有结果）

停顿律说明杠杆在"等待"而不是"重算"，粒度分析说明单位必须小于一个请求。
两者合起来指向同一类动作：**尾部 KV 回滚**——只丢弃若干请求末尾的 `m` 个块，
之后按 chunked-prefill 重算这 `m x 16` 个位置，而不是整请求重算。

它在本数据上的算术是：赤字 436 块（桥接需求）摊到 32 个请求是每请求 **13.6 块**，
即每请求丢弃约 218 个 token 的 KV，而不是让 2 个请求各丢 3780–3905 个 token。
预期效果是把 6.4 s 的集中停顿摊成每请求约 0.2 s。

**这一段是机制设计，不是结果。** 明确未知项：vLLM v1 没有部分驱逐接口，
需要实测的量是（a）尾部重算能否与正常 decode 合批、（b）32 次小重算的总 GPU 成本
是否低于 2 次大重算的 0.237 s、（c）块表操作的实现税。任何一项都可能翻盘。

---

## 6. 明确不主张

- **不主张任何策略收益。** 本轮零执行，没有任何反事实轨迹。
- 2.1–2.3 是对 4 个封存 cell 的**事后重建**，证据层级 `NATIVE_SERVING` 但性质是描述性的。
- 停顿律样本是 **4 个受害者**，来自 2 个 cell、1 个工作负载点、1 个模型、1 张 5090。
  4/4 < 1 ms 说明模型形式对，不说明它在其他 cohort 结构下成立。
- 竞赛判据用的是**声明**输出上限。请求提前 EOS 时首次释放更早，判据偏保守；
  本工作负载全部请求恰好生成满 1024 token，这是最有利于该判据的情形。
- 静态判据在 budget95 上偏保守 84 块（预测余量 233/281，实测最小空闲 317/365），
  原因是到达错开 94 步使各请求不同时达峰。偏差方向安全，但不是零误差。
- `feasible_concurrency` 由池子推出，未与 `max_num_seqs` 取交；repeat1-budget95 报 33 只是算术值。
- §3.3 的错峰结论是**模型推论**，不是执行结果。短 prompt 域的反向预测 `UNRUN`。
- §5 的尾部 KV 回滚是设计，无任何实现与实测。
- 32 请求的封闭 cohort 不支持生产尾部推断；本文只用最大值与逐请求值，不引用 p99 作为生产结论。

---

## 7. 复算

```bash
cd "<repo root>"
# 36 项原语测试
(cd refine-logs/expert_saturation/experiments/admission_capacity && \
  python3 -m unittest test_kv_deficit_model -v)

# 四个封存 cell 的赤字分析
R=refine-logs/expert_saturation/outputs/admission_capacity/20260912_kv_budget_r01
O=refine-logs/expert_saturation/outputs/admission_capacity/20260912_kv_deficit_law_r01
python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_kv_deficit.py \
  --cell $R/gpu_results/repeat0-budget90 --pause-ledger $R/cpu_analysis/current-repeat0-budget90.json \
  --cell $R/gpu_results/repeat1-budget90 --pause-ledger $R/cpu_analysis/current-repeat1-budget90.json \
  --cell $R/gpu_results/repeat0-budget95 --pause-ledger $R/cpu_analysis/current-repeat0-budget95.json \
  --cell $R/gpu_results/repeat1-budget95 --pause-ledger $R/cpu_analysis/current-repeat1-budget95.json \
  --output $O/analysis/kv_deficit.json
```

产物：[`analysis/kv_deficit.json`](analysis/kv_deficit.json)、[`analysis/summary.txt`](analysis/summary.txt)。
原始 `raw.json` 未修改；旧 attempt 目录未触碰。

---

## 8. 源码级可行性核查（本轮同场完成）

§5 判据 4 要求同预算，所以必须先确认尾部 KV 回滚在 vLLM 0.26.0 里**能不能表达**。
核查对象为 tag `v0.26.0` 的上游源码，阅读范围限定三处：
[`kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/kv_cache_manager.py)、
[`single_type_kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/single_type_kv_cache_manager.py)、
[`sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/sched/scheduler.py)。
**未运行、未安装、未实现。** 以下是阅读结论，不是测量结果。

### 8.1 所需原语已存在，但语义方向相反

`SingleTypeKVCacheManager._remove_blocks_in_range(request_id, first_block, last_block)`
正是需要的那一个操作：倒序释放某请求块表的一段连续块，归还 block pool，原地置 `null_block`。
它是私有方法，但在 SWA / chunked-local / RSWA / Mamba 的生产路径上被实际使用，不是死代码。

问题在于它的**唯一调用入口** `remove_skipped_blocks` 是**头部**语义，
由 `get_num_skipped_tokens(num_computed_tokens)` 多态驱动，而

```
SingleTypeKVCacheManager.get_num_skipped_tokens -> 0      # 全注意力
```

OLMoE 走 `FullAttentionManager`，**该路径恒返回 0，部分释放在本配置下完全不可达。**
`KVCacheManager.free(request)` 只接 `Request`、只取 `request_id`，整请求释放；
`pop_blocks_for_free` 也是整请求取出。没有任何公开 API 能按 token 位置截断尾部。

### 8.2 硬阻塞：worker 侧块表 append-only

源码注释明确："The worker block table of a running request is append-only"。
因此 `_remove_blocks_in_range` 只置 null、**不缩短列表**。而再生长走

```python
num_new_blocks = cdiv(num_tokens, block_size) - len(req_blocks)
```

被置 null 的尾部槽位仍计入 `len(req_blocks)`，于是回滚后再 decode 会算出
`num_new_blocks <= 0`，**不分配新块、直接写进 null block**。
头部/中段释放之所以安全，恰恰是因为那些位置永不再生长。
**尾部回滚 + 再生长不是参数问题，是不变量冲突。**

### 8.3 状态机也缺表示

`Request.num_computed_tokens` 是标量，没有"算到 N 但中间有洞"的表示。
`_preempt_request` 的做法是 `num_computed_tokens = 0` 全量清零、
`status = PREEMPTED`、`waiting.prepend_request(request)` 放回队首。
没有"保留 RUNNING、只回退到块边界"的中间态。

### 8.4 需要新增的实现量（诚实估计，不假装已有）

| # | 改动点 | 风险 |
|---|---|---|
| 1 | `SingleTypeKVCacheManager` 增 `truncate_tail_blocks(request_id, k)`：释放末尾 k 个非 null 块并使其可再分配 | 低，复用 `_remove_blocks_in_range` |
| 2 | `allocate_new_blocks` 按**有效长度**而非 `len(req_blocks)` 计算 `num_new_blocks` | 中，影响所有 manager |
| 3 | worker 侧解除该路径的 append-only 不变量，或对被截断请求重发块表 | **高，唯一真风险项** |
| 4 | scheduler 增部分抢占动作：`num_computed_tokens = k * block_size`，保持 RUNNING，不入 waiting | 中 |

改动点 3 有一条现成的减压路径：resumed 请求已经会重发完整 `all_token_ids` 与 block ids
（V2 runner 直接并入 `scheduled_new_reqs`），说明"重建块表"的机械装置本来就在。
再填充本身不需要新代码——chunked prefill 的 `num_new_tokens = request.num_tokens - num_computed_tokens`
已经覆盖"为一个 running 请求补算 N 个 token"。

**可行性裁决：`IMPLEMENTABLE_BUT_REQUIRES_INVARIANT_CHANGE`。** 不是接口缺失，是不变量冲突，
集中在一个点上。在改动点 3 得到确认前，不写选择器。

### 8.5 核查同时暴露的两个配置混淆项（必须记录）

**(a) 全部封存 cell 都是 `enable_prefix_caching: False`。**
而 vLLM 的 `free()` 特意逆序释放，源码注释写明目的是
"so that the tail blocks are evicted first **when caching is enabled**"——
即上游本来就设计了让被抢占请求的前缀留在缓存里、使重算变便宜的机制，
**而封存实验把这个机制关掉了。** 这不推翻 §2 的任何测量（停顿 96.4% 是容量等待，
开缓存也变不出块来），但它意味着"重算 3.6%"这个数字是在最不利于原生抢占的配置下测的，
且该配置不是默认 serving 配置。必须作为边界写清，并作为一个便宜对照。

**(b) 抢占步不调度任何 waiting 请求。** 源码为
`if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:` 才进入 WAITING 循环。
这是受害者等待的一个二阶来源，此前未被计入模型。量级未测。

另外两点与动作空间直接相关，**无需新增实现**：

- FCFS 下 victim 选择是 `self.running.pop()`，即 running 列表尾部 = 最新加入者。
  实测两个受害者正是第 31、32 个到达者（1.50 s / 1.55 s），**与源码行为一致**，
  这独立验证了账本解读没有错位。
- `scheduling_policy=PRIORITY` 是已实现的替代动作，victim 变为
  `max(priority, arrival_time)`。它是一个**零实现成本**的 victim 选择旋钮。
  但按 §2.3 的停顿律，换 victim 不改变停顿长度（停顿由下一个自然完成决定），
  只改变"谁承担"与重算量。它能检验的是公平性/归属，不是总代价。

---

## 9. 唯一下一步

核查已把 §8 的不确定性收敛到**一个点**：改动点 3。因此下一步不是实现，也不是 GPU。

**下一步 = 确认 worker 侧块表 append-only 不变量的确切作用域。**
具体读 `vllm/v1/worker/block_table.py` 与 `gpu_model_runner.py` 中
`CachedRequestData` / `NewRequestData` 的块表写入路径，判定：
被截断请求能否通过既有的 "resumed 请求重发块表" 通道绕过该不变量，
而不改 worker 内核。这是一次纯阅读，产出是一个二选一的判定与对应实现量。

若可绕过 → 第一笔 GPU 支出是"2 次大回滚 vs 32 次小回滚"的成本对照（§5 判据 1–4），
而不是完整 controller。
若不可绕过 → 该 formulation 的实现税上升到 worker 内核层，届时应先做 §8.5(a) 那个
**零实现成本**的对照（`enable_prefix_caching=True` 下重测同样四个 cell），
因为它可能直接改变"重算 3.6%"这一前提，从而改变机制的目标。

在上述判定完成前：不实现选择器，不扫参数，不启动第二条实验链。
[expert-union 采集](../20260910_expert_union_r01/DECISIONS.md) 仍为 `UNRUN` 并保持冻结；
本轮结论不改变其有效性，也不提升其优先级——它回答的是 offload 域的另一个问题。

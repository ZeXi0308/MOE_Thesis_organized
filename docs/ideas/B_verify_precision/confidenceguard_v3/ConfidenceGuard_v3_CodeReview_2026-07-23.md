# ConfidenceGuard-MoE v3 严格 Code Review

日期：2026-07-23  
结论：**CPU REVIEW PASS / TINY SMOKE ENGINEERING PASS / GPU Run Approved: MECHANISM PROBE ONLY**

## 1. 代码执行链路与实验逻辑

`finalize_confidence_guard.py` 只接受 calibration raw SHA `299589…d982`，对两模型分别执行 2,000 次 document bootstrap ridge；每个模型在完整 calibration feature set 上的 score 中位数形成 binary safe cut。`FrozenConfidenceGuard` 对新请求汇总 2,000 个 safe vote，按 `p_safe>=0.8 / <=0.2 / otherwise` 映射 period `8/2/4`。runner 再用 exact period multiset hash 重排产生唯一 matched-budget control；其余 audit/lockout、same-state diagnostic、INT4 expert proxy 和 reference logits 路径不变。

sealed raw row保存原始 9 项 features、point risk score 和 `p_safe`。analyzer 从 source-bound lock 重建全部 bootstrap ridge，独立重算 `p_safe` 与 period，并重放 phase、状态机、forward/clone ledger 和 quality 指标。

## 2. 已确认正确的关键实现

- v2 NO-GO 未被覆盖；v3 config 显式绑定原 raw SHA，并标注 calibration 数字为 exploratory。
- safe/risk/abstain 阈值、2,000 repeats、9 项 feature 顺序和 period 映射均在 config/protocol/lock 三处闭合。
- bootstrap 使用 document 为抽样单位；每次重拟合 scaler/ridge，不在 feature 或 token 级伪增样本。
- matched hash 保留 exact period multiset，common phase 不含 policy name。
- lock 序列化全部 bootstrap models/cuts；加载时检查 schema、feature order、finite 数值、模型/cut 数量和阈值顺序。
- analyzer 不信任 runner 的 `p_safe`/period，使用 raw features 重算并 fail-closed。
- 两模型使用相同 calibration 文档集合，audit threshold 沿用同一 raw discrepancies 的 P90。
- loader、cache non-alias、prepared INT4、diagnostic invariance 和统计完整性回归仍全部通过。

当前冻结绑定：

- source aggregate SHA-256: `200eaf2c0acf398539e09b41873b8671b8d6a0437d3a715ef1226cbbea3c3c03`
- config SHA-256: `c5e9646734c3c9c41ca664f427608f5a08e62f834d69c8b13cc9f8b5cad47dd1`

## 3. 潜在 bug、偏差与混杂因素

- 2,000 个 bootstrap vote 不是贝叶斯校准概率，只是 assignment stability proxy；论文不得称 posterior correctness。
- safe set 的 calibration 风险下降是同数据拟合后的探索性观察，存在明显 optimism；不能作为 pass 证据。
- 二元中位数和 0.8/0.2 阈值是在 v2 失败后根据 calibration 冻结，唯一无偏检验必须是未打开 sealed。
- point ridge 与 bootstrap ensemble 高度相关；若 sealed 分布漂移，abstain 可能覆盖过多或过少。exact multiset control 可保证主比较不混入预算差异，但不能消除分布漂移。
- 当前为 W4A16 dequantized BF16 proxy；任何 wall-clock、native INT4 或 topology claim 均无效。
- v3 修改 source manifest，因此 v2 memory certificate 按代码必然 stale；必须重跑 tiny smoke。
- 首次 v3 smoke 在 artifact 初始化阶段发现 protocol artifact 路径仍硬编码 v2 文件；没有加载模型或执行 forward。已抽取唯一 `_protocol_path(config_path)` 供 source manifest 和 initializer 共用，并增加 v2/v3 路径回归测试。

## 4. 必须修改项与建议修改项

tiny smoke 前 MUST FIX：**已关闭**。v3 smoke 的 hook max-abs/KL error 均为 0，INT4 discrepancy 非零，diagnostic invariance PASS，peak reserved `24.9648 GiB < 29.8567 GiB`。

sealed 前 MUST FIX：**已关闭**。

1. v3 source-bound smoke 及 memory certificate PASS；
2. 新 `MECHANISM PROBE ONLY` approval 已绑定 source/config；
3. 远端实际 calibration raw 生成 lock SHA `e2b5f9d3214972b82713e4d26af63684140204d3453248fd79c3bf64eccb8ed6`，两模型 reformulation gate 全 PASS；
4. 在前三项完成前 remote 仅存在 calibration manifest，没有 sealed manifest。

建议项：若 sealed GO，再把 2,000 模型压缩/解析近似为低开销 uncertainty score；当前机制 probe 不提前做压缩优化。

## 5. 最小 CPU / 小样本 dry run

- 本地新增/相关 12 tests PASS；
- 远端实际 `torch==2.8.0+cu128 / transformers==4.53.3` 下 **48/48 warning-as-error tests PASS**；
- 全部 Python `py_compile` PASS；
- 实际 calibration raw 生成 7,506,775-byte v3 lock，verdict=`REFORMULATION_GATE_PASS`；OLMoE/LLM-jp 三项 gate 均 PASS；
- 该 gate 只批准继续验证，不是科学结论。

## 6. GPU 准入结论

**GPU Run Approved: MECHANISM PROBE ONLY**

允许一次性执行冻结的 64-document sealed ConfidenceGuard 机制验证。证据边界仍为单 GPU、teacher-forced、W4A16 proxy。任何 source/config/lock 改动都会使 approval 或 preflight binding 失效。  
**Sealed GPU Run Approved: YES**

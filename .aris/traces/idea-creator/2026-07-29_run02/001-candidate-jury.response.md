# 独立 Fresh Jury 报告

## 评审边界

完整读取了本轮 `REF_PAPER_SUMMARY.md` 和 `IDEA_CANDIDATES.md`；未读取旧 `IDEA_REPORT`、`refine-logs` 或执行者摘要。评审独立性为 `same-family`，接受状态为 `provisional`。

## 总裁决

进入后续查新/资格小试的候选最多保留3个：

1. **N09 MaxRoute — PROCEED**
2. **N06 RECAP — PROCEED**
3. **N01 Frontier-Cut Bisimulation — PROCEED，高风险**

`PROCEED` 仅表示值得查新并执行冻结门槛的资格性证伪，不表示已证明新颖、有效或可发表。

- N09 是最接近独立研究贡献的主候选，最大风险是“把 max-plus tomography 迁移到 MoE”。
- N06 是路线诊断模型是否成立的关键方法学闸门，可先于 N09 执行；它可能最终只成为 N09 的模型审计组件。
- N01 有真正的形式方法难点，但可能因 cut state 退化成 full DAG 而死亡。

| 候选 | 新颖性 | 问题重要性 | 方法非平凡性 | 2–4周可证伪性 | 正式证据可达性 | 裁决 |
|---|---:|---:|---:|---:|---:|---|
| N01 | 6 | 7 | 9 | 7 | 3 | PROCEED |
| N02 | 4 | 8 | 6 | 9 | 6 | CAUTION |
| N03 | 4 | 7 | 7 | 9 | 4 | ABANDON |
| N04 | 5 | 6 | 9 | 6 | 3 | ABANDON |
| N05 | 4 | 8 | 6 | 9 | 5 | ABANDON（独立候选） |
| N06 | 6 | 8 | 8 | 8 | 5 | PROCEED |
| N07 | 5 | 7 | 8 | 8 | 3 | CAUTION |
| N08 | 3 | 7 | 5 | 9 | 6 | ABANDON（独立候选） |
| N09 | 7 | 9 | 9 | 8 | 5 | PROCEED |
| N10 | 5 | 8 | 7 | 9 | 5 | CAUTION |

## 逐项 strongest kill objection

- **N01**：为保证 suffix 双模拟，cut state 最终必须携带几乎完整的未来请求、队列与事件顺序，使证书退化成 full-DAG replay，无压缩也无独立方法贡献。
- **N02**：只是通用 Oracle 数据泄漏审计；若遮蔽未来字段几乎不改变动作或 headroom，就无问题规模也无 MoE-specific 方法贡献。
- **N03**：可行集足够诚实时几乎所有动作都永久 `UNIDENTIFIED`；为得到结论强行收窄又会把未验证假设伪装成物理约束。
- **N04**：最近反转只在人工限制的 event-order/扰动空间成立；扩到真实空间后 witness 接近完整 trace、求解爆炸且无可操作稳定半径。
- **N05**：compiler 只证明 route incidence 可分，不证明 E2E latency signature 在 batching/coalescing/barrier 下可分，可产生“编码上可诊断、系统上不可诊断”的假证书。
- **N06**：同时满足 exact route/KV/rows 控制的反事实组合在自然状态中极稀少；人工构造又改变系统状态，无法获得既有 support overlap 又不失真的因果对。
- **N07**：为获得足够定位信息，combine-boundary reference/sketch 成本接近完整输出校验或双路 inference，同时仍无法区分 expert compute、transport 与 combine corruption。
- **N08**：单卡 tracer non-interference 对真实 EP 不具传递性；在 EP 重测后仍只是常规 instrumentation validation。
- **N09**：只靠 E2E request latency，slow expert、slow rank、slow link 与正常 queue/batch skew 在 max-plus 下仍可能处于同一不可分等价类；如果需额外 telemetry 才能分开，route-probe 的核心价值消失。
- **N10**：多层 top-k route 使“其它条件匹配、仅目标 exposure 不同”的自然 pair 组合上几乎不存在；放松匹配后隐藏 route/load confounding 又进入。

## 入选三项的验证顺序

1. **先做 N06 资格闸门**：冻结 exact-route 配对规则，验证 additive、max/barrier、interaction 哪个 regime 有效。若没有足够合法反事实或所有 regime 均被 interaction 主导，N06 NO-GO，并显著降低 N09 可信度。
2. **再做 N09 主 pilot**：冻结 canary 池、噪声集合、5/10/20% 注入和 additive baseline；要求 max-plus 显著减少错误定位，且输出不可分故障类。
3. **N01 独立并行查新/小图证明**：第一闸门是 soundness，第二闸门是自然 trace 的状态压缩；任意 false certificate 或压缩 <50% 停止。

N05、N08、N10 只能作为 N09 的 codebook、测量协议或差分 estimator，不应通过拆分模块制造多个“贡献”。

## 证据边界

- RTX 5090 只能提供真实 BF16 cached decode、route/KV/logits exactness、软件 expert delay 和 N01 小型 DAG 资格性证伪；一律标记 `QUALIFICATION_PROXY`。
- 正式证据需 optimized EP dispatch/combine、NCCL/RDMA、topology/rank barrier、grouped GEMM/coalescing/continuous batching、expert/rank/link 分类 slowdown、all-arrival denominator 与 TPOT/P99/goodput。
- 当前十项均未被证明新颖或可发表；只支持 N09、N06、N01 进入下一轮证伪。


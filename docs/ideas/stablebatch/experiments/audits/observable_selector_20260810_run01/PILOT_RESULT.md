# StableBatch MaxGate-v1 vs matched shuffle Pilot 结果

**实验状态**：`COMPLETE`  
**冻结判定**：`WEAKENS_MAXGATE_V1_NOT_BETTER_THAN_SHUFFLE`  
**完整性审计**：`PASS`，`provisional_same_family`，无 P0/P1  
**研究决策**：MaxGate-v1 `NO-GO`；不进入 controller Pilot

## 本轮唯一问题

> 在相同 cell、相同 action budget、相同 M/padding/side-call/full-forward 工作面下，只看当前层 gate weight 的 MaxGate-v1，能否比 seed-frozen balanced shuffled rank 更有效地减少相对 all-M1 reference 的下游 route membership distance？

## 冻结设计

- 数据：`doc016..031 @ token offset 512`，16 个不同文档窗口；这是 window-level fresh，不冒充全局 document-level fresh。
- Cells：`16 documents × layers 0..14 = 240`；统计独立的数据簇只有 16 个文档。
- Reference `R`：目标 token 的 top-8 raw contributions 全部替换为 M1 side-call 输出；这是 synthetic self-supervised proxy，不是真值。
- Unprotected `U`：top-8 全部 M64。
- Observable `O`：MaxGate-v1 保护 rank 0 为 M1，其余 7 个 rank 为 M64。
- Shuffled `S`：冻结 balanced hash rank 保护一个 rank 为 M1，其余 7 个 rank 为 M64；每个 rank 恰好分配 30 cells。
- 主 reward：`D_U-D_policy`，其中 D 是相对 R 发生 top-k membership-set 差异的下游层数；所有正、零、负 reward 全量聚合。

## 结果

| 指标 | MaxGate-v1 O | Matched shuffle S |
|---|---:|---:|
| 总 signed reward | **-3** | **+3** |
| reward / action | -0.0125 | +0.0125 |
| positive / tie / negative cells | 13 / 209 / 18 | 10 / 220 / 10 |
| full restoration cells | 13 | 10 |
| harm cells | 18 | 10 |

共同条件：

- opportunity 为 35 cells、8 个文档，冻结机会门槛 `>=8 cells && >=4 documents` 通过；
- 有 4 个文档的聚合 O reward 高于 S，victim coverage 门槛通过；
- 但 `A_O=-3 <= A_S=+3`，且 O 不满足绝对 reward `>=8` 或正 shuffle 下的 ratio threshold `>=5`；
- 因此按冻结规则直接得到 `WEAKENS_MAXGATE_V1_NOT_BETTER_THAN_SHUFFLE`。

## 怎么解释

### 被明确削弱

- **MaxGate-v1 作为 contribution selector 被削弱。** 它虽有 13 个正 cell 和 13 次 full restoration，但产生 18 个 harm cells，负收益幅度使总 signed reward 落到 -3；不能只报告 restoration 数而忽略伤害。
- 当前结果不授权从单贡献因果现象跳到 StableBatch controller。selector/action-value 这道门没有过。

### 没有被推翻

- 上一轮已经观察到 execution-shape raw delta 可以穿过 combine 并改变下游 routing；本轮的 35 个 opportunity cells 也说明可干预差异仍存在。
- 被否定的是当前 `MaxGate-v1` 选谁保护的规则，不是此前那条窄因果传播现象。

### 不能外推

- 不是自然发生率或跨数据泛化；240 cells 不是 240 个独立样本。
- 不是 dynamic batching/controller、continuous decode、serving latency/quality、EP/NCCL/RDMA 或多 GPU 结果。
- 只比较一次冻结的 balanced shuffled assignment，不能写成对“随机策略分布”的普遍胜负。
- all-M1 R 是实验 proxy，不是 canonical output 或 ground truth。

## 当前研究决策

1. 将 `MaxGate-v1` 标记为 `NO-GO`，保留完整负结果。
2. 不改本轮阈值、不删负值、不在 `doc016..031@512` 上继续调 selector。
3. 不进入 controller/serving Pilot。
4. 若继续 StableBatch，只允许把一个事先定义、机制上不同的 online observable selector 当作新假设，并用新文档级证据单独冻结验证；否则应停止该 selector 分支。

## 完整性

- acceptance run01 因启动前发现未知 15,296 MiB Python GPU 进程而 fail-closed，且标为不具科学资格；未杀进程。进程自然退出后，新目录 run02 acceptance 通过。
- formal 与 acceptance run02 的 runner/base/config/lock/manifest/status hashes 强绑定。
- 240/240 cells 完整且唯一；native no-op、non-target closure、每 rank exactly-once、3-repeat bitwise stability、O/S exact surface、30 个同-rank O/S 等价均通过。
- 本地独立重算未调用 runner classifier，决定性字段与 summary 完全一致。
- fresh GPT-5.6-Sol ultra 审计无 P0/P1；同模型家族结论保持 provisional。

## 审计停止判断

> 当前最小实验已正确回答核心问题，负结论可信。停止继续扩展本轮审计或补无关 baseline。

## 证据入口

- [冻结配置](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/configs/observable_selector_pilot_v1.json)
- [冻结锁](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/configs/FROZEN_OBSERVABLE_SELECTOR_LOCK_V1.json)
- [正式汇总](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/observable_selector_20260810_run01/summary.json)
- [240 行原始结果](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/observable_selector_20260810_run01/cell_results.jsonl)
- [独立重算](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/audits/observable_selector_20260810_run01/INDEPENDENT_RECOMPUTE.json)
- [实验审计](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/audits/observable_selector_20260810_run01/EXPERIMENT_AUDIT.md)

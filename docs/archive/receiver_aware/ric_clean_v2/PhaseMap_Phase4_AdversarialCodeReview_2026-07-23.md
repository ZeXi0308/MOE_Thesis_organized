# PhaseMap Phase 4 对抗性代码审查（2026-07-23）

> **SUPERSEDED REVIEW SNAPSHOT**：本文记录修补前的 BLOCKED 状态。所列科学 P0 已在
> GPU 运行前关闭；最终执行与停止结论见
> `PhaseMap_Phase5_Result_2026-07-23.md`。本文保留为缺陷历史，不代表当前代码状态。

## 结论

**`BLOCKED / NOT SIGNED-OFF / FORMAL GPU RUN PROHIBITED`**

现有目标测试为 **36/36 PASS**，但未覆盖 holdout 替换、错误模型身份、聚合指标冲突、
混合 joint action 和 provenance 伪造等攻击面。以下 P0 会直接改变科学判定或使正式证据链
不可验证；修复并重新独立审查前，不得启动 Phase 5 正式实验。

本报告只审查冻结协议与以下四个 target，不审查本轮由同一实现方新增的 LUT/MILP 文件：

- 协议：`PhaseMap_Phase2_FrozenOracleGate_2026-07-23.md`，SHA-256
  `1f1c2300b62b56ce00e7fb62c35db1dad3c26e1a088b0146a224754194960a12`
- `phasemap_instances.py`：
  `e0c2b88f15ffd0c609766926bc03ad1f6d9eaf1f57b005617458eb00b282868b`
- `phasemap_oracle_core.py`：
  `4e054843b4de5d7178e307043d01470a69a79a05663ae4d87312b5d5162031a3`
- `phasemap_baselines.py`：
  `f5936af0ba4eb44dba16e98df7ee550eb4fb1a9cc820d481f5e40b843e127246`
- `run_phasemap_oracle_gate.py`：
  `c28d6824fc3d844bce04eae84d182c9ae7bf5f52928c02b0bea4e35f79ec97cb`

## P0 — 必须关闭

### P0-1：selection/holdout 输入链可被替换

`run_phasemap_oracle_gate.py:68-72,91-97,327-337`只验证对象自哈希；自哈希只能证明对象
内部自洽，不能证明对象是预注册 producer 产出的固定对象。selection manifest 在
`:386-393`只绑定 selection bundle；`:397-410`可消费任意重新自哈希的 holdout bundle。
主入口`:555-568`也未校验 holdout 与 selection 来自同一 full manifest。

**影响：** 看完 selection/数据后可替换 holdout request、pairing 或 route，属于 selection leakage。

**必须修改：** selection 前生成同时含 selection/holdout 的 immutable full manifest；selection
manifest 绑定两个 split 的 hash、route/data producer hash 和 canonical pairing hash；holdout 必须
逐项等值校验，不能只校验自哈希。

### P0-2：错误模型、数据、top-k 或 LUT 对象可以通过

`phasemap_instances.py:66-95`允许 model revision、data/placement hash 为 `None`；
`:107-140`没有验证 siblings 的 revision/data/placement 全部一致，也没有验证模型规定的 top-k。
`run_phasemap_oracle_gate.py:123-180`按调用参数 `model`选 LUT，却没有把 pair 内嵌模型身份与该参数
闭合。因此错误模型 route、错误 top-k、错误 LUT 的组合可以成为合法 scenario。

**影响：** wrong-object identity 会让服务时间、fanout 和 route 会计全部失真。

**必须修改：** 模型 revision、hidden、top-k、data/placement/route hash 设为必填；join 内所有
siblings 必须一致；split/model/LUT 必须双向闭合；错误模型或 top-k 的 adversarial fixture 必须拒绝。

### P0-3：`best_single` 的 aggregate comparator 与冻结主口径不闭合

冻结协议`:131-145`定义 request fold、CVaR90，并要求 Q/J 按同一主 aggregate metric 比较；代码
`phasemap_oracle_core.py:142-148,813-828`实际按
`(miss_count, mean_tardiness, join_close)`选 Q/J，随后才报告 CVaR。这会在 miss 相同而 mean/CVaR
排序相反时选出不同的 `best_single`，进而改变 miss/CVaR reduction 和 gate。

**影响：** 可直接翻转 Go/No-Go，不能在出结果后解释性选择。

**必须修改：** Phase 2 amendment 先明确 aggregate comparator。若主口径是 gate 本身，应固定为
`(miss_count, CVaR90, mean_tardiness, join_close, arm identity)`；随后实现并加入 mean/CVaR 排序相反
的反例测试。协议未澄清前不得自行猜测。

### P0-4：最强 myopic baseline 既重复计费，也不是 joint-action greedy

`run_phasemap_oracle_gate.py:264-283`用 `receiver_availability + deficit*unpack + combine`预测
join close，但 availability 已包含当前 queued carriers，`deficit*unpack`会再次计入这些工作；同时
忽略 sender pack/cut 及 first/second 顺序。`phasemap_baselines.py:292-296,320-341`对每个 sender
独立最小化同一个 request score，因此只能自然得到 AA/BB，无法比较冻结 action space 中的 AB/BA。

**影响：** 系统性削弱最强简单基线，可能让 `<90% capture`虚假通过。

**必须修改：** 对四个 joint actions 做 causal projected replay，使用当前 Q/J、service、deadline，
不得读取 future；去掉 queued work 双计；加入 mixed AB/BA 唯一最优的测试。其余 separable baseline
可以继续保持 sender-local。

### P0-5：kappa、线性权重和运行 provenance 可自我声明

`run_phasemap_oracle_gate.py:397-410`只检查 selection manifest 自哈希和 artifact 名称，不重算
kappa grid/selection rule，也不重新验证线性 baseline 的 selection source/examples/grid。
`:549-568`未绑定冻结协议 SHA、配置、route/data producer、full-instance manifest 或外部 Phase 4
signoff。攻击者可修改 `selected_kappa`或 weights 后重新计算自哈希。

**影响：** 允许事后调参并把未签字代码包装成“冻结运行”。

**必须修改：** 增加严格 `validate_selection_manifest()`；重算 kappa 选择与线性权重；绑定协议、
配置、producer、route/data、selection/holdout、LUT、全部 source SHA，并要求外部 Phase 4 signoff
attestation 精确匹配。

### P0-6：缺少冻结协议要求的可审计产物

协议`:200-204`要求 12 类分离产物；主入口`:568-575`只输出一个 monolithic JSON。现有 report
也未保存逐 world chosen action、stage ledger 和逐 request fold，因此无法独立重算 pack/cut/unpack/
combine 会计、32-request denominator 或 gate。

**影响：** 即使数值正常，也无法复核 acceptance test 6、统计单位和最终 decision。

**必须修改：** no-overwrite 产出 `per_pair.jsonl`、`per_request.jsonl`、baseline/control/MILP、
decision、environment、source manifest 与 summary；保存每 world action 和 stage ledger；增加从原始
产物独立重算最终 gate 的测试。

## P1/P2 — P0 关闭时一并处理

- **P1：** protocol 要求 background key 含 model，但 `phasemap_instances.py:430-436`遗漏 model。
- **P1：** `build_model_manifests()`在`:738-761`丢弃 route metadata 与父 route/data trace hashes。
- **P1：** 未实现协议列出的 full-future ceiling `C`；至少应作为明确的 diagnostic artifact，不能静默省略。
- **P1：** 缺少 acceptance test 4 的真实 counterfactual future invariance，以及上述全部 adversarial tests。
- **P1：** environment/source 记录缺 git commit、solver/runtime/GPU 版本和配置 hash。
- **P2：** writer 的单次 `os.write`、缺 directory fsync/`O_NOFOLLOW`使审计产物耐久性不足。
- **P3：** core 中存在重复不可达 return、重复字典 key，建议清理以降低 review 噪声。

## 已确认可保留的实现骨架

以下结论只表示代码结构与协议方向一致，不抵消 P0：

- route loader 限定 replay layer，selection/holdout 按 receiver 做 4/4 hash split；
- carrier、decision 和 canonical matching 选择不直接读取 outcome；
- 四世界、near/far carrier、low/high depth 的构造基本符合协议；
- pre-t0 FIFO replay、Q/J observation partition、sender history 无 ACK；
- native full-sibling accounting、deadline 公式和 B0/Q/J/R exact partition 的主体结构成立；
- 32 native-request folding 与 top-4 CVaR 计算本身存在。

## 最小重审门槛

1. 关闭 P0-1 至 P0-6，并逐项提交定向 adversarial test；
2. 现有 36 tests 与新增 tests 全通过；
3. CPU 生成一次完整 selection/holdout artifact tree，并由独立重算器得到相同 decision；
4. 更新 source SHA 后重新做独立 Phase 4 review；
5. 只有新报告写明 `SIGNED-OFF`，才允许采集 LUT 或启动正式 GPU run。

本次 review **没有修改任何 target 实现**，也没有采信或生成科学结果。

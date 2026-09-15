# CreditReduce P0 生死实验预注册

> 冻结日期：2026-07-17  
> 状态：**P0-1 sealed holdout 已执行完毕，OLMoE 与 LLM-jp 均判定 `FAIL`，CreditReduce 动态主线已判死，P0-2 不再运行。** 完整结果见 `outputs/creditreduce_p0_2026-07-17/P0-1_正式结论_2026-07-17.md`。  
> 唯一被检验的核心：动态 CreditReduce（route-collision-funded、recast-residual-gated BF16/FP32 remote partial）  
> 强制规则：PD-Full 是静态强基线，不是失败后的替代主创新。

## 1. 本轮只回答什么

本轮 Mac P0 只回答四个问题：

1. 真实模型路由映射到预注册 EP topology 后，是否存在足量的 **远端 collided partial**？
2. clean early-BF16 是否真的违反预注册完整模型质量 margin？
3. PD-Full 是否能恢复 late-BF16 质量，同时满足 remote hidden-vector payload cap？
4. 如果前三项通过，local BF16 recast residual 是否能形成非退化、优于简单 selector 的动态 Pareto 点？

本轮不回答 GPU kernel、actual NIC bytes、RDMA、overlap、TTFT、TPOT/TBT、吞吐、P99 或能耗。Torch/CPU 时间只用于估算实验成本，不是性能结果。

## 2. 方案 review 后的关键纠偏

旧 grouped-owner 实验把所有 owner groups 都计入 logical wire，未显式区分 token home domain。CreditReduce 的通信定理必须只统计远端 combine traffic。

对 token `t`：

- `h_t`：token home domain；
- `m_{t,d}`：source domain `d` 上 routed contributions 数；
- `K_{r,t}=Σ_{d≠h_t}m_{t,d}`：远端 contributions；
- `D_{r,t}=#{d≠h_t:m_{t,d}>0}`：非空远端 domains；
- `C_{r,t}=#{d≠h_t:m_{t,d}≥2}`：远端 collided groups；
- `n_{32,r,t}`：动态策略发送 FP32 的远端 collided groups。

则：

```text
B_late_remote       = 2H K_r
B_PD-Full_remote    = 2H (D_r + C_r)
B_Credit_remote     = 2H (D_r + n32_r)
n32_r <= C_r <= K_r - D_r
B_Credit_remote <= B_late_remote
```

这里的上界只覆盖 remote hidden-vector payload，不覆盖 bitmap、header、alignment、padding、ready/completion 和 transport framing。

真正的 numerical eligibility 同时要求：

```text
C_r >= 1
D_r + 1[K_home > 0] >= 2
```

第一项保证 wire 上真的有 collision credit；第二项保证 remote partial 在最终 BF16 cast 前还有其他 addend，因而 source-side BF16 recast 可能成为额外中间舍入。

旧的 `1<D<K` 只能作为直觉，不再作为实现判据。普通 routed top-2 仍应 no-op：若两条 contribution 都在唯一远端 domain，则 early16 与最终 BF16 结果相同；若分处两个 domain 或一远一近，则没有远端 collision。

## 3. 冻结的数值语义

### 3.1 Contribution contract

所有 endpoint 使用同一输入：

```text
x_bf16 = BF16(raw_expert_output_bf16 * routing_weight_bf16)
```

这模拟 backend-visible、已加权 BF16 contribution。不得让某个 endpoint 使用 FP32 routing weight、另一个使用 BF16 weight。

### 3.2 Canonical order

- source domain 内按 global expert id 升序 FP32 累计；
- receiver 按 source domain id 升序 FP32 累计；
- 所有 endpoint 最后只做一次模型 dtype/BF16 cast；
- home-domain subtotal 保持 FP32、只在 receiver 本地参与累计，不计 remote payload；
- 全部 endpoint 使用相同 token home mapping。

### 3.3 P0-1 endpoint

| Endpoint | Source subtotal | Remote record | 用途 |
|---|---|---|---|
| `late_bf16` | 不做 remote grouping；home 端按 expert-id FP32 累计全部 BF16 contributions | `K_r` 条 BF16 | 主质量 reference |
| `stock_early_bf16` | source 内按 BF16/model dtype 累计 | 每个 remote group 一条 BF16 | sensitivity；不能冒充某个真实 backend bitwise path |
| `clean_early_bf16` | source 内 FP32 累计后 cast | 每个 remote group 一条 BF16 | 直接测 intermediate recast 代价 |
| `uniform_early_fp32` | source 内 FP32 累计 | 每个 remote group 一条 FP32 | numerical upper endpoint |
| `pd_full` | source 内 FP32 累计 | singleton BF16；collided FP32 | 最强静态基线 |
| `uniform_early_fp8` | source 内 FP32 累计后 per-vector E4M3 fake quant/dequant | FP8 vector + frozen 4-byte scale proxy | 低比特强基线；不作 native kernel 性能结论 |

必须满足的 invariant：

- `pd_full` 与 `uniform_early_fp32` 数值输出逐元素一致；
- `pd_full` payload 不大于 `uniform_early_fp32`；
- `pd_full` 与任何 CreditReduce policy 的 remote hidden-vector payload 不超过 `late_bf16`；
- local/home collision 不生成 remote credit；
- `D_total=1` 与普通 top-2 lone-remote collision 对 clean early16 是 bitwise no-op。

### 3.4 P0-2 endpoint

`pd_gated` 只对 eligible remote collided groups 决策：

```text
s32       = ordered FP32 source-domain subtotal
p16       = BF16(s32)
residual  = RMS(FP32(p16) - s32)
score     = residual / frozen_global_layer_output_RMS
send FP32 iff score > theta
```

第一版只允许一个全局 `theta`，layer RMS 仅作尺度归一；不开放逐层任意 threshold 表。

对照 selector：`m-only`、subtotal norm、cancellation proxy、32 个固定 seed 的 random、leave-one-group-out local oracle。所有 selector 在相同 eligible group 集合和相同 `n32` 下比较。oracle 只叫 local upper bound，不叫 end-to-end true oracle。

## 4. Topology 与 home mapping

必须显式拆开：

```text
expert -> EP rank -> source partial domain -> token home domain
```

冻结参数：

- `ep_size=8`；
- 主 topology：`ranks_per_domain=4`，即两 source domains，模拟两节点层级 combine；
- sensitivity：`ranks_per_domain=1`，即 rank-local partial；
- 主 expert placement：`contiguous`；
- paired stress：`round_robin`，不能当第二个独立数据复现；
- token home rank：`(sample_id + layer_id + token_position) mod ep_size`；home domain 由 `home_rank // ranks_per_domain` 得到。

不能在看 holdout 后选择更有利的 placement/topology 作为主结果。主判定只看 `EP8/ranks_per_domain=4/contiguous`。

## 5. 模型、环境与数据冻结

### 5.1 模型

| 模型 | 本地 snapshot | 角色 |
|---|---|---|
| `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` | E64、top-8、H=2048 主模型 |
| `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M` | `1d5983076dfc67aee4a77ec06a27027f5bab6055` | E32、top-16、H=512 压力模型 |

当前机器是 Apple M5 Pro、48GB unified memory、CPU-only PyTorch。MPS 当前不可用。所有运行必须 `local_files_only/offline`。

### 5.2 历史排除注册表

运行前递归扫描 `experiments/idea_a_mac/outputs/**/data_manifest.{json,csv}`：

- 24 个 data manifests；
- 515 manifest rows；
- 121 个唯一 SHA-256；
- WikiText validation 60/60 已用；
- WikiText test 61/61 已用。

因此 validation/test 永久只作旧证据或 debug，不得再称 fresh confirmation。

### 5.3 Fresh pool

数据源：cached WikiText-2 article parser，`split=train`，`seed=20260717`。train 610/610 与历史 registry hash-disjoint。这里的 fresh 只表示“未被本项目实验使用”，不表示可证明未进入模型预训练语料。

- dataset cache revision：`b08601e04326c79dfdd32d625aee71d232d685c3`；
- `wikitext-train.arrow` SHA-256：`57947bc7b58df4b19662c0609cc30651bc84328dab5fd588860b752072911789`；
- frozen 192-document pool hash-of-hashes：`4fb0c938bb608213647bf72757435dd177c3c0e6d67ecc49e39d1b34683e2001`。

冻结切分（两个模型使用同一文档切分，但分别统计，不把 model 当随机样本）：

| 用途 | shuffled offset | n | subset hash-of-hashes |
|---|---:|---:|---|
| smoke/dev | 0 | 32 | `4b7ec5add131692ae13b371a197417d9d5475cd1d273ca6f9fc74e09e88ecc46` |
| P0-1 sealed holdout | 32 | 64 | `45e7cb88a18065cf8a4f5f74d1d6d7ad8af2fa48998c2d06a1511bad78ef249a` |
| P0-2 calibration | 96 | 32 | `a7f2d225a36abb0bd19e4ac6b88948078ff07aa668e0e03774f79f58aefb6b12` |
| P0-2 sealed holdout | 128 | 64 | `08c6993021bc1e09b7f09c2419a434d9614ffc85e00f6bf1e090a7eba4794cd0` |

P0-1 holdout 打开后若修改 endpoint arithmetic、home mapping、主 topology、主指标或 margin，该 holdout 作废。P0-2 只能在 P0-1 通过后使用自己的 calibration/holdout。

## 6. Workload 与运行顺序

- teacher-forced、`seq_len=256`；
- batch size 1；
- 每个模型先用 dev offset 0 的一篇 `seq_len=32` smoke；
- dev 只允许查 crash、shape、invariant、runtime，不允许据此调整质量 margin；
- P0-1 先完整运行两个模型，再读取/汇总结论；
- P0-1 不通过时，禁止启动 P0-2；
- autoregressive generation、长上下文、更多 topology 与 MMLU 留到数值 P0 通过后。

主实验采用 full-model free-route propagation。locked-route/raw replay 只作单层因果诊断，不能替代 full-model 结果。

## 7. 统计单位、主指标与三态判定

统计独立单位固定为 document。token、layer、expert、group 只作 document 内观测。

主质量指标：

```text
delta_NLL_doc = NLL_treatment - NLL_late
```

- document-cluster paired bootstrap 10,000 次；
- 单侧 95% CI；
- 非劣 margin：`0.005 nats/token`；
- relative PPL 是单调派生量，不再作为第二个独立主检验；
- KL、top-1 disagreement、router flip、document p95/p99 只作 secondary。

三态判定：

- `QUALITY_FAIL`：`LCB95(delta_NLL) > 0.005`；
- `NONINFERIOR`：`UCB95(delta_NLL) <= 0.005`；
- 其他：`INCONCLUSIVE`，不得按点估计硬判。

两个模型分别判定；contiguous/round-robin 是同文档 paired stress，不算独立样本。

## 8. P0-1 生死门

主 topology 的机会指标：

```text
p_eligible = sum(1[eligible_t] * K_r,t) / sum(K_r,t)
rho_credit = sum(1[eligible_t] * (K_r,t - D_r,t)) / sum(K_r,t)
```

以 document bootstrap 计算 CI。

OLMoE 主 topology 必须同时满足：

1. `LCB95(p_eligible) >= 20%`；
2. `LCB95(rho_credit) >= 15%`；
3. `clean_early_bf16` 为 `QUALITY_FAIL`；
4. `pd_full` 为 `NONINFERIOR`；
5. `pd_full == uniform_early_fp32` numerical invariant 全部通过；
6. `uniform_early_fp8` 不在相同质量下同时提供更低 payload 的明确支配点。

任一失败：动态 CreditReduce 停止，P0-2 不运行。任一关键质量判定为 `INCONCLUSIVE`：先扩样本，不得进入 P0-2，也不得判成功。

LLM-jp 用同一门槛作第二模型压力验证。只有 OLMoE 通过才运行；若 OLMoE 通过、LLM-jp 失败，claim 收缩为非通用、模型依赖机制。

## 9. P0-2 生死门

P0-2 calibration 只冻结：

- 一个全局 residual threshold `theta`；
- 一个全局 layer-output RMS normalization vector；
- 固定 selector family 与 random seeds；
- 其余配置沿用 P0-1。

sealed holdout 必须同时满足：

1. `0.1 <= p32 <= 0.9`，分母仅为 eligible remote collided groups；
2. `pd_gated` 为 `NONINFERIOR`；
3. 相对 PD-Full hidden-vector payload 点估计至少节省 15%；
4. document-bootstrap payload-saving LCB95 至少 10%；
5. 在 matched `n32` 下不被 `m-only`、subtotal norm、cancellation 或 32 个固定 random seeds 的最好结果支配；
6. local oracle 在 early16 与 PD-Full 间确实存在中间可行 Pareto 点。

任一失败：唯一动态主方案 CreditReduce 判死。PD-Full 只作为 baseline/negative characterization 留档，不改名救场。

## 10. FP8 威胁的解释边界

Mac fake FP8 若在更低 payload 下通过质量，CreditReduce 进入高风险状态；但 fake quant 不能证明 native FP8 kernel 的 completion-time dominance。

- 数值 P0 中 FP8 明确支配：停止把 CreditReduce 当质量侧优选方案；是否彻底杀死系统论文，等待 native optimized FP8 kernel。
- native FP8 在相同质量下 completion time 不差：CreditReduce 系统主张死亡。

## 11. 实现与测试 Gate

正式 holdout 前必须全部通过：

- pure reference 单测；
- original pretrained forward 与未启用新路径的 patched forward bitwise exact；
- late reference 重复运行 deterministic；
- home collision 不计 wire；
- `C_r <= K_r-D_r`；
- payload cap；
- `D_total=1` no-op；
- ordinary top-2 no-op；
- `D_r=1 + home addend` 有效反例；
- PD-Full 与 uniform earlyFP32 数值一致；
- contiguous/round-robin、`ranks_per_domain=1/4`；
- 历史 hash overlap hard-fail；
- source manifest、config、environment、model snapshot、dataset revision 全量落盘。

## 12. 必须产出的 artifact

每次运行至少输出：

- `config.json`；
- `environment.json`；
- `source_manifest.json`；
- `historical_exclusion_registry.json`；
- `data_manifest.json`；
- `exactness.json`；
- `sample_metrics.csv`；
- `endpoint_summary.csv`；
- `opportunity_by_document.csv`；
- `opportunity_by_layer.csv`；
- `group_diagnostics.parquet` 或分片 CSV；
- `paired_bootstrap.json`；
- `decision.json`，每个 gate 明确 `PASS/FAIL/INCONCLUSIVE/NOT_TESTED`；
- `report.md`，明确 evidence boundary。

任何 partial run 必须写 `status=PARTIAL`，不得与 sealed confirmation 合并统计。

## 13. 当前预注册判决

当前只允许进入实现与 smoke，不允许写 CreditReduce 有效。下一次 idea 修正只接受三种结果：

1. P0-1 失败：杀死动态核心，写清失败原因；
2. P0-1 通过、P0-2 失败：证明 precision dividend 存在，但动态 selector 不成立；主方案仍判死；
3. P0-1/P0-2 都通过：进入单 GPU kernel break-even，仍不声称通信加速。

## 14. 实际执行结果（2026-07-17，当天完成）

**命中结果 1：P0-1 失败。** OLMoE（EP8/D4/contiguous，64 篇全新文档，seq=256）与 LLM-jp（同拓扑，64 篇）均判定 `overall=FAIL`。机会窗口 gate（`p_eligible`、`rho_credit`）在两个模型上都远超门槛（OLMoE 99.0%/74.7%，LLM-jp 100%/87.5%），说明碰撞机会本身充分；但两个关键质量/支配 gate 均失败：

- `early_bf16_must_fail`：`clean_early_bf16` 在两个模型上都是 `NONINFERIOR`（UCB95 delta_NLL 分别为 0.00145 / 0.00112，远低于 0.005 margin），即本地合并引入的 BF16 中间舍入在完整模型质量尺度下不构成可探测伤害，CreditReduce 想修复的"问题"并不存在。
- `uniform_fp8_not_dominant`：`uniform_early_fp8` 在两个模型上都同时满足质量合格（`NONINFERIOR`）且字节数远低于 `pd_full`（OLMoE：536.8MB vs 2118.9MB；LLM-jp：135.3MB vs 536.9MB），构成明确支配。

按第13节规则，**P0-2 不再运行**，CreditReduce 动态主线正式判死。完整数据与解读见 `outputs/creditreduce_p0_2026-07-17/P0-1_正式结论_2026-07-17.md` 及各自的 `endpoint_summary.csv`/`paired_bootstrap.json`/`decision.json`。


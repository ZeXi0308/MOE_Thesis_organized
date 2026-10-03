# RouteContract 2h RTX 5090 Kill Pilot 冻结协议

- 冻结时间：2026-07-29 21:28:42 +0800
- 状态：`PRE_REGISTERED_NOT_RUN`
- 证据层级：idea-kill pilot；通过也不是论文级证据
- 运行面：1×NVIDIA GeForce RTX 5090；单机、无真实 EP/NCCL
- 预注册原则：看到结果后不得修改 capsule、mutant、tolerance、分母、gate 或 fallback 判定

## 1. 要杀的 broad claim

RouteContract 不主张“差分测试”或“metamorphic testing”本身新。唯一候选 delta 是：

> 用模型版本参数化的 executable MoE semantic contract，对 route→dispatch→expert contribution→combine→dtype/fallback 整链做带前置条件的关系检查；由不复用被测 helper 的 CPU contribution-ledger interpreter 作 oracle，并将通过语义与性能检查的 capability cell 以 runtime fail-closed certificate 强制。

近邻压力：vLLM 现有 fused-vs-reference/permutation/weighted-unpermute 测试，TENSURE 的稀疏计算领域 metamorphic fuzzing，以及 FreeFuzz/DeepREL/NNSmith/PolyJuice。若本 pilot 不能证明 unique detection，该 framing 直接放弃。

## 2. 语义边界

- exact top-k tie 不要求稳定 expert identity；跨 top-k 边界的 exact tie 只能返回 `AMBIGUOUS`/集合值。
- low margin 不是 route invariant；只有已证明扰动上界小于 selection margin，且计入 dtype 舍入误差，才允许要求 route 不变。
- within-token duplicate expert 不是普通 dense-logit top-k 的正例；只测 ABI 显式拒绝，或 ABI 明确定义的 duplicate-contribution 聚合。
- token permutation 只在隔离、无状态的 MoE block 内成立，不外推到 attention/scheduler/完整 serving。
- expert permutation 必须同步置换 router 列、expert weights、logical/local ID、quant scale 与 mapping。
- OLMoE、Qwen-MoE、Mixtral 的 normalization/shared-expert 语义不得用同一硬编码规则代替。

## 3. 冻结执行面

### 3.1 最小后端组合

1. 独立 CPU contribution-ledger interpreter；
2. PyTorch eager 与 `torch.compile`，但两者只计一个语义家族；
3. 至少一条真实 fused path：vLLM fused-MoE 或独立 Triton kernel；
4. 必须记录 actual executed backend/kernel provenance，不接受静默 eager fallback。

缺少 fused path 或 oracle 不独立，`G0 FAIL`。

### 3.2 输入 capsule

- 固定 3 个 model-version-bound route capsule：OLMoE、Qwen-MoE、Mixtral 各 1 个。
- capsule 必须在开始计时前可用，并绑定 model/config revision、layer、token input、router logits/selected IDs/gates、weights/scales、dtype 和 seed。
- 若只有随机 tiny config、伪造 natural route，或用结果反向选 capsule，`G0 FAIL`。

### 3.3 六个正向 relation

1. token permutation equivariance；
2. 同步 expert relabel invariance；
3. empty/unused expert invariance；
4. positive-margin-bounded perturbation route stability；
5. zero-gate contribution deletion；
6. dispatch/combine contribution conservation。

Exact tie 只测 `AMBIGUOUS`；duplicate 只测显式拒绝或已声明的聚合规则，不计正向 relation recall。

### 3.4 八个冻结 mutant

1. scatter offset 偏移；
2. slot/expert map 错位；
3. missing contribution；
4. duplicated contribution；
5. unconditional/wrong renormalization；
6. wrong scale index/layout；
7. premature BF16 cast；
8. silent eager fallback。

mutant 实现在执行前全部绑定 hash；不得在看到 detection 结果后替换。

## 4. 必跑强 baseline

- 原样 vLLM `tests/kernels/moe/test_moe.py` 中 reference-vs-fused、boundary shape、EP/padding/CUDA Graph、int64-overflow 相关 case；
- 原样 vLLM `tests/kernels/moe/test_moe_permute_unpermute.py` 中 offset/map、permute、weighted-unpermute case；
- 与 RouteContract 使用相同 capsule、执行次数和 wall-clock budget 的 fixed random/allclose baseline；
- 简单 multi-shape/dtype boundary baseline。

不允许只与单 shape random allclose 比较。

## 5. 两小时时间盒

- 0–20 min：冻结版本、seed、capsule、执行路径，运行 G0 smoke；
- 20–40 min：检查模型策略、margin、relation 前置条件与 oracle 独立性；
- 40–90 min：clean controls、6 relations、8 mutants 和全部 baseline；
- 90–110 min：生成 capability cell，注入 unsupported shape 与 forced fallback；
- 110–120 min：封存逐 case log、分母、hash 和 verdict。

环境/依赖安装不计入科学运行时间，但任何安装失败、后端不支持或 capsule 缺失必须如实记录，不得以替换后端调整论文门。

## 6. 冻结 gates

### G0 环境与实体门

三个真实 capsule、独立 ledger、eager/compile、至少一个 fused path、完整 provenance 全部可用。缺一即 `ABANDON_CURRENT_PILOT`。

### G1 语义门

- 6 个正向 relation 都有机械可检查的前置条件；
- clean controls 零 false positive；
- exact tie 只能判 `AMBIGUOUS`；
- 唯一 detection 不能来自非法 duplicate 正例。

任一失败，`ABANDON_PAPER_FRAMING`。

### G2 相对现有测试的增量门

- RouteContract 捕获至少 7/8 mutant；
- 至少 2 个不同 bug class 同时被 upstream vLLM baseline 和等预算 random/boundary baseline 漏掉。

不得改 tolerance、mutant 或分母抢救。任一条未满足，`ABANDON_PAPER_FRAMING`。

### G3 sealing 门

forced fallback 与 unsupported capability cell 100% 被拒绝或显式 downgrade，且 actual executed kernel provenance 可核验。

G3 单独失败：删除 capability-sealing claim，只保留 correctness tool 候选；不得声称 runtime certificate 贡献。

### G4 oracle 独立与证据门

- oracle 不复用被测 route/topk/packing/combine helper；
- version、build/code hash、GPU arch、backend、model/config revision、shape/E/top-k、dtype/quant/scale、seed、capsule hash、contract hash、actual path 和逐 mutant log 齐全；
- 分母和失败分类可由 raw log 重建。

缺证据判 `INVALID`，不是部分通过。

## 7. 决策

- `G0/G1/G2/G4` 任一失败：`ABANDON_PAPER_FRAMING`；
- 仅 `G3` 失败：删除 sealing claim，回到纯 correctness tool；
- `G0–G4 PASS`：只授权扩大到历史/真实缺陷实验，不代表达到 CCF-B。

正式 CCF-B 包仍需：至少 4 种实质不同执行路径、2 代 GPU，30–50 个历史/真实来源 mutant 与 hidden held-out 集，对最强 adapted baseline 的 recall 提升至少 15–20 个百分点，clean false-positive 不高于 1%，以及 2–3 个跨至少两个实现家族的 maintainer-confirmed 新缺陷。


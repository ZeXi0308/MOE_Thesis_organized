# CPR-MoE 单 RTX 5090 快速验证：设计、实现、Code Review 与执行门

> **专项必要条件报告，不是当前总入口。** 当前 combined verdict 仍为 `INCOMPLETE_NECESSARY_GATES`，且不授权 CPR controller；统一执行顺序见[当前研究状态](../../../current/README.md)。

> 日期：2026-07-25  
> 候选方向：Critical-Path-Budgeted MoE Return Planning（CPR-MoE）  
> 快验资源：1×RTX 5090 32GB；正式资源：8×A100  
> 证据边界：本轮只能证伪单卡必要条件，不能验证 EP/NCCL、receiver backlog、request DAG criticality、TPOT 或 P99。

## 0. 直接裁决

CPR-MoE 的核心前件是“强 overlap/fusion/zero-copy 后仍有足够 exposed EP return critical path”，这个对象在单 GPU 上不存在。因此当前状态是：

~~~text
CORE_IDEA_STATUS = BLOCKED_RESOURCE_NOT_TESTABLE_ON_SINGLE_GPU
MECHANISM_IMPLEMENTATION_AUTHORIZED = false
SCIENTIFIC_NO_GO = false
~~~

本轮只实现两个能提前杀死执行路径的必要门：

1. 冻结两模型 paired document 上的 INT4 rank-tail 质量顺序；
2. RTX 5090 上同 stream、真实连接的 INT4 pack → unpack 编解码税，与 zero-start analytic byte-transfer saving 比较。

二者全通过也只能得到：

~~~text
NOT_FALSIFIED_SINGLE_GPU_BLOCKED_EP_RETURN_PATH_GATE
~~~

不得启动 CPR controller 或宣称 TPOT/P99 改善。

## 1. 核心假设分层

### 1.1 系统核心假设 H0（本轮不可测）

> 当开启最强合法 overlap/fusion/zero-copy 后的 optimized EP return path 在至少一个两模型共同常见 workload cell 中占 request completion critical path 至少 15% 时，使用 criticality 和 quality budget 约束的 return representation planner，相比 fused uniform INT8 与最佳 static cutoff，能够使 TPOT/P99 至少改善 5%，同时不导致质量越界率和 P99 退化超过冻结边界。

H0 需要 8×A100 真实 EP serving、actual collective 与 precedence-DAG zero-return Oracle。

### 1.2 本轮可直接证伪的 H1

> 当 activation shape 为 rows ∈ {128, 512}、hidden ∈ {512, 2048} 且解析链路为 200/400 Gbps 时，使用与质量证据数值语义一致的 per-row symmetric INT4 pack → unpack，相比 BF16 byte baseline，能够使至少 75% 预注册 cell 的 analytic saved wire time 减 connected codec P95 大于 0，同时 codec P50 不超过毛节省的 30%。

这是当前 unfused INT4 executor 的必要条件，不是 CPR-MoE 的充分条件。

### 1.3 辅助假设 H2

> 在冻结 OLMoE/LLM-jp paired documents 中，相同 byte budget 下将 INT4 用于最低 gate rank 的 combine-output KL 稳定低于用于 rank 1；paired difference 95% CI 下界 > 0，且 head/tail mean KL ratio ≥ 5×。

### 1.4 已知事实

- [Observed] 旧的 128 documents/model 结果中，rank-tail 与 rank-head INT4 paired KL 有很大差异。
- [Observed] 旧 homogeneous codec 产物在 8 个解析 cell 为 0/8 viable，但旧路径是未连接的 GPU codec + unrelated pinned H2D，不作为新 connected path 的直接结论。
- [Observed] fixed RankLane 在外部给定 p_return≤20% 且 zero-tax 时，相对 uniform FP8 的代数上界仅 4.1667%；达到 5% 需要 p_return≥23.5294%。
- [Observed] 当前本地执行环境为 macOS arm64，无 CUDA/PyTorch/Triton；本轮 codec GPU 数据未运行。

### 1.5 尚未验证的推测

- optimized 8-GPU return path 有 ≥10%/15% exposed criticality；
- INT4 codec 在 RTX 5090 目标 shape 上净正；
- local codec 通过后能被 producer/consumer fusion 保留；
- criticality planner 胜过 uniform/static baseline；
- 任一局部收益可转化为 TPOT/P99。

## 2. 因果链与最脆弱环节

~~~text
自然 arrival / route / expert-ready 顺序变化
→ optimized EP dependency DAG 中仍有未被 overlap 隐藏的 return span
→ criticality × quality budget 使 planner 与 uniform/static baseline 产生不同动作
→ 真实 wire bytes/service 减少，且 codec/launch/layout 税小于节省
→ token completion 的 exposed span 缩短
→ TPOT P95/P99、goodput 或 J/completed-token 改善
~~~

最脆弱的是第二步：强 backend 优化后是否仍有 exposed return span。单卡不能验证它。本轮只从第三、四步删除明显不可行的 INT4 actuator。

## 3. 优化上限与止损门

令 optimized return 的 exposed E2E 比例为 p_r，该段局部加速比为 s：

    speedup = 1 / ((1 - p_r) + p_r / s)
    zero-return 最大 latency reduction = p_r

| 先验场景 | exposed fraction | 删除式最大 latency reduction | 裁决 |
|---|---:|---:|---|
| 最保守 | 0% | 0% | 无可优化对象 |
| 最可能先验 | <10% | <10% | 更可能是 measurement/engineering |
| 最低论文门 | 10% | 10% | 还需 strong-baseline gate |
| 常见 cell 意义门 | 15% | 15% | 才允许进入机制阶段 |
| 乐观但未观测 | 20% | 20% | fixed RankLane 仍只剩 4.17% zero-tax relative gain |

简单 heuristic 可能已接近最优：fused uniform INT8 + per-layer static cutoff。若它捕获 Oracle headroom ≥80%，即使 H0 成立也应放弃复杂 online planner。

瓶颈可能迁移为：

~~~text
return A2A → codec/unpack/launch → receiver queue/batch fragmentation
           → attention/critical expert GEMM → scheduler tail
~~~

## 4. 最小可证伪实验

### E1. Connected INT4 codec break-even（唯一的新 GPU 信息）

1. 验证假设：H1。
2. 必须先做：quality ordering 已有强数值信号；executor 税是 5090 能直接杀死的剩余条件。
3. 最小对象：synthetic BF16 activation，同 GPU 上连接的 Triton per-row INT4 pack/unpack。
4. 自变量：rows {128,512}、hidden {512,2048}；200/400 Gbps 只是预注册 analytic sensitivity。
5. 因变量：connected GPU latency 与 analytic net saving。
6. 控制变量：RTX 5090、BF16、scale contract、kernel、warmup/repeats、stream、config hash。
7. Baseline：BF16 payload bytes、codec cost=0 的乐观 baseline。
8. Oracle upper bound：BF16 bytes 减 packed INT4 bytes 减 FP32 row scales 的 zero-start byte time saving。
9. Workload：rows×hidden 冻结网格，每 seed 预生成 BF16 random tensor；不是 serving trace。
10. 指标：pack/unpack/connected raw µs、mean/std/P50/P95/P99、MSE、wire bytes、break-even Gbps、allocated/reserved memory。
11. 重复：20 warmups + 200 measured repeats/cell；3 个冻结 seeds。
12. 统计：保留全部 raw samples；组件顺序逐轮确定性打乱；不相加边际 P95。
13. 成功阈值：INT4 在 ≥75% cell 中 wire_saved - connected_P95 > 0，且 connected_P50/wire_saved ≤30%；三 seed 全通过。
14. 淘汰阈值：任一 seed 未过 INT4 gate，淘汰当前 unfused INT4 path。
15. 混杂：analytic rate≠NCCL；random tensor≠model activation；DVFS/thermal；Triton JIT；未测 fusion/overlap。
16. 能得出：当前 local INT4 codec 是否连最乐观 byte upper bound 都支付不起。
17. 不能得出：EP/NCCL 收益、fused path 收益、CPR actionability、TPOT/P99。

### E2. Paired rank-tail quality necessity（重新执行以生成同次 provenance）

1. 验证假设：H2。
2. 必须保留：它锁定 codec 的正式 decision mode 必须是 INT4，防止 INT8 快但无 matched quality 证据时被误判通过。
3. 最小对象：OLMoE/LLM-jp 各 128 个冻结 documents。
4. 自变量：被 INT4 的 gate rank（rank1 vs rankk）。
5. 因变量：document-level combine-output KL。
6. 控制变量：document/token/route/byte budget/quant contract/model checkpoint。
7. Baseline：rank1 INT4。
8. Oracle：不使用 future-known 选择；rankk 只是候选 quality-aware action。
9. Workload：offset 600、seq_len 128 的冻结文档集。
10. 指标：paired head-tail KL difference、mean ratio、95% bootstrap CI。
11. 重复：128 documents/model；5,000 paired bootstrap repeats。
12. 统计：两模型独立决策，不 pool；三 quick-validation seeds 只改 bootstrap resampling，不冒充三次 forward。
13. 成功阈值：两模型均 paired CI low>0 且 head/tail≥5×。
14. 淘汰阈值：任一模型不显著或方向翻转。
15. 混杂：fake quant、teacher forcing、只看 KL、模型代表性。
16. 能得出：gate rank 是这两模型中的局部 quality sensitivity signal。
17. 不能得出：task quality、decode drift、online selection、wire/TPOT/P99 收益。

旧 CSV 没有 producer/dependency/rounding sidecar，所以只保留为历史数值证据，不进入本次正式 matched gate。生产脚本已修改为在同次 forward 写 quantization_provenance.json；禁止为旧 CSV 追补伪 sidecar。

## 5. 实验矩阵

| 实验 | 核心假设 | 自变量 | Baseline | Oracle | 指标 | 成功阈值 | 失败结论 |
|---|---|---|---|---|---|---|---|
| E1 connected INT4 codec | local codec 不吞掉乐观 byte saving | rows、hidden；link rate 仅作解析轴 | BF16 bytes, zero codec | zero-start byte saving | raw µs、P50/P95/P99、net、break-even | 三 seed 的 INT4 viable cells≥75%，税≤30% | 淘汰当前 unfused INT4 executor |
| E2 paired quality | rank-tail INT4 稳定比 rank-head 安全 | rank1 vs rankk | rank1 INT4 | 无 future oracle | KL diff/ratio/CI | 两模型 CI low>0、ratio≥5× | 淘汰 rank quality signal |

第一轮性能网格只改 rows 和 hidden 两个结构变量；200/400 Gbps 不是实测网络变量。

## 6. 最小实现与调用链

~~~text
quick_validate.json
  ├─ frozen seeds / thresholds / exact quantization contract
  ├─ producer-emitted quality provenance paths
  └─ codec rows × hidden × analytic link grid

quality producer
  └─ same forward run → CSV + producer/dependency/CSV hashes + quant contract

run_experiment.py
  ├─ quality: provenance verify → paired bootstrap → quality decision
  └─ codec: BF16 source → Triton pack → same-buffer unpack
                         ├─ PyTorch packed-byte/scale/output oracle
                         ├─ interleaved CUDA Event samples
                         └─ analytic byte bound → INT4-only decision

analyze.py
  ├─ manifest/decision exact-match fail-closed single-run gate
  └─ exact frozen-seed/config/source-hash aggregation
~~~

实现文件：

- docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/run_experiment.py
- docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/codec_kernels.py
- docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/analyze.py
- docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/configs/quick_validate.json
- docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/test_quick_validate.py
- docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/test_codec_kernels_gpu.py
- docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/README.md
- docs/ideas/A_rank_tail_fp8/experiments/run_idea_a_rank_lut_gpu_rigorous_verify.py

代码实际测量“paired rank-tail INT4 KL ordering + homogeneous connected local codec 的乐观必要界”。它没有实现 CPR scheduler、EP collective、receiver ordering、slack estimator 或 serving loop。

## 7. 可复制运行入口

完整环境、依赖、质量 producer、smoke、正式三 seed、分析和清理命令见：

    docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/README.md

每个正式 seed 必须一次生成两个 decision：

~~~bash
python docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/run_experiment.py \
  --config docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/configs/quick_validate.json \
  --experiment all \
  --seed 20260725 \
  --output-dir docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/results/formal_seed20260725
~~~

中途失败时 run_manifest.json 保留 FAILED 和异常；原目录不允许重用，防止拼接半成品。

## 8. 独立 Code Review 与修复

代码生成与 Review 由不同角色完成。首轮 Review 给出 GPU_RUN_APPROVED=false，阻断问题及修复如下。

| 级别 | Review 发现 | 修复 |
|---|---|---|
| P0 | analyzer 对 FAILED/未知 status fail-open | 只允许 COMPLETE；SMOKE/FAILED/缺字段全 fail-closed |
| P0 | quality 只验 INT4，codec 却允许 any mode 通过 | decision mode 冻结 INT4；INT8 只表征；校验完整 quant contract |
| P1 | 无 seed override/多 seed 门 | 增加 --seed、三个预注册 seeds、config hash 与 exact-set aggregate |
| P1 | allow-other-gpu 可用于 formal | 只允许与 smoke 同用 |
| P1 | Triton kernel 无 reference oracle | PyTorch packed bytes/FP32 scale/BF16 output 对照，覆盖 zero/extreme/RTNE/signed nibble/两种 hidden |
| P1 | quality/codec 分目录，analyzer 无合法拼接 | formal 改为 experiment all，每 seed 单一不可变目录 |
| P0 | 单次分析缺 manifest 或伪 COMPLETE decisions 仍可通过 | 强制 manifest、experiment=all、seed、status、manifest decision 与文件逐字段一致 |
| P0 | 字符串 false 可作为真；任意相同 dict 可冒充 contract | 严格 bool 类型；contract 必须等于冻结完整字典 |
| P1 | 三 seed 不比较质量输入哈希 | source provenance 写入 decision/manifest；aggregate 要求完全一致 |
| P1 | 旧质量 CSV 只靠配置声明 quant contract | producer 在同次 forward 发 sidecar；校验 producer、依赖、CSV hashes；旧 CSV 正式拒绝 |
| P1 | sidecar 未绑定具体模型和数据运行身份 | 绑定 model/model_key、不可变 commit SHA、dataset/split、samples/offset/seq_len、dtype、producer seed；拒绝重复 model/CSV/sidecar |
| P1 | quality producer 可静默回退 CPU，sidecar 未绑定实际硬件 | producer 强制 RTX 5090 sm_120、Torch 2.8.0/CUDA 12.8；sidecar 记录 GPU/runtime，quick runner 严格校验 |
| P2 | 部分失败无 manifest | 启动写 RUNNING，异常写 FAILED，输入缺失写 BLOCKED |
| P2 | 显存峰值口径不清 | 分开 pre/post allocated/reserved、absolute peak 与 delta |
| P2 | 组件测量顺序固定 | 每 repeat 确定性打乱并在 raw CSV 保留顺序 |
| P2 | invalid analyzer 退出码仍为 0 | INVALID/INCOMPLETE 输出结果后退出 2 |

已完成的本地验证：

- JSON schema 与 Python 静态编译通过；
- 10 个 CPU 单测通过；
- 1 个 CUDA/Triton reference test 在无 CUDA 环境正确标记 skipped，不计作通过；
- codec 本地失败路径实测退出码 1，manifest 为 FAILED；
- 最终 preflight 因新的 producer-emitted sidecar 尚不存在，退出码 2，状态为 BLOCKED_MISSING_INPUTS。

RTX 5090 上必须先让 quality producer、GPU reference test 真正 PASS，再跑 formal codec。

最终独立复审结果：

~~~text
P0 = 0
P1 = 0
GPU_RUN_APPROVED = true
APPROVAL_SCOPE = RTX_5090_NECESSARY_CONDITION_HARNESS_ONLY
CORE_CPR_MOE_GATE = STILL_BLOCKED_REQUIRES_8xA100
~~~

## 9. 当前实际证据状态

旧 CSV 的数值重分析结果为：

| model | documents | head mean KL | tail mean KL | ratio | paired diff 95% CI | 数值检查 |
|---|---:|---:|---:|---:|---:|---|
| OLMoE | 128 | 0.208147 | 0.004326 | 48.11× | [0.193427, 0.214951] | PASS |
| LLM-jp | 128 | 0.201978 | 0.001721 | 117.35× | [0.193117, 0.207899] | PASS |

但这只是对旧 forward CSV 的 bootstrap 数值审计。由于旧产物缺少同次 producer/dependency/quantization sidecar，不能进入本次 matched formal gate。

当前严格状态：

~~~text
quality_numeric_audit = PASS_OLD_CSV_NOT_FORMAL_PROVENANCE
quality_formal = BLOCKED_REQUIRES_PRODUCER_RERUN
codec = NOT_RUN_ON_RTX_5090
core_ep_gate = BLOCKED_NOT_TESTABLE_ON_SINGLE_GPU
combined_verdict = INCOMPLETE_NECESSARY_GATES
~~~

不使用旧 0/8 H2D-proxy 结果填充新 connected path，不预设 5090 数值。

## 10. 专属结果判定树

~~~text
E2 producer-emitted paired INT4 quality
├─ provenance/hash/contract 不闭合
│  └─ INVALID，不追补 sidecar，重新执行 frozen producer
├─ 任一模型 CI low≤0 或 ratio<5×
│  └─ NO_GO_CPR_QUALITY_SIGNAL
└─ 两模型通过
   └─ E1 RTX 5090 INT4 codec（3 seeds）
      ├─ GPU reference oracle 不通过
      │  └─ INVALID_IMPLEMENTATION，修 kernel，原协议重跑
      ├─ 任一 seed viable fraction<75% 或 codec tax>30%
      │  └─ NO_GO_CURRENT_UNFUSED_INT4_CODEC_PATH
      └─ 三 seed 全通过
         └─ NOT_FALSIFIED_SINGLE_GPU
            └─ 仍等待 8×A100 optimized EP Gate 0
               ├─ 所有共同自然 cell <5%
               │  └─ 永久淘汰 return representation/receiver 整族方向
               ├─ 5%–10%
               │  └─ 降为 measurement/engineering
               └─ 共同 cell≥10%，常见 cell≥15%
                  └─ 测 fused uniform INT8 + static cutoff Captured Headroom
                     ├─ CH≥80% → 保留简单 heuristic，淘汰复杂 CPR planner
                     └─ CH<80% 且 complete fused path 净正
                        └─ 才允许冻结 CPR-MoE 机制协议
~~~

若 microbenchmark 正但 E2E 低，检查瓶颈迁移，不继续堆局部优化。若只在 slow analytic link、异常 rows、关闭 overlap 或人工 barrier 下成立，按 workload/config mismatch 淘汰。

## 11. 今天可以完成

- [x] 创建 runner、Triton codec、analyzer、frozen config、CPU/GPU tests 和 README。
- [x] 进行独立严格 Code Review，并逐轮修复 P0/P1。
- [x] 运行 Python compile、CPU 单测、config/input validation。
- [x] 对旧两模型 CSV 做 paired bootstrap 数值审计。
- [x] 实测无 CUDA 的 fail-closed 失败 manifest。
- [x] 修改 quality producer，使未来同次 forward 生成正式 sidecar。
- [ ] 在 RTX 5090 上重跑两个 frozen quality producer。
- [ ] 在 RTX 5090 上运行 test_codec_kernels_gpu.py。
- [ ] 在 RTX 5090 上运行 3 seeds formal experiment all。
- [ ] 采集 raw samples、summary、decision、manifest 和 environment。
- [ ] 画 INT4 codec_fraction 与 analytic_net_P95 的 rows×hidden heatmap；未跑前不画预设结果图。

本轮回答的问题是：matched INT4 quality signal 是否存在，以及当前 connected local INT4 codec 是否连乐观 byte bound 都付不起。

## 12. 通过或失败后的动作

### 快验通过后

1. 不实现 controller；
2. 取得 8×A100，先测 optimized EP return precedence-DAG deletion Oracle；
3. 只有 H0 达到 ≥10%/15% 才测 strong baseline 的 Captured Headroom；
4. 再测真实 producer→collective→consumer fused path、matched task quality、message conservation 和 P99 non-regression。

### 快验失败后

- quality 失败：淘汰 rank-quality actuator；若保留 CPR 问题，必须不依赖该 signal 重写假设。
- codec 失败：淘汰当前 unfused INT4 executor；不换慢链路或 synthetic sleep 救活。只有真实 fused ABI 可重开局部路径。
- 8-GPU H0 未来 <5%：淘汰整个 return representation/receiver 方向；不换 deadline、workload 或指标。
- H0 介于 5%–10%：合并到 EP measurement/engineering，不作主机制。
- simple baseline CH≥80%：保留简单机制，淘汰复杂 planner。

## 最终保留标准

| 类别 | 冻结标准 | 动作 |
|---|---|---|
| 暂时保留问题 | 单卡只能做必要门 | 不实现 CPR controller |
| 淘汰当前 executor | 任一正式 seed 的 INT4 codec gate 失败 | 停 unfused INT4 path |
| 进入机制 Gate | 8×A100 H0 共同 cell≥10%，常见 cell≥15% | 测 CH 和 complete fused path |
| 降为工程 | H0 为 5%–10% | 只保留 measurement/optimization |
| 淘汰整族方向 | H0 所有共同自然 cell<5% | 永久停止 return actuator |
| 淘汰复杂策略 | simple baseline CH≥80% | 保留 heuristic |

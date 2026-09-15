# SpectatorRoute N05 — Phase-0 冻结机制证伪协议

> 冻结时间：2026-07-29 22:00 +08:00  
> 状态：`FROZEN / PHASE0_ONLY / EXPLORATORY / NOT_CURRENT_MAINLINE`  
> 上限：`Phase-0A <= 0.5 hour hard wall-clock`（从 worker 启动前到 gate 完成，严格于 GPU active time）；全部 Phase-0 累计 `<= 2 GPU-hour`  
> 任何 planned 数字都不是结果；正式运行前不得根据预期信号调整本协议。

## 1. 不可变 Problem Anchor

### 1.1 问题

在 exact model weights、相同 victim prompt、相同 batch size、token count、top-k work 和 arrival position 下，其他请求的 prompt 会改变 per-expert token group shape。GPU runtime 可能据此选择不同 GEMM tile、algorithm 或 split-K/reduction path，使 victim row 的浮点结果依赖于 co-batched spectator 的 route shape；该差异可能经后续 router 放大成 route/output flip。

### 1.2 唯一主张

> prompt-only spectator 是否能通过 `route histogram → victim-expert group M → actual kernel/reduction regime → victim expert output delta → downstream route/token flip` 这条链产生相对 matched-random spectator 的稳定增量？

### 1.3 不属于本 Phase-0 的主张

- 不声称 production exploit、tenant identity、攻击成功率、P99/SLO 或 EP 通信影响。
- 不声称 canonical padding、batch-invariant kernel 或 margin fallback 是新方法。
- 不声称 vLLM issue #30321 的根因来自 MoE kernel；该 issue 只证明存在待复现的用户报告。
- 不把性能、occupancy、padding 或 latency 差异当作数值语义差异。

## 2. 冻结 stack

| 项 | 冻结值 |
|---|---|
| GPU | 1× NVIDIA GeForce RTX 5090, compute capability 12.0 |
| Driver | 580.76.05 |
| CUDA / Torch | CUDA 12.8 / PyTorch 2.8.0+cu128 |
| Python / Transformers | Python 3.12.3 / Transformers 4.57.6 |
| cuBLASLt | version 120804；实际加载库 SHA-256 `10b5e663...b622c` |
| OLMoE runtime source | `modeling_olmoe.py` SHA-256 `248717a8...4c0d` |
| Model | `allenai/OLMoE-1B-7B-0924`，revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`，本地路径 `/root/autodl-tmp/models/olmoe` |
| dtype | BF16 weights / BF16 hidden and outputs；router softmax 按模型原生 FP32 路径 |
| model shape | 16 layers, hidden 2048, expert intermediate 1024, 64 experts, top-8 |
| execution | Transformers eager CUDA；不安装 vLLM，不替换 expert weights |
| primary data | `docs/ideas/routeguard_kv/experiments/data/r0a_5090_v1/sealed_manifest.jsonl` |
| calibration data | 同目录 `calibration_manifest.jsonl` |

Data hashes：

- sealed manifest SHA-256：`469e5da28dc794e50f9e3d8b1d6b2b13dfb7079d1bb6fdf9d00cd41b7c4d0d11`
- calibration manifest SHA-256：`bfb8912539806d2948595eb7ba42cfb7d09aae0b31c7c00dfbc62136abc82630`

Exact model/tokenizer bindings（在任何 pretrained Phase-0 运行前冻结）：

- `config.json`：`3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335`
- `model.safetensors.index.json`：`0e2e1e0d8d357ac7af817cff28410c3dbad398f060c517a433e4076b2aae5579`
- 三个 weight shards：`5e3cff7e...d845e`、`15ef5c73...9f4a7`、`a9abac4a...6e04`
- tokenizer 三文件：`a094266a...0a40`、`78a839c7...dff`、`b77491e2...26c2`

完整 64-hex 值以 machine-readable config 为准。该 pre-run 完整性补充不改变任何 treatment、denominator、重复数或 gate。

若任一版本、shape、hash、GPU name 或 capability 不匹配，运行返回 `INVALID_ENVIRONMENT`，不能换栈救活。

运行前另建 content-addressed `FROZEN_RUN_LOCK.json`，封存 protocol、config、runner、test 的完整 SHA-256 与所有 gate 常量。启动命令必须显式传入 lock SHA-256；parent、numeric worker、trace worker 及 aggregation 前均重验同一 digest。worker 只执行输出目录中只读的 frozen runner/config 快照。当前工作树是否 tracked 不再作为证据根；detached Git seal 只用于额外审计，不替代 SHA-256 lock。

## 3. 已发生、但不进入主结果的 preflight

冻结前仅为验证 cuBLASLt logger 可用，在相同 input/output 维度的 synthetic BF16 linear 上运行 powers-of-two M sweep。日志显示实际 heuristic/config 随 M 改变，包括：

- `M=1`：`algoId=13`；
- `M=2..16`：`algoId=21, tile=16x16`；
- `M=32`：`algoId=21, tile=32x32, numSplitsK=5`；
- `M=64`：`algoId=21, tile=32x32, numSplitsK=2`。

这只说明 instrumentation 能看到真实 tactic，且 shape→regime 的物理边在 synthetic linear 上存在；没有 pretrained hidden row、expert、router、prompt 或语义结果，**不得计入 Phase-0 PASS**。M 网格使用自然 powers-of-two，而不是按结果挑某一个正点。

## 4. Phase-0A：pretrained expert arithmetic capability gate

### 4.1 冻结 victim rows

- 32 个 sealed documents 全部进入 denominator。
- 每个 document 固定取 tokenizer token offsets `0` 与 `256` 的两个长度 16 窗口；`add_special_tokens=false`。
- 每个窗口最后一个 token 是 victim token，得到 64 个不同 `(document_index, offset)` victim IDs。
- 对每个 victim，原生完整 OLMoE forward 捕获 16 层进入 sparse MLP 前的完整 16-row hidden states，以及 victim token 原生 top-8 expert IDs。router 完整性检查必须以相同 native `M=16` 重放 gate；禁止拿单 row `M=1` 与 native `M=16` 做 bitwise equality。
- 不根据 margin、route、hidden norm 或后续结果删 victim。

### 4.2 冻结 intervention

对每个 victim、每层、每个原生 top-8 expert：

1. 取完全相同的 victim hidden row `x`。
2. 构造 `x.repeat(M, 1)`，其中 `M ∈ {1,2,4,8,16,32,64}`；victim 固定为 row 0。
3. 调用真实 pretrained expert module，取 row 0 的完整 expert output。
4. 每个 M 运行 10 次；M=64 是 frozen stable-shape reference。
5. 开启 cuBLASLt trace，记录 projection shape、algoId、tile、stages、reductionScheme、numSplitsK 和 workspace。

实现分为两个新进程以限制 trace 体积，但不得直接拼接两个进程的派生布尔值：numeric worker 生成 10/10 raw output hashes；trace worker 的每个 `(victim,layer,expert,M)` 在同一次被 trace 的 expert call 中同时保存 row-0 output hash。只有 trace hash 与 numeric 的稳定 representative hash 完全相等时，tactic 与数值证据才可合并；否则整次运行 `INVALID_ARTIFACT`。

重复 filler row 不是 prompt-only 攻击证据。它只把“别的 row 内容”固定为与 victim 完全相同，从而让 row 0 的任何差异只能来自 M/config/reduction 与 uncontrolled runtime nondeterminism。

### 4.3 Phase-0A 指标

- `within_M_bitwise_stable`：同一 victim/layer/expert/M 的 10 次 row-0 output 是否逐字节相同。
- `cross_M_bitwise_equal_to_M64`：每个 M 与 M=64 reference 是否逐字节相同。
- `max_abs_delta_to_M64`、`l2_delta_to_M64` 和 changed BF16 element count。
- `regime_signature(M)`：cuBLASLt 的完整算法/config 字段；只记录实际 trace，不用 latency 代替。
- matrix descriptor 与 workspace bytes 原样保存，但 `M` 本身或只随 `M` 机械缩放的 workspace 大小不单独构成 regime change；positive 必须看到 algoId/customOption/tile/stages/split-K/reduction/compute semantics 中至少一项改变。
- aggregation 不信任 worker 写出的 `stable/equal/signature` 派生字段：必须从每 M 恰好 10 个 repeat hashes、raw trace records 与同-call trace output hash重新计算；任何字段缺失、digest 不一致、C/D dtype/layout/leading-dimension 不符或 trace 数量不为每 expert call 恰好 3 个 GEMM，均为 `INVALID_ARTIFACT`。
- victim-level positive：至少一个原生 selected expert/layer 同时满足：
  - 所有 M 内 10/10 bitwise stable；
  - 至少一个 M 与 M=64 的 regime signature 不同；
  - 对应 row-0 expert output 在 10/10 中与 M=64 非 bitwise equal。

### 4.4 Phase-0A 裁决

- **PASS_TO_PHASE0B**：64 个 frozen victims 中至少 8 个 victim-level positive，且所有进入 positive 的 cell 无 within-M nondeterminism。
- **KILL_CURRENT_STACK**：少于 8 个 victim-level positive，或所有实际 regime changes 都不改变 victim expert output。
- **INVALID**：出现 environment/hash/version mismatch、hidden capture 不完整、非有限输出，或 uncontrolled within-M nondeterminism 无法和 M treatment 分离。
- **UNSOLVED_BUDGET**：0.5 hour hard wall-clock 内未完成；parent 对每个 worker 使用剩余时间 hard timeout 并终止超时子进程。不得缩 denominator 或减少重复数换 PASS。

Phase-0A PASS 只证明 arithmetic capability，不证明 prompt-only attacker、downstream amplification 或 CCF-B idea。

## 5. Phase-0B：prompt-only matched mechanism proof

只有 Phase-0A `PASS_TO_PHASE0B` 才授权执行本节。

### 5.1 Frozen victims 与 batches

- 同一 64 个 sealed victim windows；victim 固定为 batch row 0。
- batch size 固定 8，每条 sequence 固定 16 tokens，无 arrival/order 变化。
- 7 条 spectators 均来自 8 个 calibration documents 的确定性 16-token windows：offsets `0,128,...,1920`。
- 每个 token 固定 top-8，故每个 batch 的总 routed work 固定为 `8 × 16 × 8` expert rows。

### 5.2 Frozen spectator generator

- generator 只能读取 calibration prompts 的 route histograms，不能读取 sealed victim route、hidden state、logits 或输出。
- 对每个 candidate window，预测将它复制 7 次后的 layer-wise expert M；使用 Phase-0A 实际得到、但在运行前已定义字段的 `regime_signature(M)` 计算跨 signature-boundary 的 `(layer, expert)` 数量。
- 固定 adversarial window 为该数量最大者；tie 按 candidate text SHA-256 字典序最小者。
- matched-random set 使用 seed `20260729` 从 calibration windows 无放回抽 7 个；anchor/benign set 使用 seed `20260730` 独立无放回抽 7 个。
- 生成器与三组 sets 一经写入 artifact 后不可更换；不得按 sealed positives 重选。

### 5.3 两条 execution arms

1. **native arm**：保持 Transformers 原生 per-expert M，记录每层每 expert 的 token IDs、M、victim row position、实际 regime signature、victim raw expert output、combine output、下一层 route 与 final logits/token。
2. **canonical arm**：每个 active expert 的 hidden rows 在 GEMM 前用 zero rows pad 到固定 `M=128`，丢弃 padding outputs 后按原生 expert-ID 顺序和 gate weights combine。victim route、weights、row position和 model weights不变。

canonical arm 是机制消融，不是论文方法。若它无法和 native M=128 单元测试逐字节对齐，整条 arm `INVALID`。

### 5.4 完整因果链 success

对同一 victim，adversarial 相对 anchor 必须同时满足：

1. victim 在 earliest divergent MoE 之前的 MLP input bitwise equal；
2. 至少一个 victim selected expert 的 group M 改变；
3. 该 expert 的实际 regime signature 改变；
4. 同一 victim expert raw output 非 bitwise equal；
5. 下一层 route set 或 final argmax token 改变；
6. native arm 连续 10 次都重现同一条链；
7. canonical arm 10 次中 0 次出现第 4–5 项差异。

只看到 latency、kernel name、M、occupancy 或 hidden delta 而没有 downstream route/token flip，不计 success。

### 5.5 最终 Phase-0B Gate

- 64 个 frozen victims 全部进入 denominator。
- adversarial success 至少覆盖 8 个不同 victims，且每个 10/10 重现。
- random-vs-anchor 使用同一 success 定义；adversarial success proportion 的 95% Wilson lower bound 必须严格高于 random success proportion 的 95% Wilson upper bound。
- canonical adversarial arm 对所有 native positive victims 为 0/10。
- 任一条件失败即 `KILL`；不得更换 model、kernel、victims、spectator、M、denominator 或阈值。
- 超过全部 2 GPU-hour 返回 `UNSOLVED_BUDGET`，不是 PASS，也不是把未跑完样本从分母删除。

## 6. Phase-0B PASS 后才允许讨论的方法空间

PASS 只重新打开一个问题：routing-aware adaptive dispatch 是否需要同时满足 cross-request semantic non-interference。随后必须重新查新并在至少以下 baseline 下寻找一个非平凡方法：

- vLLM batch-invariant kernels；
- LLM-42 与 MarginGate 的 verify/rollback 或 margin gate；
- static canonical kernel/config；
- RaMP/DA-MoE performance-only routing-aware dispatch；
- reduction-signature constrained dispatch（若提出）与固定-shape padding。

若方法只是“选 deterministic kernel”“固定 reduction tree”“低 margin fallback”或“pad 到固定 M”，则分别退化为已有 batch invariance、TBIK、MarginGate/LLM-42 或简单 baseline，不能算 CCF-B 新贡献。

## 7. 当前证据边界

| 层级 | 当前状态 |
|---|---|
| 文献中 route histogram 改变最佳 kernel config | 已有 primary-source evidence |
| 5090 synthetic OLMoE-shape linear 的 M→cuBLASLt regime change | preflight observed，不进入主结果 |
| pretrained OLMoE expert row 的 cross-M numerical delta | 未运行 |
| prompt-only route-shape causality | 未运行 |
| downstream route/token amplification | 未运行 |
| batch-invariant/canonical 消融 | 未运行 |
| multi-GPU EP / DP+EP / NCCL / production security | 未验证且单 5090 不可建立 |

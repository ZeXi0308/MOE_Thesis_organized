# Receiver Homogeneous Lane Codec 硬门槛测量结论（2026-07-21）

> 权威落点：单卡 codec 必要条件，不是端到端 RDMA/P99 证明。  
> **2026-07-21 晚间勘误**：纠正 serialized 会计、基线错配与 smoke hidden；旧 `NO_SYSTEM_NET_GAIN`（以未缩放 serialized 为主口径）作废。

## 1. 做了什么

1. **Phase A（AutoDL RTX 5090）**：`run_homogeneous_lane_codec_gate_gpu.py`  
   实测 homogeneous FP8/INT4 的 pack → pinned H2D → unpack。  
2. **增量栏（离线补齐）**：`incr_fp8_to_int4` = `(FP8_wire − INT4_wire) − INT4_codec`，对齐在线策略（high=FP8，low=INT4）。  
3. **Phase B**：`require_positive_net_saving` + 拦截时 **回滚** `lane_state/dwell/credit`。  
4. **重放（纠正后）**：`analyze_receiver_codec_hardgate_replay.py`  
   - 默认 `homo_int4`  
   - `serialized_tiles = tiles × unit × (tile_rows / measured_rows)`  
   - **主判决 = `once_per_step`（融合 kernel）**

产物：

- `docs/ideas/receiver_aware/outputs/homogeneous_lane_codec_gate_2026-07-21/`
- `docs/ideas/receiver_aware/outputs/receiver_codec_hardgate_replay_corrected_2026-07-21_llmjp/`

## 2. Phase A：两道不同的门

### 2.1 BF16 → FP8（表示/压缩门，非在线动作）

常用 serving（rows∈{128,512}，200–400Gbps）homo_fp8 **仍全负**。说明 host-staging 悲观界下，BF16→FP8 整 lane 很紧。

### 2.2 FP8 → INT4（在线 low 动作）

| rows | hidden | INT4 codec µs | net@200Gbps | break_even_gbps |
|---:|---:|---:|---:|---:|
| 128 | 512 | 57.9 | **-56.5** | 4.5 |
| 512 | 512 | 57.8 | **-52.5** | 18.1 |
| 128 | 2048 | 58.1 | **-52.9** | 18.0 |
| 512 | 2048 | 61.2 | **-40.2** | 68.6 |

serving 点 **0/8 viable** → 微基准“整 lane 同质 INT4 vs FP8 wire”仍建议硬门槛。  
这与重放不同：重放的 wire saving 来自 **bottleneck 字节**，hotspot 上可远大于单 lane 均质假设。

## 3. 纠正后的重放

### LLM-jp n128（hidden=512，homo_int4）

| 口径 | 含义 | 结果摘要 |
|---|---|---|
| **once_per_step（主）** | 整步付一次实测 unit | hotspot / uniform_low_hotspot：**拦截≈0，net_p50≈+188µs**；balanced 混合；总判 **`MIXED_FUSED`** |
| serialized_tiles（缩放后） | tiles×unit×0.25 | 仍近 **100% 拦截**（tile 数极大，悲观敏感性） |

### Smoke（llmjp，hidden=512，非 2048）

小样本 + 小 batch：`once_per_step` 下仍 **高拦截 / 净负** → `NO_SYSTEM_NET_GAIN_FUSED`（仅 smoke，不作大样本结论）。

## 4. 勘误了什么

| 旧错误 | 纠正 |
|---|---|
| `serialized = tiles × unit(128-row)` 未缩放 | `× (tile_rows/measured_rows)`；且主口径改为 once_per_step |
| 用 BF16→FP8 叙事杀在线 INT4 | 增加 `incr_fp8_to_int4`；重放默认 `homo_int4` |
| smoke 误用 hidden=2048 | 改为 512 |
| hard-gate 不回滚 controller 状态 | 拦截时恢复 state/dwell/credit |
| 旧结论 `NO_SYSTEM_NET_GAIN`（未缩放 serialized） | **SUPERSEDED** |

## 5. 对论文的含义（更新）

1. **H2D 悲观界**下，均质 FP8→INT4 微基准在常用 tile 仍净负 → 保留硬门槛开关有意义。  
2. **融合 codec（once_per_step）** 时，hotspot 轨迹可出现稳定正净收益 → **不能**再写“在线降级已彻底无净收益”。  
3. **逐 tile 串行**仍极悲观；部署应走融合 pack，并把 serialized 写成敏感性而非主杀。  
4. 下一步若要坚持 receiver：冻结 `once_per_step` + `homo_int4` unit，在 Existence Test 上打开硬门槛重跑；同时标注 H2D≠RDMA。

## 6. 建议默认配置

```text
require_positive_net_saving = True
codec_tax_mode = once_per_step          # 主口径
codec_mode lookup = homo_int4
codec_measured_rows = 128               # 与 lookup 一致
codec_tile_rows = 32
# serialized_tiles 仅作敏感性：tiles * unit * (32/128)
```

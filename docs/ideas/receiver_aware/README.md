# Receiver-aware 通信控制

## 主张

在 EP combine 路径上，用 **receiver/lane 压力**（而非仅 regime 分类）决定是否对部分 lane 使用更低比特；质量用任务 harm / debt 约束。  
当前可信部分是 **结构性拥塞下的静态画像**；细粒度在线控制器为 **条件性**（须过 Existence Test + codec 硬门槛）。

## 关键证据与边界

- v2：结构 hotspot 下静态 profile 可拿到大部分字节收益；瞬时拥塞因果信号弱/负。
- Codec 硬门槛（2026-07-21）：主口径 `once_per_step`；FP8→INT4；hotspot 轨迹可正净收益，均质微基准仍紧；H2D≠RDMA。见 [`../../01_current_status/Receiver_Codec硬门槛测量结论_2026-07-21.md`](../../01_current_status/Receiver_Codec硬门槛测量结论_2026-07-21.md)。
- Existence Test：`run_receiver_aware_task_quality_gpu.py`（任务质量 λ 包络）。
- **不要**再把 HHI v3「自适应全面优于固定基线」当主结果（因果审计已撤回）。

## 脚本与产物（本目录）

- 脚本：[`experiments/`](experiments/)
- 产物：[`outputs/`](outputs/)

| 脚本 | 作用 |
|---|---|
| `experiments/receiver_lane_policy.py` | lane 策略 + `require_positive_net_saving` 硬门槛 |
| `experiments/run_receiver_aware_v2_systematic.py` | v2 系统回放 |
| `experiments/run_receiver_aware_v3_causal_audit.py` | v3 因果审计 |
| `experiments/run_receiver_direct_benefit_controller.py` | direct-benefit + 滞回 |
| `experiments/run_receiver_aware_task_quality_gpu.py` | Existence Test / 任务质量 |
| `experiments/run_homogeneous_lane_codec_gate_gpu.py` | homogeneous FP8/INT4 codec 实测 |
| `experiments/analyze_receiver_codec_hardgate_replay.py` | codec 硬门槛离线重放 |
| `experiments/run_receiver_isolation_experiment.py` | hotspot 选择隔离 |

结论文档：本目录 [`原文/`](原文/)；progressive 历史在归档 `killed_ideas/progressive/`。

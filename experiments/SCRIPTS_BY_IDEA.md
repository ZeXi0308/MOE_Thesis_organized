# 实验代码索引

| Idea | 代码入口 | 证据边界 |
|---|---|---|
| BCRD | [`docs/ideas/bcrd/experiments/`](../docs/ideas/bcrd/experiments/) | harness + smoke；未正式运行 |
| DEPA-MoE | [`docs/ideas/depa_moe/experiments/`](../docs/ideas/depa_moe/experiments/) | development only |
| CPR-MoE quick validate | [`docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/`](../docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick_validate/) | 单卡必要条件；8×A100 核心 Gate 未测 |
| fixed RankLane upper bound | [`docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick/`](../docs/ideas/receiver_aware/cpr_ranklane/experiments/cpr_moe_quick/) | 冻结域局部 NO-GO |
| Rank-tail | [`docs/ideas/A_rank_tail_fp8/experiments/`](../docs/ideas/A_rank_tail_fp8/experiments/) | 结构 evidence |
| ConfidenceGuard | [`docs/ideas/B_verify_precision/confidenceguard_v3/experiments/`](../docs/ideas/B_verify_precision/confidenceguard_v3/experiments/) | sealed scientific NO-GO |
| Legacy precision verify | [`docs/ideas/B_verify_precision/legacy_precision/experiments/`](../docs/ideas/B_verify_precision/legacy_precision/experiments/) | H1 NO-GO；H2 invalid |
| Energy-SLO | [`docs/ideas/energy_slo/route_row_fp8/experiments/`](../docs/ideas/energy_slo/route_row_fp8/experiments/) | 单卡 characterization |
| JouleQueue | [`docs/ideas/energy_slo/joulequeue/experiments/`](../docs/ideas/energy_slo/joulequeue/experiments/) | Phase 4 blocked |
| Quality debt | [`docs/ideas/quality_debt/experiments/`](../docs/ideas/quality_debt/experiments/) | NO-GO |

共享代码：[`shared/`](shared/)；历史 receiver 代码：[`docs/archive/receiver_aware/`](../docs/archive/receiver_aware/README.md)；其他停止方向：[`docs/archive/killed_ideas/`](../docs/archive/killed_ideas/README.md)。

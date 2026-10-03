# CQA arrival-trace feasibility (CPU only, no execution)

prepared dir `refine-logs/expert_saturation/outputs/admission_capacity/20260906_native_knee_r01/prepared`; engine max seqs 32; prefill token budget 1024

| regime | arm | admission events | reduction | median release width | landed on capture point | max hold ms | p90 hold ms | hold/TTFT SLO | all released |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bursty | immediate_baseline | 4/32 | 0.875 | 8 | 4 | 0.7 | 0.7 | 0.003 | True |
| bursty | cqa_g4_hold25ms | 4/32 | 0.875 | 8 | 4 | 0.7 | 0.7 | 0.003 | True |
| bursty | cqa_g4_hold50ms | 4/32 | 0.875 | 8 | 4 | 0.7 | 0.7 | 0.003 | True |
| bursty | cqa_g8_hold50ms | 4/32 | 0.875 | 8 | 4 | 0.7 | 0.7 | 0.003 | True |
| bursty | cqa_g8_hold100ms | 4/32 | 0.875 | 8 | 4 | 0.7 | 0.7 | 0.003 | True |
| bursty | coalesce_only_g8_hold50ms_nosnap | 4/32 | 0.875 | 8 | 4 | 0.7 | 0.7 | 0.003 | True |
| steady | immediate_baseline | 32/32 | 0.000 | 1 | 7 | 0.0 | 0.0 | 0.000 | True |
| steady | cqa_g4_hold25ms | 32/32 | 0.000 | 1 | 7 | 26.0 | 26.0 | 0.130 | True |
| steady | cqa_g4_hold50ms | 16/32 | 0.500 | 2 | 6 | 51.0 | 51.0 | 0.255 | True |
| steady | cqa_g8_hold50ms | 16/32 | 0.500 | 2 | 6 | 51.0 | 51.0 | 0.255 | True |
| steady | cqa_g8_hold100ms | 12/32 | 0.625 | 3 | 5 | 101.0 | 100.0 | 0.505 | True |
| steady | coalesce_only_g8_hold50ms_nosnap | 16/32 | 0.500 | 2 | 6 | 51.0 | 51.0 | 0.255 | True |

## Release reasons

- `bursty/immediate_baseline`: {'hold_budget_expired': 3, 'group_complete': 1}, widths [8, 8, 8, 8]
- `bursty/cqa_g4_hold25ms`: {'group_complete': 4}, widths [8, 8, 8, 8]
- `bursty/cqa_g4_hold50ms`: {'group_complete': 4}, widths [8, 8, 8, 8]
- `bursty/cqa_g8_hold50ms`: {'group_complete': 4}, widths [8, 8, 8, 8]
- `bursty/cqa_g8_hold100ms`: {'group_complete': 4}, widths [8, 8, 8, 8]
- `bursty/coalesce_only_g8_hold50ms_nosnap`: {'group_complete': 4}, widths [8, 8, 8, 8]
- `steady/immediate_baseline`: {'hold_budget_expired': 21, 'group_complete': 11}, widths [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- `steady/cqa_g4_hold25ms`: {'hold_budget_expired': 32}, widths [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- `steady/cqa_g4_hold50ms`: {'hold_budget_expired': 14, 'lands_on_capture_point': 2}, widths [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
- `steady/cqa_g8_hold50ms`: {'hold_budget_expired': 14, 'lands_on_capture_point': 2}, widths [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
- `steady/cqa_g8_hold100ms`: {'lands_on_capture_point': 4, 'hold_budget_expired': 8}, widths [2, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2]
- `steady/coalesce_only_g8_hold50ms_nosnap`: {'hold_budget_expired': 16}, widths [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

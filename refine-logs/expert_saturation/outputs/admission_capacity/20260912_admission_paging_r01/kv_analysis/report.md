# Native cap32: fixed KV budget intervention

MEASUREMENT_ONLY

Budget95 receives more GPU memory than budget90. This is a configuration intervention, not a same-budget scheduling gain.

| Cell | Runtime / scientific status | Complete | KV GiB / usable blocks | Full reservation enough | Active/decode/wait max | Duration s | Throughput req/s | Preemptions / recomputed tokens |
|---|---|---:|---|---|---|---:|---:|---|
| repeat0-budget95 | COMPLETE | 32/32 | 16.45703 / 8425 | True | 32/32/9 | 22.55161 | 1.41897 | 0 / 0 |
| repeat0-budget90 | COMPLETE | 32/32 | 14.98438 / 7671 | False | 32/32/9 | 23.45206 | 1.36449 | 2 / 7685 |
| repeat1-budget90 | COMPLETE | 32/32 | 14.98438 / 7671 | False | 32/32/9 | 23.33146 | 1.37154 | 2 / 7685 |
| repeat1-budget95 | COMPLETE | 32/32 | 16.55078 / 8473 | True | 32/32/8 | 22.28443 | 1.43598 | 0 / 0 |

## Complete-request latency (32 requests per cell; descriptive tails)

| Cell | TTFT p50 / p99 s | TPOT p50 / p99 ms | Completion p99 s | Pooled ITL p99 ms | Request-max-ITL p99 s | Maximum ITL s |
|---|---|---|---:|---:|---:|---:|
| repeat0-budget95 | 0.37378 / 0.82609 | 20.46498 / 20.69507 | 21.33480 | 27.62687 | 0.08315 | 0.08315 |
| repeat0-budget90 | 0.37226 / 0.82651 | 20.23457 / 20.54322 | 21.73365 | 28.13158 | 3.79789 | 4.59064 |
| repeat1-budget90 | 0.35970 / 0.80885 | 20.15543 / 20.44660 | 21.61076 | 28.22914 | 3.77146 | 4.55972 |
| repeat1-budget95 | 0.31458 / 0.71748 | 20.24325 / 20.44387 | 21.04934 | 26.41719 | 0.09888 | 0.09888 |

## Frozen within-repeat comparisons

- {"repeat": 0, "base": "repeat0-budget90", "action": "repeat0-budget95", "status": "DESCRIPTIVE_FULL_EPISODE_BUDGET_INTERVENTION", "physical_kv_added_bytes": 1581252608, "usable_blocks_added": 754, "throughput_relative_change": 0.03992842994077339, "duration_relative_change": -0.03839536336461846, "budget95_minus_budget90_s": {"ttft_p50": 0.0015209377743303776, "ttft_p99": -0.000421497989446018, "tpot_p50": 0.00023040686390806558, "tpot_p99": 0.00015184391348213935, "itl_p50": 0.0004855198785662651, "itl_p99": -0.000504709780216217}, "request_max_itl_p99_change_s": -3.7147413930390063, "completion_p99_change_s": -0.39884342126920913}
- {"repeat": 1, "base": "repeat1-budget90", "action": "repeat1-budget95", "status": "DESCRIPTIVE_FULL_EPISODE_BUDGET_INTERVENTION", "physical_kv_added_bytes": 1681915904, "usable_blocks_added": 802, "throughput_relative_change": 0.046984873739942534, "duration_relative_change": -0.044876363468468794, "budget95_minus_budget90_s": {"ttft_p50": -0.04511643573641777, "ttft_p99": -0.09137241659685968, "tpot_p50": 8.781494496286693e-05, "tpot_p99": -2.7319727663975557e-06, "itl_p50": -4.898197948932648e-05, "itl_p99": -0.0018119430169463158}, "request_max_itl_p99_change_s": -3.6725720899738405, "completion_p99_change_s": -0.5614141488168372}

## Qualification and scope

Measured cells: 4. Runtime initialization failure is separate from scientific UNRUN; absent measurement has no throughput value.
Only gpu_memory_utilization differs in engine arguments: True. Five executed source hashes match: True; software equal: True; warmups: 12.
The live full-reservation threshold is a sufficient condition for capacity, not a necessary condition for native completion or zero preemption.
All request metrics include waiting, recomputation and capture overhead. Recomputed tokens come from executed interval overlap, not output receipts.
KV occupancy is within physical KV storage; expert/parameter and allocated/reserved memory are not additive. More KV is not evidence of expert reclaim.
Reference 5 s / 200 ms SLO is secondary. No quality, second-model, EP, production-SLO or same-budget method claim.

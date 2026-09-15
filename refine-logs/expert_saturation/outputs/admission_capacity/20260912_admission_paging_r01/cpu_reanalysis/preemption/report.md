# Native preemption versus safe29: full request accounting

MEASUREMENT_ONLY

| Cell | Status | Completed | Full duration s | Throughput req/s | TTFT p50/p99 s | TPOT p50/p99 ms | Completion p99 s | Max ITL s | Native preemptions | Recomputed tokens |
|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| repeat0-native32 | COMPLETE | 32/32 | 23.16106 | 1.38163 | 0.33376/0.75993 | 19.98720/20.30952 | 21.43264 | 4.47345 | 2 | 7691 |
| repeat0-safe | COMPLETE | 32/32 | 27.09503 | 1.18103 | 0.34786/17.90026 | 18.52244/18.72107 | 25.59653 | 0.10792 | 0 | 0 |
| repeat1-safe | COMPLETE | 32/32 | 27.29250 | 1.17248 | 0.37237/18.10710 | 18.69630/18.91446 | 25.78886 | 0.10801 | 0 | 0 |
| repeat1-native32 | COMPLETE | 32/32 | 23.31108 | 1.37274 | 0.36224/0.79515 | 20.09704/20.42124 | 21.58039 | 4.51215 | 2 | 7691 |

## Victims, work and no-output calls

- repeat0-native32: active/decode/wait maxima 32/32/8; scheduler-boundary KV peak 99.974%, minimum free 2; any observed allocation/preemption/scheduler point reaches zero free blocks: True; successful calls without new tokens 2; work {'scheduled_tokens': 138731, 'recomputed_tokens': 7691, 'new_prefill_tokens': 98304, 'new_decode_tokens': 32736}.
  - Pooled token ITL p99 0.027686 s versus request-max-ITL p99 3.691920 s. These have different populations; sparse victim pauses can disappear in pooled p99.
  - Victim memory-train-article-0003571: recomputed 3908 positions; longest ITL {'start_s': 18.799047585576773, 'end_s': 20.751436982303858, 'itl_s': 1.9523893967270851, 'output_index': 837, 'own_preemption_steps': [931], 'own_recomputed_tokens': 3908, 'own_recompute_steps': [1026, 1027, 1028, 1029]}.
  - Victim memory-train-article-0003640: recomputed 3783 positions; longest ITL {'start_s': 16.393898896872997, 'end_s': 20.867347300052643, 'itl_s': 4.4734484031796455, 'output_index': 712, 'own_preemption_steps': [809], 'own_recomputed_tokens': 3783, 'own_recompute_steps': [1030, 1031, 1032, 1033]}.
- repeat0-safe: active/decode/wait maxima 29/29/8; scheduler-boundary KV peak 95.832%, minimum free 320; any observed allocation/preemption/scheduler point reaches zero free blocks: False; successful calls without new tokens 2; work {'scheduled_tokens': 131040, 'recomputed_tokens': 0, 'new_prefill_tokens': 98304, 'new_decode_tokens': 32736}.
  - Pooled token ITL p99 0.028603 s versus request-max-ITL p99 0.107922 s. These have different populations; sparse victim pauses can disappear in pooled p99.
- repeat1-safe: active/decode/wait maxima 29/29/9; scheduler-boundary KV peak 95.832%, minimum free 320; any observed allocation/preemption/scheduler point reaches zero free blocks: False; successful calls without new tokens 2; work {'scheduled_tokens': 131040, 'recomputed_tokens': 0, 'new_prefill_tokens': 98304, 'new_decode_tokens': 32736}.
  - Pooled token ITL p99 0.028778 s versus request-max-ITL p99 0.108009 s. These have different populations; sparse victim pauses can disappear in pooled p99.
- repeat1-native32: active/decode/wait maxima 32/32/9; scheduler-boundary KV peak 99.974%, minimum free 2; any observed allocation/preemption/scheduler point reaches zero free blocks: True; successful calls without new tokens 2; work {'scheduled_tokens': 138731, 'recomputed_tokens': 7691, 'new_prefill_tokens': 98304, 'new_decode_tokens': 32736}.
  - Pooled token ITL p99 0.028051 s versus request-max-ITL p99 3.723924 s. These have different populations; sparse victim pauses can disappear in pooled p99.
  - Victim memory-train-article-0003571: recomputed 3908 positions; longest ITL {'start_s': 18.919567815959454, 'end_s': 20.8890445753932, 'itl_s': 1.9694767594337463, 'output_index': 837, 'own_preemption_steps': [931], 'own_recomputed_tokens': 3908, 'own_recompute_steps': [1026, 1027, 1028, 1029]}.
  - Victim memory-train-article-0003640: recomputed 3783 positions; longest ITL {'start_s': 16.492974117398262, 'end_s': 21.005127500742674, 'itl_s': 4.512153383344412, 'output_index': 712, 'own_preemption_steps': [809], 'own_recomputed_tokens': 3783, 'own_recompute_steps': [1030, 1031, 1032, 1033]}.

## Boundaries

Engine arguments equal: True; five executed source hashes match: True; software equal: True; warmup raw count: 12.
All main metrics use complete host request histories. Waiting/recomputation/telemetry already reside in that denominator; no stage is added again.
Recomputation is interval overlap in successful scheduled work. Token receipt count is not a work counter; successful no-output calls are retained.
Native victims may stop decoding while waiting. That is legal preemption, not a nonpreemptive-invariant failure.
Reference 5 s / 200 ms SLO is secondary. Request-level p99 uses 32 repeated inputs and is descriptive; output quality and expert reclaim are unmeasured.

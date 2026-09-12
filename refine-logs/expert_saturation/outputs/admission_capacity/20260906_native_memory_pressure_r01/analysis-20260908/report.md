# Natural context capacity: retained execution analysis

MEASUREMENT_ONLY

All eight planned cells remain listed. Full-episode rates are reported only for complete cells. Reference SLO 5 s / 200 ms is secondary.

| Cell | Status | Complete | Active/decode/wait max | Used/usable blocks peak | KV peak fraction | Min free | Full episode s | Full throughput req/s | TTFT p50 s | TPOT p50 ms |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| repeat0-short-cap16 | COMPLETE | 32/32 | 16/16/16 | 1152/7629 | 0.15100 | 6477 | 20.87851 | 1.53268 | 5.56162 | 9.27089 |
| repeat0-short-cap32 | COMPLETE | 32/32 | 32/32/0 | 2144/7677 | 0.27928 | 5533 | 13.13398 | 2.43643 | 0.01820 | 11.46637 |
| repeat0-long-cap16 | COMPLETE | 32/32 | 16/16/16 | 4079/7677 | 0.53133 | 3598 | 28.66478 | 1.11635 | 6.88151 | 13.68481 |
| repeat0-long-cap32 | CAPACITY_BOUNDARY_STOP | 0/32 | 32/32/9 | 7677/7677 | 1.00000 | 0 | — | — | — | — |
| repeat1-short-cap16 | COMPLETE | 32/32 | 16/16/16 | 1112/7677 | 0.14485 | 6565 | 19.72288 | 1.62248 | 4.32724 | 9.32084 |
| repeat1-short-cap32 | COMPLETE | 32/32 | 32/32/0 | 2126/7677 | 0.27693 | 5551 | 13.20477 | 2.42337 | 0.01753 | 11.53128 |
| repeat1-long-cap16 | COMPLETE | 32/32 | 16/16/16 | 4079/7677 | 0.53133 | 3598 | 28.39905 | 1.12680 | 6.79460 | 13.56982 |
| repeat1-long-cap32 | CAPACITY_BOUNDARY_STOP | 0/32 | 32/32/9 | 7677/7677 | 1.00000 | 0 | — | — | — | — |

## Capacity stops

- repeat0-long-cap32: attempt 809 (zero based), 32 running / 0 waiting, all-decode eligibility 32; computed tokens [3783, 3878]. Before {'total_blocks': 7678, 'usable_blocks': 7677, 'free_blocks': 2, 'used_blocks': 7675}; allocation failure [{'internal_request_id': 'measured/memory-train-article-0003345-8bd40fa4', 'requested_tokens': 1, 'before': {'total_blocks': 7678, 'usable_blocks': 7677, 'free_blocks': 0, 'used_blocks': 7677}, 'after': {'total_blocks': 7678, 'usable_blocks': 7677, 'free_blocks': 0, 'used_blocks': 7677}, 'computed_tokens': 3792}]; generated [712, 807] of 1024 tokens. No destructive preemption or GPU execution for failed attempt. Stop time 16.39847 s is a truncated horizon.
- repeat1-long-cap32: attempt 809 (zero based), 32 running / 0 waiting, all-decode eligibility 32; computed tokens [3783, 3878]. Before {'total_blocks': 7678, 'usable_blocks': 7677, 'free_blocks': 2, 'used_blocks': 7675}; allocation failure [{'internal_request_id': 'measured/memory-train-article-0003345-a834197b', 'requested_tokens': 1, 'before': {'total_blocks': 7678, 'usable_blocks': 7677, 'free_blocks': 0, 'used_blocks': 7677}, 'after': {'total_blocks': 7678, 'usable_blocks': 7677, 'free_blocks': 0, 'used_blocks': 7677}, 'computed_tokens': 3792}]; generated [712, 807] of 1024 tokens. No destructive preemption or GPU execution for failed attempt. Stop time 16.51773 s is a truncated horizon.

## Accounting and qualification

Identity-checked cells: 8; engine arguments equal: True.
Four execution-source hashes match archive for every observed cell: True; software fields equal: True; analysis metrics match executed metrics: True.
Retained full warmup raw files: 24; all completed with 16 outputs per request: True.
KV occupied blocks are inside the preallocated KV pool. Expert bytes are a subset of parameter bytes; Torch reserved includes allocated.
Per-engine KV pool sizes may differ despite common arguments. Boundary checks do not certify continuous GPU isolation.
Reference SLO request counts and all observed partial metrics remain in JSON; partial metrics must not be compared as completed performance.

- Truncated boundary-cell rates are not full-horizon policy throughput.
- KV occupancy is within the physical KV pool, not additive to it.
- Resident expert parameters do not establish expert transfer or reclaim headroom.
- Reference SLO is secondary; no controller, task quality or paper claim.

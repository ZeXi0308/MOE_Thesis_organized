# Prefill budget: retained request measurements

Host receipt times; pooled ITL quantiles descriptive, not independent step samples. All observed latency retained including failures; comparisons require both complete cells. Group legacy rates start at that group's first arrival; whole_episode_completion_rps uses the shared episode wall. No Oracle, no fitted counterfactual, no between-engine averaging.

All times below are ms; wall is seconds. Warmups excluded.

| Block | Cohort | Budget | Status | Group | Complete/planned | Wall s | TTFT mean | TPOT mean | ITL p50 / p95 / p99 |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| forward | mixed | 512 | COMPLETE | all | 16/16 | 2.5792 | 452.9607 | 8.6286 | 8.0729 / 14.1083 / 14.5202 |
| forward | mixed | 512 | COMPLETE | short | 8/8 | 2.5792 | 429.5225 | 8.7360 | 8.0834 / 14.2052 / 14.5245 |
| forward | mixed | 512 | COMPLETE | long | 8/8 | 2.5792 | 476.3989 | 8.5212 | 8.0656 / 14.0945 / 14.5027 |
| forward | mixed | 1024 | COMPLETE | all | 16/16 | 2.5257 | 425.8622 | 8.5512 | 7.9367 / 13.6295 / 20.7131 |
| forward | mixed | 1024 | COMPLETE | short | 8/8 | 2.5257 | 406.4931 | 8.6635 | 7.9427 / 16.3975 / 20.7342 |
| forward | mixed | 1024 | COMPLETE | long | 8/8 | 2.5257 | 445.2313 | 8.4389 | 7.9346 / 12.9699 / 20.6660 |
| reverse | mixed | 1024 | COMPLETE | all | 16/16 | 2.5288 | 425.1605 | 8.5642 | 7.9635 / 13.0797 / 20.5936 |
| reverse | mixed | 1024 | COMPLETE | short | 8/8 | 2.5288 | 405.9771 | 8.6761 | 7.9646 / 13.5019 / 20.5936 |
| reverse | mixed | 1024 | COMPLETE | long | 8/8 | 2.5288 | 444.3440 | 8.4524 | 7.9591 / 13.0556 / 20.5858 |
| reverse | mixed | 512 | COMPLETE | all | 16/16 | 2.5891 | 458.3386 | 8.6555 | 8.0392 / 14.0755 / 14.5226 |
| reverse | mixed | 512 | COMPLETE | short | 8/8 | 2.5891 | 434.8807 | 8.7687 | 8.0448 / 14.0772 / 14.7326 |
| reverse | mixed | 512 | COMPLETE | long | 8/8 | 2.5891 | 481.7964 | 8.5423 | 8.0359 / 14.0566 / 14.4934 |

| Block / cohort / budget | Steps | Token max | Ceiling hits | Prefill max | Mixed steps | Empty returns | Skips / preemptions / KV adjustments | Short negative control |
|---|---:|---:|---:|---:|---:|---:|---|---|
| forward / mixed / 512 | 290 | 512 | 32 | 511 | 41 | 0 | 0 / 0 / 0 | NOT_APPLICABLE |
| forward / mixed / 1024 | 287 | 1024 | 16 | 1023 | 27 | 0 | 0 / 0 / 0 | NOT_APPLICABLE |
| reverse / mixed / 1024 | 287 | 1024 | 16 | 1023 | 27 | 0 | 0 / 0 / 0 | NOT_APPLICABLE |
| reverse / mixed / 512 | 289 | 512 | 32 | 511 | 41 | 0 | 0 / 0 / 0 | NOT_APPLICABLE |

Budget comparisons: delta = 512 minus 1024; negative latency delta is lower. Each engine reported separately.

| Block / cohort | Valid | Actual >512 exposure at 1024 | Group / metric | 1024 ms | 512 ms | Delta ms | Delta % |
|---|---|---|---|---:|---:|---:|---:|
| forward / mixed | True | True | all / episode wall | 2525.6676 | 2579.2107 | 53.5432 | 2.1200 |
| forward / mixed | True | True | all / ttft_mean_s | 425.8622 | 452.9607 | 27.0985 | 6.3632 |
| forward / mixed | True | True | all / tpot_mean_s | 8.5512 | 8.6286 | 0.0774 | 0.9046 |
| forward / mixed | True | True | all / itl_p95_s | 13.6295 | 14.1083 | 0.4788 | 3.5133 |
| forward / mixed | True | True | all / itl_p99_s | 20.7131 | 14.5202 | -6.1928 | -29.8982 |
| forward / mixed | True | True | short / ttft_mean_s | 406.4931 | 429.5225 | 23.0295 | 5.6654 |
| forward / mixed | True | True | short / tpot_mean_s | 8.6635 | 8.7360 | 0.0724 | 0.8361 |
| forward / mixed | True | True | short / itl_p95_s | 16.3975 | 14.2052 | -2.1923 | -13.3696 |
| forward / mixed | True | True | short / itl_p99_s | 20.7342 | 14.5245 | -6.2097 | -29.9492 |
| forward / mixed | True | True | long / ttft_mean_s | 445.2313 | 476.3989 | 31.1676 | 7.0003 |
| forward / mixed | True | True | long / tpot_mean_s | 8.4389 | 8.5212 | 0.0823 | 0.9751 |
| forward / mixed | True | True | long / itl_p95_s | 12.9699 | 14.0945 | 1.1246 | 8.6708 |
| forward / mixed | True | True | long / itl_p99_s | 20.6660 | 14.5027 | -6.1633 | -29.8234 |
| reverse / mixed | True | True | all / episode wall | 2528.8128 | 2589.0628 | 60.2500 | 2.3825 |
| reverse / mixed | True | True | all / ttft_mean_s | 425.1605 | 458.3386 | 33.1780 | 7.8036 |
| reverse / mixed | True | True | all / tpot_mean_s | 8.5642 | 8.6555 | 0.0912 | 1.0654 |
| reverse / mixed | True | True | all / itl_p95_s | 13.0797 | 14.0755 | 0.9958 | 7.6132 |
| reverse / mixed | True | True | all / itl_p99_s | 20.5936 | 14.5226 | -6.0710 | -29.4801 |
| reverse / mixed | True | True | short / ttft_mean_s | 405.9771 | 434.8807 | 28.9036 | 7.1195 |
| reverse / mixed | True | True | short / tpot_mean_s | 8.6761 | 8.7687 | 0.0926 | 1.0672 |
| reverse / mixed | True | True | short / itl_p95_s | 13.5019 | 14.0772 | 0.5753 | 4.2607 |
| reverse / mixed | True | True | short / itl_p99_s | 20.5936 | 14.7326 | -5.8610 | -28.4602 |
| reverse / mixed | True | True | long / ttft_mean_s | 444.3440 | 481.7964 | 37.4525 | 8.4287 |
| reverse / mixed | True | True | long / tpot_mean_s | 8.4524 | 8.5423 | 0.0899 | 1.0635 |
| reverse / mixed | True | True | long / itl_p95_s | 13.0556 | 14.0566 | 1.0010 | 7.6672 |
| reverse / mixed | True | True | long / itl_p99_s | 20.5858 | 14.4934 | -6.0924 | -29.5951 |

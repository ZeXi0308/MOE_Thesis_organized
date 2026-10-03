# Prefill budget: retained request measurements

Host receipt times; pooled ITL quantiles descriptive, not independent step samples. All observed latency retained including failures; comparisons require both complete cells. Group legacy rates start at that group's first arrival; whole_episode_completion_rps uses the shared episode wall. No Oracle, no fitted counterfactual, no between-engine averaging.

All times below are ms; wall is seconds. Warmups excluded.

| Block | Cohort | Budget | Status | Group | Complete/planned | Wall s | TTFT mean | TPOT mean | ITL p50 / p95 / p99 |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| forward | all_short | 256 | COMPLETE | all | 16/16 | 2.1765 | 314.1208 | 7.3313 | 7.0171 / 12.0419 / 14.7428 |
| forward | all_short | 256 | COMPLETE | short | 16/16 | 2.1765 | 314.1208 | 7.3313 | 7.0171 / 12.0419 / 14.7428 |
| forward | all_short | 256 | COMPLETE | long | 0/0 | 2.1765 | NA | NA | NA / NA / NA |
| forward | all_short | 1024 | COMPLETE | all | 16/16 | 2.1680 | 307.7921 | 7.2913 | 7.0160 / 11.8832 / 13.6544 |
| forward | all_short | 1024 | COMPLETE | short | 16/16 | 2.1680 | 307.7921 | 7.2913 | 7.0160 / 11.8832 / 13.6544 |
| forward | all_short | 1024 | COMPLETE | long | 0/0 | 2.1680 | NA | NA | NA / NA / NA |
| forward | mixed | 256 | COMPLETE | all | 16/16 | 2.8044 | 582.2532 | 8.9517 | 7.9969 / 13.2452 / 20.9375 |
| forward | mixed | 256 | COMPLETE | short | 8/8 | 2.8044 | 542.8843 | 9.1019 | 8.0186 / 13.2500 / 20.9375 |
| forward | mixed | 256 | COMPLETE | long | 8/8 | 2.8044 | 621.6221 | 8.8015 | 7.9834 / 13.1920 / 13.8968 |
| forward | mixed | 1024 | COMPLETE | all | 16/16 | 2.5716 | 440.9168 | 8.7104 | 8.0227 / 13.4916 / 21.9502 |
| forward | mixed | 1024 | COMPLETE | short | 8/8 | 2.5716 | 422.0707 | 8.8165 | 8.0241 / 15.6935 / 21.9502 |
| forward | mixed | 1024 | COMPLETE | long | 8/8 | 2.5716 | 459.7629 | 8.6043 | 8.0194 / 13.1661 / 20.8420 |
| reverse | mixed | 1024 | COMPLETE | all | 16/16 | 2.5475 | 431.1313 | 8.6413 | 8.0088 / 13.8029 / 20.7120 |
| reverse | mixed | 1024 | COMPLETE | short | 8/8 | 2.5475 | 412.5580 | 8.7521 | 8.0137 / 14.6193 / 20.7268 |
| reverse | mixed | 1024 | COMPLETE | long | 8/8 | 2.5475 | 449.7047 | 8.5305 | 8.0032 / 13.5339 / 20.6791 |
| reverse | mixed | 256 | COMPLETE | all | 16/16 | 2.7802 | 580.1448 | 8.8489 | 8.0106 / 13.3927 / 14.0412 |
| reverse | mixed | 256 | COMPLETE | short | 8/8 | 2.7802 | 542.5572 | 8.9952 | 8.0180 / 13.4039 / 14.0412 |
| reverse | mixed | 256 | COMPLETE | long | 8/8 | 2.7802 | 617.7325 | 8.7025 | 7.9914 / 13.3495 / 13.7693 |
| reverse | all_short | 1024 | COMPLETE | all | 16/16 | 2.1387 | 292.8227 | 7.1463 | 6.9746 / 8.0597 / 13.5316 |
| reverse | all_short | 1024 | COMPLETE | short | 16/16 | 2.1387 | 292.8227 | 7.1463 | 6.9746 / 8.0597 / 13.5316 |
| reverse | all_short | 1024 | COMPLETE | long | 0/0 | 2.1387 | NA | NA | NA / NA / NA |
| reverse | all_short | 256 | COMPLETE | all | 16/16 | 2.1497 | 284.1708 | 7.1524 | 6.9834 / 8.3506 / 14.3723 |
| reverse | all_short | 256 | COMPLETE | short | 16/16 | 2.1497 | 284.1708 | 7.1524 | 6.9834 / 8.3506 / 14.3723 |
| reverse | all_short | 256 | COMPLETE | long | 0/0 | 2.1497 | NA | NA | NA / NA / NA |

| Block / cohort / budget | Steps | Token max | Ceiling hits | Prefill max | Mixed steps | Empty returns | Skips / preemptions / KV adjustments | Short negative control |
|---|---:|---:|---:|---:|---:|---:|---|---|
| forward / all_short / 256 | 296 | 135 | 0 | 128 | 15 | 0 | 0 / 0 / 0 | NO_OBSERVED_BINDING |
| forward / all_short / 1024 | 296 | 135 | 0 | 128 | 15 | 0 | 0 / 0 / 0 | NO_OBSERVED_BINDING |
| forward / mixed / 256 | 301 | 256 | 68 | 255 | 71 | 0 | 0 / 0 / 0 | NOT_APPLICABLE |
| forward / mixed / 1024 | 286 | 1024 | 16 | 1023 | 27 | 0 | 0 / 0 / 0 | NOT_APPLICABLE |
| reverse / mixed / 1024 | 286 | 1024 | 16 | 1023 | 27 | 0 | 0 / 0 / 0 | NOT_APPLICABLE |
| reverse / mixed / 256 | 300 | 256 | 68 | 255 | 71 | 0 | 0 / 0 / 0 | NOT_APPLICABLE |
| reverse / all_short / 1024 | 305 | 135 | 0 | 128 | 15 | 0 | 0 / 0 / 0 | NO_OBSERVED_BINDING |
| reverse / all_short / 256 | 314 | 135 | 0 | 128 | 15 | 0 | 0 / 0 / 0 | NO_OBSERVED_BINDING |

Budget comparisons: delta = 256 minus 1024; negative latency delta is lower. Each engine reported separately.

| Block / cohort | Valid | Actual >256 exposure at 1024 | Group / metric | 1024 ms | 256 ms | Delta ms | Delta % |
|---|---|---|---|---:|---:|---:|---:|
| forward / all_short | True | False | all / episode wall | 2167.9711 | 2176.5306 | 8.5595 | 0.3948 |
| forward / all_short | True | False | all / ttft_mean_s | 307.7921 | 314.1208 | 6.3287 | 2.0562 |
| forward / all_short | True | False | all / tpot_mean_s | 7.2913 | 7.3313 | 0.0401 | 0.5494 |
| forward / all_short | True | False | all / itl_p95_s | 11.8832 | 12.0419 | 0.1587 | 1.3353 |
| forward / all_short | True | False | all / itl_p99_s | 13.6544 | 14.7428 | 1.0884 | 7.9711 |
| forward / all_short | True | False | short / ttft_mean_s | 307.7921 | 314.1208 | 6.3287 | 2.0562 |
| forward / all_short | True | False | short / tpot_mean_s | 7.2913 | 7.3313 | 0.0401 | 0.5494 |
| forward / all_short | True | False | short / itl_p95_s | 11.8832 | 12.0419 | 0.1587 | 1.3353 |
| forward / all_short | True | False | short / itl_p99_s | 13.6544 | 14.7428 | 1.0884 | 7.9711 |
| forward / mixed | True | True | all / episode wall | 2571.5778 | 2804.4016 | 232.8238 | 9.0537 |
| forward / mixed | True | True | all / ttft_mean_s | 440.9168 | 582.2532 | 141.3364 | 32.0551 |
| forward / mixed | True | True | all / tpot_mean_s | 8.7104 | 8.9517 | 0.2413 | 2.7704 |
| forward / mixed | True | True | all / itl_p95_s | 13.4916 | 13.2452 | -0.2464 | -1.8264 |
| forward / mixed | True | True | all / itl_p99_s | 21.9502 | 20.9375 | -1.0127 | -4.6135 |
| forward / mixed | True | True | short / ttft_mean_s | 422.0707 | 542.8843 | 120.8136 | 28.6240 |
| forward / mixed | True | True | short / tpot_mean_s | 8.8165 | 9.1019 | 0.2854 | 3.2370 |
| forward / mixed | True | True | short / itl_p95_s | 15.6935 | 13.2500 | -2.4435 | -15.5700 |
| forward / mixed | True | True | short / itl_p99_s | 21.9502 | 20.9375 | -1.0127 | -4.6135 |
| forward / mixed | True | True | long / ttft_mean_s | 459.7629 | 621.6221 | 161.8592 | 35.2049 |
| forward / mixed | True | True | long / tpot_mean_s | 8.6043 | 8.8015 | 0.1972 | 2.2922 |
| forward / mixed | True | True | long / itl_p95_s | 13.1661 | 13.1920 | 0.0259 | 0.1969 |
| forward / mixed | True | True | long / itl_p99_s | 20.8420 | 13.8968 | -6.9451 | -33.3229 |
| reverse / all_short | True | False | all / episode wall | 2138.7114 | 2149.6925 | 10.9810 | 0.5134 |
| reverse / all_short | True | False | all / ttft_mean_s | 292.8227 | 284.1708 | -8.6519 | -2.9546 |
| reverse / all_short | True | False | all / tpot_mean_s | 7.1463 | 7.1524 | 0.0061 | 0.0853 |
| reverse / all_short | True | False | all / itl_p95_s | 8.0597 | 8.3506 | 0.2910 | 3.6101 |
| reverse / all_short | True | False | all / itl_p99_s | 13.5316 | 14.3723 | 0.8407 | 6.2127 |
| reverse / all_short | True | False | short / ttft_mean_s | 292.8227 | 284.1708 | -8.6519 | -2.9546 |
| reverse / all_short | True | False | short / tpot_mean_s | 7.1463 | 7.1524 | 0.0061 | 0.0853 |
| reverse / all_short | True | False | short / itl_p95_s | 8.0597 | 8.3506 | 0.2910 | 3.6101 |
| reverse / all_short | True | False | short / itl_p99_s | 13.5316 | 14.3723 | 0.8407 | 6.2127 |
| reverse / mixed | True | True | all / episode wall | 2547.5083 | 2780.2265 | 232.7183 | 9.1351 |
| reverse / mixed | True | True | all / ttft_mean_s | 431.1313 | 580.1448 | 149.0135 | 34.5634 |
| reverse / mixed | True | True | all / tpot_mean_s | 8.6413 | 8.8489 | 0.2076 | 2.4022 |
| reverse / mixed | True | True | all / itl_p95_s | 13.8029 | 13.3927 | -0.4102 | -2.9716 |
| reverse / mixed | True | True | all / itl_p99_s | 20.7120 | 14.0412 | -6.6708 | -32.2074 |
| reverse / mixed | True | True | short / ttft_mean_s | 412.5580 | 542.5572 | 129.9992 | 31.5105 |
| reverse / mixed | True | True | short / tpot_mean_s | 8.7521 | 8.9952 | 0.2432 | 2.7782 |
| reverse / mixed | True | True | short / itl_p95_s | 14.6193 | 13.4039 | -1.2154 | -8.3138 |
| reverse / mixed | True | True | short / itl_p99_s | 20.7268 | 14.0412 | -6.6856 | -32.2558 |
| reverse / mixed | True | True | long / ttft_mean_s | 449.7047 | 617.7325 | 168.0278 | 37.3640 |
| reverse / mixed | True | True | long / tpot_mean_s | 8.5305 | 8.7025 | 0.1720 | 2.0164 |
| reverse / mixed | True | True | long / itl_p95_s | 13.5339 | 13.3495 | -0.1844 | -1.3627 |
| reverse / mixed | True | True | long / itl_p99_s | 20.6791 | 13.7693 | -6.9098 | -33.4143 |

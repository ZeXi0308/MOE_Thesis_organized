# TokenRace-EP P0 结果

门槛 MIN_IMPROVEMENT = 0.05

**GATE: PASS**

- All gates passed with no warnings.


## OLMoE-1B-7B (E64/top8)

### regime=none

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 475.7 | 433.0 | 8.98% | 501.3 | 480.2 | 4.19% | 10.06% |
### regime=moderate

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 526.3 | 452.2 | 14.08% | 607.3 | 534.6 | 11.97% | 15.02% |
### regime=severe

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 817.7 | 555.4 | 32.07% | 995.4 | 752.0 | 24.45% | 31.69% |

## LLM-jp E32/top16

### regime=none

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 757.8 | 750.8 | 0.92% | 776.4 | 773.5 | 0.36% | 1.13% |
### regime=moderate

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 898.4 | 845.5 | 5.89% | 1052.7 | 991.3 | 5.83% | 5.86% |
### regime=severe

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 1479.4 | 1264.5 | 14.53% | 1818.0 | 1623.0 | 10.73% | 14.43% |
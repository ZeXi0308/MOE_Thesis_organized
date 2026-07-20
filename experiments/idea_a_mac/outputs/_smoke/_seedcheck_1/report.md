# TokenRace-EP P0 结果

门槛 MIN_IMPROVEMENT = 0.05

**GATE: PASS**

- All gates passed with no warnings.


## OLMoE-1B-7B (E64/top8)

### regime=none

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 475.0 | 433.0 | 8.84% | 501.6 | 479.9 | 4.33% | 10.01% |
### regime=moderate

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 526.6 | 451.7 | 14.22% | 606.6 | 534.4 | 11.90% | 14.95% |
### regime=severe

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 816.6 | 556.2 | 31.89% | 993.4 | 753.0 | 24.20% | 31.60% |

## LLM-jp E32/top16

### regime=none

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 757.4 | 750.4 | 0.92% | 776.0 | 773.2 | 0.36% | 1.14% |
### regime=moderate

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 898.8 | 845.7 | 5.90% | 1044.2 | 986.9 | 5.49% | 5.82% |
### regime=severe

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 1474.7 | 1259.5 | 14.59% | 1802.3 | 1621.1 | 10.05% | 14.42% |
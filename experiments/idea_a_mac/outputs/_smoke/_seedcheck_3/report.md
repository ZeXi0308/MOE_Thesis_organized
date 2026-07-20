# TokenRace-EP P0 结果

门槛 MIN_IMPROVEMENT = 0.05

**GATE: PASS**

- All gates passed with no warnings.


## OLMoE-1B-7B (E64/top8)

### regime=none

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 475.3 | 432.7 | 8.98% | 500.5 | 479.5 | 4.20% | 10.04% |
### regime=moderate

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 526.9 | 452.0 | 14.22% | 608.8 | 535.4 | 12.05% | 14.99% |
### regime=severe

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 817.7 | 555.6 | 32.06% | 983.8 | 752.5 | 23.51% | 31.60% |

## LLM-jp E32/top16

### regime=none

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 757.1 | 750.1 | 0.92% | 777.4 | 774.6 | 0.36% | 1.14% |
### regime=moderate

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 898.1 | 845.0 | 5.92% | 1044.1 | 985.5 | 5.61% | 5.83% |
### regime=severe

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 1477.1 | 1259.2 | 14.75% | 1806.5 | 1621.6 | 10.24% | 14.40% |
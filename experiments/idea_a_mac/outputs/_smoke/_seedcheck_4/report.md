# TokenRace-EP P0 结果

门槛 MIN_IMPROVEMENT = 0.05

**GATE: PASS**

- All gates passed with no warnings.


## OLMoE-1B-7B (E64/top8)

### regime=none

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 474.6 | 432.6 | 8.85% | 501.6 | 478.8 | 4.54% | 10.03% |
### regime=moderate

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 527.5 | 452.2 | 14.27% | 604.8 | 534.5 | 11.62% | 15.05% |
### regime=severe

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 817.7 | 556.6 | 31.94% | 992.6 | 755.9 | 23.85% | 31.60% |

## LLM-jp E32/top16

### regime=none

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 757.1 | 750.1 | 0.92% | 777.0 | 774.2 | 0.36% | 1.14% |
### regime=moderate

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 900.6 | 846.4 | 6.01% | 1041.3 | 986.2 | 5.30% | 5.88% |
### regime=severe

| B | barrier P50 | race P50 | imp P50 | barrier P99 | race P99 | imp P99 | imp mean |
|---|---|---|---|---|---|---|---|
| 128 | 1477.3 | 1261.0 | 14.64% | 1781.3 | 1599.6 | 10.20% | 14.41% |
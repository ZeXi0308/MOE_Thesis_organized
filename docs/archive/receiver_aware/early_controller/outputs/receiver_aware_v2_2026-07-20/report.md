# Receiver-Aware v2: Systematic Cross-Model Temporal Replay

models=['olmoe', 'llmjp']; placements=['contiguous', 'round_robin']; origin_modes=['balanced', 'hotspot']; budget_fractions=[0.25, 0.5, 0.75]; scenario_seeds=24; num_jobs=16

## hot - random saving (mean over scenario seeds); positive & 'frac_seeds...' close to 1.0 means hot robustly beats random

| model | placement | origin_mode | budget_fraction | info_mode | hot_minus_random_mean | hot_minus_random_std | frac_seeds_hot_beats_random_ci | n_seeds |
|---|---|---|---|---|---|---|---|---|
| llmjp | contiguous | balanced | 0.2500 | calib_static | -0.0317 | 0.0118 | 1.0000 | 24 |
| llmjp | contiguous | balanced | 0.2500 | causal_prev_step | -0.0175 | 0.0132 | 1.0000 | 24 |
| llmjp | contiguous | balanced | 0.2500 | oracle_same_step | 0.0242 | 0.0152 | 0.9583 | 24 |
| llmjp | contiguous | balanced | 0.5000 | calib_static | -0.0664 | 0.0191 | 1.0000 | 24 |
| llmjp | contiguous | balanced | 0.5000 | causal_prev_step | -0.0257 | 0.0157 | 0.9583 | 24 |
| llmjp | contiguous | balanced | 0.5000 | oracle_same_step | 0.0241 | 0.0144 | 1.0000 | 24 |
| llmjp | contiguous | balanced | 0.7500 | calib_static | -0.0453 | 0.0150 | 1.0000 | 24 |
| llmjp | contiguous | balanced | 0.7500 | causal_prev_step | -0.0222 | 0.0094 | 1.0000 | 24 |
| llmjp | contiguous | balanced | 0.7500 | oracle_same_step | 0.0278 | 0.0077 | 1.0000 | 24 |
| llmjp | contiguous | hotspot | 0.2500 | calib_static | 0.0579 | 0.0043 | 1.0000 | 24 |
| llmjp | contiguous | hotspot | 0.2500 | causal_prev_step | 0.0566 | 0.0049 | 1.0000 | 24 |
| llmjp | contiguous | hotspot | 0.2500 | oracle_same_step | 0.0592 | 0.0032 | 1.0000 | 24 |
| llmjp | contiguous | hotspot | 0.5000 | calib_static | 0.1089 | 0.0069 | 1.0000 | 24 |
| llmjp | contiguous | hotspot | 0.5000 | causal_prev_step | 0.1073 | 0.0076 | 1.0000 | 24 |
| llmjp | contiguous | hotspot | 0.5000 | oracle_same_step | 0.1105 | 0.0058 | 1.0000 | 24 |
| llmjp | contiguous | hotspot | 0.7500 | calib_static | 0.0592 | 0.0020 | 1.0000 | 24 |
| llmjp | contiguous | hotspot | 0.7500 | causal_prev_step | 0.0589 | 0.0021 | 1.0000 | 24 |
| llmjp | contiguous | hotspot | 0.7500 | oracle_same_step | 0.0605 | 0.0011 | 1.0000 | 24 |
| llmjp | round_robin | balanced | 0.2500 | calib_static | -0.0376 | 0.0129 | 1.0000 | 24 |
| llmjp | round_robin | balanced | 0.2500 | causal_prev_step | -0.0159 | 0.0122 | 0.9583 | 24 |
| llmjp | round_robin | balanced | 0.2500 | oracle_same_step | 0.0247 | 0.0148 | 1.0000 | 24 |
| llmjp | round_robin | balanced | 0.5000 | calib_static | -0.0761 | 0.0214 | 1.0000 | 24 |
| llmjp | round_robin | balanced | 0.5000 | causal_prev_step | -0.0276 | 0.0134 | 0.9583 | 24 |
| llmjp | round_robin | balanced | 0.5000 | oracle_same_step | 0.0232 | 0.0137 | 1.0000 | 24 |
| llmjp | round_robin | balanced | 0.7500 | calib_static | -0.0545 | 0.0173 | 1.0000 | 24 |
| llmjp | round_robin | balanced | 0.7500 | causal_prev_step | -0.0229 | 0.0094 | 1.0000 | 24 |
| llmjp | round_robin | balanced | 0.7500 | oracle_same_step | 0.0279 | 0.0066 | 1.0000 | 24 |
| llmjp | round_robin | hotspot | 0.2500 | calib_static | 0.0588 | 0.0042 | 1.0000 | 24 |
| llmjp | round_robin | hotspot | 0.2500 | causal_prev_step | 0.0576 | 0.0050 | 1.0000 | 24 |
| llmjp | round_robin | hotspot | 0.2500 | oracle_same_step | 0.0597 | 0.0035 | 1.0000 | 24 |
| llmjp | round_robin | hotspot | 0.5000 | calib_static | 0.1105 | 0.0068 | 1.0000 | 24 |
| llmjp | round_robin | hotspot | 0.5000 | causal_prev_step | 0.1094 | 0.0074 | 1.0000 | 24 |
| llmjp | round_robin | hotspot | 0.5000 | oracle_same_step | 0.1115 | 0.0063 | 1.0000 | 24 |
| llmjp | round_robin | hotspot | 0.7500 | calib_static | 0.0602 | 0.0019 | 1.0000 | 24 |
| llmjp | round_robin | hotspot | 0.7500 | causal_prev_step | 0.0600 | 0.0021 | 1.0000 | 24 |
| llmjp | round_robin | hotspot | 0.7500 | oracle_same_step | 0.0609 | 0.0015 | 1.0000 | 24 |
| olmoe | contiguous | balanced | 0.2500 | calib_static | -0.0166 | 0.0142 | 1.0000 | 24 |
| olmoe | contiguous | balanced | 0.2500 | causal_prev_step | -0.0150 | 0.0177 | 0.9583 | 24 |
| olmoe | contiguous | balanced | 0.2500 | oracle_same_step | 0.0529 | 0.0156 | 1.0000 | 24 |
| olmoe | contiguous | balanced | 0.5000 | calib_static | -0.0477 | 0.0235 | 1.0000 | 24 |
| olmoe | contiguous | balanced | 0.5000 | causal_prev_step | -0.0186 | 0.0209 | 0.9583 | 24 |
| olmoe | contiguous | balanced | 0.5000 | oracle_same_step | 0.0610 | 0.0118 | 1.0000 | 24 |
| olmoe | contiguous | balanced | 0.7500 | calib_static | -0.0223 | 0.0189 | 1.0000 | 24 |
| olmoe | contiguous | balanced | 0.7500 | causal_prev_step | -0.0110 | 0.0147 | 1.0000 | 24 |
| olmoe | contiguous | balanced | 0.7500 | oracle_same_step | 0.0483 | 0.0048 | 1.0000 | 24 |
| olmoe | contiguous | hotspot | 0.2500 | calib_static | 0.0570 | 0.0035 | 1.0000 | 24 |
| olmoe | contiguous | hotspot | 0.2500 | causal_prev_step | 0.0556 | 0.0047 | 1.0000 | 24 |
| olmoe | contiguous | hotspot | 0.2500 | oracle_same_step | 0.0578 | 0.0027 | 1.0000 | 24 |
| olmoe | contiguous | hotspot | 0.5000 | calib_static | 0.1069 | 0.0065 | 1.0000 | 24 |
| olmoe | contiguous | hotspot | 0.5000 | causal_prev_step | 0.1054 | 0.0073 | 1.0000 | 24 |
| olmoe | contiguous | hotspot | 0.5000 | oracle_same_step | 0.1079 | 0.0057 | 1.0000 | 24 |
| olmoe | contiguous | hotspot | 0.7500 | calib_static | 0.0594 | 0.0017 | 1.0000 | 24 |
| olmoe | contiguous | hotspot | 0.7500 | causal_prev_step | 0.0588 | 0.0021 | 1.0000 | 24 |
| olmoe | contiguous | hotspot | 0.7500 | oracle_same_step | 0.0599 | 0.0015 | 1.0000 | 24 |
| olmoe | round_robin | balanced | 0.2500 | calib_static | -0.0313 | 0.0101 | 1.0000 | 24 |
| olmoe | round_robin | balanced | 0.2500 | causal_prev_step | -0.0171 | 0.0137 | 0.9583 | 24 |
| olmoe | round_robin | balanced | 0.2500 | oracle_same_step | 0.0445 | 0.0151 | 1.0000 | 24 |
| olmoe | round_robin | balanced | 0.5000 | calib_static | -0.0675 | 0.0188 | 1.0000 | 24 |
| olmoe | round_robin | balanced | 0.5000 | causal_prev_step | -0.0219 | 0.0155 | 1.0000 | 24 |
| olmoe | round_robin | balanced | 0.5000 | oracle_same_step | 0.0551 | 0.0108 | 1.0000 | 24 |
| olmoe | round_robin | balanced | 0.7500 | calib_static | -0.0420 | 0.0196 | 1.0000 | 24 |
| olmoe | round_robin | balanced | 0.7500 | causal_prev_step | -0.0135 | 0.0113 | 0.9583 | 24 |
| olmoe | round_robin | balanced | 0.7500 | oracle_same_step | 0.0461 | 0.0049 | 1.0000 | 24 |
| olmoe | round_robin | hotspot | 0.2500 | calib_static | 0.0582 | 0.0041 | 1.0000 | 24 |
| olmoe | round_robin | hotspot | 0.2500 | causal_prev_step | 0.0568 | 0.0049 | 1.0000 | 24 |
| olmoe | round_robin | hotspot | 0.2500 | oracle_same_step | 0.0595 | 0.0031 | 1.0000 | 24 |
| olmoe | round_robin | hotspot | 0.5000 | calib_static | 0.1082 | 0.0065 | 1.0000 | 24 |
| olmoe | round_robin | hotspot | 0.5000 | causal_prev_step | 0.1066 | 0.0072 | 1.0000 | 24 |
| olmoe | round_robin | hotspot | 0.5000 | oracle_same_step | 0.1095 | 0.0060 | 1.0000 | 24 |
| olmoe | round_robin | hotspot | 0.7500 | calib_static | 0.0594 | 0.0020 | 1.0000 | 24 |
| olmoe | round_robin | hotspot | 0.7500 | causal_prev_step | 0.0589 | 0.0022 | 1.0000 | 24 |
| olmoe | round_robin | hotspot | 0.7500 | oracle_same_step | 0.0603 | 0.0017 | 1.0000 | 24 |

## Staleness cost: oracle_same_step vs causal_prev_step vs calib_static

| model | placement | origin_mode | budget_fraction | calib_static | causal_prev_step | oracle_same_step | staleness_cost_oracle_minus_causal | causal_advantage_over_calib_static |
|---|---|---|---|---|---|---|---|---|
| llmjp | contiguous | balanced | 0.2500 | -0.0317 | -0.0175 | 0.0242 | 0.0417 | 0.0141 |
| llmjp | contiguous | balanced | 0.5000 | -0.0664 | -0.0257 | 0.0241 | 0.0498 | 0.0407 |
| llmjp | contiguous | balanced | 0.7500 | -0.0453 | -0.0222 | 0.0278 | 0.0500 | 0.0231 |
| llmjp | contiguous | hotspot | 0.2500 | 0.0579 | 0.0566 | 0.0592 | 0.0027 | -0.0014 |
| llmjp | contiguous | hotspot | 0.5000 | 0.1089 | 0.1073 | 0.1105 | 0.0032 | -0.0016 |
| llmjp | contiguous | hotspot | 0.7500 | 0.0592 | 0.0589 | 0.0605 | 0.0016 | -0.0003 |
| llmjp | round_robin | balanced | 0.2500 | -0.0376 | -0.0159 | 0.0247 | 0.0406 | 0.0217 |
| llmjp | round_robin | balanced | 0.5000 | -0.0761 | -0.0276 | 0.0232 | 0.0508 | 0.0484 |
| llmjp | round_robin | balanced | 0.7500 | -0.0545 | -0.0229 | 0.0279 | 0.0508 | 0.0316 |
| llmjp | round_robin | hotspot | 0.2500 | 0.0588 | 0.0576 | 0.0597 | 0.0021 | -0.0013 |
| llmjp | round_robin | hotspot | 0.5000 | 0.1105 | 0.1094 | 0.1115 | 0.0021 | -0.0011 |
| llmjp | round_robin | hotspot | 0.7500 | 0.0602 | 0.0600 | 0.0609 | 0.0009 | -0.0001 |
| olmoe | contiguous | balanced | 0.2500 | -0.0166 | -0.0150 | 0.0529 | 0.0679 | 0.0016 |
| olmoe | contiguous | balanced | 0.5000 | -0.0477 | -0.0186 | 0.0610 | 0.0795 | 0.0291 |
| olmoe | contiguous | balanced | 0.7500 | -0.0223 | -0.0110 | 0.0483 | 0.0593 | 0.0113 |
| olmoe | contiguous | hotspot | 0.2500 | 0.0570 | 0.0556 | 0.0578 | 0.0022 | -0.0014 |
| olmoe | contiguous | hotspot | 0.5000 | 0.1069 | 0.1054 | 0.1079 | 0.0025 | -0.0015 |
| olmoe | contiguous | hotspot | 0.7500 | 0.0594 | 0.0588 | 0.0599 | 0.0012 | -0.0007 |
| olmoe | round_robin | balanced | 0.2500 | -0.0313 | -0.0171 | 0.0445 | 0.0617 | 0.0142 |
| olmoe | round_robin | balanced | 0.5000 | -0.0675 | -0.0219 | 0.0551 | 0.0769 | 0.0456 |
| olmoe | round_robin | balanced | 0.7500 | -0.0420 | -0.0135 | 0.0461 | 0.0596 | 0.0285 |
| olmoe | round_robin | hotspot | 0.2500 | 0.0582 | 0.0568 | 0.0595 | 0.0027 | -0.0014 |
| olmoe | round_robin | hotspot | 0.5000 | 0.1082 | 0.1066 | 0.1095 | 0.0029 | -0.0016 |
| olmoe | round_robin | hotspot | 0.7500 | 0.0594 | 0.0589 | 0.0603 | 0.0014 | -0.0005 |
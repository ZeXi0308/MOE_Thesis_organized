# Serving Simulation Report

model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
bandwidth_gbps: `100.0`
per_layer_overhead_us: `0.0`

## Summary

- Full BF16 total expert-output bytes: `3529244672`
- Full BF16 simulated bottleneck latency: `77497.631 us`
- Best single-rank INT4 latency strategy in this simulation: `rank1_int4`
- Its byte saving: `0.0469`
- Its bottleneck-byte saving: `0.0515`

## Accuracy-traffic comparison

- `rank8_int4`: KL `0.3507`, simulated byte saving `0.0469`
- `rank1_int4`: KL `20.4631`, simulated byte saving `0.0469`

See `serving_simulation.csv` and generated figures for the full table.

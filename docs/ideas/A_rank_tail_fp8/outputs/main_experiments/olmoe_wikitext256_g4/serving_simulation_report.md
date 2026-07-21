# Serving Simulation Report

model: `allenai/OLMoE-1B-7B-0924`
bandwidth_gbps: `100.0`
per_layer_overhead_us: `0.0`

## Summary

- Full BF16 total expert-output bytes: `12706119680`
- Full BF16 simulated bottleneck latency: `300749.292 us`
- Best single-rank INT4 latency strategy in this simulation: `rank4_int4`
- Its byte saving: `0.0938`
- Its bottleneck-byte saving: `0.0980`

## Accuracy-traffic comparison

- `rank8_int4`: KL `0.3614`, simulated byte saving `0.0938`
- `rank1_int4`: KL `20.9892`, simulated byte saving `0.0938`

See `serving_simulation.csv` and generated figures for the full table.

# Receiver-Aware v3: Adaptive Regime-Detection Controller (llmjp)

regime-detector calibration: hotspot_autocorr=0.2903, balanced_autocorr=0.1385, threshold=0.2144

## Detection accuracy (per true origin_mode, causally detected from first 30% of steps only)

- balanced: correct-regime detection rate = 1.0000
- hotspot: correct-regime detection rate = 0.9167

## Pooled mean saving across BOTH regimes (50/50 mix, unknown to controller in advance)

- adaptive_saving: 0.1729
- always_calib_static_saving: 0.1399
- always_causal_saving: 0.1594
- always_random_saving_mean: 0.1225

## Per-regime breakdown

| origin_mode | adaptive_saving | always_calib_static_saving | always_causal_saving | always_random_saving_mean |
|---|---|---|---|---|
| balanced | 0.1204 | 0.0464 | 0.0865 | 0.1205 |
| hotspot | 0.2254 | 0.2334 | 0.2324 | 0.1246 |
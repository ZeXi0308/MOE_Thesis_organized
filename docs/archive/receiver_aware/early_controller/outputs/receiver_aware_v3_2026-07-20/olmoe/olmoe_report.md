# Receiver-Aware v3: Adaptive Regime-Detection Controller (olmoe)

regime-detector calibration: hotspot_autocorr=0.2965, balanced_autocorr=0.1417, threshold=0.2191

## Detection accuracy (per true origin_mode, causally detected from first 30% of steps only)

- balanced: correct-regime detection rate = 1.0000
- hotspot: correct-regime detection rate = 0.8750

## Pooled mean saving across BOTH regimes (50/50 mix, unknown to controller in advance)

- adaptive_saving: 0.1706
- always_calib_static_saving: 0.1521
- always_causal_saving: 0.1643
- always_random_saving_mean: 0.1232

## Per-regime breakdown

| origin_mode | adaptive_saving | always_calib_static_saving | always_causal_saving | always_random_saving_mean |
|---|---|---|---|---|
| balanced | 0.1225 | 0.0733 | 0.0993 | 0.1224 |
| hotspot | 0.2187 | 0.2309 | 0.2293 | 0.1239 |
# Receiver-Aware Direct-Benefit Controller (homogeneous one-shot lane + receiver/sender credit)

## olmoe
- fitted params: {'alpha': 0.6, 'dwell_min': 1, 'high_quantile': 0.6, 'gap_ratio': 0.5, 'threshold_high': 1110.0, 'threshold_low': 555.0, 'mean_saving': 0.24900189088948252, 'lcb_saving': 0.14673678709726076, 'mean_saving_over_causal': 0.00706779004375907}
- codec lookup: {'matched_hidden': 2048, 'matched_rows': 32, 'sender_pack_us': 26.12175941467285, 'receiver_unpack_us': 25.37951946258545, 'requested_hidden': 2048, 'requested_rows': 32}
- mean saving: controller=0.2706, causal_no_hysteresis=0.2669, calib_static=0.0991, uniform_low=0.2962
- controller - causal_no_hysteresis: mean=0.0038, 95% CI=[-0.0005, 0.0085]
- GO/NO-GO (CI excludes 0 and controller beats hysteresis-free causal baseline): NO-GO

## llmjp
- fitted params: {'alpha': 0.6, 'dwell_min': 1, 'high_quantile': 0.6, 'gap_ratio': 0.5, 'threshold_high': 2178.6, 'threshold_low': 1089.3, 'mean_saving': 0.17029535304400026, 'lcb_saving': 0.03570625049154491, 'mean_saving_over_causal': 0.004667403769955137}
- codec lookup: {'matched_hidden': 512, 'matched_rows': 32, 'sender_pack_us': 17.097280025482178, 'receiver_unpack_us': 16.322879791259766, 'requested_hidden': 512, 'requested_rows': 32}
- mean saving: controller=0.2054, causal_no_hysteresis=0.1964, calib_static=0.0097, uniform_low=0.2230
- controller - causal_no_hysteresis: mean=0.0091, 95% CI=[0.0023, 0.0162]
- GO/NO-GO (CI excludes 0 and controller beats hysteresis-free causal baseline): GO

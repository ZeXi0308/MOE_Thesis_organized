# Additive-KL Modeling Audit: Why Does It Fail, And Is It Fixable?

model=allenai/OLMoE-1B-7B-0924, target_layers=[0, 3, 6, 9, 12, 15], tail=mxfp4, mean_drift_fraction(calibration)=0.6319

Interpretation: point_ratio_pred_over_true near 1.0 with a narrow CI around 1.0 means the predictor is accurate; ratio far from 1.0 (or CI excluding 1.0) means the additive assumption at that stage fails.

| predictor | true_col | point_ratio_pred_over_true | ratio_ci_low | ratio_ci_high | mean_relative_error | median_relative_error |
|---|---|---|---|---|---|---|
| naive_additive | true_joint_kl_free | 4.6347 | 4.4820 | 4.7954 | 3.6564 | 3.6755 |
| locked_additive | true_joint_kl_locked | 3.7719 | 3.6690 | 3.8691 | 2.7625 | 2.7137 |
| locked_additive_plus_global_drift | true_joint_kl_free | 2.5148 | 2.3686 | 2.6886 | 1.6020 | 1.6161 |
# Reproduction commands

Use a new output directory; raw artifacts remain read-only. Run from repository root. The plot interpreter is the already-installed brew Python; system Python lacks matplotlib.

```bash
MPLCONFIGDIR=/private/tmp/moe-window-mplcache /Users/leandrozhao/.brew/bin/python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_gap_service_tradeoff.py --workspace /Users/leandrozhao/Desktop/、++++++++ --output-dir /private/tmp/cohort3-gap-new --campaign 20260914_recovery_holdout_comparison_r01 --roles native most_output fit_scan guard_residual --label-prefix cohort3-
python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_effective_recovery_service.py --workspace /Users/leandrozhao/Desktop/、++++++++ --output-dir /private/tmp/cohort3-lifecycle-new --campaigns 20260914_recovery_holdout_comparison_r01 --expected-cells 8
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260914_service_window_model_r01/pre_resume_window_certificate_v2.py --results-root refine-logs/expert_saturation/outputs/admission_capacity/20260914_restore_token_reservation_r01/execution/readback/results --output-dir /private/tmp/pre-resume-new
```

The CLI now writes the native/most envelope alongside new eight-cell curves. The retained run computed the identical envelope in a separate call to `baseline_envelope`; the original curve analysis was never rewritten. Three boundary tests cover Q and the envelope; eight host-helper tests passed after integration. No new GPU was launched by these commands or by this branch.

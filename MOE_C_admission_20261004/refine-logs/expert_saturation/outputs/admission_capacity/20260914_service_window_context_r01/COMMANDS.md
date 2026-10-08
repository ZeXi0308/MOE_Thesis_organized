# Reproduction

Raw remains read-only in the original execution worktree. No GPU command was issued for this analysis.

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260914_service_window_context_r01/analyze.py --source /private/tmp/moe-a-recovery-components-20260914/refine-logs/expert_saturation/outputs/admission_capacity/20260914_context_victim_calibration_r01 --output-dir NEW_OUTPUT_DIRECTORY
MPLCONFIGDIR=/private/tmp/moe-window-mplcache python3 refine-logs/expert_saturation/outputs/admission_capacity/20260914_service_window_context_r01/plot.py
```

Lifecycle uses the existing main analyze_effective_recovery_service.analyze for all six COMPLETE raw files, with per-file hashes retained in lifecycle.json. No new tests or policy code were added.

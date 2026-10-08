# Reproduction and artifact roles

Run from the isolated worktree with Python3.14 (`/Users/leandrozhao/.brew/bin/python3 -B`). Replace output targets by new paths; scripts reject overwriting earlier results.

```sh
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/prepare_context_victim_calibration.py --source-bundle /Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_holdout_comparison_r01 --output-dir /private/tmp/NEW-context-bundle
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_context_victim_calibration.py --bundle refine-logs/expert_saturation/outputs/admission_capacity/20260914_context_victim_calibration_r01 --analysis-library /Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/experiments/admission_capacity --output /private/tmp/NEW-context-analysis.json
```

The measured package and commands are `preparation/pkg/run.sh` and each readback cell's `commands.txt`. The producer SHA and full input/package hashes are in `preparation/preparation.json`. The original driver, transport override and failed recovery helper remain beside this file; they are execution history, not a request to restart an existing group. `execution/execution.json` retains the SSH interruption; `execution/recovery.json` is the complete group/readback receipt. The final archive is `execution/readback-fast-r01.tar.gz`, SHA7e83a4483e37f65c4ad83ecebc3f522eefbd7263687ac4e10de86be92b1e9f5d. Original remote directory: `/root/autodl-tmp/moe-context-victim-calibration-20260914-r01`.

`analysis_r01.json` preserves an analyzer-only missing-native-field failure; `analysis_r02.json` contains all six valid cells. CPU qualification JSONs distinguish old-trajectory replays, input tests, no-action exposure checks and later current-state funding-filter checks from actual new GPU execution. No raw was changed by those checks.

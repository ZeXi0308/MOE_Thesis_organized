# Prepared continuation

The group is STAGED on the existing authorized westc:53036 host; GPU execution is UNRUN. Read the latest shared GPU_COORDINATION.md. Before launch, verify the earlier Qwen r03 full terminal state and explicit release, its controller/worker exit, empty GPU processes, and availability of `/root/autodl-tmp/moe-research-gpu.lock`. Initialization gaps are not a release. The package run.sh also holds the common lock across the entire group. Do not restart a directory with launch-once/results; inspect and retain its existing terminal state.

From `/private/tmp/moe-a-recovery-components-20260914`, using a live authorized SSH ControlPath in execute.py:

```sh
/Users/leandrozhao/.brew/bin/python3 -B refine-logs/expert_saturation/outputs/admission_capacity/20260914_funding_filter_comparison_r01/execute.py run
```

If SSH drops, preserve UNKNOWN_REMOTE and inspect the existing group before any further action. Read back the original group, including all failures, rather than launch a replacement. Current temporary socket may expire during the Qwen group; reconnect using the already authorized credentials without persisting them.

After all original artifacts are read back, run the prepared analyzer to a NEW output path:

```sh
/Users/leandrozhao/.brew/bin/python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_funding_filter_comparison.py --bundle refine-logs/expert_saturation/outputs/admission_capacity/20260914_funding_filter_comparison_r01 --context-analyzer refine-logs/expert_saturation/experiments/admission_capacity/analyze_context_victim_calibration.py --analysis-library '/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/experiments/admission_capacity' --output /private/tmp/NEW-funding-analysis.json
```

CPU qualification uses the same analyzer with `--legacy-bundle refine-logs/expert_saturation/outputs/admission_capacity/20260914_context_victim_calibration_r01 --cpu-checks-output /private/tmp/NEW-funding-cpu.json` instead of `--output`. The original selector test and native allocator commands were `python3 -B -m unittest -q test_absence_rotation.py` and `python3 -B verify_rotation_native.py` from the experiments/admission_capacity directory; 27 selector tests and the native fixtures passed.

Exact prepared producer: preparation/producer.py, matching the metadata SHA. Current prepare_funding_filter_comparison.py accepts `--implementation-dir` pointing to this frozen preparation/pkg; this reproduces all20 package file hashes while preserving the source's later comment-only EOS clarification. New tar timestamps need not reproduce the original archive bytes; the original archive and its hash are retained.


## Completed on the replacement GPU

The historical launch command above must not be run again: the original remote directory has launch-once and six COMPLETE results. `execution/recovery.json` and `execution/gpu_release_r01.json` record the recovered original group and release. The original frozen package is unchanged; `recover_readback.py` only downloaded that group using the restored authorized SSH connection. Old GPU timing is not paired with this group. Current full metrics are `analysis_r02.json`, structural checks are `model_check_r01.json`.

To reproduce structural validation into a new output (from the isolated worktree):

```sh
/Users/leandrozhao/.brew/bin/python3 -B refine-logs/expert_saturation/experiments/admission_capacity/validate_funding_progress_model.py --output /private/tmp/NEW-funding-model.json
```

The full-metrics analyzer command above remains valid with a NEW output path; do not overwrite analysis_r01/r02 or raw.


Single-action model reproduction (new output only):

```sh
/Users/leandrozhao/.brew/bin/python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_funding_action_branches.py --output /private/tmp/NEW-funding-branches.json
```

This is CPU structural branching; no GPU or new live policy is invoked.

To reproduce only the selected branch localization without writing another full candidate table:

```sh
/Users/leandrozhao/.brew/bin/python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_funding_action_branches.py --localization-output /private/tmp/NEW-funding-localization.json
```

The producer binds the existing r02 candidate table, qualified model/selector, and raw hashes. The selected candidate and its after-inspection choice rule remain exploratory.

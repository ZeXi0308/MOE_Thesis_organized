# Streaming EOS analyzer: CPU prepared

`analyze_streaming_recovery.py` supports the frozen four-cell native / most-output diagnostic. This bundle contains analyzer preparation and synthetic checks, no new GPU measurements. It reads all 64 source identities and prompt hashes, every output event, terminal receipt, actual native preemption receipt, and open-adapter decision. Raw inputs stay read-only.

Terminal completion is the engine return carrying the finished event. It may occur after the last new token or with zero tokens. Completion latency uses all 64 requests. TTFT requires at least one new output; adjacent engine-return gap requires at least two. Their defined-request counts are reported. Engine return is not client receipt. Full wall is the measured request episode, including idle and host time; initialization and warmup precede that episode.

Actual finish reasons (`stop` / `length`), native stop reasons, output volumes and per-request output hashes are retained. A `stop` without an explicit native token reason is not silently called token-confirmed EOS. A different output trajectory remains an outcome, so paired time differences do not establish equal-work speedup or output quality.

Existing `analyze_effective_recovery_service.analyze` supplies recovery intervals and exclusive recomputation accounting. The wrapper classifies re-preempted and terminal intervals into zero, 1–2 and 3+ new-output groups, adds terminal times, and corrects completed zero-output EOS from `censored_no_output` to `completed_without_new_output`. Inclusive shared recovery call durations remain nonadditive diagnostics, not pure recomputation tax. Native recompute is required; cache loading needs another lifecycle adapter.

One prerequisite fixes the reused helper's empty-residency case: initialize `totals = Counter(recompute_positions=0)` in `summarize`. This allows a valid zero-preemption run to return zero recomputation. The owned helper differs from the root helper only in that initialization at preparation time.

Checks completed:

- Three synthetic tests pass: zero/one-output termination with distinct final receipt; terminal recovery with no output; rejection of silent adapter bypass and inconsistent terminal times. The first case also exercises zero preemptions.
- The reused helper successfully analyzes existing `cohort3-block0-native/raw.json`: 32 complete requests, 6 preemptions, 21426 recomputed positions. This is compatibility verification of the existing fixed-output run, not an EOS experiment.
- The CLI with an intentionally absent result directory retains four `UNRUN` cells and overall `INCOMPLETE` in `unrun_cli/analysis.json`. Exceptions after partial capture also preserve runtime status, error, partial request/output counts and the raw file in its original location. No paired comparison is emitted until all four cells pass.

Run from a worktree root after the actual result readback exists:

```sh
python3 -m unittest discover -s refine-logs/expert_saturation/experiments/admission_capacity -p test_analyze_streaming_recovery.py -v
python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_streaming_recovery.py \
  --results-root PATH_TO_READBACK_RESULTS \
  --inputs refine-logs/expert_saturation/outputs/admission_capacity/20260914_streaming_inputs_r01/frozen \
  --output-dir PATH_TO_NEW_ANALYSIS_DIRECTORY
```

The output is descriptive. It contains no selected SLO threshold, Q metric, method GO, controller, policy simulation, or next GPU action.

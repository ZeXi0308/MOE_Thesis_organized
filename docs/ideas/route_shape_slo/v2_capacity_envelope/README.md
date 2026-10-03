# Route-Conditioned Capacity Envelope v2

This directory is a lightweight continuation of RouteShape-SLO. It is an
exploratory measurement path, not a new authority entry and not a change to
`docs/current/README.md`.

## Frozen question

After ordinary custom-runtime state and the **current per-expert load summary**
are controlled, does historical route shape improve next-window tail-latency
prediction enough to justify a capacity action?

The current producer performs pure one-token decode and schedules
`active[:max_batch_size]`. Therefore the only eligible Stage-D candidate is:

```text
running_set_budget
```

It is not authorized or frozen before a capacity-eligible P1 result that
survives every veto. Within one model call it equals the scheduled decode-token
count. The current
`max_batch_size` is only a static scheduled-batch-width cap: it does not prevent
all arrived requests from being prefetched into the custom active set and is not
yet a dynamic admission actuator.

## Files

- `LIGHTWEIGHT_STATUS.md`: current facts, measured boundary, verdict, and the
  single next experiment.
- `EXPERIMENT_AUDIT.md` / `EXPERIMENT_AUDIT.json`: provisional fresh-agent
  integrity audit, its claim ceiling, and the recorded post-audit remediation.
- `experiments/olmoe_dev_workload.json`: deterministic two-episode workload
  recipe. It references the existing formal manifest instead of duplicating
  prompt payloads.
- `experiments/olmoe_dev_capture.json`: frozen pilot/action/metric contract.
- `experiments/prepare_dev_workloads.py`: materializes two producer-native
  development manifests in a new temporary directory.
- `experiments/capture_dev_continuous_decode.py`: development-only wrapper that
  preserves token parity and measures serial-vs-batched expert-assignment
  conformance. Any assignment mismatch is emitted as `BATCH_DEPENDENT`, never
  upgraded to a conformance pass; formal producer semantics are not modified.
- `experiments/check_telemetry_overhead.py`: paired native router-output OFF/ON
  timing plus logit/token/completion parity and ON-route stability check on
  eight requests. It holds fixed preselected scheduled batches but does not
  replay the manifest's arrival timing, so P0 remains proxy-runtime evidence.
- `experiments/compare_serial_batched_router_logits.py`: fail-closed next-step
  diagnostic that reconstructs the frozen request/token prefix and compares
  native pre-top-k logits at width one versus width four; it is not a capacity
  experiment.
- `experiments/qualify_dev_capture.py`: the lightweight causal/alignment/runtime
  qualification check.
- `experiments/build_capacity_windows.py`: converts successful producer output
  into explicitly named custom-runtime fields.
- `experiments/analyze_capacity_signal.py`: directional leave-one-episode-out
  M0--M4 comparison; M3 is compared with the required M2 per-expert-load
  baseline.
- `experiments/run_dev_capture.sh`: exact end-to-end RTX 5090 command. A
  canonical six-file bundle is published only after both captures and analysis
  succeed.

## Evidence rules

- `custom_waiting_count` means arrived/prefilled requests not selected for the
  current model call.
- Producer `pending_request_count` includes future, not-yet-arrived requests. It
  is excluded from the workload baseline and is **not** renamed as a queue.
- `custom_running_set` contains only the requests scheduled in the current
  model call; the broader KV-resident set is recorded separately.
- `tokens_completed` means decode tokens executed in that model call, not
  completed requests.
- M4 is a future-route latency diagnostic, never a capacity Oracle.
- Batch-dependent route assignment forces
  `PIVOT_TO_EXECUTION_CONFORMANCE`, `capacity_claim_authorized=false`, and
  `action_oracle_authorized=false`, regardless of the diagnostic M3 score.
- A development result is custom-runtime/offline evidence and cannot establish
  vLLM, EP, A2A, receiver, NCCL/RDMA, or production SLO behavior.
- The primary M0--M4 comparison uses one frozen L2-regularized P95 linear
  quantile model solved with deterministic NumPy ADMM; ridge and one depth-4
  tree are auxiliary checks. Tiny unregularized regression fixtures retain an
  exact enumerated solver for regression testing only.

## Current retained result

The sole retained local canonical bundle is
`artifacts/route_capacity_envelope/dev/20260812T170512Z/`. Its final verdict is
`PIVOT_TO_EXECUTION_CONFORMANCE`. The raw M3-over-M2 diagnostic was nominally
`PROMISING_SINGLE_MODEL` (`+9.6288%` P95 pinball improvement; fold directions
`+1.7428%` and `+27.7089%`), but serial-vs-batched expert assignment was
batch-dependent despite exact token parity. The minimum audited layer-assignment
match was `94.7266%`; therefore `capacity_claim_authorized=false` and
`action_oracle_authorized=false`.

The logs do not prove that pilots `170328Z` and `170512Z` were isolated repeats.
The older pre-veto bundle was rejected and removed from the canonical path; it
was copied at audit time only to ephemeral local quarantine because its stored
capacity verdict omitted the conformance veto. `170512Z` is retained because it
applies that veto, not because its diagnostic metric is favorable. The
uncontrolled repeat conditions mean neither run is isolated performance
evidence.

Current execution state and the one next experiment are documented in
[LIGHTWEIGHT_STATUS.md](LIGHTWEIGHT_STATUS.md).

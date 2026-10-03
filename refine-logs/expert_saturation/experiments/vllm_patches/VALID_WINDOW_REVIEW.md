# vLLM 0.26 valid-route-window source-safety review

## Verdict

`SOURCE_SAFE_WITHIN_EXISTING_CAPTURE_SUPPORT_ENVELOPE`

Patch: `vllm-0.26-valid-route-window.patch`

Patch SHA-256:
`862b3ff7732fd4ccac4ffeba923174ab3d662e57834a981eb329aba893e0d87b`

The patch preserves the lossless routed-expert API and is the strongest simple
telemetry-overhead baseline. It changes no route computation, output shape,
slot mapping, scheduler storage, or synchronization point.

## Invariant

For each worker step, let `N = scheduler_output.total_num_scheduled_tokens`.
The only worker data that can reach the scheduler is:

```text
sync:  device_buffer[:N] -> pinned CPU -> RoutedExpertsLists
async: device_buffer[:N].clone() -> copy stream -> RoutedExpertsLists
```

Therefore the necessary and sufficient sentinel reset is
`device_buffer[:N].zero_()` before the forward. Rows `N:` are unobservable in
that step.

## Edge cases reviewed

- **Dense/non-MoE layers:** the entire visible `[N,L,K]` prefix is cleared;
  layers without a router callback remain zero exactly as before.
- **Shrinking window:** stale rows outside `[:N]` are not copied or stored.
- **Growing window:** the larger new prefix is cleared before capture, so old
  tail rows cannot become visible.
- **Zero-token step:** `[:0].zero_()` is a no-op and the runner exits before a
  route output is formed.
- **Sync:** the current D2H is covered by the existing sampled-token event
  synchronization before `ModelRunnerOutput` is returned.
- **Async:** the current output owns a private clone made on the default stream;
  the next default-stream clear cannot alter it while copy-stream D2H runs.
- **DP naive/modular:** the capturer still writes this rank's local route rows
  from offset zero and the runner still exports its local scheduled-token
  prefix.
- **SP:** the existing TP all-gather and trailing-pad trim still occur inside
  `capture()` before the local prefix write. The clear change does not alter
  collectives or ownership.

## Unchanged limitations

This patch does not qualify or repair unsupported combinations. vLLM already
rejects full-route return with PP, KV connectors, and scheduler context
parallelism; V2 routed-expert capture and DBO limitations are unchanged.

## Validation

- Patch applies cleanly to vLLM commit
  `568afb3a13806beb53bb2e6bd518269357b237c0` (`git apply --check`).
- `git diff --check` passes.
- `validate_valid_window_patch.py` carries the same expected patch SHA and
  returns `valid=true`, `source_state=patched` for the clean patched worktree.
- The patch artifact includes focused CPU regressions for default/full clear,
  dense-layer sentinels, shrink-to-grow visibility, async snapshot isolation,
  and invalid bounds.
- The modified vLLM files and test module pass `py_compile`. The focused vLLM
  pytest cases were not executed on the local Mac because that interpreter has
  no PyTorch; they still require the remote vLLM environment.
- Native token parity and the `<=5%` P95 TPOT/wall overhead Gate remain unrun
  for this optimization at the time of this review.
- The implementation Gate now interprets the 5% bound as a two-sided absolute
  timing-deviation guard and requires exact validator-approved source hashes,
  embedded producer bytes, full route parity, and two retained process repeats.

## Evidence boundary

The older four-pair smoke had token parity but 10--13% overhead. Two 36-pair
full runs had route-ON/OFF token mismatches (1 and 6 cells), so their timing is
invalid as an overhead claim. The optimized valid-window arm must re-establish
token parity before timing is interpreted.

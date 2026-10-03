# N0c Capture-Source Triage Protocol

Date: 2026-08-23  
State: `GPU_COMPLETE / NOT_REPRODUCED`  
Claim ceiling: `FRESH_PROCESS_ASSOCIATIONAL_CAPTURE_TRIAGE_ONLY`

## Repository contract

- HEAD: `b141c1d587fe2c918643c3c7c3a8f5f5157d4c8a`.
- Dirty state: the complete `refine-logs/expert_saturation/` line is
  uncommitted; unrelated `.aris` state is preserved and excluded from this
  campaign.
- Authority read: `docs/current/README.md`, `docs/ideas/README.md`,
  `EXPERIMENT_TRACKER.md`, `N0_HARNESS_QUALIFICATION_REPORT.md`, the sealed N0b
  terminal/result, and both append-only N0b correction addenda.
- No automatic edit to `docs/current/README.md`, commit, or push is authorized.

## Frozen facts inherited from N0b

N0b is `VALID_WINDOW_NOT_TRANSPARENT / TELEMETRY_TOKEN_DRIFT` in the tested
single-GPU vLLM 0.26 eager regime. Route-OFF was token-stable, while the two
route-ON implementations each had one retained repeat with a unique drift
cell. N0b did not establish causality because every arm was a fresh process.
Its saved route also omitted the prompt-tail forward, so saved route row `s`
produced output token `s+1`; the earlier localization is only partial.

N0c selects exactly the two preregistered N0b cells and replays their complete
semantic prefixes:

| Target | Runtime | Terminal shape | Source order | Target input SHA-256 |
|---|---|---:|---:|---|
| `stock_p512_b8_g2_w0` | stock | P512, B8, G2, W0 | 11 of 12 | `37b9d87d894f4b314f97c3b6e1fda893c15ab3ee40a9b7870845ebb9de1ed491` |
| `valid_window_p512_b16_g1_w0` | valid-window | P512, B16, G1, W0 | 35 of 36 | `8ddb890932ca0b4f4eb712d7df9ecebc62283b732f19ed71abd4161f3022d1d2` |

The model, revision, BF16 dtype, eager mode, seed, six warm-up shapes, 16-token
decode, prompts, prefix order, arm order, and four process rounds are constants,
not CLI choices.

## One research question

Does the retained N0b token drift already appear when routed-expert capture is
enabled without D2H/export, or only on the full route-export path?

The weakest causal link is the source of perturbation inside the telemetry
stack. N0c does not yet instrument hidden states, router logits, top-k margins,
expert outputs, or final logits.

## One experiment

For each target and round, execute four fresh-process arms in a frozen Latin
order:

1. `n_a`: sealed base runtime, route capture OFF;
2. `capture_only`: capture-enabled experiment runtime with route D2H, output
   wrapping, and async snapshot/export suppressed;
3. `full_export`: sealed base runtime with complete host-visible route export;
4. `n_b`: a second sealed base-runtime OFF control.

`capture_only` is not a pure device-kernel treatment. It retains route-capturer
initialization, buffer clear/capture, slot mapping, and scheduler
`RoutedExpertsManager` bookkeeping. A positive result is therefore named
`CAPTURE_NO_EXPORT_ASSOCIATION`, never `DEVICE_CAPTURE_CAUSALITY`.

All four campaign runtime variants are package-manifested and import-probed
before the first GPU arm. Every arm independently verifies that its actual
`vllm.__file__` belongs to the expected campaign runtime before constructing
the engine. The campaign freezes and hashes the orchestrator, common helper,
arm runner, evaluator, patch, workload, N0b runtime manifest, target specs, and
every prompt NPZ.

Full export retains route rows for the prompt tail and all 15 subsequent decode
inputs, with the explicit mapping:

```text
route row 0: input P_tail -> output y0
route row s: input D_(s-1) -> output y_s, for s=1..15
```

## Locked decision rule

All eight OFF outputs for a target must be exactly identical. Any mismatch has
highest scientific precedence and yields
`BASELINE_DISCRETE_NONDETERMINISM`.

For each round, define the first output-token divergence signature relative to
that stable OFF baseline as:

```text
(output_token_index, request_row, baseline_token, observed_token)
```

- `CAPTURE_NO_EXPORT_ASSOCIATION`: capture-only and full-export have the same
  nonempty first-divergence signature in at least 3/4 rounds.
- `EXPORT_PATH_ASSOCIATION`: capture-only equals OFF in 4/4 rounds and the same
  full-export-only signature occurs in at least 3/4 rounds.
- `INTERMITTENT_OR_UNRESOLVED`: capture-only alone drifts, the two ON arms have
  discordant signatures, signatures do not repeat, or support is only 1--2
  rounds.
- `NOT_REPRODUCED`: all three paths are token-identical in every round.
- `INVALID_CAMPAIGN`: any schedule, seal, source, prompt, package, import-root,
  route-shape, expert-range, or artifact identity check fails.

Routes are retained as diagnostics but never replace token evidence. Timing is
recorded only as execution metadata and is excluded from every N0c verdict.

## Stop, continue, and reopen conditions

- On `BASELINE_DISCRETE_NONDETERMINISM`, stop telemetry attribution and use a
  batch-invariant/deterministic negative-control Gate.
- On `CAPTURE_NO_EXPORT_ASSOCIATION`, the next and only next experiment is a
  minimal first-divergence observer spanning scheduler bookkeeping, capture
  setup, pre-router hidden state, router logits/top-k, expert output, and final
  logits. No Controller is allowed.
- On `EXPORT_PATH_ASSOCIATION`, inspect the D2H/synchronization/output
  reconstruction boundary; do not instrument upstream model state first.
- On `INTERMITTENT_OR_UNRESOLVED`, retain every round and stop. Do not add seeds,
  cells, or retries after seeing the result.
- On `NOT_REPRODUCED`, close the two historical N0b cells as non-reproduced under
  this fresh-process replay. Reopen only with a new deterministic runtime mode
  or a separately preregistered same-prestate mechanism.

No N0c outcome unlocks a decode-cap action, pressure-to-latency claim,
request-level benefit, admission policy, or Controller.

## Frozen implementation identity

| File | SHA-256 |
|---|---|
| `run_n0c_capture_stage.py` | `f8c7e3fa1dee6a99d03e735f466a8347997654e3e36c7af84a5b0f4be6bb38ab` |
| `run_n0c_capture_stage_arm.py` | `c49a9b7db3a57a5d1748a7db2cc67b9cc2c3d85e7c9b2ad230d6f40b12151c40` |
| `evaluate_n0c_capture_stage.py` | `6f5124949c3bf5050a99b88ffd5437c056ebd8ecff198d29695b7cc272b9d070` |
| `vllm-0.26-device-capture-only.patch` | `74921baa2dc40432ab0d09f4055972c2c107cfb1ba7420ba8e8b16667ac0dd94` |
| `run_valid_window_telemetry_gate.py` | `c3e1b9f12f5d49bbafe1f62610f0cd01ce4af3da2ed3354fce06ade48ddf51cf` |
| `validate_valid_window_patch.py` | `55e4a1f7d2d51054213e3af78f8b4a09fe5d58e84d30619af7a422e43532aca7` |

Local pre-run verification: `147/147 PASS` across the complete
`experiments/test_*.py` suite; the focused N0c suite is also `36/36 PASS` on
the repository Python 3.9 venv. Python compile, direct patch-apply checks, and
`git diff --check` pass. Independent adversarial review found `P0=0 / P1=0`
and reproduced every locked positive and negative classification case. This is
harness readiness only, not scientific evidence.

## Preserved preflight history

Remote source bundle `n0c-capture-stage-20260823-src-r01` passed the five
uploaded-file hash checks but failed before GPU execution because the upload
omitted the common helper's import-time dependency
`vllm_patches/validate_valid_window_patch.py`. No N0c output root was created.
The failed source directory is retained unchanged. The fix adds the validator
to the frozen manifest and a minimal isolated-upload import-closure regression;
independent delta review again found `P0=0 / P1=0`. Execution must use a new
`src-r02` directory.

The `src-r02` campaign completed all 32 arms without retry. Its sealed verdict
is `NOT_REPRODUCED`; see `../../N0C_CAPTURE_SOURCE_TRIAGE_REPORT.md` for the
evidence boundary and next Gate.

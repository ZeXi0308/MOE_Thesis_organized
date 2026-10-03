# Private uninstrumented lazy-view native full-save contrast (GPU unrun)

This candidate copies `candidate_native_backfill_only_r01` byte-for-byte except for `pkg/staged_store_rotation.py`. The frozen 128-request input, warmups, model/runtime and native-source pins, runner/CLI, full-save connector calculation, policy gates, Q1 protection, and metric capture are unchanged. The native reference still installs no scheduler adapter.

Supported cells remain:

- `pkg/run.sh native_full_native performance off ABSOLUTE_OUTPUT_DIR`
- `pkg/run.sh native_full_ordinary_only performance ordinary ABSOLUTE_OUTPUT_DIR`

The ordinary arm now constructs all-running `RequestState` rows in `begin` only when forced selection, diagnostic logging, closed-cohort activation, or a pending commit needs them. Its configured open-population, non-diagnostic, no-forced path still checks pure decode, native reservation, connector jobs, pending pushes, slots, full-history capacity, and running block ownership at the actual backfill action boundary. Tracker updates, Q1 first-output protection, native admission receipts, and no-forced assertions are unchanged. On a step with no attempted action, malformed running ownership may be detected later than in the original per-step full scan; this is an error-timing boundary, not strict equivalence for abnormal states.

The same lazy gate passed the focused CPU closure in `../test_native_backfill_lazy_r01.py` and completed a separate timed 128-request paired diagnostic in `candidate_native_backfill_lazy_r01`. That timed diagnostic showed lower `begin` CPU use but was not an uninstrumented native-reference performance result. This new package has no GPU result yet and makes no performance claim.

`manifest.json` covers the exact 25 payload files; `verify_package.py` checks their hashes without importing vLLM or CUDA.

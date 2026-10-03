# Necessary preparation checks

PASS: unchanged64document identities, token sequences and order; frozen0.2s arrivals; EOSallowed/cap1024; actual requested4096usable+1null GPU block accounting and16GiBhost configuration. Python AST, shell syntax and CLI help passed.

Actual staged-adapter closures were exercised with CPU-only fake metadata: activation before a closed cohort fills, a late partial-prefill arrival, terminal protection release without a new output, and cancellation when a mixed running population appears after preparation. Allocation, native worker transfers and GPU execution are not simulated results. The initial fixture selected the final metadata event instead of the commit event; its assertion was corrected to select the named commit record, with no runtime-source change for that issue.

The existing three EOS/late-arrival/output-boundary tests were reused against this package's EOS capture and passed. The request timing, native waiting-loop patch, saving prefix/flush/load semantics, most-output selector and configuration defaults are inherited; no broad review or additional test matrix was added.

New or newly added executable lines: 300 (194 in the focused analyzer, checks and shell entry; 106 added to inherited runner/capture/adapter/controller). The change stays below500lines and introduces no separate controller policy or predictor. The existing fixed-cohort runner could not express EOS/open membership; the minimum change directly qualifies that boundary.

Reproduce from repository root:

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_saved_recovery_gate_r01/check_package.py
```

GPU UNRUN. Remaining gates are root's acceptance of this exact archive, previous whole-host group release, package/runtime hashes and occupancy, actual4096usable/4097total GPU KV blocks and8592031744tensor bytes, actual16GiB/8192host blocks, and the single native diagnostic. A complete no-action result is valid and does not authorize pressure tuning.

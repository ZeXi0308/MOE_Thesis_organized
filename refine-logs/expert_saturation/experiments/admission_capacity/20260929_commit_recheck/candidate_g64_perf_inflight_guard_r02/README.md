# G64 corrected LTR performance candidate

Status: **CONDITIONAL_UNRUN**. This is a new package identity. No GPU performance cell has run from this package.

This package copies `candidate_g64_perf_r01` byte for byte except `pkg/ltr_style_native.py`, which is the CPU tested native in-flight reservation guard candidate (SHA-256 `ccbddc6d9860a1efc2c78dcf6614883734b2388de19a3db8034632bd9b3e9fb3`). The original performance runners, G64 inputs, warmups, native full and eager arm code, selector, model/runtime pins, and `verify_package.py` retain their original bytes. The exact 28 executable/input files are pinned by this package's new manifest.

The guard checks `target_remaining_blocks + native_inflight_prefill_reserved_blocks <= raw_free_blocks` for direct `PRIORITIZE_WAITING` admission. It rechecks an active preempted target and records bounded aggregate guard counts plus one first refusal sample. This begin-step condition can still become stale during native scheduling; it is not a global liveness proof.

**Launch gate:** keep this package unrun until the separate `candidate_g64_inflight_guard_qual_r01` native qualification receipt proves all 64 requests completed, no no-progress observer snapshot or observer error, a guard rejection occurred, and its first refusal shows `F >= N`, `R > 0`, `N + R > F`. A qualifying diagnostic is source and lifecycle evidence only; it is not a performance result. If that receipt fails or is incomplete, preserve it and do not start this performance package.

Once qualified, an externally SHA-pinned serial plan may run paired G64 cells under the common nonblocking GPU lock. The unchanged `pkg/run.sh ARM ABSOLUTE_OUTPUT_DIR` supports `native_full_native`, `eager`, `ltr_t30_q1`, `ltr_t30_q10`, `ltr_t200_q1`, and `ltr_t200_q10`; each arm needs a new output path and one-shot identity. Compare only complete episodes with the same pinned environment and resources. Earlier r01/r02 runs remain separate identities and cannot be relabeled as results from this corrected package.

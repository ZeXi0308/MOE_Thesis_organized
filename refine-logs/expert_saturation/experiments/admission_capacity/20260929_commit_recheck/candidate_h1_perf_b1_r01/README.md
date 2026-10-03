# H1 performance block 1: frozen adjacent eager off/on pair

Status: `CPU_PREPARED / GPU_UNRUN`. This is an isolated package identity for block 1. All 25 manifest payload bytes exactly match `candidate_h1_guard_qual_r02`; manifest SHA-256 is `4231a687db066f113fcc14676f91e8e5825be05b76ef0834112e053726aa13aa`. The 128 H prompts, arrivals, warmups, runner, native interface and guard adapter are unchanged. The separate identity keeps the existing `launch-once-eager-performance-off` and `launch-once-eager-performance-on` markers local to this block.

Frozen order: `eager_performance_off` followed immediately by `eager_performance_on` under one shared-lock acquisition. Use `pkg/run.sh eager performance off|on ABSOLUTE_OUTPUT_DIR` with the pinned `H1_` environment. The block controller rehashes the complete model, requires the audited native direct-action qualification and paired protocol hashes, checks GPU emptiness before and after each cell, archives with readback hashes, and reserves enough wall time for the second cell. Each cell has a 900 s limit; this block has a 3200 s total limit. Block 2 is a separate invocation and does not depend on a favorable block 1.

The only on/off mechanism difference is `commit_recheck`; both arms use this same package. Natural EOS can change output sequences or lengths, so token-rate differences are not equal-work speedups. H128 was previously seen, and two ordered blocks cannot establish statistical stability or blind validation. The native reservation check is defensive correctness hardening, not a new algorithm contribution.

CPU verification: `python3 -B verify_package.py` and `python3 -B test_cpu_guard.py`. Neither runs GPU. No performance result exists until the authorized controller completes and a separate audit checks both arms.

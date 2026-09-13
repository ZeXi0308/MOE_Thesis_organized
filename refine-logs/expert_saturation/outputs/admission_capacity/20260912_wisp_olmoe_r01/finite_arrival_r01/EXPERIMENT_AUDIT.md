# Finite-arrival bounded integrity review

2026-09-14. Overall **WARN**, P0=0, P1=1. Reviewer: GPT-5.6-Sol ultra, `/root/fullstage_qualification_integrity/source_semantics`; same-family, reused reviewer, first review of this bundle, provisional. This records the reviewer findings; phase mapping is separately attributed below.

| Check | Result | Evidence |
|---|---|---|
| A Identity / provenance | PASS | analyze_finite_arrival.py:33–40,66–69; planned arrivals and raw request/prompt identities joined; no quality ground truth claim |
| B Accounting / denominator | PASS | analyze_finite_arrival.py:110–115,129–134; raw request seconds, capture and process wall, H2D and D2D remain separate |
| C Existence / coverage | PASS | attempt01/analysis.json:1–8; all eight cells, 31 frozen input hashes and 162 retrieved hashes verified |
| D Executed source | PASS with WARN | run_attempt.py:66–79,124–133 and analyze_finite_arrival.py:46–47 directly enforce external WiSP hash |
| E Scope | WARN | Same 16 documents, two fresh engines per mode, only 128→2 short warmup; material same-arm drift and one setup overlap |
| F Claim ceiling | PASS | DESCRIPTIVE_FINITE_COHORT; F/X timing signs flip, no stable gain / quality / SLO claim |

Reviewer independently counted 128 completed requests, 4096 outputs, 20352 scheduled positions, 6960 measurement and 7856 total layer calls, zero preemptions, 24 passing boundary snapshots. Every cell reports 384 expert slots, 4,831,838,208 expert bytes and 1GiB unique KV storage.

**P1: setup was not isolated in 1_selected.** Foreign PID13921's recorded lifetime lies inside that cell's process. Boundary checks alone cannot certify continuous isolation. Evidence: `../../20260914_rotation_runtime_repeat_r01/execution/gpu_results/cohort2-block0-native-execution.json`, and `attempt01/results/execution.json:163–170,261–264`. The separate read-only phase audit maps peer exit to 16.660s before warmup and 17.694s before capture: direct overlap is zero, but downstream effects are unverified. Preserve all eight cells, qualify full-process comparisons, and do not subtract the peer lifetime. Exact timestamps are in EXPERIMENT_AUDIT.json and REPORT.md.

Two automated assertions are weaker than the protocol: the analyzer checks warmup counts but not the exact warmup document/prompt/cache handoff, and checks the reported shared byte total rather than independently summing distinct pointers. Reviewer inspected the actual raw records: all warmups are document0003733, 128→2, warmup-final residents equal measurement-initial residents; shared allocations use two distinct storage pointers with the declared sum. There is no current raw mismatch. Frozen code and evidence remain unchanged; any subsequent analyzer should enforce its own new contract.

The result supports retained descriptive measurements only. Cross-arm equal outputs are 7–11/16 and compared route hashes all differ. U repeat capture drift is −20.90%; two repeats are not a population noise floor. X/F capture and completion signs flip; X process wall is worse in both. The existing report includes these limits. No additional broad review or test expansion is required before the next focused source-localization probe.

Overall verdict: **FAIL for integrity/claim labeling**, while the narrow mechanical verdict `WEAKEN_UPPER_BOUND_TOO_SMALL / GATING_INSUFFICIENT / WEAKENS` is reproducible from the frozen artifacts. No files were edited.

P0: 0. P1: 2.

- **P1 — self-derived “correctness” reference.** Variant A’s own hash becomes `reference_output_hash`, then A/B/C are judged against it (`joinstream_real_moe_tail_pilot.cu:1125-1138`). This detects variant divergence but not a common WMMA/routing/projection error. It can change whether this executable is a valid substrate for the applicability verdict.
- **P1 — scope overreach.** Routes, inputs, and weights are synthetic (`pilot.cu:411-464,843-879`), and the seed only permutes token identities while preserving expert shape/FLOPs (`pilot.cu:422-424`). Thus tracker line 28’s broad “Freeze JoinStream as a thesis mechanism” may change under real router traces; only the exact four-cell single-5090 result is established.

A. Ground/reference provenance — **FAIL**

- `FinalizePairOrTriple` selects `A_ALL_DONE_SHAM.output_hash` as the reference and declares equality “correctness” (`pilot.cu:1125-1138`).
- The summary reports “3480/3480 correctness” without identifying it as variant-equivalence proxy (`summary.md:24-31`).
- Impact: supports **A/B/C bitwise equivalence only**, not independent numerical correctness or `real_gt`.

B. Score normalization — **PASS**

- Metrics are raw paired timestamp differences and regression relative to the paired A baseline (`analyzer.py:501-540`).
- Noise guard is frozen as `max(timer resolution, 3×1.4826×MAD)` (`analyzer.py:38-41,167-179`); candidates and 5% threshold are fixed (`analyzer.py:29,38,213-264`).
- No prediction-max self-normalization, hidden rescaling, or changed threshold was found.

C. Result existence/claim trace — **WARN**

- Full scans found exactly 1,920 calibration rows and 3,480 formal rows; all recorded correctness/timestamp flags were 1, hashes matched references, CUDA errors were empty, and producer work closed.
- Independent recomputation exactly matched all stored medians/MAD guards and reproduced the rule branch in `analyzer.py:582-645`; stored gates/verdict appear at `analysis.json:558-600`.
- Every SHA-256 in `COMPLETE.json:42-65`, including the frozen prior JoinStream and CriticalSplit evidence, recomputed correctly. `COMPLETE` is `status: COMPLETE` (`COMPLETE.json:2-9`).
- Warning: “Final independent review found no unresolved P0/P1” (`summary.md:32`; echoed at `COMPLETE.json:16`) has no bound review artifact in the supplied evidence. `formal_run_attempts: 1` is likewise self-declared (`COMPLETE.json:22`).

D. Dead code/action wiring — **WARN**

- Main producer/consumer work is live: kernels execute at `pilot.cu:542-795`, launch at `983-996`, flow through the formal permutation loop at `1643-1699`, and feed analysis at `analyzer.py:501-656`.
- A/B/C static-work parity is enforced at `analyzer.py:466-493`; all 800 measured utility triples passed it.
- `stale_read` is dead instrumentation: declared at `pilot.cu:112`, zero-initialized through `ResetTrial` at `891-901`, copied at `1046`, but never set by either kernel. Therefore “0 stale reads” is vacuous, although live hash equality still detects divergent outputs.
- The analyzer never consumes `calibration_run.csv`; its inputs at `analyzer.py:659-680` trust `calibration.json`. Current calibration medians independently recomputed correctly, so this is a fail-closed gap rather than a present mismatch.

E. Scope — **WARN**

- Honest boundary exists at `summary.md:34-36`: one single-GPU microbenchmark, not production, serving, or online policy.
- Actual scope is four locked cells (`run_lock.json:9-16`) and one declared formal run (`COMPLETE.json:18-30`).
- Calibration/formal seeds and route-table hashes are distinct, so there is no literal row reuse; however, expert counts/task topology are intentionally identical across seeds (`pilot.cu:422-424`). This does not establish cross-shape or cross-seed robustness.
- “No synthetic delay” is true, but the route distribution and tensor values remain synthetic. The broad `FREEZE` statement in `tracker:28` exceeds the evidence ceiling.
- At the median, the 50% progress gate was already satisfied before join closure in every cell (`analysis.json:72-78,210-216,348-354,486-492`), so the secondary gating result should not be generalized to regimes where the gate is actually binding.

F. Evaluation type — **WARN**

Precise classification:

`real_hardware_synthetic_grouped_expert_microbenchmark + self_supervised_variant_equivalence_proxy`

It is neither `real_gt` nor `simulation_only`. Timing comes from real RTX 5090 CUDA execution (`environment.json:3-7`), but correctness uses the run’s own A variant, and workloads/routes are generated. The current label `SINGLE_GPU_REALISTIC_MOE_TAIL_MICROBENCHMARK` (`analysis.json:556-557`) omits those two qualifications.

Bottom line: retain the exact negative mechanical result as a four-cell executable-level observation. Do not claim independently validated MoE correctness, real-router tail prevalence, or a general thesis-level freeze until those P1 boundaries are corrected.

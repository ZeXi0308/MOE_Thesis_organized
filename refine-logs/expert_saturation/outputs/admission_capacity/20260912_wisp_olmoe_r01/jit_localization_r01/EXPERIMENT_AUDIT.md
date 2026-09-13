# JIT source-localization integrity audit

2026-09-14. **Bounded PASS, P0=0/P1=0.** Reviewer GPT-5.6-Sol ultra, `/root/fullstage_qualification_integrity/source_semantics`; same-family/reused/provisional. One code-risk check was extended to the actual two-cell results, without another broad review.

| Check | Result and verified evidence |
|---|---|
| A Provenance | PASS: direct native compiler listener, five reviewed installed-source hashes, and 39 frozen inputs per cell; execution.json:55–76,892–913 |
| B Accounting | PASS: independently recomputed every phase/apply union and every engine overlap. Union/envelope is a measured time decomposition, not a score or subtractive counterfactual |
| C Existence | PASS: 89 manifested payload hashes and archive SHA; both complete16/512 cells; execution.json:787–839,2377–2429 |
| D Execution | PASS: 938/974 complete contiguous events,896/928 exact observer/pager joins,49/51 engine joins, no failed/unwrapped/untimed event or identity/phase/shape/thread error |
| E Scope | PASS at stated descriptive scope: cold21 actual misses; retained21 inventory-backed hits plus2 new misses;0→172→188 files. Single ordered pair with9/16 outputs equal and different routes |
| F Classification | Direct observational native-runtime host measurement; DESCRIPTIVE_COMPILER_LOCALIZATION / MEASUREMENT_ONLY |

Recomputed totals are32 requests,1024 outputs,5088 scheduled positions,1600 measured/1824 all layer calls. Kernel validation is explicitly disabled; this audit makes no numerical-quality claim.

Seven >100ms measurement calls independently resolve to six cold and one retained layer0 calls. Compiler/first-load unions cover97.4019%–98.2496% of their outer observed apply envelopes. Examples: attempt01/analysis.json:3246,4211,5176,5757,5954,9607,16087. Phase unions/counts are at1475 and14118. Every cache-hit bool agrees with the corresponding base32 cache-key presence in before/after inventories.

Supported: the measured source attribution for these seven new calls, and the observed coverage gap after one natural warmup/cache run. Unsupported: stable cold/retained benefit, deducting compile time, assigning old uninstrumented stalls to these events, or a positive F/X mechanism result. The report preserves these boundaries. No further JIT audit is needed; proceed to the next scientific comparison with its own explicit compiler-state contract.

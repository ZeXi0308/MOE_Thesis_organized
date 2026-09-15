# StableBatch C8 Action-Transfer Experiment Audit

**Verdict**: `PASS`  
**P0 / P1**: `0 / 0`  
**Reviewer**: fresh GPT-5.6-Sol, ultra reasoning, same-family provisional  
**Trace**: `.aris/traces/experiment-audit/2026-08-10_run11/`

## Result-relevant checks

- Reconstructed all route changes from stored per-arm top-k arrays: baseline `41`; M1 same-rank `37/0/+37`; C8 same-rank `31/6/+25`; C8 exact random `22.25/4/+18.25`; C8 oracle `36/0/+36`.
- Reconstructed transfer ratio `31/37=83.78%`, rank-specificity gap `6.75`, and oracle gap `11`.
- Confirmed 33 primary unique cells across eight documents were sealed before C8 outcomes. The 139 raw-positive M1 ranks remain secondary sensitivity evidence and are not treated as independent cells.
- Confirmed all 264 C8 candidates use exactly one fixed-C8 contribution plus seven M64 contributions; R/U/M1 closure and non-target/gate/router hashes pass.
- Confirmed packed final-logit decompositions. C8 same-rank recovers 391,066 elements and harms 145,289, but none of the 33 final-logit vectors exactly restores R.
- Confirmed all eight LODO aggregates retain positive C8 same-rank net and a positive specificity gap.
- Confirmed frozen-file hashes, output `MANIFEST.json`, clean initial GPU process snapshot, and formal runner PID at final capture.
- Recomputed from the raw ledger with `mismatch_count=0`. A second local invocation is byte-identical to the sealed recompute (`673d2f884adaf149d44b5c1e351b5673ca1d45e6b59374d71d23d2e99e4e9da4`).

M1 exact-random is stored in the manifest-bound cohort snapshot rather than duplicated into `summary.metrics.core`: recovered `175/8`, harmed `19/8`, net `39/2=19.5`. This does not change the Gate.

## Decision and claim ceiling

The evidence supports the predeclared **case A / GO** branch.

> On this retrospectively selected 33-cell M1-positive cohort, for one OLMoE revision on one RTX 5090 in BF16 eager mode, zero-pad-slot5 fixed-C8 at the frozen M1 rank retains 31/37 M1 route recoveries and has higher net recovery than exact-uniform C8 rank selection, with the direction surviving every LODO aggregation.

It does not establish unbiased prevalence, an outcome-naive selector, budget/SLO feasibility, serving or model-quality value, cross-model/GPU generality, or deployability.

For this rapid Gate, the independent recomputation is sufficient because the route decision is directly reconstructable from stored top-k arrays. It is not paper-grade independent replication: the recomputer shares aggregation definitions, and raw final-logit tensors were not retained for regeneration of packed bitsets.

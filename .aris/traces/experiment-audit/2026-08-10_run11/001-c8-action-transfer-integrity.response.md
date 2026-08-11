# StableBatch C8 action-transfer integrity audit

## Overall verdict: PASS

`review_independence=same-family`  
`acceptance_status=provisional`

No clear P0/P1 was found. The experiment can proceed as a scientifically eligible rapid-exploration Gate A / GO result.

## Independent numerical reconstruction

| Condition | Route recovered | Harmed | Net | Remaining |
|---|---:|---:|---:|---:|
| U baseline | 0 | 0 | 0 | 41 |
| M1 same rank | 37 | 0 | 37 | 4 |
| C8 same rank | 31 | 6 | 25 | 16 |
| C8 exact random | 22.25 | 4 | 18.25 | 22.75 |
| C8 best-rank oracle | 36 | 0 | 36 | 5 |

- Transfer ratio: `31/37 = 83.78%`.
- Rank-specificity gap: `25 - 18.25 = 6.75`.
- C8 oracle gap: `36 - 25 = 11`.

The reviewer rebuilt route changes from per-arm top-k arrays and checked packed final-logit bitsets rather than trusting summary counters.

## Integrity

- The primary cohort is 33 unique victim-layer cells across eight documents and was sealed before C8 outcomes. The 139 count is the secondary surface of all net-positive M1 rank actions.
- Every one of the 264 C8 actions is exactly one C8 contribution plus seven M64 contributions. R is eight M1; U is eight M64; M1 same-rank is one M1 plus seven M64.
- Fresh R/U/M1 arms and side-call hashes close against the frozen oracle ledger.
- Route recovered, harmed and persistent sets reconstruct exactly from stored top-k arrays.
- C8 same-rank recovers 391,066 final-logit elements and harms 145,289; all 33 final-logit vectors still differ bitwise from R and no greedy token changes. This is not model-quality evidence.
- C8 same-rank net is positive in 24/33 cells and exceeds exact random in 21/33. All eight documents have positive same-rank net; seven of eight have positive per-document specificity; all eight LODO aggregations retain positive net and specificity.
- All frozen-file and output-manifest bindings match. Initial GPU snapshots have no compute processes; final runtime contains only the formal runner process.
- Independent recomputation has `mismatch_count=0`.

The M1 exact-random condition is not duplicated under `summary.metrics.core`, but it is present and manifest-bound in the cohort snapshot: recovered `175/8`, harmed `19/8`, net `39/2`. This is a presentation-location issue, not P1.

## Claim ceiling

> On this retrospectively selected 33-cell M1-positive cohort, for one OLMoE revision on one RTX 5090 in BF16 eager mode, zero-pad-slot5 fixed-C8 at the frozen M1 rank retains 31/37 M1 route recoveries and has higher net recovery than exact-uniform C8 rank selection, with the direction surviving every LODO aggregation.

This supports advancing ShapeABI + StabilityBudget to an outcome-naive selector Gate. It does not establish unbiased prevalence, an observable selector, budget/SLO feasibility, serving value, model quality, cross-model/GPU generality or deployability.

Raw-ledger recomputation is independent enough for this rapid Gate because route decisions can be reconstructed from per-arm top-k arrays. It is not paper-grade independent replication: aggregation formulas are duplicated and raw final-logit tensors were not retained, so packed bitsets can be internally validated but not regenerated.

# RouteShape-SLO experiment-integrity audit

**Verdict:** `WARN / SAME_FAMILY_PROVISIONAL_FINAL_AUDIT`
**P0 unresolved:** 0
**P1 unresolved:** 0
**Scientific scope:** `SMOKE_ONLY_NOT_SCIENTIFICALLY_ELIGIBLE`

## Fixed audit findings

1. The fresh same-family auditor caught action drift between active-token and
   max-running variants. The final frozen action is the prompt default
   `next_window_active_token_budget`, identical in the
   research protocol, config, analyzer, tests, P2/P3 guards, summary, and next
   experiment. A regression test rejects the second action.
2. Eligibility now checks both the feature window and target window for real
   runtime, representativeness, hook overhead, sealing, gate weights, and
   evidence type. P2/P3 also require bound schemas, the frozen action, exact
   booleans, all eligibility checks, and an executed upstream result.

## Recomputed checks

- `26/26` RouteShape-SLO unit tests pass.
- Canonical `commands.sh` rebuilds the 256-window feature table in `/tmp`,
  recomputes M0--M4, and byte-compares `metrics.csv`, `summary.json`,
  `environment.json`, and `report.md` successfully.
- The canonical directory contains exactly six requested files.
- M1 pinball loss recomputes to `0.9717235642310368`; M3 to
  `0.9448100666024098`; their diagnostic relative difference is
  `2.7696660469404846%`; dangerous underprediction is `1/60` for both.
- Request/document identity overlap across analysis splits is zero. Arrival
  episode overlap is nonzero and is explicitly an eligibility blocker.
- M4 consumes future route only as a latency-prediction diagnostic. No
  counterfactual capacity label, Oracle headroom, or controller result is
  created.
- The final auditor recomputed SHA-256 over the full 1,337,795,800-byte source
  expert-call ledger and matched the manifest value
  `799e9799270e5b9aa267e0ff72dce80e10650ed39d236ea6e184fa99d6747e99`.

## Remaining limitations, not hidden integrity passes

- The source is one observed RTX 5090 isolated GPU primitive, not a native
  serving runtime; queue depth and running sequences are documented surrogates.
- There is one model and one arrival replay, no calibrated SLO, no dynamic cap
  intervention, no gate weights, and no hook-overhead A/B.
- The bounded canonical replay does not rehash the 1.337 GB ledger on every
  invocation; it streams the selected contiguous slice and cross-checks it
  against the smaller ledgers. The final audit separately completed the full
  hash check above.
- A fresh post-stabilization audit completed and found no P0/P1 integrity
  defect. The audit remains `WARN`, rather than an independent `PASS`, because
  the reviewer is same-family/provisional and the scientific input is still
  smoke-only.

These limitations justify `BLOCKED_RUNTIME_NOT_REPRESENTATIVE`; they do not
justify upgrading or downgrading the smoke metric into a P1 scientific result.

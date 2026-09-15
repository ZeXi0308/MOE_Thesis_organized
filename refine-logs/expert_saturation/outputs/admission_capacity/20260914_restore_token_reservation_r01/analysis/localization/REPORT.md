# Restore token reservation: actual dispatch and remaining gaps

MEASUREMENT_ONLY_DISPATCH_AND_GAP_LOCALIZATION

Actual independent trajectories only. Common state means recorded metadata/counters and returned output prefix, never KV tensor equality. Ordered dispatch can diverge before token/victim maps. Each gap is the interval between two real output receipts; nested costs are not added. Observer-only planning identifies unused reservations, not alternate-run latency.

| Pair | First ordered dispatch | First token/victim map | Visible/output prefix equal | Prior elapsed delta s | Ready selected/eligible |
|---|---:|---:|---|---:|---|
| block0-guard_all → block0-guard_residual | 406 | 406 | True/True | -0.079058 | 30/30 |
| block0-fit_scan → block0-guard_residual | 406 | 521 | True/True | -0.043675 | 30/30 |
| block1-guard_all → block1-guard_residual | 406 | 406 | True/True | -0.200380 | 30/30 |
| block1-fit_scan → block1-guard_residual | 406 | 521 | True/True | +0.198036 | 30/30 |

| Cell | Max gap request / output | Gap s | Steps | Longest unselected run | Held calls | Target recompute |
|---|---|---:|---|---:|---:|---:|
| block0-guard_residual | memory-train-article-0003475 / 435 | 4.338545 | 854–1051 | 194 | 0 | 3506 |

block0-guard_residual: withholding events 54; events with counted but actually unserved surviving ready candidates 0; those accompanied by an actual rank-prefix capacity stop 0.

| block0-guard_all | memory-train-article-0003475 / 435 | 4.292699 | 862–1065 | 200 | 0 | 3506 |
| block0-fit_scan | memory-train-article-0003571 / 311 | 4.919893 | 405–649 | 125 | 14 | 6367 |
| block1-guard_residual | memory-train-article-0003475 / 435 | 4.355581 | 854–1051 | 194 | 0 | 3506 |

block1-guard_residual: withholding events 54; events with counted but actually unserved surviving ready candidates 0; those accompanied by an actual rank-prefix capacity stop 0.

| block1-guard_all | memory-train-article-0003475 / 435 | 4.157151 | 862–1065 | 200 | 0 | 3506 |
| block1-fit_scan | memory-train-article-0003571 / 311 | 4.823253 | 405–649 | 125 | 14 | 6367 |

Complete per-request comparisons, maximal-gap victim/service timelines, actual ledger delivery, short residencies, full work/cost and reservation events are in analysis.json. No new controller or unexecuted latency claim.

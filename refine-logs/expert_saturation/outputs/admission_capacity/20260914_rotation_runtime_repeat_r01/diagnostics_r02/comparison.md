MEASUREMENT_ONLY — first forced-swap recovery comparison.

| Cell | Calls | Full wall s | Scheduler s | Engine nonscheduler s | Recovery span s | Final call s |
|---|---:|---:|---:|---:|---:|---:|
| reference/block0 | 1162 | 24.313487 | 0.880433 | 22.992491 | 0.856550307944417 | 0.7663179095834494 |
| reference/block1 | 1162 | 23.342345 | 0.872652 | 22.053836 | 0.12116756290197372 | 0.031700652092695236 |
| repeat/block0 | 1162 | 23.478504 | 0.903364 | 22.117681 | 0.12172207236289978 | 0.032029252499341965 |
| repeat/block1 | 1162 | 24.326836 | 1.066836 | 22.672997 | 0.13625854812562466 | 0.032457245513796806 |

reference/block0 → reference/block1: full execution/decision/output fingerprints equal=True; first recovery event equal=True.
reference/block0 → repeat/block0: full execution/decision/output fingerprints equal=True; first recovery event equal=True.
reference/block0 → repeat/block1: full execution/decision/output fingerprints equal=True; first recovery event equal=True.
reference/block1 → repeat/block0: full execution/decision/output fingerprints equal=True; first recovery event equal=True.
reference/block1 → repeat/block1: full execution/decision/output fingerprints equal=True; first recovery event equal=True.
repeat/block0 → repeat/block1: full execution/decision/output fingerprints equal=True; first recovery event equal=True.

wall = scheduler + engine_nonscheduler + outside_engine; recovery and final-call costs overlap these totals.
All calls retained. No trimming, significance, JIT/root-cause, new holdout, or method claim.

Actual target/victim, outputs before recovery, scheduled work and source hashes: comparison.json. Cost values are descriptive; no fixed step index or recurrence threshold was assumed.

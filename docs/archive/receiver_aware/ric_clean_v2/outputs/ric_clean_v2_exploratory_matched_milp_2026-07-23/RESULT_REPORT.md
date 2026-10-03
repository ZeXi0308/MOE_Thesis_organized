# RIC-Clean-v2 Matched-Receiver MILP Exploratory Result

## Frozen decision

`NO_GO_SYNTHETIC_MATCHED_TAIL_HEADROOM`

This is a no-go for the frozen matched-world mechanism probe, not a scientific conclusion about
real receiver congestion or MoE serving. The exact-information CVaR gap passed its 5% threshold,
but the preregistered necessity condition—receiver information changes the unique optimal first
action in at least 25% of windows—failed in all four primary cells.

## Primary gate

| Model | rho | service CV | residual scale | Median exact-information CVaR gap | First-action flip rate | Cell pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| OLMoE | 0.85 | 0.25 | 2.0 | 13.3422% | 12.5% | No |
| OLMoE | 0.95 | 0.25 | 2.0 | 15.2431% | 12.5% | No |
| LLM-JP | 0.85 | 0.25 | 2.0 | 14.4202% | 0.0% | No |
| LLM-JP | 0.95 | 0.25 | 2.0 | 15.8402% | 0.0% | No |

Frozen rule: every primary cell must have median exact-information CVaR gap >= 5%, first-action
flip rate >= 25%, and the sensitivity set must not become negative. The sensitivity minimum was
0.0%, so the decision is caused by the failed first-action necessity gate, not by a post-hoc
threshold change.

## Sensitivity boundary

At rho 0.85, the median exact-information CVaR gaps remained non-negative for service CV 0.0 and
0.5 at residual scale 2.0, and for residual scales 0.5 and 4.0 at service CV 0.25. The residual-0.5
cells had exactly 0.0% median gap for both models. The executable receiver-shadow heuristic was
also worse than the selected simple baseline in those cells: median gain -8.3293% for OLMoE and
-9.0982% for LLM-JP.

The positive oracle gap together with a low first-action flip rate means receiver information can
change later schedule positions while leaving the first dispatch decision unchanged. Under the
frozen protocol this does not establish the required receiver-aware dispatch necessity and must
not be used to rescue the idea.

## Independent accounting checks

- Result rows: 176; grouped summaries: 22; eight holdout windows per group.
- All 176 two-stage solver records have status 0 and MIP gap <= 1e-7.
- Recomputed every relative-gain identity, first-action predicate, group median/minimum, request
  cluster count, and receiver-informed optimum dominance from `per_window`: zero discrepancies.
- Producer source SHA-256:
  `f5295810718f4992a7e4595798ae5b86c1804170fee968a73bf5d7637f798619`.
- Result SHA-256:
  `d399c6683f3f61926e909b88f8dc2f04fe3394671c1d13db9237a988fcb689e2`.
- Remote environment: Python 3.12.3, SciPy 1.18.0, NumPy 2.3.2; RTX 5090, driver 580.105.08.
- Remote experiment directory was not a Git worktree, so no remote commit can be truthfully
  recorded. The source hash above is the executable identity.

## Evidence boundary

The route identities are real captured OLMoE and LLM-JP data, but the keyed post-arrival receiver
DAG tails are synthetic, non-decaying, and locked until candidate arrival. Execution is a
normalized-service, single-shared-cut L2 proxy with a clairvoyant exact small-horizon oracle.
This is not measured receiver queue state, RDMA/NCCL, multi-node serving, TPOT/P99, or production
benefit. `scientific_result` remains `false`.

Do not rerun this same construction with more seeds, looser flip thresholds, a larger synthetic
residual, or only the positive oracle-gap metric. A future receiver-aware line needs a different
mechanism in which receiver state creates an observable early control conflict under measured or
credibly calibrated contention.

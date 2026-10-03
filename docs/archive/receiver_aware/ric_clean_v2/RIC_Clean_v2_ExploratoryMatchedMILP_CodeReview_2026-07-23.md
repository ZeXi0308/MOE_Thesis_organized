# RIC-Clean-v2 Exploratory Matched-World MILP Code Review

STATUS: **SIGNED-OFF FOR CONDITIONAL EXPLORATORY RUN**  
OPEN_P0: **0**  
SCIENTIFIC_RESULT: **false**

## Reviewed snapshot

- runner: `experiments/ric_clean_v2/explore_receiver_matched_milp.py`
- runner SHA-256: `f5295810718f4992a7e4595798ae5b86c1804170fee968a73bf5d7637f798619`
- tests: `experiments/ric_clean_v2/test_explore_receiver_matched_milp.py`
- tests SHA-256: `fcb5829d67aa1cc14caf2fc8af576842b9f763e5a26ef27b1ddda9aa328e0aac`
- remote tests: `5/5 PASS`, including permutation-selection MILP versus independent exhaustive optimum
- independent reviewers: adversarial mechanism/MILP review; route/statistics/accounting review

The earlier pairwise big-M implementation and the SHA values previously recorded here are
**SUPERSEDED**. The signed-off runnable snapshot is the permutation-selection MILP identified by
the hashes above.

## Closed blockers

1. Superseded the invalid sender-reconstructible `missing` experiment. The current worlds share
   current tasks, route, release, service and aggregate receiver-tail multiset; only keyed mapping
   differs.
2. `B` uses one joint open-loop schedule across both worlds. `R0` uses world-specific schedules
   under one pooled CVaR99 then mean objective.
3. MILP CVaR and mean are replayed in flow-time units and independently checked against complete
   `8!` enumeration. Solver optimality and MIP gap are gated.
4. Route input is validated through the reviewed route signoff plus an `O_NOFOLLOW` streaming
   SHA/Cartesian/identity census.
5. Selection-half baseline choice and request-disjoint holdout-half evaluation are separated.
6. Output is no-overwrite, atomic, and records task/world instances, solver accounting and source
   SHA for independent replay.

## Mandatory claim boundary

The run is only a conditional mechanism probe for a **synthetic, non-decaying keyed receiver DAG
tail that is locked until candidate arrival**, on a normalized-service single-shared-cut L2 proxy.
The MILP is clairvoyant. Results must not be described as measured receiver congestion, RDMA/NCCL,
multi-node serving, TPOT/P99, production benefit, or a final scientific result.

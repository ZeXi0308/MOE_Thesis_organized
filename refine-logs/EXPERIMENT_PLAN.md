# Selector Failure Decomposition — Frozen exploratory plan

- Version: `v1`
- Frozen intent time: `2026-08-10T22:15:38+08:00`
- Mode: CPU-only retrospective failure decomposition
- Evidence class: `EXPLORATORY_POST_HOC_DECOMPOSITION`; no confirmatory claim is permitted because both C8 surfaces have already been observed.

## Question

Can the old broad C8 surface explain the fresh selector failure as (a) cell-opportunity transfer failure, (b) within-cell rank-residual failure, or (c) harm-prediction failure, and does one pre-existing hierarchical static profile retain positive rank gain strongly enough to justify preregistering exactly one policy on a third unseen cohort?

## Frozen decomposition

For cell `c` and rank `r`, use route utility only: `u(c,r)=recovered-harmed`. Define `mu(c)=mean_r u(c,r)` and `delta(c,r)=u(c,r)-mu(c)`. Implementation uses exact integer `residual8(c,r)=8*u(c,r)-sum_r u(c,r)` and must verify its eight-rank sum is zero in every cell. Final-logit counts are excluded.

## Frozen models

- Shared action-pre vector: layer/expert/rank one-hot plus gate weight/share/gap-to-min, top-k mass, normalized entropy, and cutoff margin. Normalizers are fitted on broad training rows only.
- Cell head: alpha-1 ridge with unpenalized intercept predicts `mu`; input is the within-cell mean action vector with the constant rank block removed.
- Rank-residual ridge: alpha-1 ridge without intercept predicts `residual8`; input is the action vector centered within its cell.
- Harm head: alpha-1 ridge with unpenalized intercept predicts route harmed count from the raw action-pre vector.
- Hierarchical static rank profile: the pre-existing no-search lambda-4 hierarchy over layer-expert, expert-rank, layer-rank, and exact layer-expert-rank tables, fitted to `residual8`. Its claim is profile-aware rank selection, not online dynamic observability.

## Frozen transfer evaluation

- Broad-side estimate: fixed 16-fold leave-one-document-out cross-fit; no tuning.
- Fresh-side estimate: fit once on all broad documents and evaluate once on the existing document-disjoint fresh surface.
- Cell head selects exact global `B=33` cells, tie by cell identity.
- Every rank policy chooses at most one action in those same cells, tie by the smallest rank.
- Global matched random and selected-cell uniform-rank baselines use exact expectations.
- Primary profile metric: fresh `profile_rank_gain_B = selected profile net - selected-cell uniform-rank net` on the cell-head top-33 cells.
- Report cell gain, ridge rank gain, profile rank gain, all-cell rank gain, per-document effects, prediction error, exact harm avoidance, and rank oracle headroom.

## Frozen decision

1. If fresh `profile_rank_gain_B > 0`, choose only `PRE_REGISTER_HYBRID_CELLGATE_PROFILEDRANK_V1` for a third, fully new document-disjoint cohort.
2. Otherwise choose only `STOP_SUPERVISED_SELECTOR_TO_WITNESSPATCH_BUDGETED_PROBING`.

A positive harm diagnostic alone does not create a third policy. This closes the previously unspecified `profile<=0, harm>0` branch before reading the new decomposition. Any selected hybrid remains an unconfirmed preregistration candidate until sealed and evaluated on new data.

## Execution stages

1. Implement pure-CPU analysis and synthetic contract tests.
2. Hash-lock code, config, dependencies, and both immutable C8 ledgers.
3. Run broad LODO and full-broad-to-fresh decomposition into a new output directory.
4. Generate a bounded result card and one independent experiment-integrity audit.
5. Record exactly one next policy; do not tune on the fresh result.

## Observed result — append-only closure

- Authoritative output: `docs/ideas/stablebatch/experiments/outputs/selector_failure_decomposition_20260810_run01/`.
- Broad LODO → fresh cell-selection gain: `-1237/320` → `-281/640`.
- Broad LODO → fresh rank-residual ridge gain: `5/8` → `3`; fresh MSE skill remains negative (`-0.00478158`).
- Broad LODO → fresh hierarchical profile rank gain: `-11/8` → `-5`; the primary profile condition fails.
- Harm head is mixed: broad LODO MSE skill is negative, while fresh MSE skill is `+0.0232576` and exact harm avoidance is `57/8`. Per the frozen rule this remains diagnostic and cannot create a post-hoc third policy.
- Unique decision: `STOP_SUPERVISED_SELECTOR_TO_WITNESSPATCH_BUDGETED_PROBING`.
- Integrity audit: `PASS`, P0 `0`, P1 `0`, same-family provisional. No GPU or model inference was used.

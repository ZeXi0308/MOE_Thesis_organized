# Sparse C8 StabilityBudget Gate — Frozen execution plan

- Version: `v1`
- Frozen intent time: `2026-08-10T21:15:03+08:00`
- Mode: corrected `GLOBAL_CELL_RANK_SELECTOR_GATE`
- Claim boundary: one OLMoE revision, BF16 eager, one RTX 5090, route-level self-supervised proxy; not model quality, serving SLO, EP, or multi-GPU evidence.

## Question

Can a fixed, outcome-naive ridge use action-pre fields to select rank-specific fixed-C8 actions on 16 document-disjoint fresh windows, under exact `B=33` and at most one rank per cell?

## Shared eligibility

Both train and fresh cohorts use the unconditional Cartesian product of 16 presealed windows and layers 0–14 at victim position 15. Every native-valid top-8 cell is included. There is no filter on `U != R`, M1/C8 reward, opportunity, recovered/harmed, oracle rank, or final outcome.

- Old broad train: document indices 16–31, 240 cells, 1,920 C8 actions.
- Fresh evaluation: separately frozen document indices 0–15, 240 cells, 1,920 C8 actions.
- Freshness qualifier: the cohort was frozen before its separate M1 run and has no C8 outcome; this Gate reads no fresh M1 artifact and claims C8-outcome-naive selection, not that the documents were never used by any experiment.
- Required order: complete broad C8 ledger → fit ridge and seal fresh selection → run fresh C8 outcomes.

## Frozen action and model

- Reference `R`: all eight contributions at M1.
- Unprotected `U`: all eight contributions at M64.
- Candidate: one rank at fixed C8 `zero_pad_slot5`, seven ranks at M64.
- Label: `route_recovered_count - route_harmed_count` from broad C8 outcomes only.
- Features: layer/expert/rank one-hot; gate weight/share/gap-to-min; top-k mass; normalized entropy; cutoff margin.
- Historical outcome-derived sensitivity: excluded.
- Ridge: L2 alpha 1.0, intercept unpenalized, train-only population mean/std, no search.
- Selection: per cell maximum predicted utility, tie lowest rank; then global score descending and cell identity ascending; exact B=33.

## Frozen comparisons

- Global matched random: exact expectation for 33 distinct uniform cells and one uniform rank per cell.
- Cell-matched random-rank: exact expectation on the selector's 33 cells with uniform rank.
- Oracle exact-B: exactly 33 unique cells.
- Oracle at-most-B: up to 33 positive-utility cells, abstention allowed.
- `cell_selection_gain = cell_matched_random_rank_net - global_random_net`.
- `rank_selection_gain = selector_net - cell_matched_random_rank_net`.
- `rank_headroom_capture = rank_selection_gain / (oracle_exact_B_net - cell_matched_random_rank_net)` when the denominator is positive.

## Mechanical classification

Use the priority and fixed thresholds in `configs/sparse_c8_stability_budget_gate_v1.json`: harm dominates; action generalization; strong selector (headroom capture at least 0.3 and cross-document counts at least 4); weakly above random; selector weak. No post-outcome changes to B, alpha, features, thresholds, cohort, or action space.

## Execution stages

1. Qualify and hash-lock runner, policy, test, config, and bound inputs.
2. `broad`: seal the unconditional old 240-cell cohort, then produce 1,920 C8 outcomes.
3. `seal`: fit ridge from broad labels, scan fresh action-pre cells, select exact B=33, and write `SELECTION_SEAL.json` while outcome paths are absent.
4. `fresh`: verify the seal, produce all 1,920 fresh C8 outcomes, then calculate both random baselines, both oracles, gains, headroom, per-document results, and classification.
5. Perform one bounded final experiment-integrity audit; do not re-audit the historical C8 transfer Gate.

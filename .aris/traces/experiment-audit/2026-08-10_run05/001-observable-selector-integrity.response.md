# StableBatch observable-selector experiment audit

## Verdict

- Overall integrity verdict: `PASS`
- Review class: `same-family provisional`
- Confidence: `0.98`
- Frozen scientific verdict: `WEAKENS_MAXGATE_V1_NOT_BETTER_THAN_SHUFFLE`
- P0/P1 findings: none
- Workspace edits by reviewer: none

## Evidence

1. R is explicitly an all-M1 synthetic self-supervised proxy, not ground truth. The config and formal summary retain this boundary.
2. The observable scan does not compute M1/M64 intervention outcomes. MaxGate-v1 receives only current-layer `gate_weights`; `expert_ids` are identity/tie-break data. The assignment ledger and `POLICY_SELECTION_LOCK.json` are written before the first `run_cell` side-call.
3. O and S have the same action budget and exact work surface: one M1-protected rank plus seven M64 ranks per cell, three side-call repeats per M and three full-forward repeats per arm. The balanced shuffle assigns 30 cells to each top-k rank.
4. Across all 240 raw rows, native no-op closure, target-pair exactly-once application, non-target contribution hashes, same-arm bitwise stability, and all integrity statuses pass.
5. All signed rewards are retained. No positive-only filtering or post-hoc threshold change was found.
6. The first acceptance failed on a non-idle GPU and is ineligible. Formal evidence binds the successful acceptance's runner, base runner, config, frozen lock, manifest, acceptance record, and status hashes.
7. Both successful manifests have exact file-set, byte-size, and SHA-256 closure. The formal manifest pre-binds `RUN_STATUS.json`.

## Independent decisive recomputation

- Rows / unique victim-layer cells: `240 / 240`
- Independent document windows: `16`
- Opportunity: `35 cells / 8 documents`, gate passed
- Observable total signed reward: `A_O = -3` (`-0.0125/action`)
- Shuffled total signed reward: `A_S = +3` (`+0.0125/action`)
- O positive/tie/negative: `13 / 209 / 18`
- S positive/tie/negative: `10 / 220 / 10`
- Documents with aggregate O above S: `4`
- Frozen ratio threshold: `ceil(1.5 * 3) = 5`
- O/S selected the same rank in `30` cells; their repeated artifacts are bitwise equal
- Shuffle counts: `[30, 30, 30, 30, 30, 30, 30, 30]`
- Verdict: opportunity passed and `A_O <= A_S`, therefore `WEAKENS_MAXGATE_V1_NOT_BETTER_THAN_SHUFFLE`

## Claim boundary and decision

The result supports only this statement: on one fixed RTX 5090, 16 document windows, 240 victim-layer action-value cells, same-layer top-8 offline replay, and an all-M1 synthetic self-supervised reference, MaxGate-v1's total signed reward is not better than the one frozen balanced matched-shuffle assignment.

It does not establish serving, a dynamic controller, EP/NCCL, multi-GPU behavior, prevalence, generalization, or 240 statistically independent samples.

Integrity passes, but MaxGate-v1 is a scientific `NO-GO`. Preserve the negative result, do not filter negative rewards or change thresholds, and do not rescue-tune on the consumed snapshot. Any new selector must be a separately preregistered hypothesis on new evidence.

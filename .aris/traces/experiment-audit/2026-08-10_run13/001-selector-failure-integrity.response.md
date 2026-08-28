## Overall verdict: PASS

**Integrity status:** `pass`  
**review_independence:** `same-family`  
**acceptance_status:** `provisional`

No P0/P1 findings. The frozen exploratory conclusion is arithmetically correct, leakage checks pass, and the authoritative run is reproducible from the locked inputs.

### A–F audit

| Check | Status | Evidence and finding |
|---|---|---|
| **A. Ground truth / proxy provenance** | **PASS** | Labels are route-level intervention outcomes: `recovered`, `harmed`, and `utility=recovered-harmed` are read and cross-checked at `docs/ideas/stablebatch/experiments/selector_failure_decomposition.py:165-173`; final-logit fields are deliberately adversarially different in `test_selector_failure_decomposition.py:41-45,71-89`. Independent scan of all 3,840 actions in `broad/cell_results.jsonl:1-240` and `fresh/cell_results.jsonl:1-240` verified recovered = unprotected route changes minus action changes, harmed = the converse, persistent = intersection, and all stored counts/net/utility. Final-logit net differed from route net in 3,839/3,840 actions, yet never affected labels. The result explicitly disclaims model-quality meaning at `PILOT_RESULT.md:5,39`. |
| **B. Leakage / normalization** | **PASS** | Feature vectors consume only layer, expert, rank, gate weights and cutoff-derived values at `sparse_c8_stability_budget_policy.py:134-209`. Normalizers fit only the passed training actions at `selector_failure_decomposition.py:205-234`; broad LODO excludes the held document at `:561-575`; fresh is scored from the one full-broad model at `:1044-1049`. Stored schema has `outcome_derived_features: []` at `models.json:280-398` and 16 broad training documents at `models.json:12566-12571`. Independently recomputed stored normalizers matched broad statistics, not fresh statistics; all 1,920 fresh scores matched independent model scoring within `4.5e-16`, and all 1,920 broad LODO scores re-executed within `1.1e-15`. MSE skill uses baseline MSE, not any model-output maximum (`selector_failure_decomposition.py:578-604`); exact decision metrics use unnormalized route counts (`:402-450`). |
| **C. Result existence / arithmetic / hashes** | **PASS** | `RUN_STATUS.json:2-6` is COMPLETE and exploratory. All seven V2-locked files match `FROZEN_SELECTOR_FAILURE_DECOMPOSITION_LOCK_V2.json:2-16`; lock/config bindings match `INPUT_BINDINGS.json:2-12`; all eight manifest entries match byte size and SHA-256 at `MANIFEST.json:2-34`. Exact metrics below independently match `summary.json`. Budget/closure records are at `summary.json:2330-2337`, and the frozen decision is at `summary.json:1130-1139`. |
| **D. Dead code / execution / failed attempt** | **PASS** | The executed call path loads and validates both ledgers, runs LODO/full-broad fitting, scores both surfaces, and evaluates both at `selector_failure_decomposition.py:1022-1052`; artifacts are written at `:1118-1146`, reached from `main` at `:1171-1176`. Fresh and broad selections reproduced exactly; six contract tests pass. The failed run is explicitly ineligible at `selector_failure_decomposition_20260810_failed_attempt01/FAILED_ATTEMPT.json:2-8`. Its V1-bound numerical artifacts—models plus both prediction ledgers—are byte-identical to run01, while only bindings changed to V2; this confirms the relock did not change scores or selections. V1/V2 changed code/test hashes are visible at `FROZEN_SELECTOR_FAILURE_DECOMPOSITION_LOCK_V1.json:10-12` versus V2 `:10-12`. |
| **E. Scope / wording** | **PASS** | The config calls the evidence exploratory/post-hoc and non-confirmatory at `selector_failure_decomposition_v1.json:3-5`; the plan repeats this at `EXPERIMENT_PLAN_20260810_221538.md:5-6,24-39`; the tracker does so at `EXPERIMENT_TRACKER_20260810_221538.md:6-9`. The result explicitly forbids interpreting it as online dynamic observability, model quality, serving SLO, or production evidence at `PILOT_RESULT.md:35-39`. |
| **F. Evaluation type** | **self_supervised_proxy** | Route-recovery/harm is an internal route-stability proxy derived from controlled model-intervention traces, with no external dataset ground truth. This run is a retrospective CPU re-analysis of already observed ledgers, not a fresh confirmation, human evaluation, production test, or model-quality evaluation. |

### Independently recomputed decisive metrics

Every policy selected exactly **B=33 distinct cells**, one rank per cell. Residual closure passed in **240/240 broad** and **240/240 fresh** cells.

| Metric | Broad 16-fold LODO | Fresh full-broad transfer |
|---|---:|---:|
| Raw action totals `(recovered, harmed, net)` | `(178, 164, 14)` | `(281, 430, -149)` |
| Global matched random `(R,H,net)` | `(979/320, 451/160, 77/320)` | `(3091/640, 473/64, -1639/640)` |
| Selected-cell uniform rank `(R,H,net)` | `(1, 37/8, -29/8)` | `(65/8, 89/8, -3)` |
| Cell-selection gain | `-1237/320` | `-281/640` |
| Ridge outcome / rank gain | `(1,4,-3)` / `5/8` | `(8,8,0)` / `3` |
| Profile outcome / rank gain | `(1,6,-5)` / `-11/8` | `(7,15,-8)` / **`-5`** |
| Min-harm outcome / harm avoidance | `(1,4,-3)` / `5/8` | `(9,4,5)` / `57/8` |

Fresh harm MSE skill independently recomputed as `0.023257565024297322`; combined with positive `57/8` avoidance, `effective=true` is correct. Nevertheless, the frozen two-branch rule at `selector_failure_decomposition_v1.json:115-121` keys only on fresh profile gain. Since `-5 <= 0`, the correct single branch is:

`STOP_SUPERVISED_SELECTOR_TO_WITNESSPATCH_BUDGETED_PROBING`

### P0/P1 findings

**None.** No detected issue can reverse or significantly change the exploratory stop decision.

Non-blocking P2: `RUN_STATUS.json` is not an entry in `MANIFEST.json`; its decision/status are redundant with the manifest-covered `summary.json` and `PILOT_RESULT.md`, so this does not affect the result.

### Claim impacts

- **Supported:** exact retrospective route-proxy decomposition; broad LODO/full-broad-to-fresh diagnostics; the frozen stop-branch decision.
- **Diagnostic only:** positive fresh harm-head result; it does not authorize a post-hoc harm-only policy.
- **Unsupported:** confirmation of a new policy, online dynamic observability, model quality, serving SLO, scientific generalization, or production proof.

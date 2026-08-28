# Experiment Tracker: Expert-Pressure-Conditioned Capacity

> Created: 2026-08-23  
> Scientific status: `N0D_POSITIVE_MEASUREMENT / NO_METHOD_GO`  
> Current Gate: `N0e layer-0 pre-router hidden-state source localization`  
> Canonical selection rule: frozen before each run; all attempts retained

## 1. Gate Board

| Gate | Research question | Dependency | Status | Exit artifact |
|---|---|---|---|---|
| E0 | Do all telemetry and action evaluators fail closed under the independent audit counterexamples? | none | `PASS` | `EXPERIMENT_AUDIT_REMEDIATION.md` |
| N0b | Is native telemetry/snapshot/actuator eligible and bounded-deviation? | E0 pass + explicit GPU authorization | `CONDITIONAL_NO_GO / VALID_WINDOW_NOT_TRANSPARENT` | `N0_HARNESS_QUALIFICATION_REPORT.md` |
| N0c | Does retained N0b drift recur in capture/no-export or only full export? | N0b valid failure + two frozen drift cells | `COMPLETE / NOT_REPRODUCED` | `N0C_CAPTURE_SOURCE_TRIAGE_REPORT.md` |
| N0d | Does custom-runtime serial-vs-batch route divergence have an earlier/same-step pre-top-k logit difference? | N0c closed + frozen custom-runtime evidence | `COMPLETE / POSITIVE_MEASUREMENT` | `N0D_ROUTER_LOGIT_CONFORMANCE_REPORT.md` |
| N0e | At the step-1/layer-0 frontier, is the batch-dependent logit difference already present in the gate input or first visible in the gate Linear shape path? | N0d positive + frozen target | `PREPARE` | `N0E_PREROUTER_LOCALIZATION_REPORT.md` |
| N0a | Does one steady and one bursty frozen A/C/D event transfer to native runtime? | N0b parity | `BLOCKED_BY_N0B` | `N0_CONFORMANCE_TRANSFER_REPORT.md` |
| G1 | Can action/cohort/SLO/budgets/common support be frozen? | N0 pass | `BLOCKED_BY_N0` | `G1_CALIBRATION_FREEZE.md` |
| I1 | Does completed pressure modify the paired request-SLO budget effect, with material Oracle? | G1 pass | `BLOCKED_BY_G1` | `I1_PAIRED_ACTION_VERDICT.md` |
| G3 | Does a one-level correction beat strongest route-free control at full cost? | positive I1 | `CONDITIONAL` | `G3_CONTROLLER_VERDICT.md` |
| G4 | Does the signal/mechanism transfer to native 8×A100 EP? | G3 pass | `CONDITIONAL` | `G4_EP_CONFIRMATORY_VERDICT.md` |

## 2. Run Queue

| Order | Run ID | Purpose | Input freeze | Required checks | Status |
|---:|---|---|---|---|---|
| 0 | E0-R01 | adversarial evaluator remediation and independent re-audit | audit request/response; exact source hashes; frozen counterexamples | zero-support; token-drift subset; source semantics; malformed NPZ; two-sided timing; GPU isolation; exit codes | `PASS` |
| 1 | N0-R01 | telemetry OFF/ON parity, overhead, snapshot/replay, branch actuator | model/runtime revision; target state; telemetry schema; tolerance | output/route/completion parity; state digest; process isolation; two-sided wall/TPOT deviation | `COMPLETE / SCIENTIFIC_FAIL` |
| 1a | N0c-R01 | two-cell capture-source triage | both unique N0b drift cells; exact prompt artifacts; explicit route-to-output index | independent OFF/OFF control; capture/no-export/full-export; 4-round matching first-divergence rule | `COMPLETE / NOT_REPRODUCED` |
| 1b | N0d-westc-r01 | launcher/capture-path attempt | frozen four-request/8-step custom-runtime trace | producer output-isolation contract | `INFRASTRUCTURE_ABORT_BEFORE_MODEL_LOAD` |
| 1c | N0d-westc-r02 | serial-vs-batch-4 pre-top-k localization | frozen four-request/8-step custom-runtime trace | matched prestate; token parity; three processes; request-step-layer identity; independent top-k replay | `COMPLETE / PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_REPRODUCED` |
| 1d | N0e-westc-r01 | layer-0 gate-input source localization | frozen request-000/step-1/layer-0 target; N0d parent hashes | three-process A/B/A; hook alignment; raw BF16 hidden comparison | `PREPARE` |
| 2 | N0-R02 | one steady native A/C/D | frozen steady event and A/C/D identities | same-arm stability; exact target/KV/position; minimum divergence chain | `BLOCKED_BY_N0B` |
| 3 | N0-R03 | one bursty native A/C/D | frozen bursty event and A/C/D identities | same checks as N0-R02 | `BLOCKED_BY_N0B` |
| 4 | G1-R01 | budget knee | median-KV homogeneous steady calibration split | actual batch semantics; KV feasibility; knee freeze | `BLOCKED_BY_N0` |
| 5 | G1-R02 | KV sensitivity and pressure common support | low/mid/high budget candidates | physical KV/padding controls; low/high pressure overlap | `BLOCKED_BY_G1_R01` |
| 6 | G1-R03 | cohort/SLO/`W`/`T_max` freeze | calibration split only | treatment-independent request IDs; no silent censoring | `BLOCKED_BY_G1_R01` |
| 7 | I1-R00 | three-branch dry run | frozen `Z_t`, budgets, cohort | digest equality; independent future state | `BLOCKED_BY_G1` |
| 8 | I1-R01 | steady low/high-pressure matched states | frozen sampling list | three budgets × process repeats; retain all | `BLOCKED_BY_I1_R00` |
| 9 | I1-R02 | bursty low/high-pressure matched states | frozen sampling list | same as I1-R01 | `BLOCKED_BY_I1_R00` |
| 10 | I1-R03 | fresh holdout + pressure permutation | sealed holdout | paired estimator; Oracle; negative control | `BLOCKED_BY_I1_R01_R02` |
| 11 | G3-R01 | one-level correction | positive I1 thresholds | route-free baselines; fallback; overhead; guards | `CONDITIONAL` |
| 12 | G4-R01 | EP parity/preflight | G3 pass; frozen model/backend/placement | max-rank signal; A2A; request denominator | `CONDITIONAL` |
| 13 | G4-R02 | four EP anchor cells | frozen positive/boundary/negative cells | signal then Controller transfer | `CONDITIONAL` |

## 3. Mandatory Pre-Run Freeze Checklist

- [ ] Repository commit and dirty diff recorded.
- [ ] Model revision, dtype, runtime/backend and hardware recorded.
- [ ] GPU process isolation checked.
- [ ] Independent unit and split frozen by document/episode.
- [ ] Request IDs, arrival timestamps and scheduler order frozen.
- [ ] Action semantics and candidate budgets frozen.
- [ ] `Z_t` digest schema frozen.
- [ ] `W`, `T_max`, SLOs and failure/censor rule frozen before holdout.
- [ ] Canonical run-selection rule declared before execution.
- [ ] Negative control declared.
- [ ] Strong route-free baselines declared.
- [ ] Output bundle path unique and append-only.

## 4. I1 State Independence Checklist

Every policy branch must own an independent copy/evolution of:

- [ ] request set and generated tokens;
- [ ] logical/physical KV and allocator/block map;
- [ ] physical batch shape and padding;
- [ ] route and expert/rank load;
- [ ] queue and scheduler state;
- [ ] RNG/sampling state;
- [ ] pressure history;
- [ ] completion wall-clock timeline.

Any unchecked item marks the experiment `INVALID_EXPERIMENT`.

## 5. Metrics Freeze

### Primary

- [ ] TTFT P50/P95/P99.
- [ ] TPOT and ITL P50/P95/P99.
- [ ] Request-level SLO attainment.
- [ ] SLO-goodput with one fixed request cohort.
- [ ] Paired `Risk(b_high)-Risk(b_low)`.
- [ ] Dangerous underprediction.
- [ ] Action-conditioned Oracle headroom.

### Guards

- [ ] Request completion time.
- [ ] Queue wait P95/P99 and max age.
- [ ] Starvation count/fairness.
- [ ] KV feasibility/OOM/failure.
- [ ] Scheduler + telemetry overhead.

### Diagnostics only

- [ ] Step latency.
- [ ] Max expert/rank routed-token load.
- [ ] Active expert union saturation.
- [ ] Sampled attention/MoE critical-path share.
- [ ] A2A/slow-rank only in G4.

## 6. Decision Ledger

| Condition | Required verdict | Controller allowed? |
|---|---|---|
| N0 identity/parity/state invalid | `INVALID_EXPERIMENT` | No |
| N0a no native transfer | `CUSTOM_RUNTIME_ONLY_ARTIFACT` or `NATIVE_RUNTIME_TRANSFER_BLOCKED` | No; explicit reopen decision required |
| no ordinary-state pressure common support | `UNIDENTIFIED_IN_TESTED_REGIME` | No |
| pressure residual absent | `SIGNAL_DEAD_IN_TESTED_REGIME` | No |
| residual exists, Oracle `<3%` | `MEASUREMENT_ONLY / ACTION_SPACE_DEAD` | No |
| Oracle exists, online policy fails | `PHENOMENON_ALIVE_MECHANISM_DEAD` | Stop mechanism search |
| only selected cells work | `CONDITIONAL_BOUNDARY` | Only guarded/fallback version |
| I1 and full-cost G3 pass | `CONDITIONAL_METHOD_GO` | Yes; then G4 confirmatory |

## 7. Result Entry Template

```text
Run ID:
Commit / dirty state:
Model / runtime / hardware:
Arrival and request cohort:
Action config:
State digest result:
Process isolation:
Raw artifact path:
Repeat retention:
Primary metrics:
Guards:
Negative control:
Strongest baseline:
Oracle/headroom:
Evidence tier:
Verdict:
What was measured:
What was not measured:
Claim ceiling:
Failure category:
One next smallest experiment:
```

## 8. Current Entry

```text
Initial independent experiment-integrity audit: FAIL at the pre-remediation
snapshot. Append-only remediation re-audit: WARN overall, E0 PASS, P0=0/P1=0.

N0b executed on the refreshed authorized RTX 5090 endpoint with vLLM 0.26.0,
OLMoE BF16 and two counterbalanced process repeats. All eight stock/optimized
route-OFF/ON bundles were retained; source identity, 464 artifact hashes,
exclusive-GPU evidence and cleanup checks passed. The corrected verdict is
VALID_WINDOW_NOT_TRANSPARENT / TELEMETRY_TOKEN_DRIFT. Repeat 0 preserved token
and lossless-route parity but optimized P95 absolute TPOT/wall deviation was
28.26%/43.77%, above the frozen 5% Gate. Repeat 1 drifted at optimized
[512,16,1,0] and stock [512,8,2,0]; its timing is diagnostic only.

The sealed v2 reducer selected the first failed repeat and masked the stronger
r1 correctness failure. Raw artifacts remain unchanged; evaluator v3 fixes
explicit cross-repeat precedence and the append-only replay returns the correct
status. A first localization pass was itself corrected append-only because the
producer drops the prompt-tail route: retained route step s produces output
token s+1. Corrected v2 localizes an earlier captured route for only 3/5 drifted
requests, so N0c must retain the prompt-tail forward and capture hidden/router/
top-k/expert/final-logit state. No native pressure-latency, capacity,
SLO-goodput, action-headroom or Controller result exists. Decode-cap remains
blocked; do not retry N0b to select a favorable repeat.

N0c then executed 32 fresh-process arms over the two preregistered drift cells.
Both OFF controls, capture-enabled/no-export and full-export were token-identical
in all four rounds for both targets; full-export route arrays were also
repeat-identical. The sealed result is `NOT_REPRODUCED`, not a source
localization. It closes retries of those two historical cells but does not
erase the independent custom-runtime serial-vs-batched conformance phenomenon.
At that point, the next unique uncertainty was N0d: whether that separate repeat-stable expert
assignment difference is already present in pre-top-k logits or only at the
top-k near-tie boundary. Capacity/action/Controller remain blocked.

N0d then executed on the authorized RTX 5090 endpoint. `westc-r01` aborted
before model loading because the capture path violated the producer isolation
contract; it produced no scientific trajectory. `westc-r02` was the first and
only scientific execution. Three counterbalanced fresh processes all preserved
serial-A/B exactness, 32/32 reference-token parity, and within-arm stability.
The first Expert-assignment frontier was request `olmoe-dev-steady-000`, decode
step 1, layer 3; the same request-step already had a nonzero BF16 router-logit
difference at layer 0. An append-only evaluator-v2 replay independently
validated all 4,608 retained Expert sets against their router logits and kept
the status `PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_REPRODUCED`.

This is a custom-runtime matched-prestate conformance measurement only. It does
not identify Attention/KV/padding versus gate-Linear causality and does not
measure capacity, latency, SLO-goodput, action headroom, or a Controller. Two
unmanifested local `.pyc` files were created after the downloaded campaign's
completion sentinel; every manifest-listed scientific byte still passes its
hash, but exact local-directory provenance remains `WARN`. The one next
uncertainty is N0e: whether the frozen step-1/layer-0 difference already exists
at the gate input.
```

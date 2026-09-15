# Experiment Tracker: Expert-Pressure-Conditioned Capacity

> Created: 2026-08-23  
> Scientific status: `UNRUN / NO_METHOD_GO`  
> Current authorized Gate: `N0` only  
> Canonical selection rule: frozen before each run; all attempts retained

## 1. Gate Board

| Gate | Research question | Dependency | Status | Exit artifact |
|---|---|---|---|---|
| N0b | Is native telemetry/snapshot/actuator eligible and low-overhead? | none | `READY_TO_RUN` | `N0_HARNESS_QUALIFICATION_REPORT.md` |
| N0a | Does one steady and one bursty frozen A/C/D event transfer to native runtime? | N0b parity | `PENDING` | `N0_CONFORMANCE_TRANSFER_REPORT.md` |
| G1 | Can action/cohort/SLO/budgets/common support be frozen? | N0 pass | `BLOCKED_BY_N0` | `G1_CALIBRATION_FREEZE.md` |
| I1 | Does completed pressure modify the paired request-SLO budget effect, with material Oracle? | G1 pass | `BLOCKED_BY_G1` | `I1_PAIRED_ACTION_VERDICT.md` |
| G3 | Does a one-level correction beat strongest route-free control at full cost? | positive I1 | `CONDITIONAL` | `G3_CONTROLLER_VERDICT.md` |
| G4 | Does the signal/mechanism transfer to native 8×A100 EP? | G3 pass | `CONDITIONAL` | `G4_EP_CONFIRMATORY_VERDICT.md` |

## 2. Run Queue

| Order | Run ID | Purpose | Input freeze | Required checks | Status |
|---:|---|---|---|---|---|
| 1 | N0-R01 | telemetry OFF/ON parity, overhead, snapshot/replay, branch actuator | model/runtime revision; target state; telemetry schema; tolerance | output/route/completion parity; state digest; process isolation; wall-clock overhead | `READY_TO_RUN` |
| 2 | N0-R02 | one steady native A/C/D | frozen steady event and A/C/D identities | same-arm stability; exact target/KV/position; minimum divergence chain | `BLOCKED_BY_N0_R01` |
| 3 | N0-R03 | one bursty native A/C/D | frozen bursty event and A/C/D identities | same checks as N0-R02 | `BLOCKED_BY_N0_R01` |
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
Planning artifact only.
No N0/G1/I1/G3/G4 run has been executed in this planning turn.
No native capacity, SLO-goodput or Controller result exists.
Next: N0-R01 only.
```

# Experiment Plan: Pressure-Conditioned MoE Decode Capacity

> Date: 2026-08-23  
> Depends on: `FINAL_PROPOSAL.md`  
> Current authority: only N0 is authorized  
> Current science status: `UNRUN / ACTION_ORACLE_STILL_PAUSED`

## 1. Claim Map

| ID | Claim | Minimum evidence | Anti-claim |
|---|---|---|---|
| C1 | In a frozen native regime, completed max expert/rank load adds stable, material information about the paired request-level effect of raising decode budget beyond a preregistered strong route-free baseline | same-`Z_t` real budget branches; independent policy state; fixed request cohort; steady+bursty; process repeats; fresh holdout; pressure negative control | Pressure is not manipulated; latency correlation alone is insufficient; single-GPU does not prove EP |
| C2, conditional | After C1 and material Oracle headroom, a one-level pressure correction improves full-cost SLO-goodput over the strongest ordinary controller | online native loop; TTFT/TPOT/fairness guards; overhead; simple baselines; conditional EP confirmation | Complex model is not required; local step saving is not request benefit |

### Claim stop rules

- C1 fails if paired effect is unstable, absorbed by route-free controls, reproduced by negative control, or has no common support.
- C2 is never tested if action Oracle headroom is `<3%` in the exploratory Gate.
- Policy-specific rerun by itself is experimental correctness, not novelty.
- If one-threshold correction captures `>=90%` of Oracle, it is the final mechanism; do not add Ridge/GBDT/RL.

## 2. Gate Sequence

```text
N0 native qualification
→ G1 calibration and branch eligibility
→ I1 one decisive paired action experiment
→ positive only: G3 minimal controller
→ G3 pass only: G4 8×A100 EP confirmatory
```

## 3. N0 — Current Only Authorized Gate

### Question

Can one representative native OLMoE serving path carry the frozen execution-conformance event, low-overhead pressure telemetry and reproducible branch state needed by a later capacity experiment?

### N0a: conformance transfer

- Reconstruct exactly one frozen steady and one frozen bursty A/C/D source event.
- A: target serial.
- C: target with original companions and original physical KV-length/padding vector.
- D: same width and KV-length/padding vector with shuffled/different-document companions.
- Freeze target request, token position, generated prefix, logical/physical KV, model revision, dtype, backend and sampling state.
- Capture only the already justified minimum chain:

```text
pre-router hidden
→ router logits
→ top-k margin
→ selected experts
→ expert output
→ next-token logits
```

Do not expand operator localization during N0.

### N0b: harness eligibility

- Telemetry OFF/ON parity for output tokens, logits within frozen tolerance, route/completion and request IDs.
- GPU process isolation and version binding.
- Same-arm repeat stability across fresh processes.
- GPU-side routed-token aggregate and asynchronous epoch export.
- Snapshot/replay digest and branch actuator dry run.
- Telemetry and snapshot overhead.

### Initial runs

| Run | Purpose | Status |
|---|---|---|
| N0-R01 | telemetry OFF/ON parity + overhead | `PENDING` |
| N0-R02 | steady A/C/D process-isolated repeats | `PENDING` |
| N0-R03 | bursty A/C/D process-isolated repeats | `PENDING` |

### N0 verdicts

| Verdict | Meaning | Next action |
|---|---|---|
| `PASS_NATIVE_QUALIFICATION` | N0a transfer and N0b harness close | permit G1 planning/execution |
| `NATIVE_RUNTIME_TRANSFER_BLOCKED` | runtime/instrumentation cannot reproduce or observe the frozen event | stop; fix only qualification blocker if in scope |
| `CUSTOM_RUNTIME_ONLY_ARTIFACT` | frozen conformance behavior does not transfer under equivalent native state | preserve narrow prior result; explicit decision required before reopening I1 |
| `INVALID_EXPERIMENT` | identity, state, repeat or accounting contract fails | repair protocol/harness; no scientific inference |

N0 never counts as pressure/capacity/Controller evidence.

## 4. G1 — Calibration, Not a Main Result

### 4.1 Freeze action semantics

```text
b_t = maximum decode sequences selected per iteration
      during the next fixed control epoch
```

Freeze FCFS + aging order, prefill policy, router/top-k, placement, precision, KV policy, sampling, arrival trace and continuation policy.

### 4.2 Freeze ordinary state and pressure

```text
Z_t:
  full active/ready identities and order
  token histories and physical KV/block state
  padding, KV occupancy, queue ages, prefill state
  previous budget/scheduled tokens/total routed tokens
  recent request ITL and step-latency EMA
  runtime/model/RNG identity

X_t(b) = g(Z_t, b)

p_t = completed max expert/rank routed-token load
```

### 4.3 Budget knee

1. Use one median-KV homogeneous steady calibration workload.
2. Scan a feasible subset of `{1, 2, 4, 8, 16, 32}`.
3. Locate the Token/KV-only SLO knee.
4. Freeze `b_low`, `b_mid`, `b_high` around the knee.
5. Verify all three actions produce distinct physical batches and are KV-feasible.

No budget may be changed after holdout inspection.

### 4.4 Fixed cohort and complete denominator

Freeze a wall-clock arrival window `W` and common timeout `T_max` on calibration data:

```text
C = pre-branch active/ready IDs
    ∪ IDs arriving from the frozen trace during [t, t+W]
```

- Replay the identical arrival IDs/timestamps in every branch.
- Do not add later arrivals to the main cohort.
- Run each branch until all `C` complete or the common `T_max`.
- Timeout/OOM/failure/uncompleted requests are preregistered SLO misses.
- TPOT/ITL denominator: every at-risk output-token interval in `C`.
- TTFT denominator: every arrival request in the frozen cohort.
- SLO-goodput denominator and wall-clock interval are identical across policies.

### 4.5 Common support

Low/high pressure states must overlap on:

- candidate token/KV/padding state;
- previous budget and scheduled tokens;
- total routed tokens;
- queue/age and prefill state;
- recent latency;
- arrival regime and workload family.

No common support means `UNIDENTIFIED_IN_TESTED_REGIME`, not permission to reuse a future route or synthesize the main result.

### G1 pass

- action semantics and state digest close;
- three nondegenerate budgets are frozen;
- `W`, `T_max`, SLOs and cohort are frozen;
- low/high pressure common support exists;
- telemetry overhead is within the preregistered qualification tolerance.

## 5. I1 — One Decisive Paired Action Experiment

### 5.1 Sampling unit and split

- Independent unit: document/request arrival episode, not decode window.
- Split calibration/development/holdout by document and episode.
- Initial design: each of four arrival×pressure strata contributes 8–12 matched prestates.
- Each state receives at least three fresh-process repeats.
- One fresh process replays all three budget branches from the same serialized prefix; branch order is randomized, and all artifacts are retained.

Initial size range:

```text
4 strata × 8–12 states × 3 budgets × 3 process repeats
= 288–432 branch executions
```

Expand only if the preregistered uncertainty rule says the effect is near noise; do not fill a larger Cartesian grid by default.

### 5.2 Same-prestate branch contract

Before branching, verify a `Z_t` digest over:

- request IDs, queue order and arrival prefix;
- generated tokens;
- logical/physical KV lengths, padding and allocator mapping;
- active/ready set and scheduler state;
- model/revision/backend/dtype;
- RNG/sampling config;
- completed pressure and previous-action controls.

Execute:

```text
branch low  → b_low
branch mid  → b_mid
branch high → b_high
```

Each branch independently evolves:

```text
request set and completion
generated tokens and KV
physical batch shape and padding
route and expert/rank load
queue and scheduler state
pressure history
wall-clock timeline
```

Cross-policy sharing of any future route/KV/completion makes the run `INVALID_EXPERIMENT`.

### 5.3 Primary estimator

For each matched state:

```text
tau_s(high, low) = Risk_s(b_high) - Risk_s(b_low)
```

Then estimate within preregistered pressure strata:

```text
tau_hat(p-bin) = mean_s tau_s(high, low)
```

Bootstrap/resample by episode, not step. The main test is whether pressure-conditioned paired risk difference remains stable beyond the strong route-free baseline. Quantile interaction is secondary.

### 5.4 Negative control

Within ordinary-state, arrival-regime and budget matched cells, permute completed-pressure labels. It should destroy the pressure-conditioned effect. A decorrelated episode pressure may be retained as a secondary control.

### 5.5 Action-conditioned Oracle

For every actual `Z_t`, choose the highest feasible budget only among the three branches that were really executed. Oracle may inspect future outcome offline, but may not share future state, ignore failures, or omit queue/TTFT/fairness/overhead.

### 5.6 I1 primary metrics

- request-level TPOT/ITL SLO attainment;
- SLO-goodput;
- `tau_s` and `tau_hat` interaction;
- dangerous underprediction;
- action Oracle headroom over strongest route-free policy.

Guards:

- TTFT P95/P99;
- request completion time;
- queue waiting P95/P99, max age and starvation;
- KV feasibility/OOM;
- scheduler/telemetry wall-clock overhead.

Diagnostics:

- step latency, max expert/rank load, active union;
- sampled attention/MoE exposed time;
- A2A/slow-rank only in G4.

## 6. Baseline Ladder

| Family | Baseline | Role |
|---|---|---|
| Fixed | native/default maximum concurrency | deployed behavior |
| Fixed | calibration-selected fixed best budget | strongest fixed comparator |
| Route-free | Token/KV/queue adaptive budget | ordinary state baseline |
| Route-free | Token/KV/queue + recent-latency feedback | strongest simple non-MoE baseline |
| MoE-aware | one-threshold, one-level pressure correction | final conditional Controller |
| Upper bound | future-known action-conditioned Oracle | material action-space check |

Gimbal is a conceptual nearest signal/action neighbor; do not compare its incompatible cross-engine/placement system numbers directly with this local cap.

## 7. Non-Cartesian Evaluation Matrix

### Matrix A: G1 budget knee

- one homogeneous steady workload;
- median physical KV;
- feasible budget subset of `{1,2,4,8,16,32}`.

### Matrix B: G1 KV sensitivity

- `b_low/b_mid/b_high`;
- short/medium/long physical KV;
- only enough cells to freeze candidate-state features.

### Matrix C: I1 decisive cells

| Arrival | Pressure | Budgets | Workload rule |
|---|---|---|---|
| steady | low / high | low, mid, high | natural episodes with ordinary-state support |
| bursty | low / high | low, mid, high | natural episodes with ordinary-state support |

### Matrix D: G3 end-to-end, positive only

1. chat, short prompt / medium output, steady;
2. chat, long prompt / short output, bursty;
3. code, medium prompt / long output, steady;
4. code, long prompt / long output, bursty;
5. math, medium prompt / medium output, steady;
6. mixed chat/code/math and mixed lengths, bursty;
7. homogeneous negative control.

### Matrix E: G4 EP, G3 pass only

- strongest positive regime;
- positive regime at a different KV band;
- boundary regime;
- expected fallback/negative regime.

Use `Qwen3-30B-A3B` on the [vLLM Expert Parallel path](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/) as the first compatibility candidate, not as a preregistered fact. Freeze one actually supported sparse-MoE revision, dtype, EP degree, placement, A2A backend and batch-invariance setting only after preflight. Redefine the primary signal as completed max-rank routed-token load; single-GPU max-expert results do not transfer automatically.

## 8. Continue / Stop / Claim Thresholds

### Exploratory

- Oracle `1%–3%`: weak signal; only controlled repeat or boundary analysis.
- Oracle `>=3%`: permit the next Gate if paired interaction is repeat-stable and the negative control fails.

### Formal, frozen before holdout

- incremental model: P95 pinball `>=5%` or dangerous underprediction `>=15%` improvement, directionally stable across frozen regimes;
- action Oracle: SLO-goodput `>=5%` at the same SLO, or violation rate `>=20%` lower at the same throughput;
- Controller: recover `>=40%` Oracle, net SLO-goodput `>=3%`, overhead `<1%`, steady/bursty same direction;
- simplicity: if one-threshold policy captures `>=90%` Oracle, stop model expansion.

### Immediate stop

- same-arm instability or treatment sign flip without resolution;
- mismatched `Z_t` digest;
- shared future policy state;
- no pressure common support;
- incomplete request denominator;
- Oracle `<3%`;
- dangerous underprediction not improved;
- route-free + recent-latency baseline absorbs pressure residual;
- TTFT/wait/fairness guard offsets TPOT gain;
- simple strategy captures `>=90%` Oracle.

### Precise verdict vocabulary

| Observation | Verdict |
|---|---|
| pressure has no residual | `SIGNAL_DEAD_IN_TESTED_REGIME` |
| residual but no Oracle headroom | `MEASUREMENT_ONLY / ACTION_SPACE_DEAD` |
| Oracle alive, online mechanism fails | `PHENOMENON_ALIVE_MECHANISM_DEAD` |
| only certain batch/KV/EP cells work | `CONDITIONAL_BOUNDARY` |
| invalid identity/state/accounting | `INVALID_EXPERIMENT` |

## 9. G3 — Minimal Controller, Positive I1 Only

```text
on_epoch_boundary(state):
    b0 = token_kv_queue_policy(candidate_state)

    if telemetry_missing
       or pressure_uncalibrated
       or outside_common_support:
        return b0

    p = completed_pressure_ema()

    if p >= high_threshold:
        target = max(b_min, b0 - one_budget_level)
    elif p <= low_threshold and dwell_passed:
        target = min(b0, current_budget + one_budget_level)
    else:
        target = current_budget

    return fairness_and_kv_guard(target)
```

Fast down, slow up, hysteresis, dwell, max-wait override and fail-closed fallback are required. No second action family is introduced.

## 10. G4 — 8×A100 EP Confirmatory, G3 Pass Only

First re-test signal transfer, then Controller transfer. Measure:

- max rank routed-token load and imbalance;
- A2A bytes/time and slow-rank critical-path survival;
- request TPOT/TTFT/SLO-goodput;
- telemetry/controller overhead.

Interpretation:

- same-direction effect plus A2A/slow-rank residual: expand the operating boundary;
- route-free state absorbs residual: fallback to ordinary control in EP;
- sign reversal: retain only the single-GPU bounded result;
- Apply/run success alone is not EP evidence.

## 11. Compute Budget — Planning Estimate, Not Measured Runtime

| Gate | Resource | Estimated cap |
|---|---|---|
| N0 + G1 | RTX 5090 | 20–60 GPU-hours |
| I1 | RTX 5090 | 20–40 GPU-hours; optional second frozen regime another 20–40 |
| G3 | RTX 5090 | 20–60 GPU-hours |
| G4 | 8×A100 | 10–20 node-hours = 80–160 A100 GPU-hours |
| Negative path | RTX 5090 | stop by W6, roughly 40–100 GPU-hours total cap |
| Positive path | RTX 5090 + 8×A100 | roughly 60–160 5090 GPU-hours + 80–160 A100 GPU-hours |

Do not spend 8×A100 time on feature search or full Cartesian sweeps.

## 12. Twelve-Week Milestones

| Week | Output | Gate |
|---|---|---|
| W1 | native adapter; telemetry parity bundle | N0b |
| W2 | steady/bursty A/C/D bundles and qualification verdict | N0a |
| W3 | frozen action, cohort, SLO, budget knee, common-support rule | G1 |
| W4 | state digest and three-branch dry run | I1 preparation |
| W5 | decisive branch artifacts with process repeats | I1 data |
| W6 | fresh-holdout paired/Oracle report and exact verdict | I1 decision |
| W7 | one-level correction implementation | G3 conditional |
| W8 | full-request single-GPU comparison | G3 decision |
| W9 | EP preflight and four-cell freeze | G4 qualification |
| W10 | EP signal/controller confirmation | G4 decision |
| W11 | controlled repeats, boundary map, figures | confirmation |
| W12 | thesis, claim audit, Resurrection Card, sealed artifacts | closeout |

## 13. Figure Contract

1. Pressure-conditioned paired treatment effect with episode bootstrap intervals.
2. Empirical feasible budget map over Token/KV load × completed pressure.
3. Full-cost SLO-goodput vs attainment Pareto for all baselines and Oracle.
4. Route-free → pressure main effect → interaction feature ladder.
5. Appendix: active-union saturation, burst timeline, critical-path breakdown, overhead/fallback, regime map.

## 14. Artifact Contract

Each Gate has one canonical bundle:

```text
config
environment + commit
state digests
raw request/route/timing data
processed metrics
report
commands and run log
```

- Retain every repeat and failed attempt.
- Declare canonical selection before running.
- Never overwrite raw evidence; corrections go to `ADDENDUM.md`.
- Record GPU process isolation and exact runtime/model/action config.
- N0, I1, G3 and G4 reports must explicitly list measured/not-measured and claim ceiling.

## 15. One Immediate Next Experiment

```text
N0-R01:
  Native OLMoE telemetry OFF/ON parity and overhead,
  together with snapshot/replay and branch-actuator eligibility.
```

Only after N0-R01 closes may N0-R02/R03 provide the one steady and one bursty A/C/D transfer verdict. No capacity branch or Controller is authorized yet.

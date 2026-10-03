# JoinStream Real-MoE Tail Applicability Gate Tracker

> Run: `20260810_205953`  
> Status: `COMPLETE / WEAKEN_UPPER_BOUND_TOO_SMALL / FREEZE`  
> Evidence: `SINGLE_GPU_REALISTIC_MOE_TAIL_MICROBENCHMARK`  
> Evaluation type: `real_hardware_synthetic_grouped_expert_microbenchmark + self_supervised_variant_equivalence_proxy`

## Run ledger

| Stage | Status | Frozen evidence |
|---|---|---|
| PRECHECK | PASS | Original JoinStream GPU bundle and CriticalSplit bundle hashes unchanged |
| PREFLIGHT | PASS | RTX 5090; 170 SM; 12 blocks/SM; MEDIUM 128/131 blocks; HIGH 2000/2000 blocks |
| CALIBRATION | LOCKED | Seed 1729; gate candidates 0/12.5/25/50%; 50% selected in 4/4 cells |
| FORMAL | COMPLETE_ONCE | Seed 314159; exact 4 cells; 3480 rows; 1160 A/B/C rows each |
| VARIANT EQUIVALENCE | PASS | 3480/3480 formal rows match A_ALL_DONE_SHAM bitwise; no CUDA errors; K=2 distinct-expert join; no independent numerical GT claim |
| ADJUDICATION | COMPLETE | `WEAKEN_UPPER_BOUND_TOO_SMALL`; `GATING_INSUFFICIENT`; `WEAKENS` |

## Mechanical result

- Safe-benefit cells: `0/4`.
- Legal GATED overlap with producer regression below 5%: `3/4`.
- Natural GATED window >=1 us: `3/4`; maximum median `161.664 us`.
- GATED producer-regression medians: `+0.0044% .. +0.0751%`.
- No cell's critical-completion gain exceeded its paired-MAD noise guard.

## Authority and decision

This real-hardware synthetic grouped-expert workload shows natural post-join headroom, but the critical-completion upper bound is not clear above paired noise and gating does not rescue feasibility in the four locked cells. Freeze this JoinStream thesis-promotion path in the current workspace; this is not a claim that JoinStream is universally impossible under real router traces. Do not promote it to an online policy, production-kernel result, or independently validated numerical-correctness result.

## Evidence

- Summary: `artifacts/joinstream_real_moe_tail/20260810_205953/summary.md`
- Analysis: `artifacts/joinstream_real_moe_tail/20260810_205953/analysis.json`
- Formal rows: `artifacts/joinstream_real_moe_tail/20260810_205953/formal_run.csv`
- Calibration/run lock: `artifacts/joinstream_real_moe_tail/20260810_205953/calibration.json`, `artifacts/joinstream_real_moe_tail/20260810_205953/run_lock.json`
- CUDA source: `docs/ideas/bcrd/experiments/joinstream_real_moe_tail_pilot.cu`

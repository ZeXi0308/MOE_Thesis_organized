# Repeated staged saving: small and delay-sensitive structural residual

Status: CPU_EXPLORATORY_SENSITIVITY. No new GPU execution. Each candidate evolves its own KV, queue, completion and load state from the same sealed step329 snapshot. Staging chooses a present-state most-output victim, schedules one preparation step, rechecks victim/target/funding, then protects the original recovery target through its first new output. All38 forced actions commit in these simulations; native preemptions remain separate. Original least/most/defer regression results are unchanged.

| Candidate | Whole-episode calls | Mean completion step |
|---|---:|---:|
| Immediate most-output | 1315 | 1168.9375 |
| Staged save-off | 1316 | 1169.0625 |
| Staged save-on, load1step | 1308 | 1165.8750 |
| Staged save-on, load2steps | 1310 | 1167.2813 |
| Staged save-on, load4steps | 1312 | 1170.0313 |
| Staged save-on, load8steps | 1316 | 1175.5000 |

Even under successful saving, adequate host capacity and no modeled store-induced wall cost, engine-call residual is small:7/5/3 fewer or1 more than immediate most-output. Mean completion step changes sign between load2 and4. This grid is not a calibrated delay envelope, a wall-time prediction, or a true Oracle bound. Delay need not be constant and transfer cost cannot be inferred from the grid.

The independent trajectories create roughly95–99K newly saved prefix tokens and160K loaded tokens; peak logical retained-prefix bytes are12.50–12.97GB, below16GiB but not proof the native host allocator can meet this budget including metadata, rounding, staging and pending work. Incremental prefixes remain cached until completion; eviction is not modeled. Native job ownership/content and store-fence timing are not simulated.

Verdict: repeated saving does not currently expose a large step-progress opportunity in this formulation. The remaining possible gain is reduced cost per call, which must compete with actual store/load and synchronization costs. This does not reject repeated saving as a system method or other operating regimes. Strongest baseline is immediate most-output, plus matched staged save-off to charge preparation/queue effects. Full-request Oracle remains unresolved; no method GO.

Next smallest experiment design must target full-cost residual, not another scheduling-threshold search: retain the same most-output selection and staging, compare measured mixed-recompute/transfer/host costs with the matched save-off arm. First extend the already qualified single-event adapter to repeated action-specific state; keep B Qwen's currently running group exclusive. No new GPU matrix queued by this CPU study.

Canonical summary: summary_corrected.json. Original summary.json retained: its immediate-baseline rotation count incorrectly counted staging records (0); corrected from forced trace markers to38, with all call/completion values unchanged. model_source.py is the executed model snapshot. Generator: experiments/admission_capacity/compare_repeated_staged_model.py.

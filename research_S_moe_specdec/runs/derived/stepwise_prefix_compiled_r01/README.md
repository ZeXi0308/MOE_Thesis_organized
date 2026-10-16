# Fixed-trajectory single-step prefix diagnostic

Status: `development_diagnostic_computed`. This directory is separate from measured F3 policy comparisons.

For cap g, identify each nonterminal request's accepted prefix only from its actual offer, verified prefix and first rejection. Progress is one target token plus that identified prefix. A short fully accepted offer stays short for all larger caps. Unseen later prefixes remain unidentified.

The score is `C(actual_B, g, proposal_budget=7) / mean_request_progress(g)`. The pre-step replay uses the existing controller and only past observations. The perfect-prefix reference uses future information from the same recorded step. Full-action minima/gaps require all eight actions to be identified and cost-covered; restricted-subset minima are separately named.

`step_diagnostics.jsonl.gz` retains every eligible step's candidate scores and identification/coverage reasons. `run_summary.csv` and `diagnostic.json` retain raw run summaries and choice histograms. Censoring counts include plugin flags and independently recovered final client chunks.

- Development diagnostic on the executed fixed-gamma-7 trajectory, not independent held-out policy evaluation.
- The perfect-prefix reference uses future accepted prefixes from this same step. It is not implementable online.
- Cap changes can change subsequent tokens, proposals, KV state, batching and execution costs. Those changed trajectories are not simulated or verified.
- Cost comes from separate budget-7 complete synchronous host-step calibration cells or the existing within-range batch interpolation; it is not this counterfactual step's measured cost.
- A score is calibrated step cost divided by mean per-request progress, not measured effective TPOT, end-to-end policy performance or a theoretical bound.
- Do not convert mean scores or their gaps into an end-to-end speedup; never plot these as measured policy curves in F3.
- No extra draft is invented beyond the observed full offer. Accepted-prefix truncation is used only on identifiable, nonterminal request outputs.
- Replayed pre-step choices use the existing estimator and cost policy with a cold prior once per run; they are not actually executed adaptive choices. Unlogged warmup state is not reconstructed.
- Steps, requests and ranks share an execution trajectory and are not independent system repetitions. Raw run-level summaries are retained.

This execution uses only compiled r01 gamma7 observations and the compiled G7 calibration. The calibration has one run per cell and nonmonotone cap costs; fine action rankings are not statistically resolved. Two retained trajectory runs produce identical replay summaries under the same prefix history and cold prior. B1 has zero score gap because cap7 has both the lowest calibrated cost and weakly maximal identified progress in this table/trace; this is not evidence of global online optimality. Small gaps elsewhere do not rule out useful online information or complete-policy improvement.

The diagnostic PDF is `../../../figures/stepwise_prefix_compiled_r01/stepwise_prefix_diagnostic.pdf`. Raw two-run plot values and coverage are in `plot_run_values.csv` and `diagnostic_summary.json`; these are separate from F3 measured policy comparisons.

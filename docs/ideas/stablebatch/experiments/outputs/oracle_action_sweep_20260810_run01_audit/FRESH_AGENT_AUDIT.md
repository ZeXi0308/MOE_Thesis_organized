# Fresh-agent audit: Oracle action sweep

**Verdict:** `WARN`, `0 P0 / 1 P1`, same-family provisional.

The sealed numerical result is internally valid. Independent reconstruction of 240 cells and 1,920 actions reproduces the abstaining-oracle reward `37`, unprotected distance `43`, recovery `37/43`, budget-matched expectations `0.309375` and `19.5`, advantages `36.690625` and `17.5`, and all four frozen strong-signal checks.

The single P1 is **partial pre-freeze same-cell outcome exposure**: rank-0 and one balanced shuffled-rank action per cell were known before this oracle protocol froze, and those exposed actions account for `20/37` of the final reward. This does not change the measured `37/43`; it changes the evidence label from confirmatory to an **exploratory retrospective upper bound**.

Authorized claim: this exact frozen single-contribution proxy action space has substantial hindsight action value.

Not authorized: online selector efficacy, model quality/accuracy, natural prevalence, serving value, EP/NCCL/RDMA, multi-GPU, or cross-model generalization. The all-M1 reference is an operational self-supervised proxy, not ground truth.

Full request/response trace: `.aris/traces/experiment-audit/2026-08-10_run10/`.

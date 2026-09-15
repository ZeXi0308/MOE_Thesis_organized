Overall verdict: **WARN**, but the frozen scientific result is interpretable and `STOP_PREACTION_STABLEBATCH` is robust. Under the rapid-experiment rubric: **P0=0, P1=0**. No identified defect can flip GO/STOP.

### A. Ground-truth/reference provenance — PASS

- Inputs are pinned WikiText documents, but there are no dataset labels. The target is the model-produced all-M1 route surface, explicitly marked “self-supervised proxy, not ground truth”: `selectability_decomposition_gate_v1.json:5-7,84-97`.
- Reward is route-membership agreement with that model proxy, not task quality: `selectability_policy.py:47-63`.
- Impact: supports causal route-surface claims only; not model quality, accuracy, or human preference.

### B. Score normalization — PASS

- Raw signed reward is `recovered - harmed`: `selectability_policy.py:47-63`.
- RecoveredOracleGap is the preregistered comparison `(R_selector - R_shuffle)/(R_oracle - R_shuffle)`: `selectability_policy.py:634-644`.
- Raw totals are fully reported: Oracle `57`, shuffle `-4`, static `-7`, online `-7`: `summary.json:1-100,102-194,196-297,299-395`.
- The diagnostic uniform expectation is separately reported and does not replace the deterministic shuffle: `summary.json:6609-6612`.
- Impact: no own-output maximum normalization or hidden score inflation.

### C. Result existence and exactness — WARN

- Formal status is COMPLETE, scientifically eligible, verdict STOP: `RUN_STATUS.json:2-6`.
- Manifest binds the selector lock, raw ledger, summary, environment, runtime, and status: `MANIFEST.json:3-51`.
- Independent recompute reports `mismatch_fields=[]`, exact ledger/lock hashes, and PASS: `INDEPENDENT_RECOMPUTE.json:6555-6616`.
- Exact formal result: Oracle `57/57/0`; shuffle `-4/5/9`; static `-7/7/14`; online `-7/7/14`; Oracle-minus-shuffle `61`; static and online gap `-3/61 = -0.0491803`; both LODO `0/16`: `summary.json:1-395,6556-6615`.
- Warning: tracker S1–S4 still say TODO and the plan checklist remains unchecked despite completion: `EXPERIMENT_TRACKER.md:6-9`; `EXPERIMENT_PLAN.md:115-119`. `IDEA_REPORT.md:45-55` still reflects the earlier state.
- Impact: narratives are stale, but no result is phantom or numerically mismatched.

### D. Dead code/checks — WARN

- Several frozen config fields are declarative rather than live switches: `selectability_decomposition_gate_v1.json:79-82,101-135,157-168`.
- Their intended behavior is nevertheless hard-enforced: feature projection at `selectability_policy.py:107-134`; exact unique B=33 at `selectability_policy.py:507-510`; reward/above-shuffle/LODO at `selectability_policy.py:686-717`; overlap and lock ordering at `run_selectability_decomposition_gate.py:119-145,352-382`.
- Impact: changing declarative booleans alone would not change behavior, but current frozen values and executed semantics agree. No verdict impact.

### E. Scope — PASS

- Fresh evaluation: 16 documents/requests × 15 layers = 240 cells, eight candidate ranks each = 1,920 actions; one OLMoE revision, one RTX 5090, one formal run.
- Calibration is a separate old 240-cell/1,920-action cohort: `selectability_decomposition_gate_v1.json:62-80`.
- Fresh text and exact-window overlap against calibration are zero: `freshness_closure.json:9-14`.
- Boundary is explicitly single-GPU, single-model, self-supervised, non-serving, non-EP, non-prevalence: `summary.json:6578`.
- Impact: claims must remain cohort- and stack-specific.

### F. Evaluation classification — PASS

**Classification:** `self_supervised_proxy on real GPU causal intervention replay`.

It is neither `real_gt`, `human_eval`, nor `simulation_only`. The intervention is physically executed on RTX 5090, but the behavioral reference is model-derived all-M1.

### G. Frozen-gate correctness — PASS

- Pre-run lock binds the exact required 19-file set: `FROZEN_SELECTABILITY_DECOMPOSITION_LOCK_V1.json:21-40`; validation is exact-set plus hashes: `run_selectability_decomposition_gate.py:41-87`.
- Native-only scanning precedes policy construction; `SELECTOR_LOCK.json` is written and hashed before `cell_results.jsonl` is created or any M1/M64 side-call executes: `run_selectability_decomposition_gate.py:330-402`.
- Persisted lock records `outcome_rows_existed_at_seal=false` and `result_path_existed_at_seal=false`: `SELECTOR_LOCK.json:3545-3548,21811`.
- R/U/A0–A7 surfaces are correct: `run_oracle_action_sweep.py:48-77,223-310`.
- Oracle, static, online, and shuffle all use exactly 33 unique cells and one rank per selected cell: `selectability_policy.py:368-409,412-463,507-510`; formal aggregates each report 33 actions: `summary.json:5,103,197,300`.
- Reward decomposition and ROG equations are correct: `selectability_policy.py:47-63,634-647`.
- LODO removes one complete victim/document, filters each frozen ranking, and refills exact B=33: `selectability_policy.py:577-585,667-717`. This matches the frozen plan: `EXPERIMENT_PLAN.md:73,78`.
- Direct mechanical re-derivation from all stored raw arm routes checked 2,160 U/action comparisons: zero route-set mismatches and zero reward mismatches.
- Runtime endpoints show an idle GPU initially and only the runner Python process at completion: `environment.json:13-27`; `runtime_final.json:3-19`.
- Independent recompute is sufficiently independent for aggregation/arithmetic but trusts stored changed-layer lists. The direct raw-route audit closes that limitation.
- Provenance caveat: remote `git_head` and `git_status_short` failed, so lineage is content-addressed/self-attested rather than commit-provenanced: `run_request.json:15-22`. Exact source/model/data hashes still close the executed result.

### Claim impact

- **Fresh oracle opportunity exists:** supported within this action space and cohort. Exact-B Oracle reward is `57` versus shuffle `-4`, recovers `57/84 = 67.86%`, with positive actions across 12 requests.
- **Frozen static selector:** fails. Reward `-7`, below shuffle, ROG `-0.04918`, LODO `0/16`.
- **Frozen online observable selector:** fails identically on decisive metrics: reward `-7`, below shuffle, ROG `-0.04918`, LODO `0/16`.
- **StableBatch as a paper mechanism:** fresh hindsight opportunity exists, but these two frozen pre-action mechanisms are rejected. It does not establish an implementable/paper-ready StableBatch system, serving benefit, general prevalence, cross-model transfer, EP behavior, or the validity/invalidity of separate canonical-shape mechanisms.


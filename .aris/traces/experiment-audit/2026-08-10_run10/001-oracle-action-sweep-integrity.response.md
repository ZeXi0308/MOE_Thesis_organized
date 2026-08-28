# Oracle action-sweep full integrity audit

## Overall verdict: WARN

`review_independence=same-family`  
`acceptance_status=provisional`

The numerical result is internally valid: independent reconstruction of all 240 rows and 1,920 actions reproduces `37/43 = 0.860465...`, both random baselines, MaxGate `-3`, shuffle `+3`, all thresholds, and the final verdict. No P0 was found. The only result-relevant P1 is partial pre-freeze outcome exposure on the same cells.

### A. Ground-truth provenance — PASS

Evidence:

- `run_oracle_action_sweep.py:62-77,282-308` defines all-M1 `R` and measures downstream route-set distance against it.
- `run_observable_selector_pilot.py:416-469` obtains M1/M64 references from model expert side-calls.
- `oracle_action_sweep_v1.json:26-40` explicitly says the reference is a self-supervised proxy, not ground truth.
- `observable_selector_pilot_v1.json:5-6,88-103` repeats that classification.
- `sealed_manifest.jsonl:1-32` contains input text and hashes, but no target labels.

Impact: `37/43` is recovery of route agreement relative to a model-derived all-M1 operational reference. It is not accuracy, quality, or external ground truth.

### B. Score computation and normalization — PASS

Evidence:

- `run_single_contribution_pilot.py:902-911` counts downstream layers whose top-k membership sets differ.
- `run_oracle_action_sweep.py:282-308` computes `reward = D_U - D_action`.
- `run_oracle_action_sweep.py:391-443` computes exact Fraction-based random expectations, abstention, and recovery.
- `run_oracle_action_sweep.py:444-469` applies the frozen thresholds.
- Thresholds are recorded at `oracle_action_sweep_v1.json:56-63`.

Independent reconstruction:

- Cells/actions: `240 / 1,920`
- `ΣD_U = 43`
- `Σall action rewards = 18`
- Uniform-rank total expectation: `18/8 = 9/4 = 2.25`
- Abstaining oracle: `37`
- Recovery: `37/43 = 0.8604651163`
- Budget-matched global random: `99/320 = 0.309375`
- Budget-matched conditional random: `39/2 = 19.5`
- Advantages: `11741/320 = 36.690625`, `35/2 = 17.5`
- MaxGate/shuffle closures: `-3 / +3`

All values match `summary.json:1-7,331-350,450-488`. No prediction-max or model-output-max normalization was found.

### C. Result existence and correspondence — WARN

Evidence:

- `RUN_STATUS.json:2-6` is COMPLETE and scientific-result eligible.
- `MANIFEST.json:3-51` binds all 12 non-manifest artifacts; every current size and SHA-256 matches.
- `static_bindings.json:6-32` closes all frozen code/config/source hashes.
- `cell_results.jsonl:1-240` contains 240 unique cells, each with all eight candidate actions.
- `summary.json:341-350,450-488` matches the raw-row reconstruction.

Warnings:

- `IDEA_REPORT_20260810_181815.md:5-6,16,90,156-160` predates the GPU run and still says no oracle GPU result. It is stale, not numerically false.
- `run_request.json:16-17` records a failed Git query, so provenance is content-hash-bound but not commit-bound.

Impact: the result files are valid, but the stale narrative must not be treated as the current result report.

### D. Dead code and actual execution — PASS

Evidence:

- Action enumeration/execution: `run_oracle_action_sweep.py:260-310`.
- Oracle selection and positive rerun: `run_oracle_action_sweep.py:311-349`.
- Classification: `run_oracle_action_sweep.py:379-469`.
- Main seals the action plan before opening result rows: `run_oracle_action_sweep.py:796-853`.
- Main classifies and verifies the manifest: `run_oracle_action_sweep.py:855-883`.
- `ACTION_SWEEP_LOCK.json:4-14,6016-6021` records eight ranks and pre-outcome sealing.
- All 33 positive rows contain confirmation objects; examples are `cell_results.jsonl:16,53,152`.
- The six bound unit tests at `test_oracle_action_sweep.py:71-142` pass.

P2 auditability limitation: each positive row stores only the repeated signature hash/status, not the repeated arm itself. Thus the second execution cannot be independently reconstructed from artifacts, although the bound code fails closed on mismatch.

### E. Leakage and controls — WARN

Within-run control passes:

- The oracle selects only after all ten arms execute: `run_oracle_action_sweep.py:260-315`.
- Workload, cell set, side-call schedule, action space, and arm order are sealed before result rows: `run_oracle_action_sweep.py:796-853`.
- Router identity, gate weights, applied raw hashes, and non-target contributions are checked at `run_oracle_action_sweep.py:170-215`.
- All raw rows satisfy these closures.

However, this is only partially pre-frozen:

- The source run completed at `observable_selector_20260810_run01/summary.json:14`.
- Oracle config froze later at `oracle_action_sweep_v1.json:4`.
- Before that freeze, rank-0 and one balanced shuffled-rank action per cell were already observed; see `observable_selector_pilot_v1.json:88-104` and `assignment_ledger.json:24985-25031`.
- The same 240 cells and prior `-3/+3` outcomes are explicitly reused at `oracle_action_sweep_v1.json:8-23`.
- The narrative explicitly uses those 35 opportunity cells to motivate the sweep at `IDEA_REPORT_20260810_181815.md:80-84`.

Independent reconstruction shows the previously exposed `{A0, shuffled-rank}` partial oracle already accounts for reward `20/37` on 17 positive cells.

Impact: the matrix remains a valid retrospective same-cell description, but the frozen “strong” classification is not a fully independent confirmatory result.

### F. Scope and evaluation type — PASS

Classification: `self_supervised_proxy`.

Exact scope:

- One RTX 5090, one OLMoE revision, BF16 eager.
- Sixteen distinct hashed document windows, offset 512.
- Fifteen layers per window, 240 same-cell prompt-forward cells.
- Artificial repeated-identical-row M1/M64 expert side-calls.
- Downstream top-k membership-set agreement against all-M1.
- Not online selection, dynamic batching, serving, quality evaluation, EP/NCCL/RDMA, multi-GPU, or prevalence.

Evidence: `oracle_action_sweep_v1.json:6-7,25-40`, `observable_selector_pilot_v1.json:5-6,87-104`, and `IDEA_REPORT_20260810_181815.md:8,80-91`.

## Findings

- P0: none.
- P1: Partial pre-freeze same-cell outcome exposure. It does not change the computed `37/43`, but it downgrades the conclusion from confirmatory evidence to an exploratory retrospective upper bound. No other result-changing P0/P1 was found.
- P2: Positive confirmation repeats are represented only by digest/status, not independently replayable evidence.
- P2: The current narrative is stale and the run lacks a valid Git commit identity, although exact file hashes close.

## Claim impact

1. Single-contribution hindsight action-space value: supported with qualification. The proxy surface shows `37/43` net recovery, 33 positive cells across eight victims, and `17.5` advantage over conditional random. However, 27/33 positive cells have tied best ranks and two cells have all eight ranks tied, so this supports action-space value more strongly than precise rank-selector necessity.
2. Online selector efficacy: unsupported. MaxGate-v1 remains weakened (`-3` versus shuffle `+3`), and the oracle uses future outcomes on reused cells.
3. Serving/system value: unsupported. There is no controller, natural serving batch, continuous decode, latency/throughput, EP, NCCL/RDMA, multi-GPU, or quality evidence.

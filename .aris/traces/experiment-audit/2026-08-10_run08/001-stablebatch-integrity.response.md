## Overall verdict: PASS

- Integrity status: `pass`
- Evaluation type: `self_supervised_proxy` — real-GPU mechanistic causal intervention with no external ground truth
- Reason code: `FROZEN_CAUSAL_TRACE_RECOMPUTES_SUPPORT`
- Acceptance status: `provisional_same_family`
- Frozen verdict independently recomputed: `SUPPORT`
- Workspace edits: none

### A. Ground-truth/proxy provenance — PASS

No dataset label, model-generated “ground truth,” or self-normalized benchmark score is used. Candidate enrichment uses native local M1/M64 raw-output distance and the native next-layer router margin; it does not use intervention route propagation, final logits, or greedy-token outcome. Candidate generation, selection, and only then target execution occur in that order at [runner lines 552–628](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/run_single_contribution_pilot.py:552) and [runner lines 1279–1296](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/run_single_contribution_pilot.py:1279).

The proxy/enrichment is explicitly frozen at [config lines 55–63](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/configs/single_contribution_pilot_v1.json:55). It limits prevalence/generalization claims but does not constitute target-outcome leakage.

### B. Score and decision integrity — PASS

Independent raw-file recomputation, without importing the runner, found zero mismatches:

- 16 workloads.
- 1,920 candidates = 16 victims × 15 layers × 8 top-k ranks. Every victim has 120 candidates; every layer has 128. All 1,920 have distinct identities and an M1/M64 raw hash difference.
- Every `gate_weighted_local_l2` equals `gate_weight × local_l2`.
- Every selection score equals `gate_weighted_local_l2 / (next_layer_topk_margin + 1e-6)`.
- Exact deterministic reselection reproduced all 32 rows byte-for-value, with 8 per band and exactly 2 per victim. Selection logic is at [runner lines 403–444](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/run_single_contribution_pilot.py:403); selected raw rows are [selected targets lines 1–32](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/selected_targets.jsonl:1).
- There are 24 zero-margin candidates. Their large ratios are transparent enrichment scores, not reported efficacy scores; examples occur at [selected target line 9](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/selected_targets.jsonl:9) and [line 17](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/selected_targets.jsonl:17).
- All 32 targets have local raw change and combine-boundary change; changed-element counts range 1,053–1,544.
- All 32 have exact three-repeat same-arm stability.
- Recomputed downstream membership changes occur in 12 targets across 8 victims:

  - Target-result lines `2, 4, 5, 6, 10, 11, 12, 13, 14, 15, 20, 25`.
  - Changed downstream layers respectively: `[1]`, `[6]`, `[11,15]`, `[10,11]`, `[6,12]`, `[6]`, `[13]`, `[9,11]`, `[14]`, `[14]`, `[9,10]`, `[15]`.
  - The remaining 20 targets have no downstream membership-set change.

- One independently recomputed greedy-token flip occurs at [target result line 9](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/target_results.jsonl:9), `337 → 608`; it does not rely on a route-membership flip.
- Frozen thresholds are ≥4 route targets and ≥2 victims. Recomputed `12` and `8` therefore yield `SUPPORT`, matching [summary lines 5–24](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/summary.json:5).

### C. Causal isolation — PASS

The implementation explicitly supplies full input IDs, an all-ones attention mask, disables cache, and captures the target hidden state and all victim router layers at [runner lines 488–531](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/run_single_contribution_pilot.py:488) and [runner lines 759–817](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/run_single_contribution_pilot.py:759).

The copied OLMoE path preserves native flattening, FP32 softmax/top-k, expert-ID traversal, `(rank, token)` dispatch, BF16 routing weights, and `index_add_`; replacement occurs once before the weight multiplication at [runner lines 655–750](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/run_single_contribution_pilot.py:655). It matches the hash-bound Transformers implementation at [native OLMoE lines 585–621](/Users/leandrozhao/Desktop/毕设论文资料/.venv/lib/python3.9/site-packages/transformers/models/olmoe/modeling_olmoe.py:585).

Across every target-result line 1–32, I independently verified:

- Input-ID hash from the recorded 16 `int64` IDs and attention-mask hash from sixteen `int64` ones.
- Target hidden/router hash against selection.
- Target top-k identity and routing-weight hashes across no-op, M1, and M64.
- All victim router hashes through the intervention layer.
- Native raw target hash equality across arms.
- Non-target contribution hash equality against both the no-op and every arm.
- Unique pair count `1` and source-level exactly-once routing-weight application.
- Native self-replacement equality for full target MoE output, all recorded routes, and final logits.
- Exact full nested trace/observation stability across all three repeats per arm.
- Membership comparison as sets, only from `target_layer + 1`.
- No propagated route was accepted before the target MoE combine hash changed.

The fail-closed combine condition is implemented at [runner lines 1096–1109](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/run_single_contribution_pilot.py:1096). All 32 raw target records are at [target results lines 1–32](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/target_results.jsonl:1).

### D. Provenance consistency — PASS

- Run03 used V1 and failed with `scientific_result_eligible: false` before producing acceptance/selection evidence: [run03 request lines 2–25](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/acceptance_20260810_run03/run_request.json:2), [failure lines 2–7](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/acceptance_20260810_run03/FAILURE.json:2).
- V2 declares only the missing candidate-gate `torch.inference_mode` wrapper at [V2 lock lines 5–11](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/configs/FROZEN_PILOT_LOCK_V2.json:5). Independently deleting current line 579 and de-indenting line 580 reconstructed V1’s exact runner SHA-256 `92ea6e…ab680`. Config, tests, and workload-manifest hashes are unchanged.
- Passing acceptance binds V2 and remains explicitly non-scientific at [run04 request lines 2–25](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/acceptance_20260810_run04/run_request.json:2), [acceptance lines 2–44](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/acceptance_20260810_run04/REAL_GPU_ACCEPTANCE.json:2), and [status lines 2–5](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/acceptance_20260810_run04/RUN_STATUS.json:2).
- Formal bindings record RTX 5090 UUID, driver `595.71.05`, Torch `2.8.0+cu128`, Transformers `4.57.6`, cuBLASLt version/path/hash, and OLMoE source hash at [environment lines 7–40](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/environment.json:7). Final runtime maps exactly that cuBLASLt path at [runtime lines 11–19](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/runtime_final.json:11).
- All eight frozen model files were independently hashed locally. They match both the config and the claimed official revision; the three LFS SHA-256 values and revision SHA also match [Hugging Face revision metadata](https://huggingface.co/api/models/allenai/OLMoE-1B-7B-0924/revision/6d84c48581ece794365f2b8e9cfb043c68ade9c5?blobs=true).
- All 32 sealed text hashes were recomputed. The first 16 workloads were independently retokenized with the hash-matched `GPTNeoXTokenizerFast`; all token IDs and canonical hashes match [workloads lines 1–16](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/workloads.jsonl:1).
- Every size and SHA-256 in both passing manifests matches the current file. Formal manifest has 10 covered files plus `MANIFEST.json` and `RUN_STATUS.json`, matching the actual 12-file directory: [formal manifest lines 3–43](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/MANIFEST.json:3).
- Timestamp/write ordering is runtime → summary → manifest → eligible status, matching source order at [runner lines 1301–1327](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/run_single_contribution_pilot.py:1301).
- Reran all five unit tests under `.venv`; all passed.
- The remote bundle was not a Git checkout, so `git_head` is unavailable at [formal request lines 20–21](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/outputs/single_contribution_20260810_run01/run_request.json:20). This does not break byte-level provenance because runner/config/lock/tests/data/model/source/library are hash-bound; it must not be presented as Git-commit provenance.

### E. Scope and dead code — PASS

Formal scope is exactly:

- 16 workloads.
- 1,920 candidates.
- 32 selected targets.
- 3 local side-call repeats for each of M1 and M64.
- 3 full-forward repeats per arm, or 192 intervention forwards total.
- One RTX 5090, OLMoE revision `6d84c…e9c5`, BF16, eager prompt forward, M1/M64 execution-shape side-calls.

No material metric or decision function is dead; an AST call-path audit found no unreferenced top-level evaluation function. Several config booleans at [config lines 62–84](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/stablebatch/experiments/configs/single_contribution_pilot_v1.json:62) are descriptive rather than runtime switches, but their corresponding invariants are hard-enforced in code, so this does not affect the result.

The permitted claim ceiling is only frozen, margin-enriched single-contribution downstream propagation. There is no evidence for a particular kernel algorithm, serving, expert parallelism, latency, StableBatch policy effectiveness, or general batch invariance.

### F. Alternative explanations — PASS

None of the listed alternatives reverses the narrow frozen conclusion:

- Fixed M1-before-M64 ordering can affect external robustness, but both local shapes are already exercised during the sweep, each selected local output is stable across three repeats, and the full-forward arms have identical execution shape. Their explicit first difference is the injected raw target hash.
- cuBLAS heuristic state may explain *how* M1 and M64 obtained different raw outputs. It cannot explain away the recorded raw-value-to-combine-to-downstream chain, and no kernel-algorithm claim is allowed.
- Offline next-layer margin enrichment can increase the observed 12/32 incidence. It cannot fabricate the 12 closed propagation traces; therefore `12/32` must not be reported as an unbiased population rate.
- Every positive route trace first changes the target MoE combine output, while native raw output, target route/weights, upstream victim routers, and non-target contribution hashes remain fixed.

### Claim impact

- C1 — artifact/integrity existence: **supported**.
- C2 — narrow execution-shape-M → one raw contribution → downstream victim top-k propagation: **supported provisionally**, specifically for 12/32 frozen, band-balanced, local-delta/margin-enriched targets across 8/16 victims on this exact setup.
- C3 — StableBatch policy, serving, EP, latency, kernel mechanism, or general batch invariance: **unsupported**.

Required before using C2: no new experiment is required for provisional use. The wording must retain the exact hardware/model/BF16/prompt-forward/M1-vs-M64/enriched-target boundary and must not interpret 12/32 as prevalence.

Optional later work: reversed/interleaved arm order, fresh-process reruns, additional GPUs/revisions, raw-vector retention, and separate serving/EP/policy experiments. A genuinely independent or cross-family audit is needed only to upgrade `provisional_same_family` to non-provisional acceptance.

```json
{
  "overall_verdict": "PASS",
  "integrity_status": "pass",
  "evaluation_type": "self_supervised_proxy_real_gpu_mechanistic_causal_intervention",
  "reason_code": "FROZEN_CAUSAL_TRACE_RECOMPUTES_SUPPORT",
  "acceptance_status": "provisional_same_family",
  "recomputed": {
    "workloads": 16,
    "candidates": 1920,
    "selected_targets": 32,
    "targets_per_band": [8, 8, 8, 8],
    "max_and_observed_targets_per_victim": 2,
    "repeats_per_arm": 3,
    "local_changed_targets": 32,
    "combine_changed_targets": 32,
    "reproducible_route_targets": 12,
    "distinct_route_victims": 8,
    "greedy_token_flips": 1,
    "verdict": "SUPPORT"
  },
  "claims": {
    "C1": "supported",
    "C2": "supported_provisionally_with_frozen_scope_and_enrichment_qualifier",
    "C3": "unsupported"
  },
  "required_before_C2": [],
  "required_to_lift_provisional_status": [
    "independent_or_cross_family_acceptance"
  ]
}
```

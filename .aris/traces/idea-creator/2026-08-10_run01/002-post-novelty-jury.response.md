{
  "review_independence": "same-family",
  "acceptance_status": "provisional",
  "final_ranked_top3": [
    {
      "rank": 1,
      "id": "C09",
      "role": "conditional_standalone_candidate",
      "decision": "proceed_with_caution",
      "strongest_objection": "The remaining delta is narrow: a cheap MoE-row predictor assembled from areas already covered by bit-exact inference reconstruction, bit-accurate MMA modeling, floating-point certification and routing-aware dispatch. An empirical classifier is not a certificate, and even the perfect frozen oracle saves only 4.7621 percent of calls before predictor overhead.",
      "likely_failure_mode": "Safety is partner-dependent or slot-dependent; alternatively, a conservative predictor reaches zero held-out false admissions only by abstaining on nearly everything, producing no latency or goodput advantage over global batch-invariant execution."
    },
    {
      "rank": 2,
      "id": "C02",
      "role": "supporting_executor_only",
      "decision": "abandon_as_standalone",
      "strongest_objection": "Its structure remains verified speculation plus selective repair, closely neighboring LLM-42. Expert-row granularity and partial commit are implementation specializations, and the entire mechanism depends on C09 providing a pre-execution rule.",
      "likely_failure_mode": "C09 fails or has negligible coverage; otherwise certificate evaluation, M2 execution, one-row M1 rescue and stitching cost at least as much as two isolated M1 calls or safe-only pooling."
    },
    {
      "rank": 3,
      "id": "C04",
      "role": "infrastructure_only",
      "decision": "abandon_as_thesis_novelty",
      "strongest_objection": "CUTLASS grouped kernels already persistently schedule multiple GEMMs, while TBIK and vLLM batch-invariant kernels already control reduction order for bitwise invariance. Preserving one legacy M1 implementation adds a qualification contract more readily than a new research mechanism.",
      "likely_failure_mode": "Preserving the exact serial-M1 reduction and epilogue removes the parallelism needed for material speedup; relaxing those constraints restores speed but breaks raw-BF16 equality. Even a positive prototype may remain an engineering result without novelty beyond existing grouped kernels."
    }
  ],
  "idea_roles": {
    "standalone": [],
    "conditional_standalone": [{"id":"C09","condition":"Survive fixed-slot partner permutation, later slot testing, document-disjoint zero-false-admission evaluation and end-to-end comparison against shape-only admission and global vLLM batch invariance."}],
    "supporting": [{"id":"C02","condition":"Retain only as C09's executor and ablation if C09 first demonstrates nontrivial safe coverage."}],
    "infrastructure": [{"id":"C04","condition":"Use as a baseline or fallback; do not present as headline novelty without a non-obvious design and strong advantage over CUTLASS and vLLM-invariant baselines."}],
    "current_standalone_verdict": "No idea is presently validated as a standalone contribution."
  },
  "immediate_pilot": {
    "count": 1,
    "id": "C09_FIXED_SLOT_PARTNER_PERMUTATION",
    "semantic_branch": "raw_BF16_exactness",
    "hypothesis": "With focal slot fixed, an anchor row's repeat-stable raw-BF16 exact/nonexact label relative to isolated M1 is invariant to same-cell partner identity.",
    "success_gate": "All identities and repeats close, sufficient exact anchors are exercised across at least four cells, and there are zero reproducible label changes across partners. This only authorizes the later slot test; it does not yet validate C09.",
    "kill_gate": "Any anchor changes its repeat-stable label for a different partner while focal slot, anchor input and M1 reference remain identical. Kill row-local C09 immediately and investigate a pair relation without tuning the gate."
  },
  "partner_slot_protocol": {
    "decision": "split_sequentially",
    "stage_1": "Test partner identity with focal slot fixed.",
    "stage_2": "Only if stage 1 passes, freeze a separate slot-permutation protocol holding pair identity fixed.",
    "rationale": "A stage-1 failure already kills row-local C09 and makes slot work unnecessary; combining factors risks causal misattribution without a larger factorial design.",
    "claim_boundary": "Passing stage 1 establishes partner invariance only at the tested slot."
  }
}

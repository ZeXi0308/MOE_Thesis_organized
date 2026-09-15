{
  "review_independence": "same-family",
  "acceptance_status": "provisional",
  "evidence_boundary": {
    "observed": [
      "M=1 versus M=64 changed raw BF16 expert contributions and post-combine values in the frozen enriched pilot.",
      "Downstream route-membership changes occurred in 12 of 32 enriched targets across 8 victims; one stable greedy-token flip was also observed.",
      "MaxGate-v1 produced total signed reward -3 versus +3 for one seed-frozen matched balanced shuffle, so MaxGate-v1 is NO-GO."
    ],
    "inferred": [
      "Expert execution shape is a legitimate MoE-specific causal control surface.",
      "Propagation is sparse, so blanket stabilization is unlikely to offer an attractive cost-benefit point.",
      "Current-layer gate-weight rank is misaligned with downstream action value."
    ],
    "hypothesized": [
      "A small set of canonical expert shapes can eliminate batch-context route divergence.",
      "Shape lanes can recover meaningful GPU efficiency relative to serial M1 and global batch-invariant execution.",
      "Execution-plan signatures or layer/shape strata provide transferable selective-stability information."
    ],
    "not_established": [
      "Natural continuous-decode incidence",
      "Request-quality improvement",
      "Serving latency or throughput benefit",
      "Cross-model or cross-GPU generality",
      "EP, NCCL, RDMA, multi-GPU, or production benefit",
      "All-M1 as model-quality ground truth"
    ]
  },
  "per_candidate": {
    "D01": {"novelty_0_10": 4.0, "evidence_fit_0_10": 7.0, "system_depth_0_10": 5.0, "two_week_falsifiability_0_10": 9.0, "collision": "vLLM Batch Invariance, fixed-shape deterministic kernels, grouped expert execution, and the previously reviewed C04 executor", "main_objection": "Fixed-M padding is an executor primitive rather than a headline contribution, and fixed M is not yet proven sufficient for row invariance.", "verdict": "supporting"},
    "D02": {"novelty_0_10": 1.0, "evidence_fit_0_10": 6.0, "system_depth_0_10": 5.5, "two_week_falsifiability_0_10": 8.5, "collision": "Direct collision with vLLM deterministic batch-invariant kernels and fixed-reduction-tree work", "main_objection": "A fixed K tree is the generic deterministic-kernel territory excluded by the novelty constraints.", "verdict": "kill"},
    "D03": {"novelty_0_10": 4.5, "evidence_fit_0_10": 5.0, "system_depth_0_10": 7.0, "two_week_falsifiability_0_10": 8.0, "collision": "vLLM modular kernel dispatch, RaMP, DA-MoE, and earlier C09", "main_objection": "Plan identity is not an equality certificate and may remain input-dependent.", "verdict": "supporting"},
    "D04": {"novelty_0_10": 1.0, "evidence_fit_0_10": 4.5, "system_depth_0_10": 6.5, "two_week_falsifiability_0_10": 7.5, "collision": "Direct collision with LLM-42 and MarginGate", "main_objection": "Next-router granularity is a specialization and misses value-only divergence.", "verdict": "kill"},
    "D05": {"novelty_0_10": 3.0, "evidence_fit_0_10": 9.0, "system_depth_0_10": 4.0, "two_week_falsifiability_0_10": 9.5, "collision": "Generic oracle analysis and policy distillation", "main_objection": "Correct experimental gate after MaxGate-v1, but not itself a paper contribution.", "verdict": "supporting"},
    "D06": {"novelty_0_10": 2.0, "evidence_fit_0_10": 3.5, "system_depth_0_10": 6.0, "two_week_falsifiability_0_10": 6.5, "collision": "MarginGate-style selective determinism and generic calibrated-risk controllers", "main_objection": "No natural risk distribution, transferable predictor, or MoE-specific action value exists yet.", "verdict": "kill"},
    "D07": {"novelty_0_10": 1.5, "evidence_fit_0_10": 2.5, "system_depth_0_10": 6.5, "two_week_falsifiability_0_10": 6.0, "collision": "Generic shadow replay plus contextual bandit", "main_objection": "No observed regime drift or cheap shadow label.", "verdict": "kill"},
    "D08": {"novelty_0_10": 3.0, "evidence_fit_0_10": 4.0, "system_depth_0_10": 6.5, "two_week_falsifiability_0_10": 8.0, "collision": "Generic quota allocation plus RaMP/DA-MoE distribution-aware scheduling", "main_objection": "No transferable stratum reward-cost heterogeneity is observed.", "verdict": "deprioritize"},
    "D09": {"novelty_0_10": 0.5, "evidence_fit_0_10": 5.5, "system_depth_0_10": 7.0, "two_week_falsifiability_0_10": 7.5, "collision": "Near-duplicate of D04 and direct LLM-42/MarginGate collision", "main_objection": "Route-barrier placement does not create a distinct method.", "verdict": "kill"},
    "D10": {"novelty_0_10": 6.5, "evidence_fit_0_10": 7.0, "system_depth_0_10": 8.5, "two_week_falsifiability_0_10": 8.5, "collision": "Touches RaMP, DA-MoE and vLLM Batch Invariance, but differs through a numerical-stability-constrained shape-lane scheduling contract", "main_objection": "It collapses into padding/queueing engineering unless fixed shape removes divergence and beats global invariance and serial M1 on a measured Pareto frontier.", "verdict": "advance"}
  },
  "semantic_overlaps": [
    {"members": ["D04", "D09"], "assessment": "Same verify-and-repair family, subsumed at headline level by LLM-42 and MarginGate."},
    {"members": ["D01", "D02", "D10"], "assessment": "Share fixed-shape arithmetic; D01 is an executor, D02 is killed, D10 adds the paper-level scheduling contract."},
    {"members": ["D03", "D08", "D10"], "assessment": "All dispatch on execution or routing state; numerical equivalence must be the hard objective."},
    {"members": ["D05", "D06", "D07", "D08"], "assessment": "Generic decision layer; D05 is useful as a falsification protocol, not a contribution."}
  ],
  "ranked_top3": [
    {"rank": 1, "candidate": "D10", "role": "sole headline method candidate", "reason": "It changes feasible expert shapes before execution, avoids replay, and directly targets the observed causal variable."},
    {"rank": 2, "candidate": "D03", "role": "conditional safety guard", "reason": "System structure exists, but held-out sufficiency is unproved and it cannot be the headline."},
    {"rank": 3, "candidate": "D01", "role": "executor primitive and baseline", "reason": "Cheapest fixed-shape correctness test but insufficient novelty alone."}
  ],
  "strongest_surviving_problem_statement": "Can a continuous-batching MoE scheduler constrain routed expert rows to a small set of deterministic, numerically stable execution-shape lanes, with bounded padding and deadline delay, so that each request preserves its route trajectory across co-request batch contexts while recovering significant goodput relative to serial M1 and vLLM global Batch Invariance?",
  "recommended_single_next_experiment": {
    "exact_hypothesis": "For held-out natural MoE arrival traces, frozen C=8 expert-row lanes with deterministic row placement and dummy masking yield zero same-row raw-BF16 and downstream-route divergence across companion, slot, and schedule perturbations, while reducing expert GPU time at bounded queue delay.",
    "cheapest_protocol": "One RTX 5090, frozen OLMoE BF16 eager, new document-disjoint decode arrivals, four arms native variable-M, serial M1, vLLM Batch Invariance, and D10 C=8; three repeats and frozen companion/slot permutations; no learned selector or post-result C tuning.",
    "frozen_go_gate": "Zero raw-hash and downstream-route mismatches; at least 20% less expert GPU time than serial M1; at least 10% less than vLLM Batch Invariance at the same zero-divergence criterion; at most 5% added p99 token-step delay.",
    "frozen_kill_gate": "Any permitted context changes a focal-row raw hash or downstream route. If correctness passes but a cost or delay threshold fails, kill D10 as headline and retain D01 only as a baseline. Do not tune C or deadlines on the same evidence.",
    "expected_gpu_time": "2-4 RTX 5090 GPU-hours after the replay harness is ready",
    "outcome_portfolio_effect": {"go": "Advance D10 as the sole method.", "correctness_kill": "Kill D01/D10 fixed-shape family; no current headline method remains.", "cost_kill": "Kill lane scheduling as a systems contribution and pivot instead of tuning."}
  },
  "direct_answer_to_user": "YES_METHOD_CANDIDATE"
}

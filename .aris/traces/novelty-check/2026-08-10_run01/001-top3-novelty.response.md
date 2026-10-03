{
  "review_independence": "same-family",
  "acceptance_status": "provisional",
  "per_idea": {
    "C09": {
      "core_claims": [
        "Batch-shape safety is sufficiently row-local to survive co-row and slot permutations.",
        "A cheap pre-execution rule using routed-row and execution-shape features can safely predict raw-BF16 equality between native M>1 execution and canonical isolated M1.",
        "Selective admission can recover enough performance to beat global batch-invariant execution without admitting any unsafe row."
      ],
      "closest_prior_work": [
        {"title":"Bit-Exact AI Inference Verification Without Performance Tradeoffs","url":"https://arxiv.org/abs/2606.00279","overlap":"Models exact GPU inference outputs from operands and execution details and identifies dynamic batch size as a numerically relevant condition.","difference":"Uses expensive bit-exact emulation rather than a cheap MoE row-level pre-execution admission rule."},
        {"title":"MMA-Sim: Bit-Accurate Reference Model of Tensor Cores and Matrix Cores","url":"https://arxiv.org/abs/2511.10909","overlap":"Predicts bit-exact matrix-accelerator arithmetic across GPU architectures.","difference":"Provides a full arithmetic simulator, not a selective, low-overhead runtime certificate."},
        {"title":"Lipschitz-Based Robustness Certification Under Floating-Point Execution","url":"https://arxiv.org/abs/2603.13334","overlap":"Develops sound pre-execution certification under floating-point error models.","difference":"Certifies robustness properties of feed-forward networks, not native M>1 versus M1 bit equality for MoE expert rows."},
        {"title":"RaMP: Runtime-Aware Megakernel Polymorphism for Mixture-of-Experts","url":"https://arxiv.org/abs/2604.26039","overlap":"Uses runtime MoE routing state to choose kernel configurations.","difference":"Optimizes performance without a numerical-equivalence safety constraint."},
        {"title":"vLLM Batch Invariance","url":"https://docs.vllm.ai/en/stable/features/batch_invariance/","overlap":"Already supplies global batch-invariant execution for tested MoE models.","difference":"Pays a global deterministic-kernel cost instead of selectively admitting native fast rows."}
      ],
      "novelty_score_0_10": 5.0,
      "recommendation": "PROCEED_WITH_CAUTION",
      "genuinely_new_delta": "No inspected source directly provides a cheap, pre-execution, MoE-expert-row rule that predicts whether a native batched BF16 kernel will be bitwise equal to the isolated M1 reference without executing either reference. That narrow intersection could be new if partner invariance holds and the rule is genuinely sound or has a rigorously declared nonzero risk bound.",
      "strongest_collision": "Bit-exact emulation already establishes that exact outputs can be predicted from operands and stack details, while floating-point certification supplies generic sound-bound machinery and RaMP supplies input-dependent MoE kernel dispatch. Merely applying a classifier or generic error bound to MoE is therefore not a new method.",
      "required_positioning": "Do not call a held-out zero-error classifier a certificate. Position it as an empirical fail-closed predictor unless soundness is proved. The paper-worthy claim must be a MoE-specific, cheap and conservative characterization of native-kernel exactness that materially outperforms global batch-invariant kernels. Run partner and slot permutations first; the current 2,768/32,234 repeat-stable rows and near-IID pair count do not establish row-locality."
    },
    "C02": {
      "core_claims": [
        "A pre-execution certificate can authorize row-level partial commit from an M2 expert call.",
        "Only the uncertified partner needs isolated M1 rescue, with exact identity-preserving stitching.",
        "M2 plus selective M1 rescue and stitching is faster than two isolated M1 executions while remaining raw-BF16 exact."
      ],
      "closest_prior_work": [
        {"title":"LLM-42: Enabling Determinism in LLM Inference with Verified Speculation","url":"https://arxiv.org/abs/2601.17768","overlap":"Runs a nondeterministic fast path, verifies under a fixed schedule, commits valid work and rolls back invalid work.","difference":"Operates at token/state level with post-execution verification rather than pre-certified expert-row partial commit."},
        {"title":"MarginGate: Sparse Margin-Triggered Verification for Batch-Invariant LLM Inference","url":"https://arxiv.org/abs/2605.30218","overlap":"Selectively invokes deterministic verification and repairs only affected K/V state.","difference":"Uses post-fast-path logit margins and token-level repair rather than raw expert-row equality."},
        {"title":"Bit-Exact AI Inference Verification Without Performance Tradeoffs","url":"https://arxiv.org/abs/2606.00279","overlap":"Supports exact recomputation and comparison when execution details are known.","difference":"Does not propose asymmetric M2 row commit with one-row M1 rescue."},
        {"title":"vLLM Batch Invariance","url":"https://docs.vllm.ai/en/stable/features/batch_invariance/","overlap":"Provides an exact global alternative that selective rescue must beat.","difference":"Avoids speculation and repair by using invariant kernels throughout."}
      ],
      "novelty_score_0_10": 3.5,
      "recommendation": "ABANDON",
      "genuinely_new_delta": "The only credible delta is expert-row-granular partial commit where safety is known before M2 and only the uncertified co-row is recomputed. This is a potentially useful execution tactic, but not a strong standalone research method.",
      "strongest_collision": "Its structure is verified speculation plus selective repair, already established by LLM-42 and MarginGate. Moving commit and repair from tokens or K/V columns to MoE expert rows is primarily an application-level specialization unless it exposes a new optimality result or a surprising hardware regime.",
      "required_positioning": "Do not headline C02 independently. If C09 succeeds, treat C02 as one downstream executor or ablation. It must beat two-M1, certified-safe-only packing, global batch-invariant execution and ordinary verify/rollback after all certificate, M2, rescue and stitching costs. The frozen 4.7621% perfect-oracle call-reduction ceiling makes a positive end-to-end result unlikely without a very favorable M2-versus-M1 latency ratio."
    },
    "C04": {
      "core_claims": [
        "Independent logical M1 expert GEMMs can share one grouped or persistent launch.",
        "Each row can retain the same K traversal, accumulator, activation and cast boundaries as serial M1.",
        "Launch scheduling and weight residency can be shared while preserving raw-bit equality and improving throughput."
      ],
      "closest_prior_work": [
        {"title":"NVIDIA CUTLASS Grouped Kernel Schedulers","url":"https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html","overlap":"A persistent kernel already launches and schedules tiles from multiple GEMM problems in one CUDA launch.","difference":"Does not promise equality to a separately designated legacy M1 implementation."},
        {"title":"vLLM Batch-Invariant BMM Implementation","url":"https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/batch_invariant/","overlap":"Each batch/tile program accumulates along K in a fixed order specifically to preserve batch invariance.","difference":"Is a general batch-invariant BMM rather than a named MoE grouped-M1 executor."},
        {"title":"Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch","url":"https://arxiv.org/abs/2511.17826","overlap":"Fixes reduction structure to guarantee bitwise-identical matrix and reduction outputs.","difference":"Targets TP-size invariance rather than grouped independent M1 expert calls."},
        {"title":"ExpertPlex","url":"https://arxiv.org/abs/2607.18002","overlap":"Uses adaptive persistent expert kernels with tile-level scheduling and isolation.","difference":"Optimizes goodput without preserving a legacy serial-M1 arithmetic identity."},
        {"title":"TMA-Adaptive FP8 Grouped GEMM","url":"https://arxiv.org/abs/2508.16584","overlap":"Optimizes variable-sized grouped GEMMs and reports numerical equivalence for valid data.","difference":"Does not establish raw-bit equality to a canonical BF16 M1 path."}
      ],
      "novelty_score_0_10": 1.5,
      "recommendation": "ABANDON",
      "genuinely_new_delta": "A carefully engineered MoE kernel might newly demonstrate positive throughput while reproducing the exact bits of a particular legacy M1 implementation and exploiting same-expert weight residency. That would be an implementation result, not a new grouped-kernel principle.",
      "strongest_collision": "CUTLASS already provides persistent grouped execution of multiple GEMMs, and vLLM already implements per-batch-tile fixed-K accumulation for batch invariance. C04 is essentially the conjunction of these existing mechanisms plus a qualification test.",
      "required_positioning": "Use C04 as infrastructure or a baseline, not the thesis novelty claim. A contribution would require a non-obvious kernel design forced by the exact legacy-M1 contract and strong measured gains over CUTLASS grouped GEMM, vLLM batch-invariant kernels and serial M1. Merely preserving accumulators and output slots is a correctness specification."
    }
  },
  "overall_recommendation": {
    "decision": "PROCEED_WITH_CAUTION",
    "scope": "Retain only C09 for a short falsification cycle. Fold C02 into a conditional executor experiment if C09 survives, and demote C04 to implementation infrastructure.",
    "reason": "C09 has a narrow unoccupied intersection, but currently lacks its defining certificate and even the partner-invariance premise. C02 is generic verified speculation specialized to rows. C04 is standard grouped/persistent scheduling combined with standard fixed-reduction arithmetic.",
    "mandatory_kill_order": [
      "Partner and slot permutation: abandon row-local C09 if safety changes with co-row identity or ordering.",
      "Construct a frozen pre-execution rule on document-disjoint data; any false admission kills raw-exact positioning.",
      "Compare with exact emulation, shape-only rules and global vLLM batch invariance.",
      "Measure complete executor latency. If C02 or C09 cannot beat the exact baseline after all overhead, retain only the characterization finding.",
      "Do not claim serving, continuous-batching or EP novelty from the current single-RTX-5090 prompt-forward expert-stage evidence."
    ]
  },
  "missing_search_risks": [
    "The audit did not exhaust patents, proprietary cuBLASLt/TensorRT implementations or unpublished vendor techniques; C04 is especially exposed to this risk.",
    "The broad pre-2024 literature on correctly rounded dot products, interval arithmetic, approximate-computing guards and selective execution was not exhaustively searched. It could further weaken C09 if its certificate is generic.",
    "Several closest 2026 sources are recent preprints or beta documentation and may change after review.",
    "No direct source collision was found for the exact C09 intersection, but absence of a keyword hit is not proof of novelty.",
    "A code-level comparison against current vLLM MoE batch-invariant kernels, CUTLASS grouped kernels and hardware-specific M1/M2 dispatch behavior remains missing.",
    "The current evidence establishes repeat stability and an oracle call-count bound only; it does not establish partner-invariant labels, a pre-execution predictor, a sound certificate, kernel speedup or end-to-end goodput."
  ]
}

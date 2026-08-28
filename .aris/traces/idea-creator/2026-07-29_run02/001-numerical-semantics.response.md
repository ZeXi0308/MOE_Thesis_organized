{
  "recovery_note": "The original JSON was returned before context compaction; this is a field-complete reconstruction from the retained turn summary, not a byte-identical response.",
  "shard_id": "lens:numerical-semantics",
  "candidates": [
    {"title":"MarginLock-MoE","hypothesis":"Router margins may certify when a fast numerical path cannot change top-k routing.","minimum_pilot":"256 prompts x 128 decode, 1-2 GPU-hours.","key_risk":"Bounds may be loose and MarginGate/LLM-42 may directly subsume it.","dedup_key":"router-margin-certified-numerical-fastpath"},
    {"title":"PermuteExact","hypothesis":"Some grouped-GEMM paths may violate within-expert token-permutation equivalence.","minimum_pilot":"10k routed rows, 0.5-1.5 GPU-hours.","key_risk":"Kernels may already be row-equivalent; likely audit-only.","dedup_key":"within-expert-permutation-semantic-contract"},
    {"title":"ReplicaSem","hypothesis":"Numerical fingerprints may make nominally identical expert replicas semantically non-interchangeable.","minimum_pilot":"1-2 GPU-hours local qualification; multi-GPU formal evidence.","key_risk":"Generic heterogeneous inference nondeterminism.","dedup_key":"expert-replica-semantic-fingerprint"},
    {"title":"BiTree-MoE","hypothesis":"A joint canonical tree across expert combine and TP reduction may preserve configuration-invariant numerics.","minimum_pilot":"Counterexample plus Triton prototype, 1-2 GPU-hours.","key_risk":"TBIK collision and separated reductions.","dedup_key":"joint-expert-tp-canonical-reduction-tree"},
    {"title":"SpectatorRoute","hypothesis":"Spectator routes change per-expert group shape and tile regime, perturbing a victim and its later routes.","minimum_pilot":"64-256 victims, matched spectator conditions, 1-2 GPU-hours.","key_risk":"Batch-conditioned refusal and batch-invariant-kernel literature may subsume the phenomenon.","dedup_key":"spectator-route-shape-semantic-interference"}
  ]
}

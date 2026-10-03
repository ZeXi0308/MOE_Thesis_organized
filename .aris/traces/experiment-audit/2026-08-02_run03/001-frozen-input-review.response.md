# Round 1 verdict: FAIL

Same-family independent review; `acceptance_status=provisional`. No formal
result exists, and no GPU result is inferred. No reviewed file changed between
initial and final hash snapshots: concurrent mutation detected: **NO**.

## Blocking findings

1. The emitted request ledger used an incompatible hash encoding under the same
   field name, `prompt_token_ids_sha256`.

   - Builder/manifest contract hashed canonical JSON token IDs.
   - Formal validation used that canonical JSON contract.
   - The output request ledger instead hashed NumPy raw bytes.
   - Fresh checks proved the mismatch for request 0 in both model manifests.
   - Existing tests validated input token hashes but never compared the emitted
     ledger field with the manifest.

2. The controlling audit documentation contradicted the executable
   preregistration.

   - Preregistration had two materialized workload paths/hashes and
     `formal_execution_authorized: true`.
   - The Gate-0 audit still said authorization was false and the manifest did
     not exist.

3. Current execution was forbidden independently: HEAD was
   `8fe396078ca365afb9ea5d06d8b88c9c01e7a825`, the tree was dirty, and the
   reviewed producer/config files were untracked. The runtime correctly
   required committed canonical bytes and a clean tree.

## Other integrity checks

- No fake scientific ground truth was found; the serial comparison was
  explicitly same-model engineering equivalence.
- The tiny random OLMoE remained development-only.
- Formal arrivals were a declared 1000x time-scaled replay of the first 128
  rows of a content-hashed BurstGPT CSV, not IID synthetic arrivals.
- No post-result tuning or cherry-picking was detected; no formal result
  existed.
- The manifest binding was non-circular.
- All 256 request records passed prompt/document/source-row/token/arrival and
  cross-model shared-workload invariants.
- Formal/smoke isolation, incomplete status and final-sentinel ordering were
  fail-closed.
- Non-blocking: the preregistration did not freeze the exact PyTorch build,
  CUDA runtime/driver, Python, OS or deterministic-runtime flags; this prevents
  a bitwise cross-environment reproducibility claim.

Test evidence: targeted 10/10 PASS; full BCRD suite 72/72 PASS. These did not
override the blocking ledger defect.

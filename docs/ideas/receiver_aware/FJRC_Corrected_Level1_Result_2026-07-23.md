# Corrected FJRC Level-1 Result (2026-07-23)

## Verdict

`NO_GO_FJRC_JOIN_PHASE_INFORMATION`

The frozen two-model AND gate failed.  The keyed sibling-completion phase changes
the exact action in every holdout pair, but those changes do not create the
required request-miss headroom beyond exact receiver queue state `Q`.

This result stops the FJRC / Join-Deficit-Credit bitmap mechanism.  It does not
reject receiver-aware scheduling in general, and it is not evidence about a
physical network, NCCL/RDMA, serving TPOT, or production P99.

## Frozen primary result

| Model | Q miss rate | R=Q+J miss rate | Absolute reduction | Paired bootstrap 95% CI | Strict flips | Gate |
|---|---:|---:|---:|---:|---:|---|
| OLMoE | 0.515625 | 0.500000 | 0.015625 | [0.000000, 0.046875] | 16/16 | FAIL |
| LLM-jp | 0.343750 | 0.343750 | 0.000000 | [0.000000, 0.000000] | 16/16 | FAIL |

The per-model threshold was an absolute miss reduction of at least 0.05 with a
paired-bootstrap lower bound greater than zero.  Pooling was disabled and both
models had to pass.

## Secondary result and interpretation

- OLMoE relative CVaR90 normalized-tardiness reduction was 0.2555, with a
  bootstrap interval of [0.2255, 0.2818].
- LLM-jp relative CVaR90 reduction was 0.0479, with a bootstrap interval of
  [0.0000, 0.1340].
- These are deterministic synthetic-workload timings over native route
  identities.  They cannot rescue the frozen request-miss gate or justify a
  physical scheduling claim.
- The 16/16 action-flip result rules out the trivial explanation that `J` never
  changes the oracle action.  The negative result is instead that those action
  changes rarely cross a request deadline, and do not transfer across both
  model routing structures.

## Validation completed

- RTX 5090 primitive LUT capture completed once; internal artifact hash:
  `9103369b8c1b79102902d960e4d7bf2cc18e8d1b2faa3615d663dfc66746e56c`.
- The LUT file SHA-256 is
  `8184e89bfe7b37e0c9a2683d603e2a7a751186ea545e963f09f40cb99652588d`.
- Post-capture preflight passed both model validators with no blocker or warning.
- Each model bundle contains 16 matched holdout scenarios and 32 distinct
  request outcomes.  Raw records contain exactly 16 `pair`, 16 `baselines`, and
  16 `negative_controls` rows.
- Q/R miss rates were independently recomputed from raw request rows and match
  the saved aggregate metrics exactly.
- Deadline calibration read selection-Q outcomes only;
  `r_outcomes_read_for_selection=false`.
- The final decision uses `OLMoE_AND_LLMJP_WITHOUT_POOLING`; its self-hash is
  `b1999523e954aed3d777d52a1d8f35d9d6d0c9cdc9ac5d35109929fd6abb1281`.

## Evidence boundary and next action

The highest valid statement is a negative conditional-information result under
native route identities, a measured single-5090 primitive LUT, analytic cut,
and deterministic synthetic arrivals/deadlines.

Do not tune deadlines, switch the primary metric, pool models, or promote the
OLMoE CVaR secondary result.  The next receiver-side experiment is the frozen
RR-credit physical timed-trace census.  It requires at least four independent
GPU ranks for an incast existence claim; a one-GPU multi-process run is allowed
only as a harness/accounting smoke test.


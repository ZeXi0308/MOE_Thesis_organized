## Verdict

- Overall verdict: **WARN**
- Integrity status: `warn`
- Accepted-result contract: **SATISFIED**
- P0/P1 findings affecting this run: **0**
- Evaluation type: `synthetic_proxy` — more precisely, a development forensic diagnostic using one same-model batched output as the comparison reference, not scientific ground truth.
- Reason code: `valid_bounded_forensic_nonblocking_evidence_hardening`
- `review_independence`: `same-family`
- `acceptance_status`: `provisional`

The result may continue only within its frozen claim ceiling: one earliest-observable-boundary classification for one target. It does not pass Gate 0 or support a scientific/system claim.

Path aliases:

- `SRC`: `/private/tmp/gate0-r2-experiment-audit-source.073fdf8/docs/ideas/bcrd/experiments/capture_continuous_decode.py`
- `TEST`: `/private/tmp/gate0-r2-experiment-audit-source.073fdf8/docs/ideas/bcrd/experiments/test_continuous_decode.py`
- `CARD`: `/private/tmp/gate0-r2-experiment-audit-source.073fdf8/docs/current/gate0_v2_olmoe_router_boundary_root_cause_card_r2_2026-08-10.md`
- `SPEC`: `/private/tmp/gate0-r2-experiment-audit-source.073fdf8/docs/spec-gate0-olmoe-router-boundary-root-cause-r2-2026-08-10.md`
- `R`: `/private/tmp/gate0-r2-capsule-audit.sYVa6E`

## A. Ground-truth/reference provenance — PASS

The artifact is a forensic comparison, not a benchmark with dataset ground truth.

- The “expected” side is the current run’s natural batched OLMoE decode output; the “observed” side is the same model’s batch-one serial cached-decode output (`SRC:2649-2701`, `SRC:2916-2933`).
- The frozen `[54] / [33]` orientation originated from the consumed revision-1 model-output anomaly and is disclosed explicitly (`CARD:13-20`, `CARD:29-42`).
- It is used as a reproduction target, not accepted as truth: membership, top-k slots, and orientation are recomputed from retained BF16 logits (`SRC:1581-1673`, `SRC:1943-1958`).
- Stored mismatch fields cannot manufacture acceptance.
- WikiText prompts and BurstGPT-derived arrivals are workload inputs, not labels or expected outputs.

There is deliberate post-hoc target selection, but it is honestly bounded to reproducing and locating one prior anomaly. It cannot support prevalence or performance claims.

## B. Score and decision integrity — PASS

No score normalization, tuned tolerance, self-denominator, or threshold rescue exists. Decisions use exact comparisons and a frozen first-match tree (`SRC:1959-1976`, duplicated at `SRC:2138-2153`).

Independent recomputation from retained JSON confirmed:

- All 18 retained value records close over canonical-value and source-storage hashes.
- Semantic inputs: exact across both sides.
- Logical layer-0 K/V: equal stored source hashes and valid all-zero difference digests; runtime raw-KV recomputation occurs at `SRC:1853-1865`.
- Pre-router activation: `488/2048` BF16 values changed; max absolute delta `0.01171875`.
- Router logits: `27/64` changed; max absolute delta `0.015625`.
- Batched top-8: `[6, 8, 31, 52, 13, 2, 63, 54]`.
- Serial top-8: `[6, 8, 31, 52, 13, 2, 63, 33]`.
- Recomputed orientation: expected-only `[54]`, observed-only `[33]`.
- Float64 54-minus-33 projection:
  - batched: `0.0028857952879661752`
  - serial: `-0.00016849176381583675`
- Saved-logit 54-minus-33 gap:
  - batched: `0.0029296875`
  - serial: `-0.000244140625`
- Recomputed classification: `UPSTREAM_LAYER0_ACTIVATION_DIVERGENCE`, matching:
  - `RUN_STATUS.forensic_classification`
  - `RUN_STATUS.serial_audit_failure.classification`
  - `RUN_STATUS.serial_audit_failure.forensic_evidence.classification`

The auxiliary CPU gate-weight recomputation differed from stored CUDA diagnostic weights by at most one float32 ULP; weights are explicitly diagnostic-only and top-k identities matched exactly.

## C. Existence, provenance, and exactly-once consistency — PASS

| Surface | Result |
|---|---|
| Source | Commit `073fdf8a46a85f7bc7c4ba16cf4a572e1e138601`, tree `ceca899d8429f445ef0f4bc02128d7f71774b10b`, clean checkout; four reviewed source hashes match. |
| Bundle | SHA `ea990b84...26fb`; `git bundle verify` reports complete history and the exact reviewed tip. |
| Inputs | Workload SHA `cce2cf61...23d0`; preregistration SHA `d72e292a...9bff`. The development workload differs from committed formal workload only by `run_class: formal → development`. |
| Workload integrity | 128 unique requests, 128 matching serial-audit IDs, 128/128 prompt hashes valid, arrival trace recomputes to `808036ca...58d`. |
| Model | Nine behavior-file hashes agree across authorization, prelaunch, frozen receipt, and provenance; three auxiliary files agree at prelaunch. Raw model shards are not retained in the capsule, so post-hoc local rehashing is unavailable. |
| Runtime | Python 3.12.3, Torch 2.8.0+cu128, CUDA 12.8, Transformers 4.57.6 agree across authorization/prelaunch/provenance. |
| GPU | One RTX 5090, UUID `GPU-64c06fb4-79b3-0d5b-4294-431e7afac4d7`, driver `595.71.05`, consistent preflight/prestart/provenance/postrun. |
| Process | One launch receipt, one PID `14677` start, one exit receipt, exit code `1`; timestamps strictly ordered and no retry evidence exists. |
| Result binding | Provenance SHA `631c934b...d60b` equals `RUN_STATUS.execution_provenance_sha256`; log SHA matches the exit receipt. |
| Capsule | 21/21 payload hashes pass. External tgz SHA `895f90e5...f179` matches its sidecar. |

The exit code `1` is valid here: the log terminates at the exact source `RootCauseForensicStop`, and `RUN_STATUS` independently records `INCOMPLETE / RootCauseForensicStop / root_cause_forensic_stop`. The throwing path is `SRC:2941-2958`; fail-closed status writing is `SRC:3664-3685`.

Exactly-once is proven within the launcher/receipt trust boundary, not against a malicious root performing an unrelated out-of-band invocation.

## D. Dead-code and evidence-path integrity — WARN

The current result’s decision path is live and coherent:

- Batched target context: `SRC:2593-2719`.
- Serial target context: `SRC:2866-2947`.
- Build, validate, then stop: `SRC:1979-2208`, `SRC:2941-2958`.
- Hook removal in `finally`: `SRC:1243-1247`.
- No success ledger or sentinel can be reached after the target stop; those writes are below the throwing call at `SRC:3555-3663`.
- The generated terminal context uses an allowlist and excludes process-local KV tensors (`SRC:1486-1497`, `SRC:2189-2199`).
- The actual JSON contains zero raw-KV value arrays.

The actual batched side is demonstrably multi-request: its target cache has `logical_length=7`, `physical_length=128`, and `left_padding=121`, while serial is `7/7/0`. Under `stack_left_padded_caches` (`SRC:1064-1100`), that padding requires at least one longer co-active request. Exact target batch cardinality was not retained.

Non-blocking implementation hardening gaps:

1. The hook transiently retains the full target-call batch tensor and later slices the target row (`SRC:1234-1258`, `SRC:2644-2647`); forensic logit preparation also processes more rows/layers than the final one-layer payload. This increases transient capture scope but cannot alter the already-produced target logits or classification.

2. Physical-cache and full-router-weight diagnostic hashes are syntax-checked rather than rebound to source tensors (`SRC:1533-1546`, `SRC:1801-1805`), and the validator does not reject arbitrary unknown JSON keys. The current builder-generated JSON is safe, while every classification-bearing value—logical KV, activation, logits, boundary rows, projections, and deltas—is bound and recomputed. Therefore these gaps do not change this run’s conclusion.

## E. Scope and claim ceiling — PASS

Exact observed scope:

| Axis | Scope |
|---|---:|
| Pretrained models | 1 OLMoE revision |
| GPUs | 1 RTX 5090 |
| Frozen workload inputs entering batched pass | 128 |
| Serially compared requests | 1 |
| Frozen target events retained | 1 |
| Compared decode step | 0 only |
| Compared layer | 0 only |
| Router width / top-k | 64 / 8 |
| Disputed experts | `{33,54}` |
| Seeds | 1 |
| Authorized runs / starts / exits | 1 / 1 / 1 |
| Retries evidenced | 0 |

The batched producer completes the 128-request pass before entering serial audit (`SRC:2534-2761`). Exact aggregate batched decode-step and target batch-size counts are not retained; the workload allows at most 16 decode steps per request and batch size 8.

Supported claim:

> For `gate0-000`, decode step 0, layer 0, in this one frozen RTX-5090 execution, semantic inputs and logical layer-0 KV matched, while the first retained differing boundary was the BF16 pre-router activation; the historical 54-to-33 membership flip reproduced.

Unsupported:

- batch-composition causality versus padding, shape, backend, or kernel effects;
- determinism, repeatability, or prevalence;
- Gate 0-A/Gate 0 completion or Gate 1 authorization;
- scientific efficacy, quality, latency, throughput, or serving claims;
- EP/NCCL/RDMA, multi-GPU, receiver, tenant-isolation, or production claims.

These exclusions agree with `CARD:117-119`, `CARD:148-154`, authorization `.claim_ceiling`, and the false eligibility fields in `RUN_STATUS`.

## F. Evaluation-type classification — PASS

Closest taxonomy label: `synthetic_proxy`.

The “expected” reference is generated by the same model under a different execution context. It is not real dataset ground truth, and no accuracy/performance score is inferred from it. The more precise description is:

> development forensic diagnostic: same-model batched-versus-serial boundary comparison.

## Frozen predicate verdict

All requested predicates are satisfied:

- Exact target and 54/33 direction: satisfied.
- Exit-1 bounded-stop contract: satisfied.
- Nested/top-level classification equality and decision-table recomputation: satisfied.
- Commit/tree/source/input/model/runtime/GPU hash closure: satisfied, subject to raw model bytes not being retained locally.
- No `CAPTURE_COMPLETE` or producer ledger; eligibility remains false: satisfied.
- Exactly-one recorded launch/start/exit and no retry evidence: satisfied.
- No raw KV tensor payload in JSON: satisfied.
- Maximum claim limited to one earliest-boundary diagnostic: satisfied.

## Actions

Blocking: none for accepting this already-consumed result within its frozen narrow scope. Do not retry this authorization.

Non-blocking for future forensic cards:

- Record and require target batch size/active IDs, and slice the target row/layer at capture time.
- Close accepted JSON key sets and bind or remove physical-cache/full-weight diagnostic hashes.

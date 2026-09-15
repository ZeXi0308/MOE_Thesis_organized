# BCRD simulator correction audit

> Date: 2026-07-29
> Scope: local simulator and formal-consumer audit
> Scientific status: `LOCAL_SIMULATOR_CORRECTED / GATE0_A_PARTIAL_IMPLEMENTED / FORMAL_GATE0_OPEN / REQUEST_DAG_OPEN / NOT_FORMALLY_RUN`

## Verdict

The five defects in the external static review are closed for the local
simulator with executable counterexamples. A second audit exposed a stricter
scientific boundary: a physical continuous-decode producer is now implemented
and development-qualified on a random tiny OLMoE fixture, but no frozen
pretrained CUDA cell has run. The repository still has no formal producer
artifact, expert/dtype-complete service surface, full-path denominator, or
counterfactual full request-DAG replay. Therefore BCRD still cannot support a
paper result, and no Gate 1 experiment is authorized yet.

Formal Gate 2 is deliberately hard-disabled with
`INVALID_REQUEST_DAG_REPLAY_NOT_IMPLEMENTED`. Smoke and single-layer exact
enumeration are local code-path evidence only.

## Original five defects closed locally

1. **Real bounded seal.** A singleton pays the actual hold. Arrivals, timeout,
   `max_batch`, seal cost, launch cost and finish are ordered as causal events.
   The deadline cap reserves seal, launch and service time.
2. **Causal route-v3.** The schema carries document/request/event identity,
   layer/decode position, route/dispatch/expert/combine times, source/target and
   legal replicas. Validators reject layer or decode-step dependency violations.
3. **Real queue lifecycle and online isolation.** Assignment and evaluation use
   one dispatch/seal/launch/finish engine. Gate 1 replays each
   model/phase/layer stream once instead of resetting per wave. Online policies
   receive a causal view without observed dispatch/expert/combine suffixes or
   access to the engine; remote cost is charged once. Least-load and
   min-predicted-finish are now distinct baselines.
4. **Exact local Oracle action space.** Hold is independent for every active
   `(replica, expert)` queue. Symmetry reduction preserves request/event/timing,
   legal-target and serialized-controller distinctions. State-budget overflow
   is `UNSOLVED`, never relabeled exact.
5. **Frozen split and Gate quantifiers.** Documents are split globally once
   across models/layers/windows. The 15% witness is a subset of the 10%
   LCB/actionability witness; `KILL` requires every complete preregistered common
   cell to be below 5%.

Additional local hardening fixes one run-wide hash salt, exact exposure-matrix
coverage and byte hashes, and actionable expert-work accounting that requires a
common legal replica and counts only the relevant expert work.

## Verification

- BCRD unit/counterexample suite: `69/69 PASS` on 2026-08-02, including seven
  Gate-0 continuous-decode producer tests.
- Randomized exactness audit: 1,971 builder-valid raw-versus-structured cases
  matched on objective sets; a later 400-case cost-varied audit also matched.
- Full CPU smoke: two synthetic models -> Gate 1 -> 128 frozen instances ->
  exact local Oracle -> 96 evaluation instances -> Gate 3; all scientific
  statuses remain `SMOKE_ONLY`.
- Shared numerical-reference/import and archived compatibility suites:
  `32/32 PASS` from the artifact/import audit.
- No GPU experiment was run and no prior characterization was reclassified.

## Remaining formal blockers

1. The Gate-0 A runner physically batches left-padded per-request KV caches and
   closes natural greedy token/route identity against serial cached decode in a
   development fixture. Formal execution remains blocked by its own
   preregistration because dataset/tokenizer/document/prompt/arrival bytes are
   unresolved; neither pretrained model cell has run. Model-call boundaries
   are scheduler provenance, not per-layer service timing.
2. The current service catalog is a layer-level latency proxy. It does not yet
   key and bind the formal `(model, layer, expert, dtype, rows)` surface, and it
   contains no Joule/completed-token measurement. BCRD must remain a latency/SLO
   line unless a separate physical energy protocol is implemented. Formal
   Gate 1 is hard-blocked until this consumer exists.
3. Formal Gate 1 still needs an independently measured, complete full-path
   exposure denominator for every preregistered route cell.
4. Gate-2 instances are `single_layer_window`; an assignment/hold delay is not
   propagated through later layers and autoregressive decode steps. Request SLO
   and completion claims are forbidden until that DAG exists.
5. The preregistered top-16, six-hold exact space can exceed the current state
   budget even for a small nontrivial instance. That is an honest `UNSOLVED`
   result, not evidence for or against BCRD.

Gate-3 controller-tax and final policy predicates remain downstream work and
cannot authorize anything because formal Gate 2 cannot produce `PASS_GATE2`.

## Next authorized work

The next step is input freezing and formal qualification, not Gate 1:

1. freeze one immutable workload manifest for the existing producer, including
   dataset and tokenizer revisions, exact document/prompt bytes, arrival trace,
   seed, code SHA, expected counts and preregistration hash; then remove the
   preregistration blockers without changing thresholds;
2. from a clean commit, run and independently audit both frozen pretrained CUDA
   cells; do not treat the tiny-model tests as formal evidence;
3. implement the expert/dtype-complete service surface and exact full-path
   exposure denominator, with content-hash provenance;
4. run the remaining frozen Gate 0 checks, then Gate 1 only;
5. if Gate 1 passes, implement and test full counterfactual request-DAG replay
   before attempting Gate 2;
6. stop BCRD if Gate 1 misses the preregistered cross-model thresholds.

Controller work, formal Oracle claims, Gate 3 and 8xA100 remain unauthorized.

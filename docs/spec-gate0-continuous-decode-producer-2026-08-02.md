# Gate 0 continuous-decode producer specification

## Scope

This component is an engineering qualification producer for Gate 0-A. It
captures natural router output during real KV-cached decode under a mutable
active set. It is not a serving engine, an EP runtime, a service-surface
benchmark, a full request-DAG ledger or a Gate 1 evaluator.

## Roles and data flow

1. The canonical preregistration freezes the allowed cells and thresholds.
2. One immutable workload manifest supplies model/tokenizer/data revisions,
   exact prompt bytes and token-ID hashes, document/request/source-row IDs,
   arrival/deadline trace, generation, batching parameters and seed.
3. The runner admits requests in arrival order, performs one real prefill per
   admitted request, and retains its KV cache and greedy next token.
4. The scheduler selects the bounded active prefix. Per-request caches are
   left-padded and stacked into one physical cached-decode batch.
5. The model consumes exactly one input token per active request. Native router
   logits are converted to logical expert/top-k identities; the updated cache
   is split back into per-request state.
6. A serial cached-decode rerun checks greedy token and logical route identity
   for the frozen audit request set.
7. Route, batch and request ledgers are written with manifest, preregistration,
   environment and hashes. The completion sentinel is the last write.

## Identity contract

The producer must preserve:

`request_id -> document_id -> decode_step -> token_position -> layer_id -> topk_slot -> logical expert`.

It also records source rank and a legal replica set, but it does not claim to
observe physical dispatch, target replica, expert execution or combine. Those
unobserved stages keep Gate 0-D partial.

## Integrity contract

- Formal and development output roots are disjoint.
- Existing output directories are never overwritten.
- Formal mode requires CUDA, one visible RTX 5090, frozen dependency versions,
  a clean worktree, the canonical preregistration file exactly as committed,
  128 unique documents and all preregistration blockers resolved.
- The canonical preregistration fixes each per-model workload path and full
  SHA-256; formal mode requires that file to be committed unchanged. This is a
  one-way chain because embedding a manifest SHA in preregistration while also
  embedding the preregistration SHA in that manifest would be circular and
  unsatisfiable. The actual executing `HEAD` is recorded in output provenance.
- Prompt bytes, model-specific prompt token IDs and arrival-trace hashes are
  checked before model execution.
- `prompt_token_ids_sha256` always means SHA-256 over the compact canonical
  JSON integer list in both the input manifest and emitted request ledger; raw
  tensor/NumPy byte encodings must not reuse this field name.
- Every output begins with `RUN_STATUS=INCOMPLETE`.
- Consumers must require both `RUN_STATUS=COMPLETE` and
  `CAPTURE_COMPLETE.json`.
- Candidate outputs always keep `scientific_result_eligible=false`,
  `gate0_complete=false` and `gate1_authorized=false`.

## Current boundary

Development tests prove cache/batch/identity mechanics on a tiny random CPU
fixture. They do not prove pretrained-model compatibility, full-logit or
gate-weight equivalence, service timing, dispatch/expert/combine identity,
request-level performance, cross-model consistency or any multi-GPU property.

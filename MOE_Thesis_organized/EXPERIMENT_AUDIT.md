# Experiment integrity audit: Gate 0-B frozen inputs

> Audit date: 2026-08-02  
> Final reviewer: fresh `gpt-5.6-sol`, reasoning `ultra`  
> Acceptance: provisional same-family review  
> Verdict: `PASS_CANDIDATE_ONLY`  
> Current formal execution: `NOT_PERMITTED_DIRTY_UNTRACKED_CHECKOUT`

## Verdict

The two Gate 0-A producer cells now have reproducible, materialized frozen
inputs and are safe to retain. Gate 0-B is `PASS`. This is not a formal capture,
full Gate 0 pass or scientific result.

Formal execution is not permitted from the current checkout. HEAD is
`8fe396078ca365afb9ea5d06d8b88c9c01e7a825`, the tree is dirty, and the new
producer/config files are untracked. The runner may execute only after these
exact reviewed bytes are committed, the entire checkout is clean, and exactly
one visible RTX 5090 satisfies the frozen dependency guards.

## Frozen-source integrity

| Input | Frozen evidence |
|---|---|
| Models | OLMoE `6d84c485…`; LLM-jp `1d598307…`; BF16 |
| Tokenizers | Same exact revisions; tokenizer files plus every selected prompt's token count and canonical token-ID SHA |
| Dataset | WikiText-103 raw test `b08601e…`; Arrow SHA `2b8a3efa…`; first 128 non-empty rows in index order; raw prompt bytes preserved |
| Arrival | BurstGPT commit `d895a53…`, blob `f95dd2…`, CSV SHA `46fc9480…`; first 128 data rows; declared 1000x replay and fixed deadline transform |
| Generation | Greedy, max 16 decode steps, max batch 8, seed `20260725` |
| Target | One visible RTX 5090; PyTorch `2.8.0*`; Transformers `4.57.6`; single-GPU producer qualification only |

The builder reproduces both canonical manifest files byte-for-byte and never
reads route outputs. The integrity chain is non-circular: clean committed
`HEAD` binds the canonical preregistration, which binds each full manifest SHA;
the executing commit is recorded in output provenance.

## Independent review loop

Round 1 correctly returned `FAIL` for two blockers:

1. the emitted request ledger used NumPy bytes for
   `prompt_token_ids_sha256`, while the manifest used canonical JSON token IDs;
2. the Gate-0 ledger retained a stale statement that authorization was false
   and manifests did not exist.

Both were fixed. The ledger now calls the same canonical hash helper and the
test checks every emitted request field. The documentation now distinguishes
authorized cells from current execution readiness. Round 2 returned
`PASS_CANDIDATE_ONLY`; targeted tests were 10/10, the full suite was 72/72, and
initial/final reviewed hashes matched with no concurrent mutation.

## Evidence boundary

- Natural routes will come from native pretrained-model router logits; no IID
  synthetic route is accepted as formal input.
- Serial replay is same-model engineering equivalence, not scientific ground
  truth.
- No pretrained CUDA cell, formal output directory, `RUN_STATUS=COMPLETE` or
  `CAPTURE_COMPLETE.json` exists yet.
- Physical dispatch, rank/replica, expert execution, combine, full request DAG,
  service surface and full-path denominator remain outside this producer.
- Single RTX 5090 evidence cannot establish EP, NCCL/RDMA, multi-GPU TPOT/P99,
  SLO-goodput or scalability.

## Authorization

- Retain frozen inputs and producer candidate: **YES**.
- Run the two formal cells from the current checkout: **NO**.
- Run them after committing these exact bytes in an entirely clean checkout on
  the frozen single RTX 5090 environment: **YES, producer qualification only**.
- Claim Gate 0-A or full Gate 0 pass: **NO, until both cells and independent
  completion-sentinel audit pass**.
- Start Gate 1: **NO**.

Full review trace:
`.aris/traces/experiment-audit/2026-08-02_run03/`.

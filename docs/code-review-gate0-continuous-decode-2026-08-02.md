# Gate-0 continuous-decode producer code review

> Date: 2026-08-02  
> Scope: Gate-0 A/B only  
> Verdict: `PASS_CANDIDATE_FOR_SINGLE_5090_AFTER_COMMIT / FORMAL_NOT_RUN`

## Requirement fit

| Requirement | Status | Evidence |
|---|---|---|
| natural greedy cached one-token decode | implemented | `capture_continuous_decode.py` executes cached decode and reads native router logits |
| one physical decode call for multiple requests | implemented | per-request caches are left-padded, stacked, decoded once and split back |
| request/step/token/layer/top-k/expert identity | implemented | contribution construction plus episode identity closure |
| mutable active set | implemented | arrival-ordered admission and natural EOS/max-step retirement |
| frozen real-world arrivals | implemented | first 128 BurstGPT rows at fixed commit/blob/full-file SHA and a fixed 1000x replay transform |
| frozen prompts/tokenizers | implemented | WikiText revision/Arrow hash/fingerprint, dataset rows, raw prompt hashes, tokenizer files and per-model token-ID hashes |
| serial equivalence audit | implemented | serial cached-decode replay checks every frozen request and step |
| fail-closed formal contract | implemented | canonical committed preregistration/workload binding, clean Git, exact dependency versions and single RTX 5090 guards |
| isolated, non-overwriting artifacts | implemented | disjoint roots, initial incomplete status and final completion sentinel |
| scientific or Gate-1 verdict | deliberately absent | every candidate output keeps all scientific/Gate-1 eligibility fields false |

## Findings resolved before execution

1. **Critical — caller-selected preregistration could self-authorize a formal run.**
   Formal mode accepts only the canonical repository preregistration path and
   requires its bytes to equal the copy in the executing commit.
2. **Major — final process exit could falsely count as an active-set change.**
   Acceptance counts only changes between observed scheduler iterations.
3. **Major — arrival provenance was named but not content-bound.**
   The model-independent `(sample_id, arrival_us, deadline_us)` trace is
   canonically hashed and checked against both manifest and preregistration.
4. **Major — router capture did not explicitly close the expected layer set.**
   The producer requires exactly `config.num_hidden_layers` router outputs.
5. **Major — formal and development outputs could share ambiguous paths.**
   Formal and development cells use disjoint canonical roots; existing paths
   are never overwritten.
6. **Major — self-produced complete output could look externally qualified.**
   `producer_formal_eligible`, `scientific_result_eligible`, `gate0_complete`
   and `gate1_authorized` remain false pending independent result audit.
7. **Major — final-step EOS was mislabeled as max-step termination.**
   EOS now takes precedence and a deterministic boundary test locks it.
8. **Critical — authorized metadata did not bind exact workload bytes.**
   Formal mode accepts only canonical model paths and requires each full SHA to
   match both the preregistration entry and executing commit.
9. **Critical — preregistration and manifest hashes were mutually recursive.**
   Requiring each file to contain the other's final SHA has no constructible
   fixed point. The chain is one-way: clean committed `HEAD` binds canonical
   preregistration bytes, which bind each canonical manifest SHA; output
   provenance records the executing commit.
10. **Critical — embedding the executing Git SHA in a committed manifest was
    self-referential.** A file cannot predict the commit hash that contains the
    file itself. Formal mode instead requires all control/input files to equal
    their committed bytes and a clean tree, then records the observed `HEAD`.
11. **Critical — raw prompts were stripped before hash validation.**
    `_require_resolved` removed leading/trailing WikiText whitespace. Prompt
    validation now checks non-emptiness without changing bytes; canonical
    manifest tests cover the exact raw strings.
12. **Major — tokenizer revision alone did not detect tokenization drift.**
    Manifests store tokenizer artifact hashes plus every request's token count
    and token-ID SHA; the runner re-tokenizes and checks all 128 before loading
    results into the producer.
13. **Major — a locally invented burst could be mislabeled as natural traffic.**
    Formal arrivals require real-world trace provenance. The source is
    BurstGPT commit `d895a53…`, blob `f95dd2…`, CSV SHA `46fc9480…`; the replay
    transform was fixed before pretrained route output was observed.
14. **Critical — the emitted request ledger reused the manifest token-hash
    field name with a different byte encoding.** The manifest and preflight use
    canonical JSON token IDs, while the ledger used raw NumPy bytes. The ledger
    now calls the same canonical helper, and the tiny producer test asserts the
    exact output field for every request.
15. **Critical — the audit ledger retained a stale authorization statement.**
    It said authorization was false and manifests were absent after the
    preregistration had authorized two materialized cells. The ledger now
    distinguishes cell authorization from current execution readiness: the
    dirty, untracked checkout remains forbidden.

## Open evidence risk

The runner has not executed either frozen pretrained model revision. The tiny
OLMoE test exercises the same Transformers cache shape and router path, but
cannot prove OLMoE-1B or LLM-jp compatibility, BF16/RTX 5090 behavior or formal
artifact completeness. Mocks cannot close that evidence gap.

The serial audit checks greedy tokens and selected expert/top-k identities, not
full router logits, gate weights or cache tensors. This qualifies only the
Gate-0 A producer candidate and does not satisfy full Gate 0-D.

## Verification

- Canonical input generator reproduction check: PASS.
- Canonical JSON parse and input-contract validation: PASS.
- Targeted Gate-0 producer/input tests: 10/10 PASS.
- Full BCRD experiment suite: 72/72 PASS.
- Python compile, long-line scan and `git diff --check`: PASS.
- Python lint: not run because neither `ruff` nor `flake8` is installed.
- Independent experiment-integrity review: the first pass found findings 14
  and 15; both were fixed and retested. The final stable-hash verdict is kept in
  `EXPERIMENT_AUDIT.md` so this review does not embed a self-staling hash.

## Decision

The input contract is constructible and frozen, and no known code defect
requires GPU retuning. `formal_execution_authorized=true` permits only the two
declared cells; it does not make this dirty checkout executable. Formal mode
must fail until the reviewed bytes are committed and the tree is clean on the
single RTX 5090 host. The consolidated integrity report is
[`../EXPERIMENT_AUDIT.md`](../EXPERIMENT_AUDIT.md).

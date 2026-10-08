# Gate 0 natural continuous-decode producer: Experiment Card

> Date frozen: 2026-08-02  
> Scope: Gate 0-A only  
> Scientific status: `PREREGISTERED_ENGINEERING_QUALIFICATION / INPUTS_FROZEN`  
> This card does not authorize Gate 1, a service-latency claim, or a controller experiment.

## Research question

Can the repository execute identity-safe, cached one-token decode for naturally
routed pretrained MoE requests under a mutable continuous-batching active set,
without substituting IID routes or relabeling full-prefix forwards as decode?

## Hypothesis and null

- **H1:** A frozen arrival replay can admit and retire requests while the model
  executes actual batched cached one-token decode. Every executed token can be
  mapped exactly to request, decode step, layer, top-k slot and logical expert,
  and batched execution preserves the serial cached-decode token and route
  semantics.
- **H0:** The producer cannot create a real mutable active set, loses or
  duplicates route identity, executes full-prefix work under a decode label, or
  changes token/route semantics relative to serial cached decode.

## Frozen formal cells

The formal qualification has exactly two cells and one shared arrival trace:

| Cell | Model revision | Dtype | Documents | Decode steps | Generation |
|---|---|---:|---:|---:|---|
| `olmoe` | `allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5` | BF16 | 128 | max 16 | greedy |
| `llmjp` | `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M@1d5983076dfc67aee4a77ec06a27027f5bab6055` | BF16 | 128 | max 16 | greedy |

The shared dataset is now frozen to
`wikitext/wikitext-103-raw-v1@test@b08601e04326c79dfdd32d625aee71d232d685c3`.
The selection is the first 128 non-empty raw test rows in dataset index order;
prompt bytes are preserved verbatim before model-specific right truncation at
128 tokens. Each request stores the dataset row, document/prompt hash, frozen
token count and token-ID hash. The two tokenizer snapshots equal their exact
model revisions and bind all tokenizer-file hashes.

Arrival times come from the first 128 data rows of the public real-world
`BurstGPT_1.csv` trace at repository commit
`d895a53bb7b8ec137d0d2fe203b335835a78c10a`, full CSV SHA-256
`46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a`.
The original timestamps are normalized to the first selected row and replayed
with a predeclared 1000x time scale; every deadline is scaled arrival plus 60 s.
This source and transform were fixed before any pretrained route output was
observed.

Each model manifest uses its canonical path under
`docs/ideas/bcrd/experiments/configs/workloads/`. The integrity chain is
one-way and non-circular: a clean executing `HEAD` must contain the canonical
preregistration; that file binds each full manifest SHA-256; and the runner
requires the manifest bytes to equal both that SHA and the bytes in the same
commit. The actual executing Git SHA is recorded in the output environment.

## Variables

- **Independent:** request arrival time from the frozen trace; scheduler
  `max_batch_size`; natural EOS retirement.
- **Controlled:** model and revision, tokenizer snapshot, prompt bytes and
  document IDs, BF16, greedy decoding, maximum 16 decode steps, seed
  `20260725`, exact committed implementation, PyTorch `2.8.0*`, Transformers
  `4.57.6`, one visible RTX 5090 and the arrival trace.
- **Development-only fixture:** a randomly initialized tiny OLMoE model may be
  used only for deterministic cache, batching, identity and fail-closed tests.
  It is never a scientific result.

## Metrics and denominators

Every qualification artifact must report these absolute counts:

- total, admitted, completed, EOS-stopped and failed requests / all frozen requests;
- executed decode steps / all scheduled decode steps;
- router invocations / all executed `(request, decode_step, layer)` events;
- route contributions / all executed top-k contributions;
- identity-closed contributions / all route contributions;
- observed batch-size histogram / all decode scheduler iterations;
- serial-parity audited steps / all steps selected before execution by the
  frozen audit rule.

No subset-only latency or improvement ratio is a metric in this card. Producer
wall-clock timestamps are provenance for scheduling order only and are not a
service surface, TPOT, P99 or SLO denominator.

## Acceptance threshold

The implementation qualification passes only if all are true:

1. deterministic tests observe at least one decode batch with size at least 2
   and at least one admission or retirement that changes the active set;
2. cached decode advances every retained request by exactly one token per
   scheduler iteration and never recomputes a full prefix under a decode label;
3. serial-versus-batched audit has identical greedy token IDs and logical
   expert/top-k identities on 100% of audited steps;
4. route identity coverage is 100%, with zero duplicate, missing or cross-request
   contributions;
5. a non-empty frozen arrival trace is mandatory outside smoke mode, output
   directories are new, and all non-GPU/development outputs retain
   `formal_eligible=false` and `scientific_result_eligible=false`;
6. both frozen real-model cells later complete with all expected requests and a
   validated integrity sentinel before Gate 0-A may be marked `PASS`.

## Expected cells and output isolation

- Formal: 2 model cells, written only below a new
  `artifacts/bcrd_gate0/formal/<run-id>/` directory.
- Smoke/development: tiny-model and CPU checks, written only below a new
  `/tmp/bcrd-gate0-smoke-<run-id>/` directory.
- Formal and smoke files must never be merged or jointly summarized.

## Stop rule and failure conditions

- Dirty/uncommitted inputs, a non-5090 CUDA environment, or either real-model
  cell absent: stop at `CONTINUE`; do not claim Gate 0-A `PASS`.
- Any serial-parity, cache-length, active-set, route-closure or identity failure:
  mark the run `INVALID` and do not analyze route prevalence.
- If the frozen formal trace yields no batch larger than one, the producer
  qualification fails for that frozen trace. Do not retune its arrivals after
  observing the result.
- Partial output, exception, config/hash mismatch or missing integrity sentinel:
  mark `INCOMPLETE` or `INVALID` and exclude it from formal statistics.

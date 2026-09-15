# ErrorToken Cross-Artifact CPU Selector Audit

**Overall verdict:** `PASS_WITH_LIMITATIONS`  
**Evidence status:** `deterministically_recomputed`  
**Acceptance:** `provisional_same_family`  
**Mechanical decision:** `NO_RETROSPECTIVE_ENRICHMENT` is correct.

## Independent recomputation

- SemanticFence calibration: 29,803 calls and 6,793 `(layer, expert, route-rank)` keys; the persisted risk table matches exactly, maximum error `0.0`.
- StableBatch join: 32 targets, 16 victims, two candidates per victim. Converting zero-based `topk_rank` to one-based `route_rank` with `+1` covers `28/32 = 87.5%`; missing targets are `03`, `14`, `23`, and `29`.
- ErrorToken causal-first `B=1`: `6/16` route-positive targets.
- Gate-weight baseline: `6/16`, sharing 15/16 selections with ErrorToken.
- Top-k-rank baseline: `6/16`, sharing 14/16 selections with ErrorToken.
- Exact 65,536-assignment matched null:

```text
hits:        4      5       6       7      8
assignments: 4096   16384   24576   16384  4096
```

The null mean is `6.0`; `P(H >= 6) = 0.6875`. Since ErrorToken obtains `6`, the frozen no-enrichment decision follows mechanically.

## Integrity and leakage

- SemanticFence `COMPLETE` and calibration hash, plus StableBatch manifest/status and selected/result hashes, pass before selection.
- `target_results.jsonl` is byte-hashed before freeze for provenance, but its JSON contents are decoded only after `SELECTION_FROZEN.json` is written.
- The selector statically rejects `next_layer_topk_margin`, `selection_score`, and `earliest_changed_downstream_layer` as online inputs.
- The 12 tests pass, including rank conversion, causal first-eligible behavior, exact-null enumeration, leakage-field rejection, tamper failure, and output-directory non-reuse.
- The emitted action is explicitly `NOT_EXECUTED_PLAN_ONLY`; no replay, GPU call, pack surgery, or runtime guard is executed.

## Material limitations

- The thresholds are retrospective and were not preregistered before the source experiments. The 32 StableBatch targets are themselves outcome-informed enriched targets.
- “Causal first” is only simulated within two preselected candidates per victim, not over a natural online decode stream.
- Only 6,528 of 32,234 calibration key-row histories cover all six M values. Elsewhere, “onset” means the first observed mismatch on an incomplete grid, not a proven global onset.
- The exact null describes these fixed 16 pairs; it is not an independent replication or a population prevalence estimate.
- Output order is supported mainly by sealed source control flow because output mtimes fall in the same second and no syscall trace exists.

## Claim boundary

Allowed: in this retrospective, outcome-enriched, fixed 16-pair CPU screen, ErrorToken does not enrich route-positive targets over the exact matched null or either simple baseline.

Not allowed: online effectiveness, an executed guard, latency/quality benefit, natural prevalence, cross-workload generalization, serving, EP, multi-GPU, or production claims.

## Next step

Do not tune the thresholds. The standalone ErrorToken formulation stops here. Any successor must first compile propagation witnesses on an exploration split, freeze the resulting selector, and then execute actual pack surgery on new natural heterogeneous held-out co-batches.

# Streaming recovery: targeted integrity review

Integrity **PASS**; scope **WARN**; unresolved **P0=0, P1=0** for the final `MEASUREMENT_ONLY` claim.

One fresh GPT-5.6-Sol reviewer (`/root/streaming_audit`, same-family/provisional) reviewed the frozen package, retained execution/readback records, raw examples, analyzer, resource supplement and RESULTS.md. The bounded review is complete; no additional audit round or GPU rerun is required for this iteration. This is integrity review, not independent experimental replication.

| Check | Finding |
|---|---|
| Sources / future information | Frozen WikiText prompts supply no outcome labels. Runtime selectors use present state and declared length caps, not future EOS. |
| Identity / time / volume | Request IDs, prompt hashes, arrivals, output prefixes, terminal events and engine-return boundaries are reconciled. EOS IDs are separated from non-EOS IDs and client-visible text. |
| Real execution / policy state | Four fresh engines completed N/M/M/N with per-cell GPU checks, warmup/drain/reset and actual schedule receipts. Archive/readback retention is consistent; transfer retry was not a GPU retry. |
| Accounting | Scheduler, remaining engine and outside-engine wall buckets conserve the request window. Recovery call spans are inclusive, not pure recomputation tax; host command wall is separate. |
| Actual intervention | Open adapter coverage is complete, but forced rotations are zero. Nine natural recoveries all complete after 996–1024 further returned IDs; no method treatment effect follows. |
| Scope / baseline | Same frozen resources, input and runtime across arms. One model and one domain; 58/64 requests per cell hit the length cap, trajectories differ, and the host limit is shared. These limits remain explicit. |

Reviewer checked actual stop-event tails: EOS50279 is returned as a new ID while the native specific stop reason is None. The first analysis's stricter reason-based count and the supplemental ID-based confirmation are distinct definitions; neither raw nor first analysis was overwritten.

No unverified performance, quality, client-receipt/SLO, full natural-EOS distribution, connector-byte or independently hard-capped host claim is accepted. Such extensions would need additional evidence, but are absent from the final report.

Reviewed entry points: [RESULTS.md](RESULTS.md), [analysis.json](analysis/analysis.json), [resource_summary.json](analysis/resource_summary.json), [frozen protocol](PROTOCOL.md). Machine-readable verdict: [EXPERIMENT_AUDIT.json](EXPERIMENT_AUDIT.json).

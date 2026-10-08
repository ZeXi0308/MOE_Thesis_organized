# Suffix prototype: per-cell results

Group status: **COMPLETE**. Baseline: `00_n2fixed0c`.

All discovered cells are retained. Time includes complete served/drained cost; initialization and warmup follow the existing capture boundary.

| Cell | Analysis / run | Drained s | Returned tokens | Token/s | Mean flow s | Mean TTFT s | Max chunk gap s | Correct | Expert GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_n2fixed0c | COMPLETE / COMPLETE | 18.2983 | 1504 | 82.1933 | 12.1432 | 2.5467 | 0.8246 | 9/16 | 643.3140 |

| Cell | Cut steps / request-steps / unique requests | Removed draft rows | Sampler tokens before stop | Returned tokens | Decision / repack s | Local union eliminated GB | Decision-call after-load matches |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00_n2fixed0c | 195/797/16 | 1594 | 1504 | 1504 | 0.0465 / 0.0125 | 18.4843 | 195/195 |

Local union eliminated bytes describe the same decision-layer state, not a measured whole-episode byte reduction. Later layers contribute only their actual loads. Horizon time savings remain predictions.
Decision/repack and CUDA spans already lie inside measured service time. MoE-local gather/scatter has no isolated timer. The archived r01 source records route-to-host before controller.select (D2H/tolist only); host_apply includes controller and MoE-local gather/scatter. Optional cross-layer compaction timers appear separately below. These diagnostics are not added to engine wall.

| Cell | Compact / measured forwards | Fallback / failed | Tail query row-layer positions: original → actual | LM-head rows: original → executed | Causal q/seq matches | Pager tail matches |
|---|---:|---:|---:|---:|---:|---:|
| 00_n2fixed0c | 195/201 | 0/0 | 182400 → 158490 | 3090 → 1496 | 195/195 | 195/195 |

Tail rows come from recorded per-layer execution inputs; causal checks verify the recorded prefix plan, not independent GPU metadata readback. The decision-layer attention keeps its original shape. LM-head rows refer to compacted calls only.

| Cell | Compact setup host s | Hidden scatter host s | Logits adapter host s | Forward host s (inclusive) | Fallback reasons |
|---|---:|---:|---:|---:|---|
| 00_n2fixed0c | 0.0373 | 0.0058 | 0.0274 | 17.8362 | {"no_cut": 6} |

Forward host includes compact setup and hidden scatter. The logits adapter includes the reduced LM-head call plus its gather/scatter and also lies inside engine.step. These host timers include submissions/possible waits; their sums are never added to engine or CUDA time.


Output lengths and contents may differ. Multi-token chunks do not resolve token ITL; scores on this small task do not establish quality equivalence.

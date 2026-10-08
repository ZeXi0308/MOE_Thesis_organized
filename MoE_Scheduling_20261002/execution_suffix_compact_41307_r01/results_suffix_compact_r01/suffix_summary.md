# Suffix prototype: per-cell results

Group status: **COMPLETE**. Baseline: `00_ar16`.

All discovered cells are retained. Time includes complete served/drained cost; initialization and warmup follow the existing capture boundary.

| Cell | Analysis / run | Drained s | Returned tokens | Token/s | Mean flow s | Mean TTFT s | Max chunk gap s | Correct | Expert GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_ar16 | COMPLETE / COMPLETE | 16.6931 | 1416 | 84.8252 | 11.7361 | 2.4782 | 1.1570 | 8/16 | 595.6121 |
| 01_n1off | COMPLETE / COMPLETE | 16.7699 | 1423 | 84.8546 | 11.0203 | 2.4159 | 1.4625 | 8/16 | 609.3275 |
| 02_n4horizon | COMPLETE / COMPLETE | 18.9222 | 1620 | 85.6136 | 12.9500 | 2.5792 | 0.4160 | 8/16 | 710.2551 |
| 03_n4horizonc | COMPLETE / COMPLETE | 18.2190 | 1498 | 82.2220 | 12.3554 | 2.6107 | 0.4236 | 8/16 | 666.9069 |
| 04_n4horizonc | COMPLETE / COMPLETE | 17.0589 | 1416 | 83.0068 | 11.7600 | 2.5916 | 0.4320 | 8/16 | 619.9852 |
| 05_n4horizon | COMPLETE / COMPLETE | 19.1762 | 1620 | 84.4796 | 13.1155 | 2.6140 | 0.4242 | 8/16 | 710.2551 |
| 06_n1off | COMPLETE / COMPLETE | 16.7887 | 1423 | 84.7595 | 11.0494 | 2.4279 | 1.4615 | 8/16 | 609.3275 |
| 07_ar16 | COMPLETE / COMPLETE | 16.6143 | 1416 | 85.2279 | 11.7107 | 2.4807 | 1.1485 | 8/16 | 595.6121 |

| Cell | Cut steps / request-steps / unique requests | Removed draft rows | Sampler tokens before stop | Returned tokens | Decision / repack s | Local union eliminated GB | Decision-call after-load matches |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00_ar16 | 0/0/0 | — | — | 1416 | 0 / — | 0.0000 | 0/0 |
| 01_n1off | 0/0/0 | 0 | 1423 | 1423 | 0.0360 / 0.0006 | 0.0000 | 154/154 |
| 02_n4horizon | 87/210/16 | 615 | 1620 | 1620 | 0.0510 / 0.0284 | 8.7200 | 143/143 |
| 03_n4horizonc | 103/196/14 | 572 | 1498 | 1498 | 0.0502 / 0.0140 | 10.4816 | 146/146 |
| 04_n4horizonc | 93/186/14 | 558 | 1416 | 1416 | 0.0467 / 0.0125 | 9.3114 | 136/136 |
| 05_n4horizon | 87/210/16 | 615 | 1620 | 1620 | 0.0530 / 0.0288 | 8.7200 | 143/143 |
| 06_n1off | 0/0/0 | 0 | 1423 | 1423 | 0.0356 / 0.0006 | 0.0000 | 154/154 |
| 07_ar16 | 0/0/0 | — | — | 1416 | 0 / — | 0.0000 | 0/0 |

Local union eliminated bytes describe the same decision-layer state, not a measured whole-episode byte reduction. Later layers contribute only their actual loads. Horizon time savings remain predictions.
Decision/repack and CUDA spans already lie inside measured service time. MoE-local gather/scatter has no isolated timer. The archived r01 source records route-to-host before controller.select (D2H/tolist only); host_apply includes controller and MoE-local gather/scatter. Optional cross-layer compaction timers appear separately below. These diagnostics are not added to engine wall.

| Cell | Compact / measured forwards | Fallback / failed | Tail query row-layer positions: original → actual | LM-head rows: original → executed | Causal q/seq matches | Pager tail matches |
|---|---:|---:|---:|---:|---:|---:|
| 03_n4horizonc | 103/153 | 0/0 | 27420 → 18840 | 1826 → 1254 | 103/103 | 103/103 |
| 04_n4horizonc | 93/151 | 0/0 | 25365 → 16995 | 1689 → 1131 | 93/93 | 93/93 |

Tail rows come from recorded per-layer execution inputs; causal checks verify the recorded prefix plan, not independent GPU metadata readback. The decision-layer attention keeps its original shape. LM-head rows refer to compacted calls only.

| Cell | Compact setup host s | Hidden scatter host s | Logits adapter host s | Forward host s (inclusive) | Fallback reasons |
|---|---:|---:|---:|---:|---|
| 03_n4horizonc | 0.0369 | 0.0035 | 0.0163 | 17.7964 | {"no_cut": 50} |
| 04_n4horizonc | 0.0331 | 0.0032 | 0.0142 | 16.6659 | {"no_cut": 58} |

Forward host includes compact setup and hidden scatter. The logits adapter includes the reduced LM-head call plus its gather/scatter and also lies inside engine.step. These host timers include submissions/possible waits; their sums are never added to engine or CUDA time.

- `01_n1off` / `00_ar16`: drained ratio 1.0046, actual output-rate ratio 1.0003; identical complete outputs 7/16, same extracted answers 12/16.
- `02_n4horizon` / `00_ar16`: drained ratio 1.1335, actual output-rate ratio 1.0093; identical complete outputs 5/16, same extracted answers 10/16.
- `03_n4horizonc` / `00_ar16`: drained ratio 1.0914, actual output-rate ratio 0.9693; identical complete outputs 10/16, same extracted answers 13/16.
- `04_n4horizonc` / `00_ar16`: drained ratio 1.0219, actual output-rate ratio 0.9786; identical complete outputs 8/16, same extracted answers 13/16.
- `05_n4horizon` / `00_ar16`: drained ratio 1.1487, actual output-rate ratio 0.9959; identical complete outputs 5/16, same extracted answers 10/16.
- `06_n1off` / `00_ar16`: drained ratio 1.0057, actual output-rate ratio 0.9992; identical complete outputs 7/16, same extracted answers 12/16.
- `07_ar16` / `00_ar16`: drained ratio 0.9953, actual output-rate ratio 1.0047; identical complete outputs 16/16, same extracted answers 16/16.

Output lengths and contents may differ. Multi-token chunks do not resolve token ITL; scores on this small task do not establish quality equivalence.

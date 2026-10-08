# Suffix prototype: per-cell results

Group status: **COMPLETE**. Baseline: `00_expert24`.

All discovered cells are retained. Time includes complete served/drained cost; initialization and warmup follow the existing capture boundary.

| Cell | Analysis / run | Drained s | Returned tokens | Token/s | Mean flow s | Mean TTFT s | Max chunk gap s | Correct | Expert GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_expert24 | COMPLETE / COMPLETE | 16.5777 | 1416 | 85.4160 | 11.6648 | 2.4586 | 1.1957 | 8/16 | 595.6121 |
| 01_expert23 | COMPLETE / COMPLETE | 16.3772 | 1441 | 87.9883 | 11.6762 | 1.6642 | 0.6164 | 8/16 | 604.9235 |
| 02_expert20 | COMPLETE / COMPLETE | 17.0311 | 1416 | 83.1419 | 11.5021 | 0.8146 | 0.2684 | 9/16 | 615.0024 |
| 03_expert20 | COMPLETE / COMPLETE | 17.1448 | 1416 | 82.5907 | 11.5699 | 0.8201 | 0.2711 | 9/16 | 615.0024 |
| 04_expert23 | COMPLETE / COMPLETE | 16.3426 | 1441 | 88.1744 | 11.6370 | 1.6462 | 0.6150 | 8/16 | 604.9235 |
| 05_expert24 | COMPLETE / COMPLETE | 16.5521 | 1416 | 85.5481 | 11.6510 | 2.4591 | 1.1868 | 8/16 | 595.6121 |

| Cell | Cut steps / request-steps / unique requests | Removed draft rows | Sampler tokens before stop | Returned tokens | Decision / repack s | Local union eliminated GB | Decision-call after-load matches |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00_expert24 | 0/0/0 | — | — | 1416 | 0 / — | 0.0000 | 0/0 |
| 01_expert23 | 0/0/0 | — | — | 1441 | 0 / — | 0.0000 | 0/0 |
| 02_expert20 | 0/0/0 | — | — | 1416 | 0 / — | 0.0000 | 0/0 |
| 03_expert20 | 0/0/0 | — | — | 1416 | 0 / — | 0.0000 | 0/0 |
| 04_expert23 | 0/0/0 | — | — | 1441 | 0 / — | 0.0000 | 0/0 |
| 05_expert24 | 0/0/0 | — | — | 1416 | 0 / — | 0.0000 | 0/0 |

Local union eliminated bytes describe the same decision-layer state, not a measured whole-episode byte reduction. Later layers contribute only their actual loads. Horizon time savings remain predictions.
Decision/repack and CUDA spans already lie inside measured service time. MoE-local gather/scatter has no isolated timer. The archived r01 source records route-to-host before controller.select (D2H/tolist only); host_apply includes controller and MoE-local gather/scatter. Optional cross-layer compaction timers appear separately below. These diagnostics are not added to engine wall.

- `01_expert23` / `00_expert24`: drained ratio 0.9879, actual output-rate ratio 1.0301; identical complete outputs 8/16, same extracted answers 12/16.
- `02_expert20` / `00_expert24`: drained ratio 1.0274, actual output-rate ratio 0.9734; identical complete outputs 7/16, same extracted answers 12/16.
- `03_expert20` / `00_expert24`: drained ratio 1.0342, actual output-rate ratio 0.9669; identical complete outputs 7/16, same extracted answers 12/16.
- `04_expert23` / `00_expert24`: drained ratio 0.9858, actual output-rate ratio 1.0323; identical complete outputs 8/16, same extracted answers 12/16.
- `05_expert24` / `00_expert24`: drained ratio 0.9985, actual output-rate ratio 1.0015; identical complete outputs 16/16, same extracted answers 16/16.

Output lengths and contents may differ. Multi-token chunks do not resolve token ITL; scores on this small task do not establish quality equivalence.

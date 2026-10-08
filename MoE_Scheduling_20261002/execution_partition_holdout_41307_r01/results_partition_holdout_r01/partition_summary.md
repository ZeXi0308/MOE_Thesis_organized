# Suffix prototype: per-cell results

Group status: **COMPLETE**. Baseline: `00_expert24`.

All discovered cells are retained. Time includes complete served/drained cost; initialization and warmup follow the existing capture boundary.

| Cell | Analysis / run | Drained s | Returned tokens | Token/s | Mean flow s | Mean TTFT s | Max chunk gap s | Correct | Expert GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_expert24 | COMPLETE / COMPLETE | 15.5452 | 1270 | 81.6972 | 10.4485 | 2.6900 | 1.0957 | 8/16 | 541.7824 |
| 01_expert23 | COMPLETE / COMPLETE | 21.3268 | 1543 | 72.3504 | 11.2459 | 1.8021 | 0.8762 | 8/16 | 674.8342 |
| 02_expert20 | COMPLETE / COMPLETE | 14.9190 | 1310 | 87.8072 | 11.5206 | 0.8517 | 0.2705 | 9/16 | 562.3429 |
| 03_expert20 | COMPLETE / COMPLETE | 15.0014 | 1310 | 87.3250 | 11.5648 | 0.8477 | 0.2669 | 9/16 | 562.3429 |
| 04_expert23 | COMPLETE / COMPLETE | 21.3347 | 1543 | 72.3234 | 11.2169 | 1.7846 | 0.8776 | 8/16 | 674.8342 |
| 05_expert24 | COMPLETE / COMPLETE | 15.5309 | 1270 | 81.7722 | 10.4293 | 2.6836 | 1.0964 | 8/16 | 541.7824 |

| Cell | Cut steps / request-steps / unique requests | Removed draft rows | Sampler tokens before stop | Returned tokens | Decision / repack s | Local union eliminated GB | Decision-call after-load matches |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00_expert24 | 0/0/0 | — | — | 1270 | 0 / — | 0.0000 | 0/0 |
| 01_expert23 | 0/0/0 | — | — | 1543 | 0 / — | 0.0000 | 0/0 |
| 02_expert20 | 0/0/0 | — | — | 1310 | 0 / — | 0.0000 | 0/0 |
| 03_expert20 | 0/0/0 | — | — | 1310 | 0 / — | 0.0000 | 0/0 |
| 04_expert23 | 0/0/0 | — | — | 1543 | 0 / — | 0.0000 | 0/0 |
| 05_expert24 | 0/0/0 | — | — | 1270 | 0 / — | 0.0000 | 0/0 |

Local union eliminated bytes describe the same decision-layer state, not a measured whole-episode byte reduction. Later layers contribute only their actual loads. Horizon time savings remain predictions.
Decision/repack and CUDA spans already lie inside measured service time. MoE-local gather/scatter has no isolated timer. The archived r01 source records route-to-host before controller.select (D2H/tolist only); host_apply includes controller and MoE-local gather/scatter. Optional cross-layer compaction timers appear separately below. These diagnostics are not added to engine wall.

- `01_expert23` / `00_expert24`: drained ratio 1.3719, actual output-rate ratio 0.8856; identical complete outputs 8/16, same extracted answers 13/16.
- `02_expert20` / `00_expert24`: drained ratio 0.9597, actual output-rate ratio 1.0748; identical complete outputs 8/16, same extracted answers 13/16.
- `03_expert20` / `00_expert24`: drained ratio 0.9650, actual output-rate ratio 1.0689; identical complete outputs 8/16, same extracted answers 13/16.
- `04_expert23` / `00_expert24`: drained ratio 1.3724, actual output-rate ratio 0.8853; identical complete outputs 8/16, same extracted answers 13/16.
- `05_expert24` / `00_expert24`: drained ratio 0.9991, actual output-rate ratio 1.0009; identical complete outputs 16/16, same extracted answers 16/16.

Output lengths and contents may differ. Multi-token chunks do not resolve token ITL; scores on this small task do not establish quality equivalence.

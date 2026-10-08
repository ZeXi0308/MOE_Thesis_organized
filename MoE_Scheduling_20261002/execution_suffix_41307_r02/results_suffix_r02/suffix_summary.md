# Suffix prototype: per-cell results

Group status: **COMPLETE**. Baseline: `00_n1off`.

All discovered cells are retained. Time includes complete served/drained cost; initialization and warmup follow the existing capture boundary.

| Cell | Analysis / run | Drained s | Returned tokens | Token/s | Mean flow s | Mean TTFT s | Max chunk gap s | Correct | Expert GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_n1off | COMPLETE / COMPLETE | 16.9399 | 1423 | 84.0027 | 11.1398 | 2.4412 | 1.4762 | 8/16 | 609.3275 |
| 01_n4horizon | COMPLETE / COMPLETE | 18.8731 | 1620 | 85.8366 | 12.9164 | 2.5705 | 0.4143 | 8/16 | 710.2551 |
| 02_n4horizon | COMPLETE / COMPLETE | 19.1303 | 1636 | 85.5190 | 12.8981 | 2.5901 | 0.4163 | 8/16 | 704.2027 |
| 03_n1off | COMPLETE / COMPLETE | 16.8607 | 1423 | 84.3972 | 11.0935 | 2.4342 | 1.4701 | 8/16 | 609.3275 |

| Cell | Cut steps / request-steps / unique requests | Removed draft rows | Sampler tokens before stop | Returned tokens | Decision / repack s | Local union eliminated GB | Decision-call after-load matches |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00_n1off | 0/0/0 | 0 | 1423 | 1423 | 0.0356 / 0.0006 | 0.0000 | 154/154 |
| 01_n4horizon | 87/210/16 | 615 | 1620 | 1620 | 0.0520 / 0.0278 | 8.7200 | 143/143 |
| 02_n4horizon | 101/231/16 | 669 | 1636 | 1636 | 0.0524 / 0.0302 | 10.9849 | 155/155 |
| 03_n1off | 0/0/0 | 0 | 1423 | 1423 | 0.0358 / 0.0006 | 0.0000 | 154/154 |

Local union eliminated bytes describe the same decision-layer state, not a measured whole-episode byte reduction. Later layers contribute only their actual loads. Horizon time savings remain predictions.
Decision/repack and CUDA spans already lie inside measured service time. Gather/scatter has no isolated timer. The archived r01 source records route-to-host before controller.select (D2H/tolist only); host_apply includes controller and gather/scatter. These diagnostics are not added to engine wall.

- `01_n4horizon` / `00_n1off`: drained ratio 1.1141, actual output-rate ratio 1.0218; identical complete outputs 6/16, same extracted answers 11/16.
- `02_n4horizon` / `00_n1off`: drained ratio 1.1293, actual output-rate ratio 1.0180; identical complete outputs 7/16, same extracted answers 10/16.
- `03_n1off` / `00_n1off`: drained ratio 0.9953, actual output-rate ratio 1.0047; identical complete outputs 16/16, same extracted answers 16/16.

Output lengths and contents may differ. Multi-token chunks do not resolve token ITL; scores on this small task do not establish quality equivalence.

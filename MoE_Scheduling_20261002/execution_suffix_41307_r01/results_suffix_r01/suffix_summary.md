# Suffix prototype: per-cell results

Group status: **COMPLETE**. Baseline: `00_n2off`.

All discovered cells are retained. Time includes complete served/drained cost; initialization and warmup follow the existing capture boundary.

| Cell | Analysis / run | Drained s | Returned tokens | Token/s | Mean flow s | Mean TTFT s | Max chunk gap s | Correct | Expert GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_n2off | COMPLETE / COMPLETE | 16.0889 | 1324 | 82.2927 | 10.5991 | 2.3799 | 0.2599 | 8/16 | 597.1347 |
| 01_n2fixed0 | COMPLETE / COMPLETE | 17.8211 | 1436 | 80.5787 | 11.7287 | 2.3697 | 0.8011 | 9/16 | 612.5613 |
| 02_n2horizon | COMPLETE / COMPLETE | 16.9993 | 1407 | 82.7680 | 11.3930 | 2.4907 | 0.2609 | 7/16 | 630.0516 |

| Cell | Cut steps / request-steps / unique requests | Removed draft rows | Sampler tokens before stop | Returned tokens | Decision / repack s | Local union eliminated GB | Decision-call after-load matches |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00_n2off | 0/0/0 | 0 | 1324 | 1324 | 0.1356 / 0.0005 | 0.0000 | 127/127 |
| 01_n2fixed0 | 190/766/16 | 1532 | 1436 | 1436 | 0.0440 / 0.0162 | 17.4651 | 190/190 |
| 02_n2horizon | 52/89/15 | 134 | 1407 | 1407 | 0.0428 / 0.0219 | 3.9133 | 137/137 |

Local union eliminated bytes describe the same decision-layer state, not a measured whole-episode byte reduction. Later layers contribute only their actual loads. Horizon time savings remain predictions.
Decision/repack and CUDA spans already lie inside measured service time. Gather/scatter has no isolated timer. The archived r01 source records route-to-host before controller.select (D2H/tolist only); host_apply includes controller and gather/scatter. These diagnostics are not added to engine wall.

- `01_n2fixed0` / `00_n2off`: drained ratio 1.1077, actual output-rate ratio 0.9792; identical complete outputs 6/16, same extracted answers 10/16.
- `02_n2horizon` / `00_n2off`: drained ratio 1.0566, actual output-rate ratio 1.0058; identical complete outputs 6/16, same extracted answers 10/16.

Output lengths and contents may differ. Multi-token chunks do not resolve token ITL; scores on this small task do not establish quality equivalence.

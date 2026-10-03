# SemanticFence row-safety predictability probe

- Decision: `KILL_C09_V1_ZERO_ERROR_ADMISSION`
- Paper result: `false`
- Evidence boundary: `exploratory_reused_run03_calibration_m2_labels_document_disjoint_cpu_prediction_only_not_fresh_not_certificate_not_serving_not_ep_not_latency`

| Model | Held-out FP | Held-out TP | Safe coverage | Documents | Layer/expert cells |
|---|---:|---:|---:|---:|---:|
| M0_shape_control | 5 | 1 | 0.036127% | 5 | 2 |
| M1_input_value | 14 | 0 | 0.000000% | 6 | 12 |

## Interpretation

- input-value model admitted at least one held-out unsafe row
- Thresholds were selected only from validation unsafe scores; each document was tested exactly once.
- This is reused calibration evidence, not fresh generalization or a sound certificate.
- No latency, serving, EP, multi-GPU, or paper claim is authorized.

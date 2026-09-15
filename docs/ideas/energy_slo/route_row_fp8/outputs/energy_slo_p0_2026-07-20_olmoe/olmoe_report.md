# Energy-SLO P0: Real GPU Power Measurement (olmoe)

## Batch size vs energy-per-token (real nvidia-smi power draw)

| batch_size | seq_len | throughput_tokens_per_s | mean_power_w | energy_per_token_mj |
|---|---|---|---|---|
| 1 | 64 | 450.0654 | 106.2329 | 235.8716 |
| 4 | 64 | 1705.2009 | 130.8836 | 75.9275 |
| 16 | 64 | 6213.3511 | 169.4497 | 27.2231 |
| 64 | 64 | 19550.9898 | 267.4634 | 13.5869 |

## Real FP8 tensor-core GEMM vs real bf16 GEMM (matched expert gate_proj matmul size)

| precision | matmuls_per_s | mean_power_w | energy_per_matmul_mj |
|---|---|---|---|
| bf16 | 9974.8125 | 430.3195 | 42.7065 |
| fp8_e4m3_real_tensor_core | 20229.4561 | 573.9243 | 28.0766 |
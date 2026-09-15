# Homogeneous Lane Codec Gate

GPU: `NVIDIA GeForce RTX 5090` (AutoDL)

## Evidence boundary

single-GPU Triton pack/unpack plus pinned H2D; analytic wire time at given Gbps; not NCCL/RDMA; no incast, collective headers, or multi-node queueing

Note: local tables rebuilt from GPU stdout p50 timings after remote scp was blocked; p95 columns mirror p50.

## Pre-registered serving-point summary (homo_fp8)

- cells: 8
- viable: 0 (0.0%)
- recommend `require_positive_net_saving=True`: **True**

### Per-cell

| rows | hidden | gbps | net_p50_us | viable |
|---:|---:|---:|---:|:---:|
| 128 | 512 | 200.0 | -56.167 | N |
| 128 | 512 | 400.0 | -57.468 | N |
| 512 | 512 | 200.0 | -48.012 | N |
| 512 | 512 | 400.0 | -53.214 | N |
| 128 | 2048 | 200.0 | -47.871 | N |
| 128 | 2048 | 400.0 | -53.103 | N |
| 512 | 2048 | 200.0 | -29.115 | N |
| 512 | 2048 | 400.0 | -50.045 | N |

## Codec tax table (p50)

| mode | rows | hidden | pack_us | unpack_us | h2d_us | codec_total | net@200Gbps | break_even_gbps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_bf16 | 32 | 512 | 0.000 | 0.000 | 11.328 | 0.000 | 0.000 | 0.000 |
| homo_fp8 | 32 | 512 | 24.192 | 23.424 | 11.232 | 58.848 | -58.198 | 2.210 |
| homo_int4 | 32 | 512 | 23.488 | 23.392 | 11.296 | 58.176 | -57.198 | 3.362 |
| baseline_bf16 | 128 | 512 | 0.000 | 0.000 | 11.232 | 0.000 | 0.000 | 0.000 |
| homo_fp8 | 128 | 512 | 23.984 | 23.392 | 11.392 | 58.768 | -56.167 | 8.852 |
| homo_int4 | 128 | 512 | 23.328 | 23.296 | 11.232 | 57.856 | -53.944 | 13.522 |
| baseline_bf16 | 512 | 512 | 0.000 | 0.000 | 13.952 | 0.000 | 0.000 | 0.000 |
| homo_fp8 | 512 | 512 | 24.080 | 23.296 | 11.040 | 58.416 | -48.012 | 35.620 |
| homo_int4 | 512 | 512 | 23.312 | 23.296 | 11.168 | 57.776 | -42.129 | 54.163 |
| baseline_bf16 | 2048 | 512 | 0.000 | 0.000 | 42.080 | 0.000 | 0.000 | 0.000 |
| homo_fp8 | 2048 | 512 | 24.128 | 23.680 | 23.520 | 71.328 | -29.713 | 116.687 |
| homo_int4 | 2048 | 512 | 23.504 | 23.616 | 14.080 | 61.200 | 1.387 | 204.532 |
| baseline_bf16 | 32 | 2048 | 0.000 | 0.000 | 11.264 | 0.000 | 0.000 | 0.000 |
| homo_fp8 | 32 | 2048 | 24.192 | 23.584 | 11.264 | 59.040 | -56.424 | 8.863 |
| homo_int4 | 32 | 2048 | 23.328 | 23.232 | 11.200 | 57.760 | -53.833 | 13.598 |
| baseline_bf16 | 128 | 2048 | 0.000 | 0.000 | 13.952 | 0.000 | 0.000 | 0.000 |
| homo_fp8 | 128 | 2048 | 23.968 | 23.296 | 11.072 | 58.336 | -47.871 | 35.879 |
| homo_int4 | 128 | 2048 | 23.424 | 23.440 | 11.232 | 58.096 | -42.388 | 54.077 |
| baseline_bf16 | 512 | 2048 | 0.000 | 0.000 | 41.952 | 0.000 | 0.000 | 0.000 |
| homo_fp8 | 512 | 2048 | 24.192 | 23.360 | 23.424 | 70.976 | -29.115 | 117.959 |
| homo_int4 | 512 | 2048 | 23.680 | 23.520 | 13.984 | 61.184 | 1.649 | 205.389 |
| baseline_bf16 | 2048 | 2048 | 0.000 | 0.000 | 153.504 | 0.000 | 0.000 | 0.000 |
| homo_fp8 | 2048 | 2048 | 24.192 | 23.296 | 79.232 | 126.720 | 40.724 | 264.275 |
| homo_int4 | 2048 | 2048 | 23.424 | 23.648 | 42.048 | 89.120 | 162.211 | 564.027 |

## Interpretation

At common serving points (rows∈{128,512}, 200–400Gbps) **homo_fp8 is never lane-viable**: codec+H2D tax exceeds analytic BF16→FP8 wire saving. Only very large tiles (e.g. rows=2048, hidden=2048 at ≤~264Gbps break-even) show positive net. Default online policies should set `require_positive_net_saving=True`.

## Incremental FP8→INT4 (online policy question)

Baseline = FP8 wire; action = homogeneous INT4; tax = INT4 pack+h2d+unpack.

- serving cells viable: 0/8 (0.0%)
- recommend hard-gate for online INT4: **True**

| rows | hidden | gbps | net_p50_us | viable |
|---:|---:|---:|---:|:---:|
| 128 | 512 | 200.0 | -56.545 | N |
| 128 | 512 | 400.0 | -57.201 | N |
| 128 | 2048 | 200.0 | -52.853 | N |
| 128 | 2048 | 400.0 | -55.475 | N |
| 512 | 512 | 200.0 | -52.533 | N |
| 512 | 512 | 400.0 | -55.155 | N |
| 512 | 2048 | 200.0 | -40.212 | N |
| 512 | 2048 | 400.0 | -50.698 | N |

Note: BF16→FP8 Phase-A cells answer a different question than online FP8→INT4.

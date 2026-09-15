# Logical-alignment performance r01 — bounded integrity audit

## Verdict

**PASS_BOUNDED_INTEGRITY — P0: 0, P1: 0.** The retained six-cell run supports only the frozen, descriptive conclusion: on this one 16-document OLMoE finite-arrival cohort, Y had lower complete-capture time than both F and X in both mirrored blocks, with no repeated adverse maximum-ITL result. This is enough to follow the frozen rule and run one independent workload check. It is not evidence of a stable benefit, a noise floor, statistical significance, task-quality preservation, an SLO result, or method GO.

Reviewer independence is **same-family fresh requested-route**. Therefore this PASS is **provisional** under the experiment-audit reviewer-independence policy. No `.aris` trace was created because the repository task explicitly prohibited creating an unrequested trace.

## Audit scope and evidence

This audit read the frozen protocol, six-cell plan, input hashes and execution freeze; all 11 frozen runtime source files; `instrumentation/{run_covered,compile_domain,host_probe,logical_execution}.py`; `run_attempt.py`; and the three analysis files. It independently rebuilt request metrics and denominator invariants from retained `raw.json`, `workload.json`, pager, compiler, logical-execution, resource and execution artifacts. The author analyzer was used only as a value to cross-check after the independent reconstruction.

The previously sealed `logical_alignment_qualification_r01` result is inherited only as a bounded numerical qualification for the frozen invocation domain. Its measurements were not re-audited and are not promoted to cohort-wide output equivalence or quality evidence.

Evidence identity:

- Repository HEAD recorded by the protocol: `de64dae5ea4fa3c92ec605e9847e817d8e68f3ac`.
- Frozen input manifest: 40 files; SHA-256 `32d4a8c15cab53c2eb926cba7d80da046ceebdce45bafba67f233353f1f68d84`.
- Readback: 1470 payload members plus the manifest itself, 1471 files total. Every payload size and SHA-256 matched `readback_manifest.json`; manifest SHA-256 `73ad80c4ddad3305b5559cbe929851c65ef11cd23d9b9c0985b31cc942ebfc3c`.
- `results/execution.json` SHA-256: `3443f39a480286a6b001c4c87caca8fb5f49a205e26896f7b7dbc27af07db45c`.
- Retained `analysis.json` SHA-256: `46de429eb53719041dcfba0058ed89b10b7731602b9f989eaa9ad3610e9269ab`.
- Strict local parsing covered 437 JSON files and 12 JSONL files, 6503 objects total, with no duplicate key or non-finite value found.

## P0/P1 findings

No P0 or P1 finding was found.

The cross-arm generated-token and route differences are a material claim boundary, not an integrity failure for this policy-specific full-request measurement. Same-arm repeats had identical request outputs and route hashes. Across arms, F/Y had 13 of 16 equal request outputs and X/Y had 14 of 16 in each block; route hashes differed. Thus the timing comparison retains each arm's actual downstream trajectory, while this run cannot establish task-quality equivalence or attribute the entire completion difference to a trajectory-invariant local kernel saving.

## Execution and call-path integrity

The retained order is exactly `0_F, 1_X, 2_Y, 3_Y, 4_X, 5_F`; every subprocess returned 0, every cell status is COMPLETE, and every cell passed the recorded compile-coverage qualification. The driver hashes the 40 frozen inputs before launch, uses a private Triton cache per engine, holds the declared whole-group cooperative flock, checks the GPU before every fresh engine, rejects a nonempty private cache, and keeps all six outputs without replacement or retry.

The actual call path is consistent with the intended arms:

- F uses fullstage with private `20×16` plus stage 64 and the original global-64 fused-experts path.
- X uses oneshot with private `21×16` plus stage 48 and the original physical-384 alignment path.
- Y uses the same physical oneshot resources and copy plan as X, changes alignment to global 64 with a 64-entry physical-slot map, then invokes the original fused-experts function once.

`run_shared_pool_pager.py` performs copies before one top-level `fused_experts` invocation per pager record and returns that result. In Y, `logical_execution.py` records exactly one alignment-helper call for that one top-level invocation, changes `ignore_invalid_experts=True` to the qualified effective value `False`, then restores the assignment helper. This wording is deliberately top-level: the fused-experts implementation contains its two GEMM kernel invocations; the evidence does not claim one CUDA kernel launch in total.

Per cell, the retained pager accounting is:

| Phase | Pager/top-level calls | Token rows across 16 layers | Ordered identity SHA-256 |
|---|---:|---:|---|
| initialization | 32 | 2816 | `37d98f01c7b6c4d92ee94490363a8d04f65db09307c94cc62e39179291920572` |
| warmup | 80 | 2064 | `a1b8b03d4a3a0edd55d0b881650be84333821874339711cfa8a93d049a3d72e2` |
| measurement F | 896 | 40704 | `7af3871f19785751d983dddfa0bc007f8b9f3768005f62603854dc84e8a052e3` |
| measurement X | 896 | 40704 | `9ef46d96560fbde655e33601db13823bcbb1835f98743890250e8fc3a05befac` |
| measurement Y | 896 | 40704 | `222b7109f1c701c78321b5b954a08c8cf0f50ea99bb2bfec074c51cb7c1b3b38` |

The initialization and warmup identities are the same in all arms; each measurement identity is the same in both repeats of its arm. The measurement identity was independently rebuilt in trace order from `[call_id, layer_name, step_id, rows]`; the rebuilt Y digest equals the separately recorded logical-wrapper digest.

For both Y cells, every phase has `calls == completed_calls == helper_calls`, failed calls are zero, helper-token rows equal top-level token rows, and the declared helper parameters are global experts 64, map shape `[64]`, top-k 8, incoming invalid-expert handling true, and effective handling false. `assignment_restored` is true for all three phases and `hooks_restored` is true at shutdown. F/X have logical alignment disabled and empty logical phase records. Reference validation, prefix comparison, numerical readback, negative controls, kernel validation, CPU diagnostics and the runtime variation observer are all disabled/empty in the performance run. There is no qualification double execution hidden in the measurement counts.

## Compile-domain integrity

Every fresh engine started with an empty private Triton disk cache and completed exactly 320 compile-and-load-only invocations: `{M=1..160} × {GEMM1,GEMM2}`. Every invocation is COMPLETE, the 320 pairs are complete and unique by `(M, GEMM)`, and the pager/KV state comparison before and after compile setup is equal.

The realized domains match the protocol:

| Arm | Local experts | Global experts | Mapped | Temporary compile buffers (bytes) |
|---|---:|---:|---|---:|
| F | 64 | 64 | false | 8572420 |
| X | 384 | 384 | true | 8733700 |
| Y | 384 | 64 | true | 8572420 |

All measurement row sizes are within the compiled `M=1..160` domain. Direct filtering of every `jit_probe.json` found **zero events of any kind attributed to the measurement phase in all six cells**, so in particular there was no measurement compiler-pipeline miss. The compile claim remains limited to this MoE Triton family; it does not establish that every runtime backend path was warm.

## Independent denominators and metrics

For every cell, raw data independently gives 16 completed requests, 512 output tokens, 512 one-token output events with valid cumulative prefixes, 56 scheduler steps, 2544 scheduled token positions, no preemption, and no recomputed tokens. The measurement pager denominator is exactly `56 steps × 16 layers = 896` calls, and the layer-token denominator is `2544 × 16 = 40704` rows. The workload identity, 128-token prompt prefixes, arrival times `0..3.75 s` at `0.25 s`, and requested 32 output tokens match across all six cells.

Metrics were rebuilt per request as follows: TTFT is first received token minus planned arrival; TPOT is `(last token - first token)/(32-1)`; request maximum ITL is the maximum adjacent received-token gap; completion latency is completion minus planned arrival. Complete capture is the retained observation end from the first planned arrival through final drain. Requests share an engine, arrival episode and execution timeline, so the 16 requests are not treated as independent statistical samples.

| Cell | Capture (s) | Last completion (s) | Mean completion (s) | Mean TTFT (s) | Mean TPOT (s) | Max request ITL (s) | Process wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0_F | 8.176383 | 8.176209 | 5.474759 | 0.820231 | 0.150146 | 0.190423 | 60.730738 |
| 1_X | 8.271075 | 8.270718 | 5.560276 | 0.834699 | 0.152438 | 0.212861 | 59.046425 |
| 2_Y | 8.006667 | 8.006342 | 5.326670 | 0.792387 | 0.146267 | 0.186193 | 60.040813 |
| 3_Y | 8.033705 | 8.033570 | 5.339523 | 0.800831 | 0.146409 | 0.190086 | 58.164760 |
| 4_X | 8.231974 | 8.231843 | 5.520317 | 0.837149 | 0.151070 | 0.200036 | 60.666077 |
| 5_F | 8.172975 | 8.172833 | 5.481042 | 0.825004 | 0.150195 | 0.198490 | 60.034148 |

The independent values agree with every corresponding retained analyzer value to floating-point serialization tolerance.

Primary capture deltas, where a negative value means the target completed sooner:

| Comparison | Forward | Reverse | Ratio of two-engine means |
|---|---:|---:|---:|
| Y vs F | -2.075680% | -1.704036% | -1.889897% |
| Y vs X | -3.196768% | -2.408519% | -2.803577% |
| X vs F | +1.158111% | +0.721869% | +0.940035% |

Y's maximum request ITL was below both F and X in each block. These are two ordered engines per arm on one repeated cohort. The mirrored order does not produce a noise estimate, and neither the requests nor the two block differences supply an independent `n` for significance.

## Resource, isolation and cost accounting

All arms have 384 unique physical expert slots totaling 4,831,838,208 bytes and 16 unique KV storages totaling 1,073,741,824 bytes on `cuda:0`. F allocates 320 private plus 64 stage slots; X/Y allocate 336 private plus 48 stage slots. X and Y therefore match in physical expert/KV capacity while differing in alignment indexing. The recorded CPU affinity is cores 0–7.

The expected RTX 5090 UUID is present in every cell. The outer pre-cell samples returned successfully with no compute processes; three inner samples per cell also passed, with the post-run sample allowing only the cell process. This proves the recorded sampled boundary plus cooperative group-lock use; it cannot prove the absence of an uncooperative process between samples.

Measurement transfer and host-apply costs were retained:

| Cell | H2D bytes | D2D bytes | Host apply (ms) |
|---|---:|---:|---:|
| 0_F | 352749355008 | 390359678976 | 7536.100 |
| 1_X | 344494964736 | 178375360512 | 7454.509 |
| 2_Y | 346281738240 | 177771380736 | 7340.551 |
| 3_Y | 346281738240 | 177771380736 | 7353.413 |
| 4_X | 344494964736 | 178375360512 | 7404.879 |
| 5_F | 352749355008 | 390359678976 | 7540.634 |

Capture includes the initial finite-arrival transient and final drain but excludes engine setup, compile preparation, warmup, finalization and shutdown. Process wall retains engine startup, compile preparation, warmup, capture, finalization, artifact I/O and shutdown. The parent GPU/input/cache checks are outside each subprocess wall but remain inside the complete six-cell group elapsed time. Nested host plan/apply/load and CUDA spans overlap and are not summed into wall time.

## Required closing fields

- **Verdict:** `PASS_BOUNDED_INTEGRITY`, provisional same-family fresh requested-route review; P0=0, P1=0.
- **Evidence type:** `REQUEST_LEVEL / NATIVE_SERVING_INPROCESS_HOST_CAPTURE` with integrated WiSP pager and covered MoE compile domain.
- **What was measured:** One frozen 16-document, 512-output-token finite-arrival cohort in six fresh engines, including full capture, request/token latency, pager transfers, resource allocation, compile preparation and subprocess wall.
- **What was not measured:** An independent cohort, statistical noise/significance, task quality, output equivalence, a production SLO, steady state, true model-OOM deployment, multi-GPU EP, or all-runtime compile warmth.
- **Strongest baseline:** F is the stronger completion baseline; X is the same-budget original-alignment control. Y beat both on capture in both blocks.
- **Oracle/headroom status:** No oracle was defined or measured. This is a direct policy execution comparison, not an action-oracle result.
- **Claim ceiling:** `MEASUREMENT_ONLY / one-cohort directional existence signal`; proceed only to the frozen independent workload check.
- **Failure category:** None for bounded integrity. Cross-arm trajectories constrain quality and causal-attribution claims.
- **Resurrection condition:** Not applicable; no formulation-level NO-GO is issued here.
- **One next smallest experiment:** Execute the already specified independent new workload cohort with the same six-engine mirrored design and unchanged accounting. Do not use this run to tune a threshold or declare a noise floor.

The research question is answered only at this ceiling: **yes, Y was directionally faster than both F and X on complete capture in both blocks of this single cohort, so the frozen independent-cohort check is warranted; no stable-performance claim is yet supported.**

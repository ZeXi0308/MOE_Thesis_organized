# CRQM Exploratory Pilot Result

## Frozen verdict

`NO_GO_CRQM_L2_EARLY_RECEIVER_CONFLICT`

Both models failed the preregistered primary gate with exact zero headroom.

| Model | Depth pair | Median exact-information CVaR99 gap | Minimum gap | First-admission flip rate |
| --- | --- | ---: | ---: | ---: |
| OLMoE | 2 / 8 | 0.0% | 0.0% | 0.0% |
| LLM-JP | 2 / 8 | 0.0% | 0.0% | 0.0% |

The depth-0 negative control passed with gap/flip both zero. Both sensitivity pairs, 1/4 and
4/16, also produced gap/flip equal to zero for both models. Thresholds were not changed after the
run.

## Execution and independent checks

- RTX 5090 calibration: 1,440 raw trials; depths 0/1/2/4/8/16; 20 warmups + 100 measured trials
  per model/depth.
- Pilot: 64 per-window rows = 2 models x 4 depth pairs x 8 holdout windows.
- All MILP stages reported status 0 and MIP gap <= 1e-7.
- Independently recomputed relative gaps, singleton first-action predicates, matched depth/work
  multisets, route-window sender/receiver invariants, history totals, all summaries, and the frozen
  model-AND gate: zero discrepancies.
- Calibration file SHA-256:
  `4635d03a7d2d1732509b70b9b8c59473b4a60203f28fafdc7274d17d2c26e212`.
- Result file SHA-256:
  `627d1a9934fd4f99e159f0d08b8efa752531210a1f740eb3a0d37948a0801c32`.

## Why the value is exactly zero

With 12 equally weighted completion samples, empirical CVaR99 equals the single maximum
completion. All six equal-size contributions are ready, have fixed destinations, must full-drain,
and can only be reordered. The measured unpack backlog/service dominates the short 200 Gb/s
launch horizon. Once a receiver remains busy, its last completion is determined primarily by its
initial availability plus its fixed number of candidate unpacks. Swapping which receiver owns the
backlog changes the slow receiver's identity but does not create a controllable reduction in the
full-drain maximum. The joint B optimum therefore matches the world-specific R0 optimum.

## Kill boundary

This no-go applies to the frozen object only: clean-v2 native route windows; one expert sender;
six all-ready, equal-size BF16 contributions; at least three token-owner receivers; analytic
200 Gb/s serial cut; measured RTX 5090 unpack-only FIFO availability/service; depths
0/1/2/4/8/16; no future arrival, reroute, replication or changing work; full-drain 12-sample
CVaR99 then mean.

It does not establish that receiver awareness is universally useless. Multi-sender incast,
continuous arrivals, finite deadlines, bounded buffers, heterogeneous service, or actions that
change work/destination are different mechanisms and require a new Phase 1/2 protocol.

Do not rescue CRQM with more seeds, larger synthetic depths, changed bandwidth/accounting,
post-hoc P95 or mean-only metrics, selected windows, relaxed 5%/25% gates, additive tails,
last-missing/combine assumptions, or a controller/bandit on this zero-headroom action space.

## Evidence boundary

The route identities are native OLMoE/LLM-JP calibration data and receiver unpack timings are real
RTX 5090 measurements. Receiver ranks and histories remain virtual L2 replay; the cut is analytic.
This is not real receiver queue measurement, RDMA/NCCL, multi-rank serving, TPOT/P99, or a final
scientific result.

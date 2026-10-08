# Native versus maximum-bound admission: one matched development block

Registered after the completed maximum-bound pilot, before dispatching this block.

**Question:** Does the simple bound rule reproducibly trade longer flow time for enough additional 20 s TTFT / 4 s gap qualified requests to improve complete-episode goodput? The pilot returned 67/128 and 0.7768 requests/s versus the previous matched native runs' 58/60 and 0.7244/0.7471; mean flow worsened 4.7–5.4%. It is a simple reference, not a proposed novel method.

Run exactly four cells in one adjacent shared-lock block: `native_recompute_1`, `native_max_bound_1`, `native_max_bound_2`, `native_recompute_2`. Use the already frozen native V2 cell and maximum-bound V1 cell/adapter; preserve the same 128 viewed articles, external arrivals, cap-1,024 EOS generation, model/runtime, 25 CPU threads, 4,096 usable KV blocks, batch 1,024, engine maximum 32 sequences, no offload and application warmups. No retries or overwriting outputs. The only policy difference is the admission gate during measurement.

Retain every request and complete drain. Report the primary joint goodput, all 20 existing frontier points, TTFT, per-request maximum host-return gap and flow, output tokens/IDs/finish reasons, preemptions and reservation state. Use adjacent native→bound pairs, plus same-arm variation; tokens are not independent repeats. Any mechanism failure remains a failed cell and stops this block.

If both adjacent comparisons preserve the primary sign, retain the bound rule as a stronger development baseline and report its flow/throughput cost. A sign reversal means the pilot's advantage did not repeat; report all cells without additional seeds or tuning. Neither outcome establishes new-method novelty, semantic quality or unseen-data confirmation. Do not expand a parameter grid.

Runner: `C_NATIVE_BOUND_MATCHED_BLOCK_V1.py`, frozen before execution. Remote root: `/root/autodl-tmp/c-research-20260930/c-native-bound-matched-dev-v1`. Allow 900 seconds per cell; check sufficient free disk before launch and each cell. Lock covers the entire necessary adjacent block and all child exits; release it before analysis.

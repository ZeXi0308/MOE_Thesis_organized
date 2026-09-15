# Interpretation and retention addendum

2026-09-08, written after discovering the existing remote campaign and after viewing
its output. This is not a preregistration. The original decisions are retained in
`readback/DECISIONS.md`; no original raw result is modified.

The campaign was already running at `/root/autodl-tmp/moe-aligned-ladder-20260908-r01`
when this task inspected the GPU. The existing runner completed forward and reverse;
this task retrieved each completed engine, verified archive SHA256 against the remote,
and recomputed the retained results. It did not launch a second copy of that campaign.

The frozen F1 interval mixes two quantities: an episode cap and its realized decode
width. A static24 episode can spend much of its time below width17. Moreover, the old
9.30–9.51 ms range comprises medians pooled by width, not a confidence or tolerance
interval; individual old cells already fall outside it. We report the literal frozen
numeric check as well as a separate diagnostic restricted to realized widths17–24.
Neither substitutes for the other; failure of the fixed interval cannot alone falsify
the bucket structure.

F2/F3 compare against historical feedback runs. Both historical orders are retained;
0.301 is only forward steady padding, while reverse steady is 0.157. Changing `--caps`
also changes the warmup set. Therefore old/new differences cannot isolate ladder
alignment. F4, comparison with every static arm in the same engine and arrival regime,
is the primary directly measured request-level result. The reverse engine changes
condition/feedback order but does not fully reverse all static caps; it is retained as
the predeclared repeat, with that limitation.

Step padding is a diagnostic derived from logged capture buckets, not measured GPU
utilization or exposed HBM traffic. Step observations within a request or episode are
not independent samples. Passing mean request TPOT does not guarantee each token ITL.
There is no expert-signal or task-quality result in this campaign.

All 24 measured raw cells, their input IDs, token timestamps, scheduler/output events,
metrics, process-boundary checks, actual source, environment, combined stdout/stderr,
and zero-exit campaign status are locally available. `TRANSFER.json` records archive
verification. The two engines also ran 12 warmups each: their summaries and logs exist,
but the inherited runner discarded successful warmup raw objects without persisting
them. Those missing warmup trajectories cannot be recovered by downloading again.
Thus measured-data readback is complete; all-phase raw retention is not complete.
Remote originals have not been deleted.

While this task handled readback, another workspace task changed the shared runner
and prepared `../20260908_capture_ladder_paired_r01/`. Those future code changes are
not the executed source of this campaign. Analysis uses the source in `readback/`
and checks it against the environment hashes. This task does not overwrite the
concurrent runner or relabel the paired experiment as executed.

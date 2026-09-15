# LTR packing baseline

MEASUREMENT_ONLY

Both arms are the same custom FCFS/current-history-reservation/native-recompute backend with LTR200/10 on; only fit-scan versus ranked-prefix capacity stop changes. Not full LTR or a holdout. All wall, nested decision/scheduler cost, request losses and outputs remain. No SLO/Oracle/quality/method GO. Recovery ends at the next actual preemption of that request; completion endings are separate. Counts of 0/1/2 new outputs apply only to actually resumed, subsequently re-preempted residencies. Future outputs are outcomes only; exact planner replay requires recorded resolved chunk threshold.

- block0-d6-packing-fit_scan: COMPLETE
- block0-d6-packing-rank_prefix: COMPLETE
- block1-d6-packing-rank_prefix: COMPLETE
- block1-d6-packing-fit_scan: COMPLETE

Per-request changes, output differences, exact preemption-bounded recovery counts, quantum epochs and full cost are retained in analysis.json.

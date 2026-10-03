# First-output restore obligation

UNRUN

Same rank-prefix/FCFS/LTR200/10/native-recompute backend; only an actual resumed PREEMPTED request with past output and pending recovery can create a first-output obligation. No future output is used for decisions. Status is reconstructed from captured running membership/preemption count under the frozen three-state install scope. Actual completed requests alone release completion obligations. The independent helper replay supplies real priorities, including -2; original LTR quantum, actual ordering, victims, ownership and full costs remain checked. No SLO, Oracle, quality, full-LTR, significance or method-GO claim.

- block0-d6-restore-off: UNRUN
- block0-d6-restore-on: UNRUN
- block1-d6-restore-on: UNRUN
- block1-d6-restore-off: UNRUN

Full metrics, raw-derived residency counts, actual priority/ledger replay and per-request changes are retained in analysis.json.

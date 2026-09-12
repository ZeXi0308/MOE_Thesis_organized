# Native cap transfer: PARTIAL_OR_UNRUN

Four observed episodes per cap/condition reuse one workload; they are not four independent workloads.
Small-sample TTFT/TPOT/ITL tail quantiles are descriptive. ABBA rankings do not identify a mechanism.
queue_s is client submission lag; native waiting is scheduler queue count and core scheduled_ts minus queued_ts.
max_native_waiting_after counts requests still waiting after scheduling; the legacy maximum also includes newly submitted requests before scheduling.
Core and host clocks are never subtracted; multi-token host chunks do not resolve generation ITL.
GPU isolation checks cover episode boundaries; this report does not certify continuous isolation.

| Group/cell | Scale/regime/repeat | Cap | Status | Goodput | TTFT p50 s | TPOT p50 s | Max active/decode/wait after schedule | Prefill/decode tokens | Preempt/adjust/chunks>1 |
|---|---|---|---|---|---|---|---|---|---|

missing group config: a0_cap6
missing group config: b0_cap8
missing group config: b1_cap8
missing group config: a1_cap6

One next question: At a separately frozen native load/SLO setting, does cap 6 versus 8 change violation risk reproducibly while actual scheduling reaches both limits?

# Native cap transfer: MEASUREMENT_ONLY

Four observed episodes per cap/condition reuse one workload; they are not four independent workloads.
Small-sample TTFT/TPOT/ITL tail quantiles are descriptive. ABBA rankings do not identify a mechanism.
queue_s is client submission lag; native waiting is scheduler queue count and core scheduled_ts minus queued_ts.
max_native_waiting_after counts requests still waiting after scheduling; the legacy maximum also includes newly submitted requests before scheduling.
Core and host clocks are never subtracted; multi-token host chunks do not resolve generation ITL.
GPU isolation checks cover episode boundaries; this report does not certify continuous isolation.

| Group/cell | Scale/regime/repeat | Cap | Status | Goodput | TTFT p50 s | TPOT p50 s | Max active/decode/wait after schedule | Prefill/decode tokens | Preempt/adjust/chunks>1 |
|---|---|---|---|---|---|---|---|---|---|
| a0_cap6/0 | 0.02/steady/0 | 6 | COMPLETE | 30.5514 | 0.1638 | 0.0076 | 6/6/10 | 2048/240 | 0/0/0 |
| a0_cap6/1 | 0.02/bursty/0 | 6 | COMPLETE | 19.6490 | 0.1841 | 0.0076 | 6/6/10 | 2048/240 | 0/0/0 |
| a0_cap6/2 | 0.02/bursty/1 | 6 | COMPLETE | 34.6596 | 0.1241 | 0.0073 | 6/6/10 | 2048/240 | 0/0/0 |
| a0_cap6/3 | 0.02/steady/1 | 6 | COMPLETE | 23.7403 | 0.1930 | 0.0075 | 6/6/10 | 2048/240 | 0/0/0 |
| b0_cap8/0 | 0.02/steady/0 | 8 | COMPLETE | 48.2491 | 0.1227 | 0.0077 | 8/8/8 | 2048/240 | 0/0/0 |
| b0_cap8/1 | 0.02/bursty/0 | 8 | COMPLETE | 38.8014 | 0.1146 | 0.0077 | 8/8/8 | 2048/240 | 0/0/0 |
| b0_cap8/2 | 0.02/bursty/1 | 8 | COMPLETE | 37.5091 | 0.1281 | 0.0076 | 8/8/8 | 2048/240 | 0/0/0 |
| b0_cap8/3 | 0.02/steady/1 | 8 | COMPLETE | 46.2610 | 0.1329 | 0.0074 | 8/8/8 | 2048/240 | 0/0/0 |
| b1_cap8/0 | 0.02/steady/0 | 8 | COMPLETE | 60.5340 | 0.0793 | 0.0073 | 8/8/8 | 2048/240 | 0/0/0 |
| b1_cap8/1 | 0.02/bursty/0 | 8 | COMPLETE | 54.9342 | 0.1024 | 0.0075 | 8/8/8 | 2048/240 | 0/0/0 |
| b1_cap8/2 | 0.02/bursty/1 | 8 | COMPLETE | 51.2114 | 0.1242 | 0.0075 | 8/8/8 | 2048/240 | 0/0/0 |
| b1_cap8/3 | 0.02/steady/1 | 8 | COMPLETE | 47.1823 | 0.1319 | 0.0075 | 8/8/8 | 2048/240 | 0/0/0 |
| a1_cap6/0 | 0.02/steady/0 | 6 | COMPLETE | 26.5073 | 0.1858 | 0.0075 | 6/6/10 | 2048/240 | 0/0/0 |
| a1_cap6/1 | 0.02/bursty/0 | 6 | COMPLETE | 29.7243 | 0.1814 | 0.0076 | 6/6/10 | 2048/240 | 0/0/0 |
| a1_cap6/2 | 0.02/bursty/1 | 6 | COMPLETE | 34.5756 | 0.1243 | 0.0073 | 6/6/10 | 2048/240 | 0/0/0 |
| a1_cap6/3 | 0.02/steady/1 | 6 | COMPLETE | 32.9271 | 0.1362 | 0.0073 | 6/6/10 | 2048/240 | 0/0/0 |

Scale 0.02, bursty: Some violations observed; this alone does not establish a calibrated capacity boundary.
Per-cap four-episode ranges: {"6": {"goodput_rps": {"n": 4, "minimum": 19.648960014487365, "maximum": 34.65956448899399}, "attainment": {"n": 4, "minimum": 0.5, "maximum": 0.75}, "ttft_p50_s": {"n": 4, "minimum": 0.12407395675778388, "maximum": 0.1840760788321495}, "tpot_p50_s": {"n": 4, "minimum": 0.007290928314129512, "maximum": 0.0076272855202356975}, "max_active": {"n": 4, "minimum": 6, "maximum": 6}, "max_decode_requests": {"n": 4, "minimum": 6, "maximum": 6}, "max_native_waiting": {"n": 4, "minimum": 10, "maximum": 12}, "max_native_waiting_after": {"n": 4, "minimum": 10, "maximum": 10}}, "8": {"goodput_rps": {"n": 4, "minimum": 37.50911225976781, "maximum": 54.9342490356152}, "attainment": {"n": 4, "minimum": 0.75, "maximum": 1.0}, "ttft_p50_s": {"n": 4, "minimum": 0.1024311935901642, "maximum": 0.12812066696584226}, "tpot_p50_s": {"n": 4, "minimum": 0.007493225981791814, "maximum": 0.007737436021367709}, "max_active": {"n": 4, "minimum": 8, "maximum": 8}, "max_decode_requests": {"n": 4, "minimum": 8, "maximum": 8}, "max_native_waiting": {"n": 4, "minimum": 12, "maximum": 12}, "max_native_waiting_after": {"n": 4, "minimum": 8, "maximum": 8}}}

{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -19.15245800158455, "attainment": -0.25, "ttft_p50_s": 0.06945004500448701, "tpot_p50_s": -0.00011015050113201159}, "observed_goodput_winner": 8}
{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -2.8495477707738175, "attainment": 0.0, "ttft_p50_s": -0.004046710208058385, "tpot_p50_s": -0.00033371510605017304}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -25.20990359952203, "attainment": -0.25, "ttft_p50_s": 0.07893246784806249, "tpot_p50_s": 8.271497984727299e-05}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -16.635791246261746, "attainment": -0.25, "ttft_p50_s": 4.049763083455171e-05, "tpot_p50_s": -0.00017536741991837704}, "observed_goodput_winner": 8}

Scale 0.02, steady: Some violations observed; this alone does not establish a calibrated capacity boundary.
Per-cap four-episode ranges: {"6": {"goodput_rps": {"n": 4, "minimum": 23.74029579968299, "maximum": 32.92713812063681}, "attainment": {"n": 4, "minimum": 0.625, "maximum": 0.75}, "ttft_p50_s": {"n": 4, "minimum": 0.1361565680205822, "maximum": 0.19302745693922044}, "tpot_p50_s": {"n": 4, "minimum": 0.007348661869764328, "maximum": 0.007568178325891494}, "max_active": {"n": 4, "minimum": 6, "maximum": 6}, "max_decode_requests": {"n": 4, "minimum": 6, "maximum": 6}, "max_native_waiting": {"n": 4, "minimum": 10, "maximum": 15}, "max_native_waiting_after": {"n": 4, "minimum": 10, "maximum": 10}}, "8": {"goodput_rps": {"n": 4, "minimum": 46.261032218219, "maximum": 60.53397654215144}, "attainment": {"n": 4, "minimum": 0.9375, "maximum": 1.0}, "ttft_p50_s": {"n": 4, "minimum": 0.07927167376875877, "maximum": 0.13285385219752788}, "tpot_p50_s": {"n": 4, "minimum": 0.007257106900215149, "maximum": 0.007662089665730795}, "max_active": {"n": 4, "minimum": 8, "maximum": 8}, "max_decode_requests": {"n": 4, "minimum": 8, "maximum": 8}, "max_native_waiting": {"n": 4, "minimum": 8, "maximum": 15}, "max_native_waiting_after": {"n": 4, "minimum": 8, "maximum": 8}}}

{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -17.697669127083294, "attainment": -0.1875, "ttft_p50_s": 0.04109298075735571, "tpot_p50_s": -9.391133983930027e-05}, "observed_goodput_winner": 8}
{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -22.52073641853601, "attainment": -0.3125, "ttft_p50_s": 0.06017360474169256, "tpot_p50_s": 0.00011900266011555943}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -34.026631550879245, "attainment": -0.3125, "ttft_p50_s": 0.10656982484459879, "tpot_p50_s": 0.00025940189758936506}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -14.25515768926629, "attainment": -0.1875, "ttft_p50_s": 0.004273839399218576, "tpot_p50_s": -0.00010539342959721941}, "observed_goodput_winner": 8}


One next question: At a separately frozen native load/SLO setting, does cap 6 versus 8 change violation risk reproducibly while actual scheduling reaches both limits?

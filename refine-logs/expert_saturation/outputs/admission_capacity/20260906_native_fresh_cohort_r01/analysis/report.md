# Native cap transfer: MEASUREMENT_ONLY

Four observed episodes per cap/condition reuse one workload; they are not four independent workloads.
Small-sample TTFT/TPOT/ITL tail quantiles are descriptive. ABBA rankings do not identify a mechanism.
queue_s is client submission lag; native waiting is scheduler queue count and core scheduled_ts minus queued_ts.
max_native_waiting_after counts requests still waiting after scheduling; the legacy maximum also includes newly submitted requests before scheduling.
Core and host clocks are never subtracted; multi-token host chunks do not resolve generation ITL.
GPU isolation checks cover episode boundaries; this report does not certify continuous isolation.

| Group/cell | Scale/regime/repeat | Cap | Status | Goodput | TTFT p50 s | TPOT p50 s | Max active/decode/wait after schedule | Prefill/decode tokens | Preempt/adjust/chunks>1 |
|---|---|---|---|---|---|---|---|---|---|
| a0_cap6/0 | 0.02/steady/0 | 6 | COMPLETE | 30.3214 | 0.1700 | 0.0075 | 6/6/10 | 2048/240 | 0/0/0 |
| a0_cap6/1 | 0.02/bursty/0 | 6 | COMPLETE | 14.6053 | 0.1872 | 0.0078 | 6/6/10 | 2048/240 | 0/0/0 |
| a0_cap6/2 | 0.02/bursty/1 | 6 | COMPLETE | 20.4753 | 0.1685 | 0.0076 | 6/6/10 | 2048/240 | 0/0/0 |
| a0_cap6/3 | 0.02/steady/1 | 6 | COMPLETE | 33.1999 | 0.1366 | 0.0073 | 6/6/10 | 2048/240 | 0/0/0 |
| b0_cap8/0 | 0.02/steady/0 | 8 | COMPLETE | 54.1935 | 0.1074 | 0.0076 | 8/8/8 | 2048/240 | 0/0/0 |
| b0_cap8/1 | 0.02/bursty/0 | 8 | COMPLETE | 37.8494 | 0.1294 | 0.0075 | 8/8/8 | 2048/240 | 0/0/0 |
| b0_cap8/2 | 0.02/bursty/1 | 8 | COMPLETE | 37.0870 | 0.1325 | 0.0077 | 8/8/8 | 2048/240 | 0/0/0 |
| b0_cap8/3 | 0.02/steady/1 | 8 | COMPLETE | 60.1894 | 0.0805 | 0.0074 | 8/8/8 | 2048/240 | 0/0/0 |
| b1_cap8/0 | 0.02/steady/0 | 8 | COMPLETE | 55.2009 | 0.0994 | 0.0076 | 8/8/8 | 2048/240 | 0/0/0 |
| b1_cap8/1 | 0.02/bursty/0 | 8 | COMPLETE | 37.8516 | 0.1294 | 0.0075 | 8/8/8 | 2048/240 | 0/0/0 |
| b1_cap8/2 | 0.02/bursty/1 | 8 | COMPLETE | 37.9062 | 0.1292 | 0.0075 | 8/8/8 | 2048/240 | 0/0/0 |
| b1_cap8/3 | 0.02/steady/1 | 8 | COMPLETE | 55.5731 | 0.1010 | 0.0076 | 8/8/8 | 2048/240 | 0/0/0 |
| a1_cap6/0 | 0.02/steady/0 | 6 | COMPLETE | 30.7236 | 0.1659 | 0.0076 | 6/6/10 | 2048/240 | 0/0/0 |
| a1_cap6/1 | 0.02/bursty/0 | 6 | COMPLETE | 19.5581 | 0.1892 | 0.0076 | 6/6/10 | 2048/240 | 0/0/0 |
| a1_cap6/2 | 0.02/bursty/1 | 6 | COMPLETE | 33.7320 | 0.1333 | 0.0075 | 6/6/10 | 2048/240 | 0/0/0 |
| a1_cap6/3 | 0.02/steady/1 | 6 | COMPLETE | 32.4357 | 0.1435 | 0.0076 | 6/6/10 | 2048/240 | 0/0/0 |

Scale 0.02, bursty: Some violations observed; this alone does not establish a calibrated capacity boundary.
Per-cap four-episode ranges: {"6": {"goodput_rps": {"n": 4, "minimum": 14.605250751972674, "maximum": 33.73196650207227}, "attainment": {"n": 4, "minimum": 0.375, "maximum": 0.75}, "ttft_p50_s": {"n": 4, "minimum": 0.13329619228839873, "maximum": 0.18919629663228987}, "tpot_p50_s": {"n": 4, "minimum": 0.007522601634263992, "maximum": 0.0077761373172203704}, "max_active": {"n": 4, "minimum": 6, "maximum": 6}, "max_decode_requests": {"n": 4, "minimum": 6, "maximum": 6}, "max_native_waiting": {"n": 4, "minimum": 10, "maximum": 12}, "max_native_waiting_after": {"n": 4, "minimum": 10, "maximum": 10}}, "8": {"goodput_rps": {"n": 4, "minimum": 37.08698809075215, "maximum": 37.906167866770076}, "attainment": {"n": 4, "minimum": 0.75, "maximum": 0.75}, "ttft_p50_s": {"n": 4, "minimum": 0.1292063916474581, "maximum": 0.1325161649286747}, "tpot_p50_s": {"n": 4, "minimum": 0.007496521001060803, "maximum": 0.007715489715337753}, "max_active": {"n": 4, "minimum": 8, "maximum": 8}, "max_decode_requests": {"n": 4, "minimum": 8, "maximum": 8}, "max_native_waiting": {"n": 4, "minimum": 12, "maximum": 12}, "max_native_waiting_after": {"n": 4, "minimum": 8, "maximum": 8}}}

{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -23.244132524275855, "attainment": -0.375, "ttft_p50_s": 0.057773906737565967, "tpot_p50_s": 0.0002662779142459243}, "observed_goodput_winner": 8}
{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -16.61169501652152, "attainment": -0.25, "ttft_p50_s": 0.03602718561887738, "tpot_p50_s": -0.00016296754280726092}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -18.293536754986945, "attainment": -0.25, "ttft_p50_s": 0.05980855040252206, "tpot_p50_s": 3.288922210534402e-05}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -4.174201364697808, "attainment": 0.0, "ttft_p50_s": 0.0040898006409406384, "tpot_p50_s": 2.608063320318904e-05}, "observed_goodput_winner": 8}

Scale 0.02, steady: Some violations observed; this alone does not establish a calibrated capacity boundary.
Per-cap four-episode ranges: {"6": {"goodput_rps": {"n": 4, "minimum": 30.321380103243616, "maximum": 33.199887975484636}, "attainment": {"n": 4, "minimum": 0.75, "maximum": 0.75}, "ttft_p50_s": {"n": 4, "minimum": 0.13655679830908776, "maximum": 0.1699942552447319}, "tpot_p50_s": {"n": 4, "minimum": 0.0073477002481619515, "maximum": 0.007592287411292394}, "max_active": {"n": 4, "minimum": 6, "maximum": 6}, "max_decode_requests": {"n": 4, "minimum": 6, "maximum": 6}, "max_native_waiting": {"n": 4, "minimum": 10, "maximum": 15}, "max_native_waiting_after": {"n": 4, "minimum": 10, "maximum": 10}}, "8": {"goodput_rps": {"n": 4, "minimum": 54.19350922110527, "maximum": 60.18941797510746}, "attainment": {"n": 4, "minimum": 1.0, "maximum": 1.0}, "ttft_p50_s": {"n": 4, "minimum": 0.08054020030796527, "maximum": 0.10742052792012691}, "tpot_p50_s": {"n": 4, "minimum": 0.007394647846619288, "maximum": 0.007593905677398046}, "max_active": {"n": 4, "minimum": 8, "maximum": 8}, "max_decode_requests": {"n": 4, "minimum": 8, "maximum": 8}, "max_native_waiting": {"n": 4, "minimum": 8, "maximum": 15}, "max_native_waiting_after": {"n": 4, "minimum": 8, "maximum": 8}}}

{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -23.872129117861657, "attainment": -0.25, "ttft_p50_s": 0.062573727324605, "tpot_p50_s": -0.0001222451527913412}, "observed_goodput_winner": 8}
{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -26.989529999622825, "attainment": -0.25, "ttft_p50_s": 0.05601659800112249, "tpot_p50_s": -4.6947598457336946e-05}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -24.47727415717728, "attainment": -0.25, "ttft_p50_s": 0.06644639196991922, "tpot_p50_s": -1.598671078681911e-05}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -23.137415229155735, "attainment": -0.25, "ttft_p50_s": 0.04246365103125574, "tpot_p50_s": 9.153534968693935e-06}, "observed_goodput_winner": 8}


One next question: At a separately frozen native load/SLO setting, does cap 6 versus 8 change violation risk reproducibly while actual scheduling reaches both limits?

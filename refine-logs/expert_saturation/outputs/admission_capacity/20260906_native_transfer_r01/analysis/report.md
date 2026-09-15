# Native cap transfer: MEASUREMENT_ONLY

Four observed episodes per cap/condition reuse one workload; they are not four independent workloads.
Small-sample TTFT/TPOT/ITL tail quantiles are descriptive. ABBA rankings do not identify a mechanism.
queue_s is client submission lag; native waiting is scheduler queue count and core scheduled_ts minus queued_ts.
max_native_waiting_after counts requests still waiting after scheduling; the legacy maximum also includes newly submitted requests before scheduling.
Core and host clocks are never subtracted; multi-token host chunks do not resolve generation ITL.
GPU isolation checks cover episode boundaries; this report does not certify continuous isolation.

| Group/cell | Scale/regime/repeat | Cap | Status | Goodput | TTFT p50 s | TPOT p50 s | Max active/decode/wait after schedule | Prefill/decode tokens | Preempt/adjust/chunks>1 |
|---|---|---|---|---|---|---|---|---|---|
| a0_cap6/0 | 1.0/steady/0 | 6 | COMPLETE | 7.2210 | 0.9783 | 0.0079 | 6/6/9 | 2048/240 | 0/0/0 |
| a0_cap6/1 | 1.0/bursty/0 | 6 | COMPLETE | 10.0278 | 0.0188 | 0.0054 | 4/4/0 | 2048/240 | 0/0/0 |
| a0_cap6/2 | 0.02/steady/0 | 6 | COMPLETE | 37.0810 | 0.1993 | 0.0079 | 6/6/10 | 2048/240 | 0/0/0 |
| a0_cap6/3 | 0.02/bursty/0 | 6 | COMPLETE | 37.6805 | 0.1939 | 0.0082 | 6/6/10 | 2048/240 | 0/0/0 |
| a0_cap6/4 | 0.02/bursty/1 | 6 | COMPLETE | 38.4258 | 0.1887 | 0.0080 | 6/6/10 | 2048/240 | 0/0/0 |
| a0_cap6/5 | 0.02/steady/1 | 6 | COMPLETE | 42.5773 | 0.1407 | 0.0076 | 6/6/10 | 2048/240 | 0/0/0 |
| a0_cap6/6 | 1.0/bursty/1 | 6 | COMPLETE | 10.0295 | 0.0179 | 0.0053 | 4/4/0 | 2048/240 | 0/0/0 |
| a0_cap6/7 | 1.0/steady/1 | 6 | COMPLETE | 10.2122 | 0.0144 | 0.0030 | 2/2/0 | 2048/240 | 0/0/0 |
| b0_cap8/0 | 1.0/steady/0 | 8 | COMPLETE | 7.5934 | 0.8927 | 0.0239 | 8/8/7 | 2048/240 | 0/0/0 |
| b0_cap8/1 | 1.0/bursty/0 | 8 | COMPLETE | 9.7391 | 0.0555 | 0.0065 | 8/8/0 | 2048/240 | 0/0/0 |
| b0_cap8/2 | 0.02/steady/0 | 8 | COMPLETE | 50.2802 | 0.1328 | 0.0074 | 8/8/8 | 2048/240 | 0/0/0 |
| b0_cap8/3 | 0.02/bursty/0 | 8 | COMPLETE | 50.5212 | 0.1287 | 0.0075 | 8/8/8 | 2048/240 | 0/0/0 |
| b0_cap8/4 | 0.02/bursty/1 | 8 | COMPLETE | 51.4485 | 0.1237 | 0.0075 | 8/8/8 | 2048/240 | 0/0/0 |
| b0_cap8/5 | 0.02/steady/1 | 8 | COMPLETE | 50.3154 | 0.1327 | 0.0074 | 8/8/8 | 2048/240 | 0/0/0 |
| b0_cap8/6 | 1.0/bursty/1 | 8 | COMPLETE | 10.0299 | 0.0356 | 0.0057 | 4/4/0 | 2048/240 | 0/0/0 |
| b0_cap8/7 | 1.0/steady/1 | 8 | COMPLETE | 10.2688 | 0.0139 | 0.0028 | 1/1/0 | 2048/240 | 0/0/0 |
| b1_cap8/0 | 1.0/steady/0 | 8 | COMPLETE | 10.2770 | 0.0143 | 0.0028 | 1/1/0 | 2048/240 | 0/0/0 |
| b1_cap8/1 | 1.0/bursty/0 | 8 | COMPLETE | 9.7191 | 0.0462 | 0.0059 | 4/4/0 | 2048/240 | 0/0/0 |
| b1_cap8/2 | 0.02/steady/0 | 8 | COMPLETE | 47.0985 | 0.1494 | 0.0078 | 8/8/8 | 2048/240 | 0/0/0 |
| b1_cap8/3 | 0.02/bursty/0 | 8 | COMPLETE | 51.9900 | 0.1198 | 0.0075 | 8/8/8 | 2048/240 | 0/0/0 |
| b1_cap8/4 | 0.02/bursty/1 | 8 | COMPLETE | 60.7584 | 0.0740 | 0.0076 | 8/8/8 | 2048/240 | 0/0/0 |
| b1_cap8/5 | 0.02/steady/1 | 8 | COMPLETE | 50.0820 | 0.1300 | 0.0076 | 8/8/8 | 2048/240 | 0/0/0 |
| b1_cap8/6 | 1.0/bursty/1 | 8 | COMPLETE | 9.7210 | 0.0503 | 0.0061 | 4/4/0 | 2048/240 | 0/0/0 |
| b1_cap8/7 | 1.0/steady/1 | 8 | COMPLETE | 10.2854 | 0.0136 | 0.0028 | 2/2/0 | 2048/240 | 0/0/0 |
| a1_cap6/0 | 1.0/steady/0 | 6 | COMPLETE | 10.2332 | 0.0139 | 0.0028 | 1/1/0 | 2048/240 | 0/0/0 |
| a1_cap6/1 | 1.0/bursty/0 | 6 | COMPLETE | 9.9719 | 0.0205 | 0.0055 | 4/4/0 | 2048/240 | 0/0/0 |
| a1_cap6/2 | 0.02/steady/0 | 6 | COMPLETE | 38.0195 | 0.1725 | 0.0085 | 6/6/10 | 2048/240 | 0/0/0 |
| a1_cap6/3 | 0.02/bursty/0 | 6 | COMPLETE | 44.8080 | 0.1299 | 0.0077 | 6/6/10 | 2048/240 | 0/0/0 |
| a1_cap6/4 | 0.02/bursty/1 | 6 | COMPLETE | 38.2218 | 0.1909 | 0.0080 | 6/6/10 | 2048/240 | 0/0/0 |
| a1_cap6/5 | 0.02/steady/1 | 6 | COMPLETE | 38.1888 | 0.1843 | 0.0080 | 6/6/10 | 2048/240 | 0/0/0 |
| a1_cap6/6 | 1.0/bursty/1 | 6 | COMPLETE | 10.0230 | 0.0153 | 0.0054 | 4/4/0 | 2048/240 | 0/0/0 |
| a1_cap6/7 | 1.0/steady/1 | 6 | COMPLETE | 10.2771 | 0.0139 | 0.0028 | 1/1/0 | 2048/240 | 0/0/0 |

Scale 0.02, bursty: ALL_PASS: goodput equals completion throughput; these SLOs do not separate outcomes.
Per-cap four-episode ranges: {"6": {"goodput_rps": {"n": 4, "minimum": 37.6804610284693, "maximum": 44.808029059026204}, "attainment": {"n": 4, "minimum": 1.0, "maximum": 1.0}, "ttft_p50_s": {"n": 4, "minimum": 0.12985856890678404, "maximum": 0.19389129862189292}, "tpot_p50_s": {"n": 4, "minimum": 0.00766839658220609, "maximum": 0.008199173212051391}, "max_active": {"n": 4, "minimum": 6, "maximum": 6}, "max_decode_requests": {"n": 4, "minimum": 6, "maximum": 6}, "max_native_waiting": {"n": 4, "minimum": 10, "maximum": 12}, "max_native_waiting_after": {"n": 4, "minimum": 10, "maximum": 10}}, "8": {"goodput_rps": {"n": 4, "minimum": 50.52120335764945, "maximum": 60.75841961436486}, "attainment": {"n": 4, "minimum": 1.0, "maximum": 1.0}, "ttft_p50_s": {"n": 4, "minimum": 0.07400314569473268, "maximum": 0.12872418820858003}, "tpot_p50_s": {"n": 4, "minimum": 0.007452475031216939, "maximum": 0.007562917843461036}, "max_active": {"n": 4, "minimum": 8, "maximum": 8}, "max_decode_requests": {"n": 4, "minimum": 8, "maximum": 8}, "max_native_waiting": {"n": 4, "minimum": 8, "maximum": 12}, "max_native_waiting_after": {"n": 4, "minimum": 8, "maximum": 8}}}

{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -12.840742329180145, "attainment": 0.0, "ttft_p50_s": 0.06516711041331288, "tpot_p50_s": 0.0007056709378957743}, "observed_goodput_winner": 8}
{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -13.022601826072865, "attainment": 0.0, "ttft_p50_s": 0.0650688614696264, "tpot_p50_s": 0.0005371145904064173}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -7.181950634344126, "attainment": 0.0, "ttft_p50_s": 0.010094916447997065, "tpot_p50_s": 0.000215921550989151}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -22.536641482058727, "attainment": 0.0, "ttft_p50_s": 0.1169142313301563, "tpot_p50_s": 0.0004089811195929858}, "observed_goodput_winner": 8}

Scale 0.02, steady: ALL_PASS: goodput equals completion throughput; these SLOs do not separate outcomes.
Per-cap four-episode ranges: {"6": {"goodput_rps": {"n": 4, "minimum": 37.08104539023021, "maximum": 42.577287321583825}, "attainment": {"n": 4, "minimum": 1.0, "maximum": 1.0}, "ttft_p50_s": {"n": 4, "minimum": 0.14072059625387193, "maximum": 0.19925610879063607}, "tpot_p50_s": {"n": 4, "minimum": 0.007645186533530553, "maximum": 0.008496303111314774}, "max_active": {"n": 4, "minimum": 6, "maximum": 6}, "max_decode_requests": {"n": 4, "minimum": 6, "maximum": 6}, "max_native_waiting": {"n": 4, "minimum": 10, "maximum": 15}, "max_native_waiting_after": {"n": 4, "minimum": 10, "maximum": 10}}, "8": {"goodput_rps": {"n": 4, "minimum": 47.098504382458735, "maximum": 50.31538315930482}, "attainment": {"n": 4, "minimum": 1.0, "maximum": 1.0}, "ttft_p50_s": {"n": 4, "minimum": 0.13004494792222976, "maximum": 0.14936132504045962}, "tpot_p50_s": {"n": 4, "minimum": 0.007413343340158462, "maximum": 0.007810158158342044}, "max_active": {"n": 4, "minimum": 8, "maximum": 8}, "max_decode_requests": {"n": 4, "minimum": 8, "maximum": 8}, "max_native_waiting": {"n": 4, "minimum": 15, "maximum": 15}, "max_native_waiting_after": {"n": 4, "minimum": 8, "maximum": 8}}}

{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -13.199193226263432, "attainment": 0.0, "ttft_p50_s": 0.06640806245803835, "tpot_p50_s": 0.00047610079248746294}, "observed_goodput_winner": 8}
{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -7.738095837720998, "attainment": 0.0, "ttft_p50_s": 0.007982343539595621, "tpot_p50_s": 0.00023184319337209072}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -9.07898519857504, "attainment": 0.0, "ttft_p50_s": 0.023118984058499353, "tpot_p50_s": 0.0006861449529727299}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -11.893201363516148, "attainment": 0.0, "ttft_p50_s": 0.054253036260604875, "tpot_p50_s": 0.0003691790004571281}, "observed_goodput_winner": 8}

Scale 1.0, bursty: ALL_PASS: goodput equals completion throughput; these SLOs do not separate outcomes.
Per-cap four-episode ranges: {"6": {"goodput_rps": {"n": 4, "minimum": 9.971882272607742, "maximum": 10.029466390308668}, "attainment": {"n": 4, "minimum": 1.0, "maximum": 1.0}, "ttft_p50_s": {"n": 4, "minimum": 0.015334932133555412, "maximum": 0.020546618849039078}, "tpot_p50_s": {"n": 4, "minimum": 0.005295041824380557, "maximum": 0.005546978116035462}, "max_active": {"n": 4, "minimum": 4, "maximum": 4}, "max_decode_requests": {"n": 4, "minimum": 4, "maximum": 4}, "max_native_waiting": {"n": 4, "minimum": 4, "maximum": 4}, "max_native_waiting_after": {"n": 4, "minimum": 0, "maximum": 0}}, "8": {"goodput_rps": {"n": 4, "minimum": 9.719066723347096, "maximum": 10.02989767357426}, "attainment": {"n": 4, "minimum": 1.0, "maximum": 1.0}, "ttft_p50_s": {"n": 4, "minimum": 0.03555741906166077, "maximum": 0.055456217378377914}, "tpot_p50_s": {"n": 4, "minimum": 0.0057095464318990706, "maximum": 0.006473082055648168}, "max_active": {"n": 4, "minimum": 4, "maximum": 8}, "max_decode_requests": {"n": 4, "minimum": 4, "maximum": 8}, "max_native_waiting": {"n": 4, "minimum": 4, "maximum": 4}, "max_native_waiting_after": {"n": 4, "minimum": 0, "maximum": 0}}}

{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": 0.2886864113936678, "attainment": 0.0, "ttft_p50_s": -0.03662676550447941, "tpot_p50_s": -0.0011171846340099975}, "observed_goodput_winner": 6}
{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -0.0004312832655912757, "attainment": 0.0, "ttft_p50_s": -0.017655931413173676, "tpot_p50_s": -0.00041450460751851336}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": 0.25281554926064587, "attainment": 0.0, "ttft_p50_s": -0.025664564222097397, "tpot_p50_s": -0.0003227433810631432}, "observed_goodput_winner": 6}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": 0.3020182476856288, "attainment": 0.0, "ttft_p50_s": -0.03497379086911678, "tpot_p50_s": -0.0007127113640308389}, "observed_goodput_winner": 6}

Scale 1.0, steady: ALL_PASS: goodput equals completion throughput; these SLOs do not separate outcomes.
Per-cap four-episode ranges: {"6": {"goodput_rps": {"n": 4, "minimum": 7.220997741278246, "maximum": 10.277128913654408}, "attainment": {"n": 4, "minimum": 1.0, "maximum": 1.0}, "ttft_p50_s": {"n": 4, "minimum": 0.013853264600038495, "maximum": 0.9782948169857264}, "tpot_p50_s": {"n": 4, "minimum": 0.0028314133485158287, "maximum": 0.00793463240067164}, "max_active": {"n": 4, "minimum": 1, "maximum": 6}, "max_decode_requests": {"n": 4, "minimum": 1, "maximum": 6}, "max_native_waiting": {"n": 4, "minimum": 1, "maximum": 9}, "max_native_waiting_after": {"n": 4, "minimum": 0, "maximum": 9}}, "8": {"goodput_rps": {"n": 4, "minimum": 7.593363266522643, "maximum": 10.285407493945094}, "attainment": {"n": 4, "minimum": 1.0, "maximum": 1.0}, "ttft_p50_s": {"n": 4, "minimum": 0.013593390956521012, "maximum": 0.8927497945725917}, "tpot_p50_s": {"n": 4, "minimum": 0.0028077139208714168, "maximum": 0.023936110734939575}, "max_active": {"n": 4, "minimum": 1, "maximum": 8}, "max_decode_requests": {"n": 4, "minimum": 1, "maximum": 8}, "max_native_waiting": {"n": 4, "minimum": 1, "maximum": 7}, "max_native_waiting_after": {"n": 4, "minimum": 0, "maximum": 7}}}

{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -0.3723655252443976, "attainment": 0.0, "ttft_p50_s": 0.08554502241313466, "tpot_p50_s": -0.016001478334267932}, "observed_goodput_winner": 8}
{"cap6_group": "a0_cap6", "cap8_group": "b0_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -0.056597019678696014, "attainment": 0.0, "ttft_p50_s": 0.0004351973533630593, "tpot_p50_s": 0.00016064941883087158}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 0, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -0.043831974473491186, "attainment": 0.0, "ttft_p50_s": -0.0003973376005887985, "tpot_p50_s": -5.43221831321699e-06}, "observed_goodput_winner": 8}
{"cap6_group": "a1_cap6", "cap8_group": "b1_cap8", "repeat": 1, "status": "DESCRIPTIVE_PAIRED_RERUN", "cap6_minus_cap8": {"goodput_rps": -0.008278580290685511, "attainment": 0.0, "ttft_p50_s": 0.0003202024847268614, "tpot_p50_s": 3.039923806985199e-05}, "observed_goodput_winner": 8}


One next question: At a separately frozen native load/SLO setting, does cap 6 versus 8 change violation risk reproducibly while actual scheduling reaches both limits?

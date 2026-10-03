# Arrival-slot identity swap, frozen before execution

The fresh cohort's cap8 bursty runs failed mean TPOT for the same first four
arrival-zero requests. Their token1-to2 host interval aligned with a mixed
4-decode-token /512-prefill-token scheduling step. This is a temporal association,
not a kernel-causal result. Test whether the long interval moves with arrival slot.

Keep cap8, engine max8, model/revision/BF16, all engine arguments, output length16,
prompt length128, identical arrival arrays/scale .02, SLO .20/.009 and reference
5/.2, CPU threads, and the three warmups used in the preceding native experiments.
Use the exact previous runner/capture/metrics bytes. Only permute the source
requests and their token arrays: [4,5,6,7,0,1,2,3,8,9,10,11,12,13,14,15].
Request IDs stay with their text; arrival values stay with positions. All future
KV, batches, queues, outputs and timings are rerun independently.

Four fresh processes run original / swapped / swapped / original. One repeat
per process gives steady then bursty, two episodes each, eight total. Bursty is
the primary diagnostic; steady is retained as a geometry-matched supplementary
condition. Retain all episodes, failures, warmups and phase logs.

Report identity and arrival position for every failed request, actual token1-to2
ITL, mean TPOT, subsequent intervals, and the scheduled prefill/decode work within
that interval. A newly failing first cohort after the swap supports a stage/slot
association; persistence on the moved original identities motivates content
localization. If the event fails to reproduce within arm, report instability,
not a content mechanism. Do not infer that the entire interval is removable,
or claim a new scheduling method, GPU-kernel cause, or U/C contribution.

This is an exploratory causal negative control, not a new controller or paper GO.

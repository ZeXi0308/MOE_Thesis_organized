# Ordinary-feedback continuation, frozen before its GPU execution

Stage1 completed20 episodes. Throughput/TTFT versus TPOT tradeoff is repeatable,
but steady main-goodput ranking of caps8/16/32 changes across the two runs.
This batch is the one controlled followup: retain those static anchors, add the
simple midpoint12 (not a fitted predictor), and test one ordinary feedback plus
a shadow-cost control. Do not search thresholds or replace stage1 results.

Same32 retained texts,128prompt/128output, arrivals gap.05s or groups8 over1.55s.
Same native engine32/model/software/precision/FCFS/memory/token/graph settings.
MainSLO .20/.009s, reference5/.2s unchanged. All future states independently run.

Two fresh engines, all arms in one engine each. Forward:
static8,static12,static16,static32,shadow32,feedback32; each steady then bursty.
Reverse:feedback32,shadow32,static32,static16,static12,static8; each bursty then
steady.24 episodes total. Both engines warm static8/12/16/32 in identical
ascending order and cover both primary arrival shapes. No extra underload run.

Feedback starts32 and uses only completed-step host ITL: first take a median
across request ITLs delivered in that step, then a median over4 completed-step
observations. After at least4 steps since last intent change, above9ms lowers
one cap rung; below7.2ms with waiting and active==target raises one rung.
No reverse-up while active>target drains. Ladders8/12/16/32; no trained parameters.
Shadow computes the same observations/intents but always keeps actual target32.
Static32 has no feedback collection and is the capture-cost comparator.

Native scheduler limit=max(target,len(running)); preserve all active requests,
do not pause/evict/slice. Check that existing decode identities are all scheduled,
and preserve target/effective-limit/actual/decode/waiting separately. Signal
availability must precede decision and action. All in-memory collection/decision
cost enters the full request/episode denominator. No post-hoc fixed trace policy.

Analyze all anchors and both conditions; do not choose one favorable condition.
If rankings or feedback gain remain unstable, report that after this controlled
followup rather than invent a mechanism or launch a third selector. Compare
feedback against best actually measured static and shadow, not only cap4/8.
This is ordinary feedback qualification, not a novel U/C controller or paper GO.

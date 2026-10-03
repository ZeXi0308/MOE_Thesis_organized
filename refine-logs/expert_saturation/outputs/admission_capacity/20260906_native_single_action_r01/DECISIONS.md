# One-shot admission action qualification, frozen before GPU execution

2026-09-06; HEAD 2a37765 plus the minimal uncommitted one-shot extension.
Primary question: can one real 32->16 action improve complete-request outcomes
when the existing four-step ITL rule first signals pressure, without its repeated
lowering to12/8? This tests an action, not a new predictor or U/C contribution.

Reuse native_knee_r01 policy_probe workload, environment and warmups:32 existing
WikiText texts,128 prompt/128 fixed output tokens,ignore_eos,BF16 OLMoE pinned
revision,engine32,max_model_len256,token budget1024,FA2/Triton,compiled,FCFS,
memory utilization.70,no prefix cache,synchronous in-process capture.
Same steady .05s spacing and bursty groups8 over1.55s; no new input selection.
Main TTFT<=.20s and request mean TPOT<=.009s; reference5/.2s unchanged.

One canonical bundle: two fresh sequential engines; 6 episodes each,12 total.
Forward arms:hold32,single_down32_to16,static16; each steady then bursty.
Reverse arms:static16,single_down32_to16,hold32; each bursty then steady.
Both engines warm all8/12/16/32 shapes in the same ascending order and both
arrival conditions, exactly as the preceding ordinary-feedback stage.

hold32 and down share identical completed-step ITL observation/decision code:
median of request ITLs within step, then median of last4 available step medians.
At the first existing >9ms trigger with completed-step cooldown>=4, latch one
intent32->16. No further up/down decisions change intent. hold32 records this
same intention but keeps admission32; down applies16 for the episode remainder.
If no trigger occurs, keep32 and report it; do not force a post-hoc trigger.
static16 remains a separately rerun simple baseline.

No preemption, KV eviction or active-request skipping: effective scheduler limit
is max(target,current running). All future states independently execute. Inputs
available after engine.step cannot affect a decision before availability.
Target-write, first observable capacity-binding opportunity and target-reached
times are separate. A binding opportunity within one arm is not an exact causal
divergence from another arm. Record frontier alignment from available raw fields;
wall-clock-triggered repeats need not have identical prefixes or KV tensors.
admission_s is client submission; first scheduled prefill identifies native entry.

All measurement/decision/apply/host costs remain in original request/episode
wall-clock denominator. Initialization and identical warmups remain outside.
No replay-generated policy result or rescaled performance metric. Compare all
four paired domains/repeats against hold32 and static16; historical multi-step
feedback is context, not a newly randomized causal arm.

Retain every attempt. GPU process isolation is checked before initialization and
at episode boundaries; do not kill other work or change memory settings to fit.
On sign reversal or noise, perform at most one unchanged controlled repeat of
this same12-episode block, preserving both. No thresholds, domains or predictors
will be searched based on outcomes. If a one-shot action cannot reliably exceed
the contemporaneous simple baselines, stop this triggering/downward formulation
in this workload. This is not zero dynamic Oracle or a family-wide NO-GO.

Allowed ceiling: single-model/single-GPU native in-process request measurement;
finite reused cohort and fixed generation length. U/C,quality,natural EOS,
second model,HTTP/production,steady-state capacity and dynamic Oracle are UNRUN.

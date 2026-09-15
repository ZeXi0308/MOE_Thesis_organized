# Staged single-event recovery: service-window input mapping

Read-only source: `/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation/outputs/admission_capacity/20260914_staged_store_probe_r01`; latest shared `RESULT_LEDGER.md` consulted. This note adds only the service-window input boundary and one victim's service/resumption split. It does not duplicate the full-rotation model or change source artifacts.

Verdict: this observed victim has a long resumed service interval and a delay dominated by time before restoration begins. Native saving/loading and notification-gated execution are qualified for one event. `C_remaining` in milliseconds, avoidable marginal time, and an executable future-time bound remain UNKNOWN.

## What this execution establishes

- Both arms use one staged most-output event: prepare step329, preempt step330, original absent request3640 first, protection released after its new output at333 is observed by the step334 hook. Both original targets first output at333. This differs from the old selective-store event, which resumed its newly preempted victim3571 first. Do not attribute the staged ordering's improvement to saving or import the old two-load sequence.
- New victim0000001: before prepare, computed3398/history3399, held213 blocks; the preparation call performs one more decode. At actual preemption it has computed3399/history3400 and328 returned outputs. Actual store covers only3392 tokens/212 complete blocks; the seven computed tail positions and pending first-output input are outside the saved prefix.
- Save-on registers store job114 at329, requires its native flush metadata at330, and has observed completed store114/load115. Store/load are each444596224 bytes. This is runtime interface/coverage evidence, not a byte-level KV fidelity test or direct timing of the physical flush fence.
- Load starts in1048: victim changes from0 blocks/computed0 to212 held blocks/native computed3392, but receives no scheduled compute or new output in that call. Its own completion notification is observed at1048 call end;1049 executes8 positions =7 recompute+1 decode, then returns its first new output and holds213 blocks.

## Known fields versus unknown costs

| Window input | Permitted value/use | Boundary |
|---|---|---|
| Request identity/history | victim0000001, history3400 at restore; source IDs joined through raw.internal_to_source | Preserve per-arm internal identity; do not share future state. |
| Saved host prefix |3392 tokens,212 blocks,444596224 bytes; actual store/load job ownership observed | Runtime-declared source validity only; no bitwise/numerical-epoch fidelity result for this victim. |
| Valid GPU prefix | During1048 load pending, allocated prefix is not ready; after own notification,3392 is runnable at1049 | Native computed must not itself clear load blocking. |
| Remaining execution positions |8 positions at1049, of which7 are old-prefix recompute and1 first-output decode | This is work coverage; it does not fill `C_remaining` or `prospective_tax_ms`. |
| Actual GPU holdings |212 blocks while load pending;213 after first-output compute | Charge pending holdings immediately, independently of ready. |
| Global free blocks | on1048 before691/after470; on1049 before470/after468 |212 load blocks plus peer/target growth are already included; do not add212 again. At1049 victim grows1 block and peers together grow1 block. |
| Host budget |17179869184 bytes configured in both arms | Budget is known; actual host occupancy/peak, staging and metadata are not measured. Do not set `host_used_bytes` equal to transferred bytes. |
| Transfer statistics | Worker reports store0.007948416s/load0.008301408s in offload-events.transfers | Preserve as reported transfer statistics; not an exposed request cost or causal saving. |
| New outputs/age reset | Use raw.output_events received_s and cumulative/new output IDs | An allocated/computed prefix or completed DMA job does not reset output age. |
| First-output timing/remaining marginal tax | Keep unknown for prospective choices | Retrospective call spans and this arm's future notification cannot qualify an unexecuted action. |

Per-arm raw source: `readback/results/save-{off,on}/raw.json` (`memory_trace[1048:1050]`, `scheduler_steps[1048:1050]`, `engine_steps`, `output_events`, `preemption_events`). Store registration/flush source: each arm's `selective-store.json`. Completed jobs/transfer counts: `offload-events.json`. Notification boundary: `structural_replay.json` observations.

## Victim service after restoration

| Observed field | Save-off | Save-on |
|---|---:|---:|
| Last pre-pause output cumulative count |328|328|
| First restored output call |1051|1049|
| Completion output call |1746|1744|
| New outputs from restoration through completion |696|696|
| Later preemptions of this victim |0|0|
| Executed saved-prefix loads |0|1|

Thus the loaded prefix supports this uninterrupted residency until completion. No later discard, second load or repeated-prefix reuse is established. This event does not exhibit a1–2-output restoration window.

## Delay before restoration versus elapsed restoration span

All boundaries below use the same arm's host perf_counter origin. Define L = victim output event with cumulative_tokens328; S = engine_steps[1048].start_s, the first call performing victim recompute(off) or allocating/submitting its load(on); F = victim output event with cumulative_tokens329. This is descriptive interval accounting, not a causal decomposition or a deduction of peer service.

| Seconds | Save-off | Save-on |
|---|---:|---:|
| L received_s |7.766037500|8.904956779|
| S engine-call start_s |23.389129490|26.274112156|
| F received_s |23.504791116|26.334749684|
| F−L total output gap |15.738753617|17.429792905|
| S−L before first restoration call |15.623091990|17.369155377|
| F−S after restoration call begins |0.115661627|0.060637528|
| Fraction before restoration call |99.2651%|99.6521%|

The first interval includes the short last-output-to-preemption interval and subsequent waiting; it is not an isolated scheduler-queue timer. The second includes scheduling, asynchronous completion and peer computation/output; it is not all removable recovery cost. Neither difference between arms is drift-corrected or a stable timing effect.

The observed recovery saves3392 recompute positions but replaces two recompute calls with two decode calls:1794 total calls in both arms.47 output positions move from recompute-containing calls to pure-decode calls (`time_and_calls.json`). Full-batch remaining service, async notification timing under other actions, actual host peaks and repeated-rotation costs remain unmeasured. Use these exact state fields to check structural eligibility; retain unknown `C_remaining` and defer prospective amortization eligibility.

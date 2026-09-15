# Staged single-event recovery: execution qualified, net benefit unproven

Verdict: SINGLE_EVENT_EXECUTION_QUALIFIED / MEASUREMENT_ONLY. New physical RTX5090 (GPU-bd5e9bb9-f98b-db5b-cc1c-5857c39f0bdc), native vLLM0.26 in-process, fixed d6 closed cohort, 32 requests per arm, 1024 output IDs each. Native16GiB host cache in both arms. One save-off then one save-on, one staged most-output event each. Both COMPLETE; group exit0, GPU released. This is not full repeated most-output rotation.

| Metric | save-off | save-on |
|---|---:|---:|
| Mean completion (s) | 23.8405 | 26.8345 |
| Episode wall (s) | 30.3769 | 34.1901 |
| Maximum ITL (s) | 15.7388 | 17.4298 |
| Engine calls | 1794 | 1794 |
| Recomputed tokens | 24792 | 21400 |
| Original target first new output step | 333 | 333 |
| Victim first new output step | 1051 | 1049 |
| Store / load bytes | 0 / 0 | 444596224 / 444596224 |

Actual native metadata registered one store covering 3392tokens/212 complete blocks, required its flush at preemption, and worker observations recorded completed store and load jobs. Original target3640 receives priority, then releases protection after a new output. Victim0000001 later loads its saved prefix. This qualifies the single-event interface, not byte-level fidelity for this victim or a directly timed physical fence.

Saved3392tokens reduces recompute calls26→24, but pure-decode calls1670→1672. Counts do not support an engine-step gain. Original target's first-new-output step is unchanged between arms; moving it forward is common to both staged queue policies, not attributable to saving. Victim's missing-service gap remains large (15.739s/17.430s).

Save-on mean completion is +12.558%, wall +12.553%, maxITL +10.744% in this pair. Before preparation at step329, the same scheduling prefix already differs by +1.140924s; post-cutoff mean duration differs +1.853034s. Their sum equals the +2.993959s mean difference. This arithmetic partition is not drift correction or proof of the cause of slowdown. JIT warnings and warmup logs are retained. n=1/arm with fixed order; no stable performance effect, noise bound, quality or transfer-to-other-device claim.

Strongest comparison here: identical staged action without saving. The broader strong baseline remains full most-output rotation; this pair does not beat or replace it. Oracle/headroom is unresolved. Failure category: local work reduction does not reduce total calls in this observed pair; net effect confounded. The research problem is not rejected.

Next smallest step: feed this observed single-event transition into the existing full-rotation structural model and determine whether repeated saving can change useful service/total steps relative to most-output, before another GPU matrix. No third same-event timing repeat to chase a positive sign. Any model replay using actual completion notifications is descriptive only; alternative policies require their own execution.

Raw archive SHA256 bfad2152a754197ae558608e3d4f312d106d5f902b75327b3ce99c0b78e80c75;19 packaged files match local manifest. analysis.json and time_and_calls.json derived from both retained raw files. Fresh integrity review pending; no method GO.

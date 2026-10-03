# Before1026 allocation certificate

**POSTHOC_PRESTATE_STRUCTURAL_CERTIFICATE.** Candidate arithmetic uses only `memory_trace[1026].before`, current request aliases, declared output cap1024, model context4096 and token budget1024. It reuses `service_window_model.growth`. No state or scheduled token count from1027 onward generates candidates; actual1029 allocation failure appears only under `validation_only_1029`. This is not prospective GPU ranking or an executed alternate policy.

The target `0020902` has history3772, computed0 and held0. Initial free blocks are242. The27 current running peers all have exactly one input pending; their actual held block tables total6414 blocks. Blocks held by paused peers stay allocated. The finite action keeps this target and a fixed FCFS peer prefix; it makes no additional admission from the existing waiting queue. Unknown EOS and new arrivals are absent by assumption. Declared output-cap completion is respected and releases blocks only after a successfully allocated batch returns.

| Recovery batch | All27 target positions | All27 cumulative peer growth | All27 joint additional blocks | All27 free at allocation peak | Prefix22 target positions | Prefix22 joint additional blocks |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 997 | 2 | 65 | 177 | 1002 | 65 |
| 2 | 997 | 3 | 128 | 114 | 1002 | 129 |
| 3 | 997 | 5 | 192 | 50 | 1002 | 192 |
| 4 / first output | 781 | 7 | 243 | −1 | 766 | 242 |

Initially the current full history fits: after the first batch's peer allocation,240 blocks remain for the236-block target history. This fit is an admission check, not a persistent future reservation. By batch4, peers require7 additional blocks in total and the target236, exceeding242 by1. Raw1029 subsequently reports target requested781 positions at computed2991 and free48:49 additional target blocks are required. This is the sole future-state comparison.

Peer `0017190` starts with1020 outputs, so it would reach the declared1024 cap in batch4. Its256 blocks are unavailable until that batch returns. The all27 action therefore fails before its first output; the script does not advance that infeasible action using the later release.

The longest feasible fixed FCFS prefix has22 peers. It pauses the following five, keeping all their KV:

| Paused source ID suffix | Retained blocks | Foregone output opportunities through first output | Through the7-batch horizon |
|---|---:|---:|---:|
| 0020184 | 252 | 4 | 7 |
| 0020257 | 220 | 4 | 7 |
| 0020323 | 252 | 4 | 7 |
| 0020958 | 203 | 4 | 7 |
| 0020734 | 243 | 4 | 7 |

Thus the prefix action transfers20 peer output opportunities by the target's first output, or35 if retained throughout7 batches. No milliseconds or output-age guarantee is inferred. Prefixes23–27 do not fit through first output under the same assumptions; this is a prefix-family comparison, not an optimal subset search.

The prefix22 action reaches its first output with zero free blocks at the allocation peak. After that successful return, the known cap releases256 blocks. Three further target outputs fit in batches5–7; free blocks at their allocation peaks are256,255,478. Peer `0017245` reaches its declared cap at batch6 and releases224 blocks only afterward. These are **four total target output opportunities**, including the first, respecting the parent's seven-batch bound. Other queued admissions or earlier unknown EOS would change this conditional continuation; it is not the original future trajectory.

In JSON, `joint_new_blocks_before_return` is net occupancy growth relative to before1026 after subtracting blocks released at earlier returns. Its negative values after batch4 denote lower occupancy, not negative allocation. The first four batches have no prior release and therefore give the exact joint new-block demand shown above.

Reproduce into a new directory:

```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/certify_funding_resume_window.py \
  --raw /private/tmp/moe-a-recovery-components-20260914/refine-logs/expert_saturation/outputs/admission_capacity/20260914_funding_filter_comparison_r01/execution/readback/results/funding-block0-least_feasible/raw.json \
  --output-dir PATH_TO_NEW_DIRECTORY
```

The conclusion is local: full current-history fit does not guarantee completion of restoration while every current peer keeps growing. A fixed peer prefix can fund the first output in the same pool, at explicit peer-service cost. This establishes neither elapsed-time benefit nor a new scheduling mechanism.

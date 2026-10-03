# Existing-data CPU recomputation

Executed from repository root during the 2026-09-12 asset audit. All four commands
exited 0 and wrote fresh `/tmp` directories. No GPU run was started; old raw was
not changed. Existing expert-union work was preserved.

```bash
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260908_native_preemption_r01/analyze_native_preemption.py --output-dir /tmp/moe-asset-audit-preemption-20260912

python3 refine-logs/independent_ideas_20260908/host_timing_boundary_r01/prefill_budget_midpoint_r01/analyze_results.py --results-dir refine-logs/independent_ideas_20260908/host_timing_boundary_r01/prefill_budget_midpoint_r01/readback/results --output-dir /tmp/moe-asset-audit-prefill512-20260912

python3 refine-logs/independent_ideas_20260911/per_request_prefill_share_r01/analyze_results.py --results-dir refine-logs/independent_ideas_20260911/per_request_prefill_share_r01/readback/results --output-dir /tmp/moe-asset-audit-share512-20260912

python3 refine-logs/expert_saturation/outputs/admission_capacity/20260912_feasibility_envelope_r01/analyze_envelope.py --campaign refine-logs/expert_saturation/outputs/admission_capacity/20260908_capture_ladder_paired_r01 --campaign refine-logs/expert_saturation/outputs/admission_capacity/20260908_capture_ladder_paired_repeat_r01 --output-dir /tmp/moe-asset-audit-envelope-20260912
```

Use new output-directory names when reproducing. Native preemption returned
`MEASUREMENT_ONLY`, `engine_args_equal=true`. Midpoint returned four measured
cells, four warmups, two comparisons, no issues. Per-request share returned
`DESCRIPTIVE_MEASUREMENT_ONLY`, 12 cells, 12 warmups, no issues. Envelope reproduced
64 validation rows, median relative error 3.25%, p90 6.09%, and 60/64 verdict
agreement.

The envelope conditions on realized width, prefill counts and episode duration;
these statistics are retrospective. Reproducing its numerical results does not
validate prospective action prediction, universal infeasibility, TTFT, or tails.

Existing targeted CPU tests, also exit 0:

```bash
python3 -B -m unittest -v refine-logs/expert_saturation/experiments/gpu_pressure_sketch/test_route_pressure_sketch.py

python3 -B -m unittest discover -s refine-logs/expert_saturation/experiments/admission_capacity -p 'test_kv_feasibility_admission.py' -v
```

The sketch suite ran 5 tests; the KV-feasibility suite ran 14 tests. These verify
CPU contracts and arithmetic, not native integration or GPU performance.

## Retained processed output

Reports, tables, `envelope/envelope.json`, and `preemption/pause_recompute.json`
are byte copies of the `/tmp` output. `metrics_compact.json` files copy existing
processed values while omitting duplicated per-request/token histories and
identities; each file records its original source and omitted keys. No fitting or
new measurement occurred while retaining these files. Full request/raw data
remains in the original campaign directories referenced by the commands.

The pause decomposition was independently reconstructed from old raw with the
following command; it matches the existing analyzer's interval definition and
writes only a fresh processed file:

```bash
python3 -B - <<'PY'
import json
from pathlib import Path
root=Path('refine-logs/expert_saturation/outputs/admission_capacity/20260908_native_preemption_r01')
rows=[]
for rep in (0,1):
 raw=json.loads((root/'gpu_results'/f'repeat{rep}-native32'/'raw.json').read_text())
 for ev in raw['preemption_events']:
  rid=ev['victim_internal_request_id']; src=raw['internal_to_source'][rid]
  req=next(r for r in raw['requests'] if r['request_id']==src); ts=req['token_times_s']
  i=max(range(len(ts)-1),key=lambda i:ts[i+1]-ts[i]); a,b=ts[i:i+2]
  steps=[s['step'] for s in raw['scheduler_steps'] if any(r['internal_request_id']==rid and r['recompute_tokens']>0 for r in s['scheduled'])]
  calls=[c for c in raw['engine_steps'] if any(c['scheduler_step_start']<=s<c['scheduler_step_end'] for s in steps)]
  first=min(c['start_s'] for c in calls); last=max(c['returned_s'] for c in calls)
  assert a<=first<=last<=b and all(c['completed'] for c in calls)
  rows.append(dict(repeat=rep,request=src,gap_s=b-a,before_recompute_s=first-a,recompute_span_s=last-first,after_s=b-last))
print(json.dumps(rows,indent=2))
Path('/tmp/moe-asset-audit-preemption-20260912/pause_recompute.json').write_text(json.dumps(rows,indent=2)+'\n')
PY
```

Recomputation-call spans include concurrent work and host overhead. They are not
isolated GPU recomputation time and must not be added again to request latency.

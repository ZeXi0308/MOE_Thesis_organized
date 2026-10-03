#!/usr/bin/env python3
"""Post-hoc mean-TPOT versus request-max-ITL readout; no new SLO or execution."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
LABELS=['repeat0-native32','repeat0-safe','repeat1-safe','repeat1-native32']
METRICS=['mean_tpot_s','max_itl_s']
cells={}
for label in LABELS:
    path=ROOT/'gpu_results'/label/'raw.json';raw=json.loads(path.read_text())
    assert raw['status']=='COMPLETE' and all(r['status']=='completed' for r in raw['requests'])
    requests=[]
    for r in raw['requests']:
        t=r['token_times_s'];assert len(t)==len(r['output_token_ids'])==1024
        requests.append(dict(request_id=r['request_id'],document_id=r['document_id'],
            ttft_s=t[0]-r['arrival_s'],mean_tpot_s=(t[-1]-t[0])/(len(t)-1),
            max_itl_s=max(b-a for a,b in zip(t,t[1:]))))
    assert len(requests)==32 and min(r['arrival_s'] for r in raw['requests'])==0
    cells[label]=dict(raw_path=str(path),duration_s=raw['observation_end_s'],requests=requests)

def evaluate(threshold):
    result=dict(threshold_s=threshold,readouts={},winners={})
    for metric in METRICS:
        by_cell={}
        for label,c in cells.items():
            passed=sum(r['ttft_s']<=5.0 and r[metric]<=threshold for r in c['requests'])
            by_cell[label]=dict(passed=passed,goodput_rps=passed/c['duration_s'])
        result['readouts'][metric]=by_cell
        result['winners'][metric]={}
        for repeat in [0,1]:
            a,b=[by_cell[f'repeat{repeat}-{arm}']['goodput_rps'] for arm in ['native32','safe']]
            result['winners'][metric][str(repeat)]='native32' if a>b else 'safe' if b>a else 'TIE'
    return result

thresholds=sorted({r[m] for c in cells.values() for r in c['requests'] for m in METRICS})
points=[evaluate(t) for t in thresholds];zero=evaluate(0.0);reference=evaluate(0.2)
regions=[]
for repeat in [0,1]:
    current=None
    for point in [zero,*points]:
        winners=[point['winners'][m][str(repeat)] for m in METRICS]
        if current is None or winners!=current['winners']:
            if current is not None:current['upper_exclusive_s']=point['threshold_s']
            current=dict(repeat=repeat,lower_inclusive_s=point['threshold_s'],upper_exclusive_s=None,winners=winners)
            regions.append(current)
limits=['Post-hoc description of the same 32 documents and two repeats; no independent-token inference.',
    'TTFT <=5s is fixed. The generation threshold applies separately to mean TPOT or maximum ITL, both inclusive.',
    'All observed values from both metrics and all four cells form the exact breakpoint union; none are selected for a desired winner.',
    'Goodput uses the full original episode duration; no stage cost is added or removed.',
    'This readout authorizes no new SLO, threshold or policy and contains no budget95 measurement.']
result=dict(status='POSTHOC_READOUT_ONLY',ttft_threshold_s=5.0,generation_metrics=METRICS,
    cells=cells,breakpoints=points,zero_threshold=zero,reference_200ms=reference,winner_regions=regions,limits=limits)
lines=['# Mean TPOT versus request-max ITL: post-hoc readout','',*['- '+x for x in limits],'',
    '| Cell | Mean-TPOT 200ms passes | Max-ITL 200ms passes | Mean-rule goodput | Max-rule goodput |',
    '|---|---:|---:|---:|---:|']
for label in LABELS:
    a,b=[reference['readouts'][m][label] for m in METRICS]
    lines.append(f"| {label} | {a['passed']}/32 | {b['passed']}/32 | {a['goodput_rps']:.6f} | {b['goodput_rps']:.6f} |")
lines+=['','## All winner intervals for the common generation threshold','',
    'Intervals are left-inclusive and right-exclusive; infinity means no further observed breakpoint changes the winner. These are descriptive comparisons, not threshold recommendations.','',
    '| Repeat | Threshold interval (s) | Mean-TPOT winner | Max-ITL winner |','|---|---|---|---|']
for r in regions:
    upper='infinity' if r['upper_exclusive_s'] is None else repr(r['upper_exclusive_s'])
    lines.append(f"| {r['repeat']} | [{r['lower_inclusive_s']!r}, {upper}) | {r['winners'][0]} | {r['winners'][1]} |")
lines+=['',f'{len(thresholds)} exact observed breakpoints and all 128 per-request values are retained in readout.json.',
    'At 200ms, mean TPOT hides two long-paused native32 requests in each repeat; max ITL removes those two passes. Native32 still has higher full-episode goodput under both definitions.','']
out=ROOT/'slo-readout';out.mkdir(exist_ok=False)
(out/'readout.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
(out/'report.md').write_text('\n'.join(lines))
print(json.dumps(dict(status=result['status'],breakpoints=len(thresholds),reference_200ms=reference),indent=2))

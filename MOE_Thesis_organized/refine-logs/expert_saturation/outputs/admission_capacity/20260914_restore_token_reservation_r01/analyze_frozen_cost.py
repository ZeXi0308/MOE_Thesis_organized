"""Cost transfer on actual T schedules; no refit, alternate future, or Oracle."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent/'20260914_recovery_progress_model_r01/cost_baseline.json'
MODEL_SHA = '207ad5a01595954fd7fddbba9933c1a297dfdd2caa6f819cc7276fe982f679f1'
CUTOFF = 299  # First preemption in the earlier S package, fixed before T readback.
load = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def inspect(folder, beta):
    raw = load(folder/'raw.json')
    steps = {s['step']:s for s in raw['scheduler_steps']}
    calls = [c for c in raw['engine_steps'] if c['scheduler_step_start'] >= CUTOFF]
    assert calls[0]['scheduler_step_start'] == CUTOFF and all(c['completed'] for c in calls)
    origin = calls[0]['start_s']
    predicted_clock, clock_at_return, categories = 0., {}, {}
    features = [0, 0, 0]
    for i, call in enumerate(calls):
        s = steps[call['scheduler_step_start']]
        assert not any(r['prefill_tokens'] for r in s['scheduled'])
        x = [1, len(s['scheduled']), sum(r['scheduled_tokens'] for r in s['scheduled'])]
        features = [a+b for a,b in zip(features,x)]
        prediction = sum(b*v for b,v in zip(beta,x))
        actual = (calls[i+1]['start_s'] if i+1 < len(calls) else call['returned_s'])-call['start_s']
        assert actual > 0 and prediction > 0
        predicted_clock += prediction
        clock_at_return[call['returned_s']] = predicted_clock
        category = 'recompute_mixed' if s['recompute_tokens'] and s['decode_requests'] else (
            'recompute_only' if s['recompute_tokens'] else 'decode_or_empty')
        row = categories.setdefault(category,dict(calls=0,actual_s=0.,predicted_s=0.))
        row['calls'] += 1; row['actual_s'] += actual; row['predicted_s'] += prediction
    requests = []
    for q in raw['requests']:
        assert q['status'] == 'completed' and q['completion_s'] in clock_at_return
        requests.append(dict(request_id=q['request_id'],actual_remaining_s=q['completion_s']-origin,
                             estimated_remaining_s=clock_at_return[q['completion_s']]))
    assert len(requests) == 32
    actual_mean = statistics.mean(q['actual_remaining_s'] for q in requests)
    estimated_mean = statistics.mean(q['estimated_remaining_s'] for q in requests)
    actual_last = calls[-1]['returned_s']-origin
    assert abs(sum(v['actual_s'] for v in categories.values())-actual_last) < 1e-8
    return dict(label=folder.name,raw_sha256=sha(folder/'raw.json'),cutoff=CUTOFF,
                actual_origin_s=origin,features=features,estimated_terms_s=[b*x for b,x in zip(beta,features)],
                actual_mean_remaining_s=actual_mean,estimated_mean_remaining_s=estimated_mean,
                mean_error_pct=100*(estimated_mean/actual_mean-1),actual_last_remaining_s=actual_last,
                estimated_last_remaining_s=predicted_clock,last_error_pct=100*(predicted_clock/actual_last-1),
                categories=categories,requests=requests)


def main():
    assert sha(MODEL) == MODEL_SHA, 'frozen model changed'
    primary = load(ROOT/'analysis/analysis.json')
    assert primary['status'] == 'MEASUREMENT_ONLY' and len(primary['cells']) == 6
    model = load(MODEL); beta = model['beta_seconds']
    rows = [inspect(ROOT/'execution/readback/results'/r['label'], beta) for r in primary['cells']]
    pairs = []
    for a,b in ((1,2),(0,2),(4,3),(5,3)):
        left,right=rows[a],rows[b]
        pairs.append(dict(baseline=left['label'],action=right['label'],
            actual_mean_delta_s=right['actual_mean_remaining_s']-left['actual_mean_remaining_s'],
            estimated_mean_delta_s=right['estimated_mean_remaining_s']-left['estimated_mean_remaining_s'],
            actual_last_delta_s=right['actual_last_remaining_s']-left['actual_last_remaining_s'],
            estimated_last_delta_s=right['estimated_last_remaining_s']-left['estimated_last_remaining_s']))
    output=ROOT/'analysis/cost_transfer'; output.mkdir()
    result=dict(status='OBSERVED_SCHEDULE_COST_TRANSFER',model_sha256=MODEL_SHA,beta_seconds=beta,
        training_paths=model['training_paths'],cutoff=CUTOFF,cells=rows,pairs=pairs,
        scope='Frozen native-calibrated coefficients applied to independently executed T schedules. '
        'Actual future schedule is an input here, so this is no action predictor or counterfactual. '
        'Intervals include host gaps; final interval ends at return. Model completion uses interval end, '
        'which includes the next-call gap for non-final completion calls, matching the old model convention. '
        'Neither coefficients nor category spans are physical kernel costs. No refit or excluded outlier.')
    (output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],pairs=pairs)))


if __name__ == '__main__': main()

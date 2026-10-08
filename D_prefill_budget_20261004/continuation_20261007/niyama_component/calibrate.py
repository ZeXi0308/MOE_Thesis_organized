"""CPU-only transfer calibration of Niyama's two linear cost regimes.

No GPU profiling, margin search, clipping, or random per-step train/test split.
The held-out units are complete reverse-order runs. Costs are host engine-step
completion durations, not CUDA-only kernel times or full request latencies.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
FEATURES = ('P', 'D', 'Cdecode', 'Cprefill')
TRAIN = ('pressure01/00_fixed384', 'pressure01/01_fixed2048',
         'highrate01/00_fixed1024', 'highrate01/01_fixed2048', 'upper01/01_fixed4096')
VALIDATE = ('pressure01/02_fixed2048', 'pressure01/03_fixed384',
            'highrate01/02_fixed2048', 'highrate01/03_fixed1024', 'upper01/02_fixed4096')
REGIMES = ('total_le_512', 'total_gt_512')
MARGIN = 1.2
TOTAL_TIERS = [128] + list(range(256, 2049, 256)) + [2552]
LOW_MEMORY_TIERS = list(range(128, 513, 128))


class RankDeficientError(ValueError):
    pass


def quantile(values, q):
    if not values:
        return None
    a = sorted(values)
    position = (len(a)-1)*q
    lo, hi = math.floor(position), math.ceil(position)
    return a[lo]*(hi-position)+a[hi]*(position-lo) if lo != hi else a[lo]


def fit_linear(samples, min_samples=100):
    """Centered/scaled least squares, column-pivoted Householder QR; no ridge."""
    n, p = len(samples), len(FEATURES)
    if n < max(min_samples, p+1):
        raise RankDeficientError(f'{n} samples; require at least {max(min_samples, p+1)}')
    means = [statistics.fmean(row['x'][j] for row in samples) for j in range(p)]
    scales = [math.sqrt(statistics.fmean((row['x'][j]-means[j])**2 for row in samples))
              for j in range(p)]
    if any(s == 0 or not math.isfinite(s) for s in scales):
        raise RankDeficientError(f'Constant/nonfinite feature; scales={scales}')
    ymean = statistics.fmean(row['y'] for row in samples)
    a = [[(row['x'][j]-means[j])/scales[j] for j in range(p)] for row in samples]
    b = [row['y']-ymean for row in samples]
    permutation = list(range(p))
    tolerance = 1e-10*math.sqrt(n)
    diagonals = []
    for k in range(p):
        pivot = max(range(k,p), key=lambda j: math.fsum(a[i][j]**2 for i in range(k,n)))
        if pivot != k:
            for row in a:
                row[k], row[pivot] = row[pivot], row[k]
            permutation[k], permutation[pivot] = permutation[pivot], permutation[k]
        norm = math.sqrt(math.fsum(a[i][k]**2 for i in range(k,n)))
        if norm <= tolerance or not math.isfinite(norm):
            raise RankDeficientError(f'Rank {k}/{p}; residual column norm {norm}, tolerance {tolerance}')
        alpha = -math.copysign(norm, a[k][k])
        v = [a[i][k] for i in range(k,n)]
        v[0] -= alpha
        vnorm = math.sqrt(math.fsum(t*t for t in v))
        v = [t/vnorm for t in v]
        for j in range(k,p):
            dot = math.fsum(v[i-k]*a[i][j] for i in range(k,n))
            for i in range(k,n):
                a[i][j] -= 2*v[i-k]*dot
        dot = math.fsum(v[i-k]*b[i] for i in range(k,n))
        for i in range(k,n):
            b[i] -= 2*v[i-k]*dot
        diagonals.append(abs(a[k][k]))
    solution = [0.]*p
    for j in reversed(range(p)):
        solution[j] = (b[j]-math.fsum(a[j][k]*solution[k] for k in range(j+1,p)))/a[j][j]
    coefficients = [0.]*p
    for j, original in enumerate(permutation):
        coefficients[original] = solution[j]/scales[original]
    intercept = ymean-math.fsum(c*m for c,m in zip(coefficients,means))
    return dict(intercept_s=intercept, coefficients_s_per_unit=dict(zip(FEATURES,coefficients)),
                fit_rows=n, rank=p, standardized_qr_diagonal=diagonals,
                rank_tolerance=tolerance, centered_feature_means=dict(zip(FEATURES,means)),
                feature_scales=dict(zip(FEATURES,scales)))


def predict(model, x):
    return model['intercept_s'] + math.fsum(model['coefficients_s_per_unit'][k]*v for k,v in zip(FEATURES,x))


def regime(x):
    return REGIMES[int(x[0]+x[1] > 512)]


def describe(samples):
    return dict(rows=len(samples), pure_decode_rows=sum(r['x'][0]==0 and r['x'][1]>0 for r in samples),
        mixed_rows=sum(r['x'][0]>0 and r['x'][1]>0 for r in samples),
        pure_prefill_rows=sum(r['x'][0]>0 and r['x'][1]==0 for r in samples),
        empty_rows=sum(r['x'][0]+r['x'][1]==0 for r in samples),
        exact_total_513_rows=sum(r['x'][0]+r['x'][1]==513 for r in samples),
        feature_ranges={k:dict(min=min((r['x'][j] for r in samples),default=None),
                                   max=max((r['x'][j] for r in samples),default=None)) for j,k in enumerate(FEATURES)},
        cost_s=dict(min=min((r['y'] for r in samples),default=None),
                    max=max((r['y'] for r in samples),default=None),
                    mean=statistics.fmean(r['y'] for r in samples) if samples else None))


def load_run(name):
    path = DATA/name/'raw.json'
    blob = path.read_bytes()
    raw = json.loads(blob)
    status = json.loads((path.parent.parent/'status.json').read_text())
    protocol = json.loads((path.parent.parent/'protocol.json').read_text())
    requests, steps = raw['requests'], raw['steps']
    checks = dict(parent_status=status.get('status'), requests=len(requests),
        expected_requests=protocol['request_count'], unfinished_requests=sum(not r['finished'] for r in requests),
        missing_completion_requests=sum(r['completion_s'] is None for r in requests),
        output_count_mismatches=sum(len(r['output_token_ids'])!=r['max_tokens'] for r in requests),
        token_timestamp_count_mismatches=sum(len(r['token_times_s'])!=len(r['output_token_ids']) for r in requests),
        prompt_tokens=sum(r['prompt_tokens'] for r in requests),
        output_tokens=sum(len(r['output_token_ids']) for r in requests),
        preempted_steps=sum(bool(s['preempted']) for s in steps),
        raw_error_fields={k:raw[k] for k in ('error','errors','traceback') if k in raw})
    checks['passed'] = (checks['parent_status']=='COMPLETE' and checks['requests']==checks['expected_requests']
        and all(checks[k]==0 for k in ('unfinished_requests','missing_completion_requests','output_count_mismatches',
                                      'token_timestamp_count_mismatches','preempted_steps'))
        and not checks['raw_error_fields'])
    samples, excluded, extraction_errors = [], [], []
    for i, step in enumerate(steps):
        try:
            duration = step['end_s']-step['start_s']
            assert math.isfinite(duration) and duration > 0, 'Nonpositive/nonfinite completed-step duration'
            assert step['prefill_tokens']==sum(r['prefill_tokens'] for r in step['requests']), 'P total mismatch'
            assert step['decode_tokens']==sum(r['decode_tokens'] for r in step['requests']), 'D total mismatch'
            x = [step['prefill_tokens'], step['decode_tokens'],
                 sum(r['computed_start']+1 for r in step['requests'] if r['decode_tokens']>0),
                 sum(r['computed_start'] for r in step['requests'] if r['prefill_tokens']>0)]
            assert all(math.isfinite(v) and v >= 0 for v in x), 'Invalid feature value'
            assert x[0]+x[1] > 0, 'Empty native step has no positive-token regime'
            row = dict(run=name, step_index=i, x=x, y=duration)
            samples.append(row)
            if duration > 1.:
                excluded.append(dict(step_index=i,cost_s=duration,features=dict(zip(FEATURES,x)),
                                     reason='Artifact training filter: host step duration > 1 second'))
        except (AssertionError, KeyError, TypeError) as exc:
            extraction_errors.append(dict(step_index=i,error=repr(exc)))
    checks['extraction_errors'] = extraction_errors
    checks['passed'] = checks['passed'] and not extraction_errors
    return samples, dict(run=name,raw_sha256=hashlib.sha256(blob).hexdigest(),checks=checks,
        excluded_from_artifact_fit_or_filtered_metrics=excluded,
        all_steps=describe(samples), artifact_filtered_steps=describe([r for r in samples if r['y']<=1.]))


def error_stats(samples, models, training_ranges):
    if not samples:
        return dict(rows=0)
    predictions = [predict(models[regime(row['x'])],row['x']) for row in samples]
    actual = [r['y'] for r in samples]
    outside = []
    for row in samples:
        ranges = training_ranges[regime(row['x'])]['feature_ranges']
        if any(v < ranges[k]['min'] or v > ranges[k]['max'] for k,v in zip(FEATURES,row['x'])):
            outside.append(row['step_index'])
    def at_margin(margin):
        residuals = [y-margin*p for y,p in zip(actual,predictions)]
        return dict(mean_absolute_error_s=statistics.fmean(abs(x) for x in residuals),
            mean_signed_underprediction_s=statistics.fmean(residuals),
            p95_positive_underprediction_s=quantile([max(0.,x) for x in residuals],.95),
            maximum_positive_underprediction_s=max(max(0.,x) for x in residuals),
            actual_above_prediction_count=sum(x>0 for x in residuals),
            actual_above_prediction_fraction=sum(x>0 for x in residuals)/len(residuals))
    return dict(rows=len(samples), actual_mean_s=statistics.fmean(actual),actual_p95_s=quantile(actual,.95),
        prediction_min_s=min(predictions),prediction_max_s=max(predictions),
        negative_prediction_count=sum(p<0 for p in predictions),
        negative_prediction_step_indices=[r['step_index'] for r,p in zip(samples,predictions) if p<0],
        zero_prediction_count=sum(p==0 for p in predictions),
        outside_training_coordinate_ranges_count=len(outside),
        outside_training_coordinate_ranges_step_indices=outside,
        unscaled=at_margin(1.), fixed_margin_1_2=at_margin(MARGIN))


def monotonicity(samples, models):
    """Hold observed D/contexts fixed; substitute legal cap candidates only.

    These are predictor-domain checks, not replayed decisions or action outcomes.
    """
    transitions = {}
    cross = []
    negatives = []
    for row in samples:
        _, d, cd, cp = row['x']
        def predicted(total):
            x = [max(0,total-d),d,cd,cp]
            return predict(models[regime(x)],x)
        cross.append(dict(step_index=row['step_index'],
            decline_s=predicted(512)-predicted(513), D=d,Cdecode=cd,Cprefill=cp))
        for domain, tiers in [('normal',TOTAL_TIERS),('low_memory',LOW_MEMORY_TIERS)]:
            values=[predicted(total) for total in tiers]
            for total,value in zip(tiers,values):
                if value < 0:
                    negatives.append(dict(step_index=row['step_index'],domain=domain,total_tier=total,prediction_s=value))
            for lower,upper,left,right in zip(tiers,tiers[1:],values,values[1:]):
                key=f'{domain}:{lower}->{upper}'
                entry=transitions.setdefault(key,dict(comparisons=0,decline_count=0,maximum_decline_s=0.,worst_example=None))
                entry['comparisons']+=1
                if right < left:
                    entry['decline_count']+=1
                    if left-right > entry['maximum_decline_s']:
                        entry['maximum_decline_s']=left-right
                        entry['worst_example']=dict(step_index=row['step_index'],D=d,Cdecode=cd,Cprefill=cp,
                            lower_prediction_s=left,upper_prediction_s=right)
    declines=[r for r in cross if r['decline_s']>0]
    return dict(rows=len(samples), cross_512_to_513_decline_count=len(declines),
        cross_512_to_513_decline_fraction=len(declines)/len(samples) if samples else None,
        cross_512_to_513_maximum_decline_s=max((r['decline_s'] for r in declines),default=0.),
        cross_512_to_513_worst_examples=sorted(declines,key=lambda r:r['decline_s'],reverse=True)[:5],
        legal_tier_transitions=transitions,negative_candidate_prediction_count=len(negatives),
        negative_candidate_examples=sorted(negatives,key=lambda r:r['prediction_s'])[:5])


def self_test():
    samples=[]
    coeff=[.002,-.003,.00004,.00005]
    for i in range(40):
        x=[(i*7)%17,(i*11)%19,(i*i+3*i)%23,(i*i*i+5)%29]
        samples.append(dict(x=x,y=.03+sum(a*b for a,b in zip(coeff,x))))
    model=fit_linear(samples,min_samples=5)
    assert max(abs(predict(model,r['x'])-r['y']) for r in samples)<1e-12
    # Any coefficient t on total can be absorbed in both P and D coefficients.
    t=.0017
    for row in samples:
        p,d,cd,cp=row['x']
        original=.03+t*(p+d)+(coeff[0]-t)*p+(coeff[1]-t)*d+coeff[2]*cd+coeff[3]*cp
        assert abs(original-predict(model,row['x']))<1e-12
    singular=[dict(x=[i,i,2*i,3*i],y=float(i)) for i in range(20)]
    try:
        fit_linear(singular,min_samples=5)
    except RankDeficientError:
        pass
    else:
        raise AssertionError('Singular design unexpectedly fitted')
    assert regime([512,0,0,0])=='total_le_512' and regime([513,0,0,0])=='total_gt_512'
    print('PASS: synthetic exact fit; total=P+D prediction equivalence; singular rank error retained; 512/513 boundary.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'calibration.json')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    result=dict(status='DIAGNOSTIC_NOT_ONLINE_VALIDATED',
        artifact=dict(name='Niyama two-regime linear cost predictor',
            commit='72339db5b120614033b317880ad3a05a5e4c9415', margin=MARGIN,
            regimes='total <=512 and total >512', filter='Exclude only completed steps with y >1s from artifact fit and filtered metrics',
            boundary_adaptation='Pinned code literally uses >513 for upper bin; intended 513+ is implemented as >512 here. Count exact-total-513 rows explicitly.',
            min_rows_per_regime=100),
        estimator=dict(method='Centered/scaled column-pivoted Householder QR, rank errors retained, no regularization/clipping',
            identifiable_features=list(FEATURES), intercept=True,
            collinearity='Artifact total=P+D. Setting total coefficient to zero and adding its coefficient to both P and D preserves every prediction; these four columns span the same linear predictor space.',
            rank_note='Intercept plus four features; rank below four after centering is unsupported.'),
        cost_label='Host engine.step completion: end_s-start_s, including scheduler/output handling; not GPU-only time or full-service effect',
        feature_definition=dict(P='Actual step prefill tokens',D='Actual step decode tokens',
            Cdecode='Sum(computed_start+1) over actual scheduled requests with decode_tokens>0',
            Cprefill='Sum(computed_start) over actual scheduled requests with prefill_tokens>0'),
        split_note='Five frozen complete training runs and their five reverse-order held-out runs; requests or steps within a run are not independent validation repeats.',
        training_runs=list(TRAIN),validation_runs=list(VALIDATE),input_runs=[],models={},fit_errors={})
    runs={}
    for name in TRAIN+VALIDATE:
        rows,info=load_run(name);runs[name]=rows;result['input_runs'].append(info)
    if not all(info['checks']['passed'] for info in result['input_runs']):
        result['status']='UNSUPPORTED_INPUT_VALIDATION_FAILED'
    else:
        training=[r for name in TRAIN for r in runs[name] if r['y']<=1.]
        grouped={key:[r for r in training if regime(r['x'])==key] for key in REGIMES}
        result['training_ranges']={key:describe(rows) for key,rows in grouped.items()}
        for key,rows in grouped.items():
            try:
                result['models'][key]=fit_linear(rows)
            except RankDeficientError as exc:
                result['fit_errors'][key]=dict(type=type(exc).__name__,message=str(exc))
        if result['fit_errors']:
            result['status']='UNSUPPORTED_RANK_OR_SAMPLE_COUNT'
        else:
            result['runtime_models']={runtime:dict(intercept_s=result['models'][reg]['intercept_s'],
                coefficients=[result['models'][reg]['coefficients_s_per_unit'][k] for k in FEATURES])
                for runtime,reg in [('le512','total_le_512'),('gt512','total_gt_512')]}
            result['monotonicity_diagnostic']=dict(
                scope='Hold actual validation-step D and executed-request contexts constant; substitute legal total tiers. Predictor-domain check only, not scheduler-state replay or measured action outcomes. Both tier lists checked without claiming either was active.',
                normal_total_tiers=TOTAL_TIERS, low_memory_total_tiers=LOW_MEMORY_TIERS,
                P_slopes_s_per_token={key:result['models'][key]['coefficients_s_per_unit']['P'] for key in REGIMES},
                per_held_out_run=[dict(run=name,**monotonicity(runs[name],result['models'])) for name in VALIDATE])
            result['per_run_metrics']=[]
            for name in TRAIN+VALIDATE:
                rows=runs[name]
                result['per_run_metrics'].append(dict(run=name,split='training' if name in TRAIN else 'held_out_run',
                    all_completed_steps=error_stats(rows,result['models'],result['training_ranges']),
                    artifact_filtered_steps=error_stats([r for r in rows if r['y']<=1.],result['models'],result['training_ranges']),
                    artifact_filtered_regimes={key:error_stats([r for r in rows if r['y']<=1. and regime(r['x'])==key],result['models'],result['training_ranges']) for key in REGIMES}))
            if any(r['all_completed_steps']['negative_prediction_count'] for r in result['per_run_metrics']):
                result['status']='DIAGNOSTIC_NEGATIVE_PREDICTIONS_NOT_READY'
            elif any(any(t['decline_count'] for t in r['legal_tier_transitions'].values())
                     for r in result['monotonicity_diagnostic']['per_held_out_run']):
                result['status']='DIAGNOSTIC_NONMONOTONIC_PREDICTOR_NOT_READY'
    result['interpretation']='Transfer diagnostic only. Error statistics neither prove online decision utility nor provide an SLO guarantee. No thresholds or safety margins searched.'
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],output=str(args.output),fit_errors=result['fit_errors'])))
    return 2 if result['status'].startswith('UNSUPPORTED') else 0


if __name__=='__main__':
    raise SystemExit(main())

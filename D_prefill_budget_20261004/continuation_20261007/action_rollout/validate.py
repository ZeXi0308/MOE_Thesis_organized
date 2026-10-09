"""Bounded historical direction check; no fitting, future-observation oracle or GPU.

Whole-run simulations see the exogenous workload for validation only. Separate
teacher-forced errors use actual past states to diagnose accumulated clock error;
they are never supplied to candidate simulations or an online selector.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from action_rollout.model import predict, simulate


def actual(path,models):
    raw = json.loads(path.read_text())
    groups = {}
    for s in raw['steps']:
        cd = s.get('component_decision',{}).get('decode_context_source',s['decode_context_sum']+s['decode_tokens'])
        cp = sum(q['computed_start'] for q in s['requests'] if q['prefill_tokens']>0)
        hat = predict(models,s['prefill_tokens'],s['decode_tokens'],cd,cp)
        cost = s['end_s']-s['start_s']
        group = 'mixed' if s['prefill_tokens'] and s['decode_tokens'] else 'prefill' if s['prefill_tokens'] else 'decode'
        groups.setdefault(group,[]).append(cost-hat)
    rows=raw['requests']
    qualified=[]
    for q in rows:
        ts=q['token_times_s']
        gap=max((b-a for a,b in zip(ts,ts[1:])),default=0.)
        if q['finished'] and ts[0]-q['arrival_s']<=4 and gap<=.1 and q['completion_s']-q['arrival_s']<=20:
            qualified.append(q['request_id'])
    mean = statistics.mean
    return dict(raw=str(path.relative_to(ROOT.parent)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        qualified_count=len(qualified),qualified_ids=sorted(qualified),elapsed_s=raw['elapsed_s'],
        conditional_goodput=len(qualified)/raw['elapsed_s'],
        mean_ttft_s=mean(q['token_times_s'][0]-q['arrival_s'] for q in rows),
        mean_generation_s=mean(q['completion_s']-q['token_times_s'][0] for q in rows),
        mean_flow_s=mean(q['completion_s']-q['arrival_s'] for q in rows),
        teacher_forced_clock_error=dict(
            groups={k:dict(steps=len(v),mean_actual_minus_prediction_s=mean(v),
                           mean_absolute_error_s=mean(abs(x) for x in v),sum_actual_minus_prediction_s=sum(v)) for k,v in groups.items()},
            sum_actual_minus_prediction_s=sum(sum(v) for v in groups.values()),
            loop_outside_engine_step_s=raw['elapsed_s']-sum(s['end_s']-s['start_s'] for s in raw['steps']),
            interpretation='Clock residual along the OBSERVED trajectory; not an endpoint saving, independent samples, or corrected counterfactual.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'validation.json')
    args=parser.parse_args()
    calibration=ROOT.parent/'niyama_component/calibration.json'
    models=json.loads(calibration.read_text())['runtime_models']
    workload=ROOT.parent/'mixed_arrivals/workload_high_rate.json'
    work=json.loads(workload.read_text())
    predictions={}
    for policy in ('fixed512','fixed1024','prefill_demand'):
        predictions[policy]=simulate(dict(now=0.,running=[],waiting=[],total_blocks=36752),models,policy,
                                    future_arrivals=work)['metrics']
    observations={}
    for name,rel in (
        ('half512a','fixedhalf53005_01/00_fixed512/raw.json'),
        ('half1024a','fixedhalf53005_01/01_fixed1024/raw.json'),
        ('half1024b','fixedhalf53005_01/02_fixed1024/raw.json'),
        ('half512b','fixedhalf53005_01/03_fixed512/raw.json'),
        ('demand1024a','demand53005_02/00_fixed1024/raw.json'),
        ('demanda','demand53005_02/01_prefill_demand/raw.json'),
        ('demandb','demand53005_02/02_prefill_demand/raw.json'),
        ('demand1024b','demand53005_02/03_fixed1024/raw.json')):
        observations[name]=actual(ROOT.parent/rel,models)
    comparisons=[]
    keys=('mean_ttft_s','mean_generation_s','mean_flow_s','qualified_count','elapsed_s','conditional_goodput')
    for policy, pairs in (
        ('fixed512',[('half512a','half1024a'),('half512b','half1024b')]),
        ('prefill_demand',[('demanda','demand1024a'),('demandb','demand1024b')])):
        pred={k:predictions[policy][k]-predictions['fixed1024'][k] for k in keys}
        actual_deltas=[{k:observations[a][k]-observations[b][k] for k in keys} for a,b in pairs]
        matches={k:all(math.copysign(1,pred[k])==math.copysign(1,d[k]) for d in actual_deltas) for k in keys}
        comparisons.append(dict(policy=policy,predicted_delta=pred,actual_paired_delta=actual_deltas,
                                direction_matches_both_pairs=matches))
    passed=all(all(c['direction_matches_both_pairs'].values()) for c in comparisons)
    out=dict(status='DIRECTION_EXPLANATION_SUPPORTED' if passed else 'MODEL_NOT_SUPPORTED_FOR_ACTION_SELECTION',
        model_scope='Approximate state transitions including new decoders/exits, known declared output budgets, fixed unscaled host cost, full drain. Not native replay/theoretical bound/service evidence.',
        future_arrivals='Exogenous trace used only in offline whole-policy validation. No measured future actions/costs/EOS/timestamps enter simulation. Online variant must supply no future arrivals.',
        parameters='No fitting or threshold search; inherited 4s/.1s/20s and demand margin1.2.',
        calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'model.py',Path(__file__))},
        predicted=predictions,observed=observations,comparisons=comparisons,
        action='Do not select/submit a GPU intervention from this model unless its explanation failure is resolved by specific implementation evidence, not outcome fitting.' if not passed else 'Only supports a bounded real action probe, not a controller contribution.')
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],comparisons=comparisons),indent=2))


if __name__=='__main__':
    main()

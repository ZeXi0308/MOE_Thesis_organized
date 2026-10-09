"""Single warm-only re-calibration check of the existing two affine regimes.

Historical formal outputs are evaluation only, not fit targets. The simulated
demand CONTROLLER keeps its original frozen predictor; only the execution-cost
environment changes. This is model development, not independent confirmation.
"""
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from action_rollout import model
from niyama_component.calibrate import load_run, fit_linear, regime, REGIMES, FEATURES, describe

WARMS=('demand53005_02/warm_fixed512','demand53005_02/warm_fixed1024','demand53005_02/warm_fixed2048')


def calibrated_simulator(runtime_models,busy_gap_s):
    """Reuse identical state transitions, with explicit cost/clock substitutions.

    Keeps model.py unchanged so the failed original result remains reproducible.
    """
    source=inspect.getsource(model.simulate)
    old='duration = predict(models,p,model_d,model_cd,cp)'
    new='duration = predict(runtime_models,p,model_d,model_cd,cp)'
    assert source.count(old)==1
    source=source.replace(old,new)
    old='        if not running and not waiting:\n'
    new='        if running or waiting:\n            now += busy_gap_s\n'+old
    assert source.count(old)==1
    source=source.replace(old,new)
    namespace=dict(model.__dict__,runtime_models=runtime_models,busy_gap_s=busy_gap_s)
    exec(compile(source,str(ROOT/'warm_validate.py')+'::warm_cost_simulate','exec'),namespace)
    return namespace['simulate'],source


def fit_warms():
    samples,provenance,gaps=[],[],[]
    for run in WARMS:
        rows,info=load_run(run)
        assert info['checks']['passed'],info
        samples.extend(r for r in rows if r['y']<=1.)
        provenance.append(info)
        raw=json.loads((ROOT.parent/run/'raw.json').read_text())
        for previous,current in zip(raw['steps'],raw['steps'][1:]):
            # A currently running decoder proves the engine did not become
            # empty between these two steps; excludes empty-engine arrival idle.
            if current['decode_count_before']>0:
                gap=current['start_s']-previous['end_s']
                assert gap>=0
                gaps.append(gap)
    fits={key:fit_linear([r for r in samples if regime(r['x'])==key]) for key in REGIMES}
    runtime={name:dict(intercept_s=fits[key]['intercept_s'],
                     coefficients=[fits[key]['coefficients_s_per_unit'][f] for f in FEATURES])
             for name,key in zip(('le512','gt512'),REGIMES)}
    result=dict(status='WARM_ONLY_FIXED_FORM_FIT',training_runs=list(WARMS),
        scope='Same two affine feature/regime families and QR as existing Niyama adaptation. No added feature, ridge, threshold/margin search, or formal target fit.',
        artifact_filter='Inherited duration>1s exclusion only; excluded steps retained in provenance.',
        warm_jit_caveat='The original log records fused_moe_kernel JIT during first warm_fixed512; no exact step/shape/duration alignment is available. No >1s step is present, so inherited filtering removes none. Do not claim JIT-free calibration or selectively drop step0.',
        input_runs=provenance,models=fits,runtime_models=runtime,
        training_ranges={key:describe([r for r in samples if regime(r['x'])==key]) for key in REGIMES},
        busy_host_gap=dict(samples=len(gaps),mean_s=statistics.mean(gaps),median_s=statistics.median(gaps),
                           max_s=max(gaps),rule='Current decode_count_before>0, consecutive engine starts/ends, no residual-based filter.'),
        interpretation='Ordinary local calibration, not a new predictor contribution. Warm workload is already seen development input.')
    return result


def main():
    fit=fit_warms()
    # Persist the fit before loading formal evaluation outcomes.
    fit_path=ROOT/'warm_calibration.json'
    fit_path.write_text(json.dumps(fit,indent=2)+'\n')
    old=json.loads((ROOT.parent/'niyama_component/calibration.json').read_text())['runtime_models']
    simulate,source=calibrated_simulator(fit['runtime_models'],fit['busy_host_gap']['mean_s'])
    work=json.loads((ROOT.parent/'mixed_arrivals/workload_high_rate.json').read_text())
    predictions={policy:simulate(dict(now=0.,running=[],waiting=[],total_blocks=36752),old,policy,
                                 future_arrivals=work)['metrics']
                 for policy in ('fixed512','fixed1024','prefill_demand')}
    previous=json.loads((ROOT/'validation.json').read_text())
    comparisons=[]
    for comparison in previous['comparisons']:
        policy=comparison['policy']
        pred={k:predictions[policy][k]-predictions['fixed1024'][k] for k in comparison['predicted_delta']}
        actual=comparison['actual_paired_delta']
        matches={k:all((pred[k]>0)==(x[k]>0) and (pred[k]<0)==(x[k]<0) for x in actual) for k in pred}
        comparisons.append(dict(policy=policy,predicted_delta=pred,actual_paired_delta=actual,
                                direction_matches_both_pairs=matches))
    passed=all(all(x['direction_matches_both_pairs'].values()) for x in comparisons)
    result=dict(status='DIRECTION_EXPLANATION_SUPPORTED_FOR_BOUNDED_PROBE' if passed else 'WARM_CALIBRATION_DIRECTION_NOT_SUPPORTED',
        parameters='Execution cost warm-refit; original demand controller model/1.2 margin and all SLOs fixed. No formal correction or policy tuning.',
        calibration_sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        derived_simulator_sha256=hashlib.sha256(source.encode()).hexdigest(),
        old_validation_sha256=hashlib.sha256((ROOT/'validation.json').read_bytes()).hexdigest(),
        predicted=predictions,comparisons=comparisons,
        interpretation='Seen-data explanatory development check, never a GPU service result or independent validation. Failed original validation is preserved.')
    (ROOT/'warm_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],runtime_models=fit['runtime_models'],busy_host_gap=fit['busy_host_gap'],comparisons=comparisons),indent=2))


if __name__=='__main__':
    main()

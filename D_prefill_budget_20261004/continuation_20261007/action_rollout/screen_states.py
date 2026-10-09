"""Three predeclared representative prefix states; no future arrivals in rollouts."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from action_rollout.history import iter_states
from action_rollout.warm_validate import calibrated_simulator


def main():
    path=ROOT.parent/'demand53005_02/00_fixed1024/raw.json'
    fit_path=ROOT/'warm_calibration.json'
    fit=json.loads(fit_path.read_text())
    old=json.loads((ROOT.parent/'niyama_component/calibration.json').read_text())['runtime_models']
    simulate,_=calibrated_simulator(fit['runtime_models'],fit['busy_host_gap']['mean_s'])
    picked={}
    for state in iter_states(json.loads(path.read_text())):
        prefills=[q for q in state['running']+state['waiting'] if q['prefill_done']<q['prompt']]
        if not prefills or prefills[0]['prompt']-prefills[0]['prefill_done']<=1024:continue
        for name,minimum in (('low',1),('middle',64),('high',128)):
            if name not in picked and state['decode_count']>=minimum:picked[name]=state
        if len(picked)==3:break
    output={}
    for name,state in picked.items():
        forecasts={}
        for cap in (512,1024,2048):
            try:
                result=simulate(state,old,candidate_cap=cap,hold_until_head=True,trace=True)
                forecasts[str(cap)]=dict(result['metrics'],head_completion_s=result['requests'][state['head_id']]['completion'],
                                          first_steps=result['steps'][:12])
            except ValueError as exc:forecasts[str(cap)]=dict(invalid=str(exc))
        output[name]=dict(state=state,predictions=forecasts)
    result=dict(status='THREE_REPRESENTATIVE_PREFIX_STATES_NO_FUTURE_ARRIVALS',
        selection='First head remaining>1024 states at D>=1/64/128 of F00, chosen before reading action predictions. Candidate held through original head first output, then fixed1024; currently known requests drain. No future arrivals, observed EOS or future measured costs.',
        scope='Seen-data diagnostic; conditional U does not include future requests and is not full service goodput or an online-controller result.',
        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),calibration_sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest(),states=output)
    (ROOT/'three_states.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,row in output.items():
        print(name,row['state']['step_index'],row['state']['decode_count'],
              {cap:{k:v[k] for k in ('qualified_count','elapsed_s','mean_flow_s')} if 'invalid' not in v else v for cap,v in row['predictions'].items()})


if __name__=='__main__':main()

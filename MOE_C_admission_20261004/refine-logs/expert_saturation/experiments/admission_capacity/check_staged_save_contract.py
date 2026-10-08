"""Targeted CPU qualification, including explicit unverified backend assumptions."""
import json
from dataclasses import replace
from pathlib import Path
from staged_save_contract import RequestState, StoreEvidence, prepare, commit_reason, recovery_guard

ROOT = Path(__file__).resolve().parents[2] / 'outputs/admission_capacity'

def main():
    src = ROOT / '20260914_recovery_progress_model_r01/input.json'
    state = json.loads(src.read_text())['state']['requests']
    def req(suffix):
        key = next(k for k in state if k.endswith(suffix))
        r = state[key]
        return RequestState(key,r['computed'],r['prompt'],r['output'],r['max_tokens'],r['status'],tuple(r['blocks']))
    victim,target = req('0000001'),req('0003640')
    raw = json.loads((ROOT/'20260914_d6_strong_baselines_r01/readback/results/block0-d6-native/raw.json').read_text())
    observed=[]
    for step in (329,330):
        b=raw['memory_trace'][step]['before']
        rows={name:next(v for k,v in b['requests'].items() if f'{name}-' in k) for name in ('0000001','0003640')}
        observed.append(dict(step=step,free=b['pool']['free_blocks'],requests=rows))
    for i,row in enumerate(observed):
        for name,r in [('0000001',victim),('0003640',target)]:
            q=row['requests'][name]; advance=i if name=='0000001' else 0
            assert (q['computed_tokens'],q['output_tokens'],q['block_counts'])==(r.computed+advance,r.output+advance,[len(r.blocks)])
    plan=prepare(329,victim,target,observed[0]['free'])
    # Raw next-state captures counts only: unchanged physical IDs and store job
    # below are fixtures, not observed backend evidence.
    nxt=replace(victim,computed=victim.computed+1,output=victim.output+1)
    store=StoreEvidence(victim.request_id,plan.saved_tokens,plan.source_blocks,(1,))
    def check(expected,**kw):
        args=dict(plan=plan,step=330,victim=nxt,target=target,free_blocks=145,store=store)
        args.update(kw); got=commit_reason(**args); assert got==expected,(expected,got)
        return got
    cases={
        'conditional_commit':check('READY'),
        'missing_store':check('CANCEL_STORE_NOT_REGISTERED',store=None),
        'matched_save_off':check('READY',store=None,save_enabled=False),
        'stale_step':check('CANCEL_STEP_CHANGED',step=331),
        'finished':check('CANCEL_REQUEST_FINISHED',victim=None),
        'victim_changed':check('CANCEL_VICTIM_CHANGED',victim=replace(nxt,status='PREEMPTED')),
        'target_changed':check('CANCEL_TARGET_CHANGED',target=replace(target,status='RUNNING')),
        'ownership_changed':check('CANCEL_BLOCK_OWNERSHIP_CHANGED',victim=replace(nxt,blocks=(99999,)+nxt.blocks[1:])),
    }
    large=replace(target,prompt=4096)
    p2=prepare(329,victim,large,100)
    assert commit_reason(p2,330,nxt,large,0,store)=='CANCEL_TARGET_UNFUNDED'
    cases['lost_funding']='CANCEL_TARGET_UNFUNDED'
    pending=replace(target,computed=3264,status='WAITING_FOR_REMOTE_KVS',blocks=tuple(range(204)))
    guard=recovery_guard(pending,1,1)
    assert guard['reserved_free_blocks']==1 and not guard['target_schedulable'] and not guard['other_growth_allowed']
    assert recovery_guard(pending,1,0)['other_growth_allowed']
    assert recovery_guard(pending,1,0,load_completion_visible=True)['target_schedulable']
    cases['async_reservation']='PASS'
    result=dict(status='CPU_CONDITIONAL_CONTRACT_PASS',cases=cases,observed=observed,saved_tokens=plan.saved_tokens,saved_blocks=len(plan.source_blocks),unverified=['next-step physical block ownership','native store registration for this victim','store completion fence before block reuse','in-loop integration and request performance'])
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

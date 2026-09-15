"""Independent-state repeated staging sensitivity; structural exploration only."""
import hashlib
import json
from pathlib import Path
import statistics
from absence_rotation import AbsenceRotation,RequestView
from recovery_progress_model import simulate

O=Path(__file__).resolve().parents[2]/'outputs/admission_capacity'
def main():
    out=O/'20260914_repeated_staged_model_r01';out.mkdir(exist_ok=False)
    inp=json.loads((O/'20260914_recovery_progress_model_r01/input.json').read_text())
    original=json.loads((O/'20260914_recovery_progress_model_r01/predictions.json').read_text())
    def run(action,delay=2):return simulate(inp['state'],inp['history'],action,AbsenceRotation,RequestView,restore_delay_steps=delay)
    assert all(run(a)==original[a] for a in ('least','most','defer'))
    cases={'immediate_most':run('continuous_most'),'staged_off':run('staged_most_off')}
    cases.update({f'staged_on_delay{d}':run('staged_most_on',d) for d in (1,2,4,8)})
    rows={}
    for name,r in cases.items():
        assert r['status']=='COMPLETE' and len(r['completed'])==32
        commits=[e for e in r.get('staging_events',[]) if e['event']=='commit']
        rows[name]=dict(total_calls=r['last_step']+1,mean_completion_step=statistics.mean(r['completed'].values()),
            committed_rotations=sum(bool(s['forced']) for s in r['trace']),cancellations=sum(e['event']=='cancel' for e in r.get('staging_events',[])),
            first_commit=commits[0] if commits else None,
            loaded_tokens=sum(q['tokens'] for s in r['trace'] for q in s.get('loads',[])),
            stored_tokens=r.get('stored_tokens',0),peak_logical_host_bytes=r.get('peak_host_tokens',0)*131072,
            completion_step_delta_vs_immediate={k:v-cases['immediate_most']['completed'][k] for k,v in r['completed'].items()},
            completion_step_delta_vs_staged_off={k:v-cases['staged_off']['completed'][k] for k,v in r['completed'].items()})
    summary=dict(status='CPU_EXPLORATORY_SENSITIVITY',rows=rows,legacy_regression=True,
        assumptions=['Fixed closed cohort lengths, no route/token identity or wall-time model','Save succeeds; complete-prefix host storage, no eviction or physical store cost modeled','Each action chooses using its own present state; one-step preparation and next-step cancellation checks','Load readiness delay1/2/4/8 steps is a sensitivity grid, not a calibrated range or oracle','No observed future completion events supplied to any candidate'],
        scope='No repeated-policy GPU execution. Differences in calls or completion steps are not throughput/latency predictions.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'model_source.py').write_text(Path(__file__).with_name('recovery_progress_model.py').read_text())
    (out/'input.json').write_text(json.dumps(inp))
    print(json.dumps({k:{n:v for n,v in row.items() if not n.startswith('completion_step_delta')} for k,row in rows.items()},indent=2))
if __name__=='__main__':main()

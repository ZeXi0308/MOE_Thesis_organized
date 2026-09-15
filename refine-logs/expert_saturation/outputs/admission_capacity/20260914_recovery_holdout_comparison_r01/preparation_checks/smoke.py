#!/usr/bin/env python3
"""CPU existing-data API smoke and canonical configuration fixture; no holdout run."""
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'experiments/admission_capacity'))
import analyze_recovery_holdout_comparison as h

source=HERE.parent/'preparation/pkg';specs=h.read(source/'campaign.json')['cells']
components=h.module('holdout_smoke_counters',source/'recovery_service_components.py')
metrics=h.module('holdout_smoke_metrics',source/'metrics.py')
selector=h.module('holdout_smoke_selector',source/'absence_rotation.py')
results=[]
for variant,path in [('native',ROOT/'outputs/admission_capacity/20260914_d6_strong_baselines_r01/readback/results/block0-d6-native'),
    ('most_output',ROOT/'outputs/admission_capacity/20260914_d6_strong_baselines_r01/readback/results/block0-d6-most_output'),
    ('guard_residual',ROOT/'outputs/admission_capacity/20260914_restore_token_reservation_r01/execution/readback/results/block0-guard_residual')]:
    spec=next(s for s in specs if s['variant']==variant)
    raw=h.read(path/'raw.json');decision_file='component-decisions.json' if spec['adapter']=='component' else 'headroom-decisions.json'
    decisions=h.read(path/decision_file)
    # Real old-cohort raw exercises reused interfaces, not cohort3 identity acceptance.
    h.base.work_accounting(raw,{r['request_id']:r for r in raw['requests']})
    checked=h.action_accounting(path,source,raw,decisions,spec,components,metrics)
    assert checked['action_eligible']
    results.append(dict(variant=variant,source_cell=str(path),steps=len(decisions),
        raw_sha256=h.sha(path/'raw.json'),totals=checked['component']['totals'],
        rotation_replay=checked.get('rotation_replay'),
        component_replayed_steps=checked.get('planner_replay',{}).get('replayed_steps')))
    del raw,decisions,checked
config=h.read(source/'inputs_preparation/prepared/long/config.json')
engine=h.read(ROOT/'outputs/admission_capacity/20260914_d6_strong_baselines_r01/readback/results/block0-d6-most_output/engine_args.json')
rows=[]
for spec in specs[:4]:
    cfg=dict(config,variant=spec['variant'],policy_family=spec['policy_family'],completion_policy=spec['completion_policy'],
        rotation_victim_order=spec['victim_order'],rotation_config=h.rotation.ROTATION,headroom_observer='fast',
        reservation_policy='full',fixed_kv_cache_memory_bytes=h.token.KV,preemption_mode='native_recompute')
    if spec['adapter']=='component':
        cfg['component']=dict(boost=True,threshold=200,quantum=10,base_order='FCFS',
            backend='priority packing / current-history reservation / native recompute',packing=spec['packing'],
            complete_restores=spec['complete_restores'],reserve_ready_tokens=spec['reserve_ready_tokens'])
    assert h.spec_config(cfg,spec,config)
    rows.append(dict(config=cfg,engine=engine))
assert all(h.pair_settings(rows[0],r) for r in rows[1:])
bad=deepcopy(rows[-1]['config']);bad['variant']='native'
assert not h.spec_config(bad,specs[3],config)
bad=deepcopy(rows[0]['config']);bad['component']=rows[-1]['config']['component']
assert not h.spec_config(bad,specs[0],config)
out=dict(status='PASS',scope='CPU only: three existing old-cohort actual traces plus constructed canonical configs; no cohort3 GPU data.',
    holdout_gpu_executions=0,existing_raw_checks=results,valid_configs=4,shared_settings_pass=True,
    wrong_variant_rejected=True,native_component_rejected=True,analyzer_sha256=h.sha(Path(h.__file__)))
path=HERE/'smoke.json';assert not path.exists(),'preserve smoke receipt'
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],cases=[dict(variant=r['variant'],steps=r['steps']) for r in results],analyzer_sha256=out['analyzer_sha256'])))

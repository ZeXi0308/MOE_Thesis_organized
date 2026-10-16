#!/usr/bin/env python3
"""Build one ten-arm frontier from a measured W2 pressure point; never submit."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(session, index, inputs, remote_inputs, destination, caps, mc_policy):
    receipt_path=session/'receipt.json'
    metrics_path=session/'metrics.json'
    receipt=json.loads(receipt_path.read_text())
    plan=receipt['plan']
    cell=plan['cells'][index]
    name=f'cell-{index:02d}-{cell["policy"]}'
    matches=[r for r in json.loads(metrics_path.read_text()) if Path(r['cell']).name==name]
    if len(matches)!=1:
        raise ValueError('One measured calibration cell required')
    measured=matches[0]
    if (cell['policy']!='native' or cell['cap']!=256 or not measured.get('complete')
            or not .05<=measured['preempted_request_fraction']<=.15):
        raise ValueError('Complete native cap256 calibration with 5–15% preemption required')
    source_path=session/name/'inputs.json'
    source=json.loads(source_path.read_text())
    target=json.loads(inputs.read_text())
    if source['config']['workload']!='W2':
        raise ValueError('Calibrate on W2, then transfer its rate to W1/W3')
    for field in ('model_path','model_revision','output_tokens','ignore_eos','min_tokens','sampling'):
        if source['config'].get(field)!=target['config'].get(field):
            raise ValueError('Input identity differs: '+field)
    if target['config']['workload'] not in ('W1','W2','W3'):
        raise ValueError('Unsupported workload')
    if (target['config']['workload']=='W2' and
            source['workload']['actual_prompt_token_ids']!=target['workload']['actual_prompt_token_ids']):
        raise ValueError('W2 prompts differ from calibration')
    if len(caps)!=3 or len(set(caps))!=3 or any(c<=0 or c>=256 for c in caps):
        raise ValueError('Three distinct static caps between 1 and 255 required')
    if not Path(destination).is_absolute() or not Path(remote_inputs).is_absolute():
        raise ValueError('Absolute remote session and input paths required')
    common=dict(kv_gib=cell['kv_gib'],arrival_rate=cell['arrival_rate'],max_seconds=240)
    cells=[dict(common,policy='native',cap=c) for c in [256,*sorted(caps)]]
    cells += [dict(common,policy=p,cap=256) for p in ('full-budget',mc_policy)]
    cells += [dict(common,policy='horizon',cap=256,ha=h,hr=32) for h in (16,64,256,1024)]
    result={k:plan[k] for k in ('runtime_overlay','library_path','hf_home','per_cell_seconds')}
    result.update(session=destination,inputs=remote_inputs,cells=cells,
        provenance=dict(calibration_session=str(session),calibration_cell=name,
            receipt_sha256=digest(receipt_path),metrics_sha256=digest(metrics_path),
            calibration_inputs_sha256=digest(source_path),inputs_sha256=digest(inputs),
            workload=target['config']['workload'],model_revision=target['config']['model_revision'],
            preempted_request_fraction=measured['preempted_request_fraction'],
            mc_policy=mc_policy,
            note='MC variants are component adaptations, not a full MC-SF reproduction.'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration-session',type=Path,required=True)
    p.add_argument('--calibration-cell',type=int,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--remote-inputs',required=True)
    p.add_argument('--session',required=True)
    p.add_argument('--caps',type=int,nargs=3,required=True)
    p.add_argument('--mc-policy',choices=('mc','mc-load'),default='mc-load')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=build(a.calibration_session,a.calibration_cell,a.inputs,a.remote_inputs,
                 a.session,a.caps,a.mc_policy)
    with a.output.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(output=str(a.output),cells=len(result['cells']),
        rate=result['cells'][0]['arrival_rate'],submitted=False)))

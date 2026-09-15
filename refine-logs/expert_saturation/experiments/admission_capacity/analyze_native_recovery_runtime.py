"""Reuse full-request accounting; reconcile the default-off native restore hook."""
import argparse
from bisect import bisect_right
import inspect
import json
from pathlib import Path
import sys

import analyze_funding_filter_comparison as base

read, sha, require = base.read, base.sha, base.require
VARIANTS = ('most_output', 'least_feasible', 'least_feasible_native_guard')
GPU = 'GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5'
LIMIT = 3.0
OLD_VALIDATE = base.validate_arm_contract


def validate(cfg, raw, decisions, variant):
    guarded = variant == VARIANTS[2]
    mapped = 'least_feasible' if guarded else variant
    OLD_VALIDATE(dict(cfg, variant=mapped), raw, decisions, mapped)
    require(cfg['variant'] == variant and cfg['protect_native_recovery'] is guarded,
            'native protection flag differs')
    require(guarded or not any('native_recovery_started' in d for d in decisions),
            'disabled arm registered a natural recovery')


def replay_factory(context, library, variant):
    filtered, guarded = variant != 'most_output', variant == VARIANTS[2]
    order = 'least_progress' if filtered else 'most_output'
    source = inspect.getsource(library.rotation.replay)
    source = '\n'.join(line for line in source.splitlines() if not any(x in line for x in
        ("q=d['pre_exchange_ownership']", "q['live_owned_blocks']", 'qualification_receipts+=1')))
    replace = lambda old, new: base.replace_once(source, old, new, old)
    source = replace("victim_order='most_output'", 'victim_order=order')
    source = replace("d['effective_victim_order']=='most_output'", "d['effective_victim_order']==order")
    source = replace("states[r]['prompt_tokens'],4096,states[r]['output_tokens']",
                     "states[r]['prompt_tokens'],states[r]['prompt_tokens']+1024,states[r]['output_tokens']")
    source = replace("{r:need(r) for r in waiting})",
                     "{r:need(r) for r in waiting},**({'released_blocks':{r:owned(r) for r in running}} if filtered else {}))")
    source = replace("protected in states and states[protected]['output_tokens']>output_start",
                     "bisect_right(requests[raw['internal_to_source'][protected]]['token_times_s'],s['start_s'])>output_start")
    source = replace('    events={(e[', "    requests={r['request_id']:r for r in raw['requests']}\n    events={(e[")
    source = replace("        require(d['recovery_target']==protected, 'recovery target changed')", '''        candidates=[rid for rid,n in d['actual_scheduled'].items() if rid not in running
            and states[rid]['num_preemptions']>0 and states[rid]['output_tokens']>0
            and states[rid]['prompt_tokens']+states[rid]['output_tokens']-states[rid]['computed_tokens']>1 and n>0]
        expected_start=guarded and protected is None and bool(candidates)
        event=d.get('native_recovery_started')
        require(bool(event)==expected_start, 'native protection start eligibility differs')
        if event:
            target=candidates[0]; v=states[target]; post=after['requests'][target]
            remaining=max(0,(v['prompt_tokens']+v['output_tokens']+15)//16-post['block_counts'][0])
            require(event==dict(request_id=target,output_tokens=v['output_tokens'],
                pending_tokens=v['prompt_tokens']+v['output_tokens']-v['computed_tokens'],
                scheduled_tokens=d['actual_scheduled'][target],remaining_blocks=remaining),
                'native restore event differs from actual allocation')
            require(after['pool']['free_blocks']>=remaining, 'native restore underfunded')
            protected,output_start=target,v['output_tokens']
        require(d['recovery_target']==protected, 'recovery target changed')''')
    source = replace("    require(forced>0, 'INVALID_NO_ACTION')", '    # Complete no-action cells remain evidence of applicability.')
    ns = dict(library.rotation.replay.__globals__, order=order, filtered=filtered,
              guarded=guarded, check_release=library.old_release, bisect_right=bisect_right)
    exec(compile(source, __file__+':native-recovery-replay', 'exec'), ns)
    return ns['replay']


def recovery_flows(raw, decisions):
    calls={k:c for c in raw['engine_steps'] for k in range(c['scheduler_step_start'],c['scheduler_step_end'])}
    requests={r['request_id']:r for r in raw['requests']}; aliases=raw['internal_to_source']; flows=[]
    for d in decisions:
        if not (event:=d.get('native_recovery_started')): continue
        rid=event['request_id']; request=requests[aliases[rid]]; k=d['step']; count=event['output_tokens']
        require(len(request['token_times_s'])>count, 'native protected recovery never returned new output')
        returned=request['token_times_s'][count]
        first=[i for i,c in calls.items() if i>=k and c['returned_s']==returned]
        require(first, 'first new output has no engine return')
        last=min(first); held={}
        for decision in decisions[k:last+1]:
            for peer in decision['held']: held[aliases[peer]]=held.get(aliases[peer],0)+1
        peers={r['request_id']:sum(calls[k]['start_s']<t<=returned for t in r['token_times_s'])
               for r in raw['requests'] if r['request_id']!=aliases[rid]}
        release=[x['step'] for x in decisions[last+1:] if x.get('recovery_completed')==rid]
        flows.append(dict(request_id=aliases[rid],start_step=k,first_new_output_step=last,
            observed_release_step=release[0] if release else None,
            initial_outputs=count,pending_tokens=event['pending_tokens'],
            time_to_new_output_s=returned-calls[k]['start_s'],
            gap_from_previous_output_s=returned-request['token_times_s'][count-1],
            peer_new_outputs=sum(peers.values()),peer_outputs=peers,held_peer_calls=held))
    return flows


def primary_pair(pair, a, b):
    left,right=a['primary_eligible'],b['primary_eligible']
    decision=('NO_FEASIBLE_WINNER' if not left and not right else
              'FEASIBILITY_ONLY' if left!=right else 'COMPARE_MEAN_COMPLETION')
    pair.update(primary=dict(max_started_itl_limit_s=LIMIT,baseline_eligible=left,action_eligible=right,
        decision=decision,mean_completion_delta_pct=pair['mean_completion_delta_pct'] if left and right else None,
        inference='Observed pair only; reversed order blocks are not a confidence bound.'),
        request_completion_better=sum(r['completion_s']<0 for r in pair['per_request_deltas']),
        request_completion_worse=sum(r['completion_s']>0 for r in pair['per_request_deltas']))
    return pair


def analyze(bundle, context_path, library_path):
    context,library=base.import_dependencies(context_path,library_path)
    prep=bundle/'gpu_preparation'; source=prep/'pkg'; meta=read(prep/'preparation.json')
    require(all(sha(source/n)==h for n,h in meta['files_sha256'].items()), 'prepared source differs')
    back=bundle/'execution/readback/pkg'
    require(not back.exists() or all(sha(back/n)==h for n,h in meta['files_sha256'].items()), 'readback source differs')
    campaign=read(source/'campaign.json'); specs=campaign['cells']
    require(specs==meta['cells'] and [(r['block'],r['variant']) for r in specs]==
        [(b,v) for b,vs in ((0,VARIANTS),(1,VARIANTS[::-1])) for v in vs], 'campaign differs')
    require(campaign['primary_max_started_request_itl_s']==meta['primary_max_started_request_itl_s']==LIMIT,
            'primary requirement differs')
    inputs=read(source/'inputs_preparation/prepared/heterogeneous/workload.json')
    cfg=read(source/'inputs_preparation/prepared/heterogeneous/config.json')
    require(base.hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()==cfg['workload_sha256'], 'input differs')
    metrics=library.module('native_recovery_metrics',source/'metrics.py')
    selector=library.module('native_recovery_selector',source/'absence_rotation.py')
    base.validate_arm_contract,base.replay_factory=validate,replay_factory
    inspector=base.make_inspector(context); cells=[]
    for spec in specs:
        row=inspector(bundle,spec,library,meta,source,inputs,cfg,metrics,selector)
        folder=bundle/'execution/readback/results'/spec['label']
        row['primary_eligible']=False
        if (folder/'raw.json').exists():
            raw=read(folder/'raw.json')
            row.update(raw_sha256=sha(folder/'raw.json'),raw_path=str(folder/'raw.json'),
                retained_requests=[dict(id=r['request_id'],status=r['status'],outputs=len(r['output_token_ids'])) for r in raw['requests']],
                terminal_raw_status=raw['status'],terminal_error=raw['error'])
        if row['eligible']:
            try:
                env=read(folder/'environment.json')
                require(env['gpu_before']['selected_gpu_uuid']==env['gpu_before']['cuda_visible_devices']==row['gpu_uuid']==GPU,
                        'selected GPU differs')
                decisions=read(folder/'headroom-decisions.json')
                row.update(native_recovery_flows=recovery_flows(raw,decisions),
                    primary_eligible=row['max_itl_s']<=LIMIT,
                    primary_max_started_itl_s=LIMIT,config=read(folder/'config.json'))
            except (ValueError,KeyError,TypeError) as exc:
                row.update(status='INVALID_OR_INCOMPLETE',eligible=False,error=str(exc))
        cells.append(row)
    comparisons=[]
    if all(r['eligible'] for r in cells):
        for block in (0,1):
            rows={r['variant']:r for r in cells if r['block']==block}
            trim=lambda r:{k:v for k,v in r['config'].items() if k not in ('variant','protect_native_recovery')}
            require(trim(rows[VARIANTS[1]])==trim(rows[VARIANTS[2]]), 'matched pair differs beyond hook flag')
            for a,b in ((VARIANTS[1],VARIANTS[2]),(VARIANTS[0],VARIANTS[2]),(VARIANTS[0],VARIANTS[1])):
                comparisons.append(primary_pair(context.compare(rows[a],rows[b]),rows[a],rows[b]))
        for variant in VARIANTS:
            a,b=[r for r in cells if r['variant']==variant]
            comparisons.append(primary_pair(context.compare(a,b),a,b))
    for row in cells: row.pop('outputs',None)
    return dict(status='MEASUREMENT_ONLY' if comparisons else ('UNRUN' if all(r['status']=='UNRUN' for r in cells) else 'INCOMPLETE'),
        scope='Same reused workload; six actual independent runtime states. Fixed 3s pause requirement then all-arrival mean completion. '
              'Full failures/no-action cells retained; no significance, quality, optimality, or new-method claim.',
        analyzer_sha256=sha(Path(__file__)),preparation_sha256=sha(prep/'preparation.json'),
        reused_analysis={str(Path(m.__file__)):sha(Path(m.__file__)) for m in
                         (base,context,library,library.rotation,library.token,library.base,library.headroom)},
        cells=cells,comparisons=comparisons)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--analysis-library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); require(not a.output.exists(),'preserve previous analysis')
    result=analyze(a.bundle,Path(__file__).with_name('analyze_context_victim_calibration.py'),a.analysis_library)
    base.write_exclusive(a.output,result)
    print(json.dumps(dict(status=result['status'],cells=[{k:r.get(k) for k in ('label','status','error','primary_eligible')}
                         for r in result['cells']],comparisons=len(result['comparisons']))))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()

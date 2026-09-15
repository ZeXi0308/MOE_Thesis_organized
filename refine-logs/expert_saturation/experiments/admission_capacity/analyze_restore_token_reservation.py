#!/usr/bin/env python3
"""Six real runs: simple packing, full restore priority, and residual restore tokens."""
import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import inspect as python_inspect
import json
from pathlib import Path
import analyze_restore_completion_probe as prior

base, read, require, sha, digest, module = prior.base, prior.read, prior.require, prior.sha, prior.digest, prior.module
KV, BLOCKS = prior.KV, prior.BLOCKS
DEFAULT = Path(__file__).resolve().parents[2]/'outputs/admission_capacity/20260914_restore_token_reservation_r01'
VARIANTS = ['fit_scan', 'guard_all', 'guard_residual', 'guard_residual', 'guard_all', 'fit_scan']
LABELS = [f'block{i//3}-{v}' for i,v in enumerate(VARIANTS)]
FLAGS = {'fit_scan': ('fit_scan','off','off'), 'guard_all': ('rank_prefix','on','off'),
         'guard_residual': ('rank_prefix','on','on')}
PRIOR_SHA = 'd536a7b97d50eca3e538b7861d31a408af2cb1b8cf6f5c584df3f15266809913'
SCOPE = ('Same d6/APC-off/6656-block/custom recompute backend, original LTR200/10 and first-output helper. '
    'Every raw run independently checks identities, clocks, ownership, actual release, past-state priorities, '
    'the full frozen planner including reserve_ready_tokens, and all nested costs. One-step flag-off plans '
    'on observed prestates identify executed budget changes only; they are not alternate future trajectories '
    'or latency Oracles. Zero/one/two-output residencies end at actual next preemption. '
    'No full-LTR, SLO, Oracle, quality, significance or method-GO claim.')


def spec_config(cfg, spec, config):
    return (all(cfg[k] == v for k,v in config.items())
        and cfg['variant'] == spec['variant']
        and cfg['completion_policy'] == 'restore_token_reservation_'+spec['variant']
        and cfg['component'] == dict(boost=True, threshold=200, quantum=10, base_order='FCFS',
            backend='priority packing / current-history reservation / native recompute',
            packing=spec['packing'], complete_restores=spec['complete_restores']=='on',
            reserve_ready_tokens=spec['reserve_ready_tokens']=='on'))


def replay(folder, source, raw, decisions, spec):
    path = folder/'resolved-scheduler-config.json'
    require('resolved-scheduler-config.json' in (source/'run_probe.py').read_text(), 'resolved config lacks provenance')
    chunk = read(path)['long_prefill_token_threshold']
    require(type(chunk) is int and chunk >= 0, 'invalid resolved chunk threshold')
    tree = ast.parse((source/'ltr_recompute_native.py').read_text())
    ns = dict(dataclass=dataclass, __name__=__name__)
    nodes = [n for n in tree.body if getattr(n,'name',None) in ('Candidate','Plan','plan')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source/'ltr_recompute_native.py'),'exec'),ns)
    requests = {q['request_id']:q for q in raw['requests']}; aliases = raw['internal_to_source']
    flag = spec['reserve_ready_tokens']=='on'; totals = Counter(); actions = []
    for d,m in zip(decisions,raw['memory_trace']):
        before = m['before']; live = before['requests']; running = before['running_ids']
        require(d['reserve_ready_tokens'] is flag, 'executed token reservation flag differs')
        rows = [ns['Candidate'](rid,d['priorities'][rid],requests[aliases[rid]]['arrival_s'],rid in running,
            v['block_counts'][0],(v['prompt_tokens']+v['output_tokens']+15)//16,
            v['prompt_tokens']+v['output_tokens']-v['computed_tokens']) for rid,v in live.items()]
        plan = ns['plan'](rows,before['pool']['free_blocks'],1024,32,chunk,
                         packing=spec['packing'],reserve_ready_tokens=flag)
        require(plan.tokens == d['tokens'] == d['actual_scheduled'] and plan.victims == d['victims']
                and plan.free_after_reservation == d['free_after_reservation'], 'exact token-reservation plan differs')
        ordered = sorted(rows,key=lambda r:(r.priority,r.arrival,r.request_id))
        protected = {r.request_id for r in ordered if r.priority == -2 and r.request_id in plan.tokens}
        later_ready = {r.request_id for i,r in enumerate(ordered) if r.resident and r.pending_tokens == 1
                       and r.request_id not in plan.victims
                       and any(p.request_id in protected for p in ordered[:i])}
        totals.update(protected_selected_calls=len(protected), protected_scheduled_tokens=sum(plan.tokens[r] for r in protected),
            later_ready_candidates=len(later_ready), later_ready_selected_calls=len(later_ready & plan.tokens.keys()))
        if flag:
            off = ns['plan'](rows,before['pool']['free_blocks'],1024,32,chunk,
                            packing=spec['packing'],reserve_ready_tokens=False)
            changed = plan.tokens != off.tokens or plan.victims != off.victims
            effective = changed and bool(protected)
            totals.update(changed_steps=int(changed), protected_changed_steps=int(effective))
            if changed:
                actions.append(dict(step=d['step'], actual_tokens={aliases[r]:n for r,n in plan.tokens.items()},
                    actual_victims=[aliases[r] for r in plan.victims],
                    same_prestate_flag_off_tokens={aliases[r]:n for r,n in off.tokens.items()},
                    same_prestate_flag_off_victims=[aliases[r] for r in off.victims],
                    actual_protected_requests=[aliases[r] for r in sorted(protected)],
                    actual_later_ready_tokens_scheduled=len(later_ready & plan.tokens.keys()),
                    actual_free_reservation_slack=plan.free_after_reservation))
    return dict(status='EXACT_FROZEN_PLAN_REPLAY', replayed_steps=len(decisions),
        reserve_ready_tokens=flag, resolved_chunk_threshold=chunk, receipt_sha256=sha(path),
        token_budget_action_counts=dict(totals), changed_actual_steps=actions,
        boundary='Flag-off replay stays at the same recorded prestate and never supplies future outcomes.')


def replace_once(text, old, new):
    require(text.count(old)==1, 'audited source replacement no longer unique: '+old[:80])
    return text.replace(old,new)


def make_inspector():
    require(sha(Path(prior.__file__))==PRIOR_SHA, 'reused restore analyzer changed')
    source = python_inspect.getsource(prior.inspect)
    start = source.index('        require(all(cfg[k] == v for k,v in config.items())')
    end = source.index("        require(q['status']",start)
    old = source[start:end]
    require("packing='rank_prefix'" in old and "'workload/component configuration differs'" in old, 'config source boundary differs')
    source = replace_once(source,old,"        require(spec_config(cfg,spec,config) and cfg['cap'] == raw['target_cap'] == 32, 'workload/component configuration differs')\n")
    source = replace_once(source,"policy='restore_completion_'+spec['complete_restores']","policy='restore_token_reservation_'+spec['variant']")
    source = replace_once(source,"d['packing']=='rank_prefix' and d['boost'] is True and d['complete_restores'] is enabled",
        "d['packing']==spec['packing'] and d['boost'] is True and d['complete_restores'] is enabled and d['reserve_ready_tokens'] is (spec['reserve_ready_tokens']=='on')")
    source = replace_once(source,"packing.replay(folder,source,raw,decisions,'rank_prefix')","replay(folder,source,raw,decisions,spec)")
    ns = dict(prior.inspect.__globals__,spec_config=spec_config,replay=replay)
    exec(compile(source,__file__+':audited-inspect','exec'),ns)
    return ns['inspect']


inspect_cell = make_inspector()


def pair_settings(a,b):
    # Per-cell exact flags were already checked. Exclude only the enumerated arm fields
    # for this pair's shared-input check; preserve the actual configs in all output rows.
    common = lambda row:{k:v for k,v in row['config'].items() if k not in ('completion_policy','component','variant')}
    components = lambda row:{k:v for k,v in row['config']['component'].items()
                            if k not in ('packing','complete_restores','reserve_ready_tokens')}
    return a['engine']==b['engine'] and common(a)==common(b) and components(a)==components(b)


def compare(a,b,kind):
    source = python_inspect.getsource(base.compare)
    line = next(line for line in source.splitlines() if line.strip().startswith('same = all('))
    source = replace_once(source,line,'    same = pair_settings(a,b)')
    source = replace_once(source,"'pair differs beyond APC'","'pair shared settings/runtime differ'")
    ns = dict(base.compare.__globals__,pair_settings=pair_settings)
    exec(compile(source,__file__+':fixed-metrics-compare','exec'),ns)
    pair = ns['compare'](a,b,kind)
    left={q['request_id']:q for q in a['per_request']}; right={q['request_id']:q for q in b['per_request']}
    for change in pair['per_request_changes']:
        rid=change['request_id']; change['mean_tpot_s']=right[rid]['mean_tpot_s']-left[rid]['mean_tpot_s']
    pair.update(baseline_policy=a['policy'],action_policy=b['policy'],
        baseline_component=a['config']['component'],action_component=b['config']['component'],
        schedule_path_equal=a['component']['schedule_path_sha256']==b['component']['schedule_path_sha256'])
    pair['per_request_change_counts']={metric:dict(improved=sum(q[metric]<0 for q in pair['per_request_changes']),
        harmed=sum(q[metric]>0 for q in pair['per_request_changes']),equal=sum(q[metric]==0 for q in pair['per_request_changes']))
        for metric in ('ttft_s','completion_s','max_itl_s','mean_tpot_s')}
    return pair


def finish(result,args):
    for row in result['cells']: row.pop('_outputs',None)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'REPORT.md').write_text('# Restore token reservation\n\n'+result['status']+'\n\n'+SCOPE+'\n\n'+
        '\n'.join(f"- {r['label']}: {r['status']}" for r in result['cells'])+'\n\n'+result.get('error',
        'All actual flags, independent ledger/resource checks, token-budget changes, zero/tiny-service residencies, nested costs, output fingerprints and request changes remain in analysis.json.')+'\n')
    print(json.dumps(dict(status=result['status'],preparation_checks=result.get('preparation_checks'),error=result.get('error'),
        cells=[dict(label=r['label'],status=r['status'],error=r.get('error')) for r in result['cells']])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,default=DEFAULT)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--results-dir',type=Path)
    p.add_argument('--source-dir',type=Path);p.add_argument('--preparation-dir',type=Path)
    p.add_argument('--expected-metadata-sha256',required=True);args=p.parse_args()
    require(not args.output_dir.exists(),'output exists; refuse overwrite')
    bundle=args.run_dir.parent if args.run_dir.name.startswith('execution') else args.run_dir
    run=args.run_dir if args.run_dir.name.startswith('execution') else bundle/'execution'
    prep=args.preparation_dir or bundle/'preparation'
    source=args.source_dir or (run/'readback/pkg' if (run/'readback/pkg').is_dir() else prep/'pkg')
    results=args.results_dir or run/'readback/results';meta_path=prep/'preparation.json'
    result=dict(status='UNRUN',cells=[dict(label=n,policy='restore_token_reservation_'+v,status='UNRUN',eligible=False)
        for n,v in zip(LABELS,VARIANTS)],comparisons=[],scope=SCOPE,source=str(source),results_dir=str(results),
        reused_restore_accounting_sha256=PRIOR_SHA,reused_base_sha256=sha(Path(base.__file__)))
    try:
        if not meta_path.exists():
            require(not results.exists() or not any(results.iterdir()),'retained attempt lacks metadata')
            result['preparation_checks']='WAITING_FOR_FROZEN_PACKAGE';finish(result,args);return
        require(sha(meta_path)==args.expected_metadata_sha256,'frozen metadata changed');meta=read(meta_path)
        for package in {source,prep/'pkg'}:
            require(all(sha(package/n)==h for n,h in meta['files_sha256'].items()),'frozen source differs')
            sums=dict((n.lstrip('*'),h) for h,n in (line.split(maxsplit=1) for line in (package/'SHA256SUMS').read_text().splitlines()))
            require(sums==meta['files_sha256'],'source inventory differs')
        campaign=read(source/'campaign.json'); cells=campaign['cells']
        expected=[dict(label=label,variant=v,boost='on',packing=FLAGS[v][0],complete_restores=FLAGS[v][1],
            reserve_ready_tokens=FLAGS[v][2],usable_blocks=BLOCKS,kv_cache_bytes=KV) for label,v in zip(LABELS,VARIANTS)]
        require(campaign['comparison']==meta['comparison']=='restore_token_reservation'
            and cells==meta['cells']==expected,'frozen arms differ')
        parent=DEFAULT.parent/'20260914_restore_completion_r01/preparation'
        require(meta['parent_archive_sha256']==read(parent/'preparation.json')['archive_sha256'],'parent package differs')
        require(all(sha(source/n)==sha(parent/'pkg'/n) for n in ('restore_obligation.py','recovery_service_components.py')),
                'first-output helper or original 200/10 counters changed')
        inp=source/'inputs_preparation/prepared/long';workload=read(inp/'workload.json');config=read(inp/'config.json')
        require(hashlib.sha256(json.dumps(workload,sort_keys=True).encode()).hexdigest()==config['workload_sha256']
            and len(workload['source_requests'])==len(workload['actual_prompt_token_ids'])==len(workload['arrival_traces_s']['steady'])==32,'frozen inputs differ')
        ref=Path(meta['reference_cell']);require(ref.is_absolute() and sha(ref/'engine_args.json')==meta['reference_engine_args_sha256'],'reference engine differs')
        reference=dict(engine=read(ref/'engine_args.json'),runtime=read(ref/'environment.json')['vllm_source_sha256'])
        require(len(reference['runtime'])==7 and all(reference['runtime'][n]==h for n,h in meta['expected_runtime_sources'].items()),'pinned runtime differs')
        metrics=module('token_reservation_metrics',source/'metrics.py');components=module('token_reservation_counters',source/'recovery_service_components.py')
        rows=[inspect_cell(results,c,source,meta,workload,config,metrics,components,reference) for c in cells]
        for row,c in zip(rows,cells):
            if row['eligible'] and c['reserve_ready_tokens']=='on' and not row['planner_replay']['token_budget_action_counts'].get('protected_changed_steps',0):
                row.update(status='INVALID_NO_ACTION',eligible=False,error='no actual changed protected token allocation')
        result.update(cells=rows,status='UNRUN' if all(r['status']=='UNRUN' for r in rows) else 'INCOMPLETE',
            preparation_checks='PASS',metadata_sha256=args.expected_metadata_sha256)
        if all(r['eligible'] for r in rows):
            pairs=[(1,2,'block0_residual_vs_guard_all'),(0,2,'block0_residual_vs_fit_scan'),
                   (4,3,'block1_residual_vs_guard_all'),(5,3,'block1_residual_vs_fit_scan'),
                   (0,5,'fit_scan_repeat'),(1,4,'guard_all_repeat'),(2,3,'guard_residual_repeat')]
            result.update(status='MEASUREMENT_ONLY',comparisons=[compare(rows[i],rows[j],kind) for i,j,kind in pairs])
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,RuntimeError) as error:
        result.update(status='INCOMPLETE',error=str(error),comparisons=[])
    finish(result,args)


if __name__=='__main__': main()

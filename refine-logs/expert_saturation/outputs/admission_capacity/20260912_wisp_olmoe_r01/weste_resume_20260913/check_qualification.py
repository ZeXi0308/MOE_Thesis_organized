"""Read-only qualification of two new-host engines; no performance/quality claim."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HELPERS = (Path(os.environ['MOE_QUALIFICATION_HELPERS']) if 'MOE_QUALIFICATION_HELPERS' in os.environ
           else Path(__file__).resolve().parents[4] / 'experiments/admission_capacity')
sys.path.insert(0, str(HELPERS))
from analyze_native_pager_transfer import requests, call_accounting

LAYERS = [f'model.layers.{i}.mlp.experts' for i in range(16)]
MODES = {'fullstage20': ('fullstage', 20), 'oneshot21': ('oneshot', 21)}
ERRORS = (OSError, KeyError, ValueError, TypeError, IndexError, AssertionError)


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def check(ok, message, issues):
    if not ok: issues.append(message)


def check_calls(records, mode, cap, sizes, raw, validation):
    """Start every layer empty; follow all initialization/warmup/measurement calls."""
    from analyze_layer_budget import LRU
    from full_stage_plan import plan_full_stage
    from shared_pool_plan import plan_shared_pool
    planner = plan_full_stage if mode == 'fullstage' else plan_shared_pool
    key = 'full_stage_plan' if mode == 'fullstage' else 'shared_plan'
    states = {name: LRU(cap) for name in LAYERS}; issues = []; measured = []; steps = Counter()
    counts = Counter(); phase_counts = Counter()
    for i, r in enumerate(records):
        name = r['layer_name']; state = states[name]; active = set(r['active_experts']); groups = r['groups']; tag = f'call {i}'
        check(r['call_id'] == i and r['status'] == 'complete' and r['validation_run'] and not r['measurement'], tag+': status/qualification flag', issues)
        check(r['execution'] == 'shared_pool_'+mode and len(r['row_topk_experts']) == r['rows'] and
              {e for row in r['row_topk_experts'] for e in row} == active, tag+': execution/active rows', issues)
        check(r['entry_resident_experts'] == sorted(state.mapping), tag+': sequential entry', issues)
        plan = planner(active, state.snapshot(), int(name.split('.')[2]))
        check(r.get(key) == plan and not (key == 'full_stage_plan' and 'shared_plan' in r) and
              not (key == 'shared_plan' and 'full_stage_plan' in r), tag+': exact current-state plan', issues)
        stage = plan['hit_stage'] if mode == 'fullstage' else plan['d2d']; writeback = plan['writeback'] if mode == 'fullstage' else []
        check(len(groups) == r['group_count'] == int(bool(active)), tag+': actual groups', issues)
        for g in groups:
            check(g['required_experts'] == g['ensure_experts'] == sorted(active) and not g['protected_after'] and
                  (g['start'], g['stop']) == (0, r['rows']), tag+': kernel coverage', issues)
            check(g['loaded_experts'] == [c['expert'] for c in plan['h2d']] and
                  g['d2d_experts'] == [c['expert'] for c in stage+writeback] and
                  g['evicted_experts'] == plan['entry_evicted_experts'] and g['evict'] == plan['entry_evict'], tag+': copy order/entry evictions', issues)
            check(g['d2d_copy_bytes'] == len(stage+writeback)*sizes[name], tag+': group D2D bytes', issues)
        check(r['miss'] == plan['canonical_miss'] and r['evict'] == plan['entry_evict'] and
              r['canonical_group_count'] == plan['canonical_group_count'] and r['canonical_evict'] == plan['canonical_evict'] and
              r['d2d_copy_bytes'] == len(stage+writeback)*sizes[name], tag+': call accounting', issues)
        for group in plan['canonical_groups']: state.ensure(group)
        check(state.snapshot() == plan['final_state'] and sorted(state.mapping) == r['final_resident_experts'], tag+': final metadata', issues)
        counts.update(calls=1, h2d_bytes=r['weight_copy_bytes'], stage_d2d_bytes=len(stage)*sizes[name],
                      writeback_d2d_bytes=len(writeback)*sizes[name], groups=len(groups), canonical_groups=plan['canonical_group_count'])
        phase_counts[r['context']['phase']] += 1
        if r['context']['phase'] != 'measurement': continue
        measured.append(r); ctx = r['context']; step = raw['scheduler_steps'][ctx['step_id']]; steps[(step['step'], name)] += 1
        expected = Counter((s['internal_request_id'], pos, s['prompt_tokens']) for s in step['scheduled']
                           for pos in range(s['scheduled_start_computed'], s['scheduled_start_computed']+s['scheduled_tokens']))
        check(ctx['row_request_order_verified'] and len(ctx['rows']) == r['rows'] and
              Counter((x['internal_request_id'], x['computed_position'], x['prompt_tokens']) for x in ctx['rows']) == expected,
              tag+': request/position/topk row join', issues)
    check(steps == Counter({(s['step'], n): 1 for s in raw['scheduler_steps'] if s['total_scheduled_tokens'] for n in LAYERS}), 'measurement step/layer coverage', issues)
    check(all(c['byte_count_valid'] for c in call_accounting(records, sizes)), 'actual H2D bytes', issues)
    check(Counter(v['call_id'] for v in validation) == Counter(r['call_id'] for r in measured), 'validation coverage/duplicates', issues)
    by_id = {r['call_id']: r for r in measured}
    for v in validation:
        r = by_id[v['call_id']]
        check(v['mode'] == mode and v['layer_name'] == r['layer_name'] and v['context'] == r['context'] and
              v['rows'] == r['rows'] and v['actual_group_ranges'] == [[g['start'], g['stop']] for g in r['groups']], 'validation identity', issues)
        check(v['allfinite'] is True and v['allclose'] is True and v['bit_equal'] is True and v['maxabs'] == 0 and
              v['weight_byte_checks'] == [True]*4 and v['atol'] == v['rtol'] == .01, 'numerical/weight qualification', issues)
    check({v['layer_name'] for v in validation} == set(LAYERS), 'all16 validation layers', issues)
    return dict(issues=issues, **counts, by_phase=dict(phase_counts), validation_calls=len(validation),
                measurement_context_calls=len(measured), final_state_sha256=hashlib.sha256(json.dumps(
                    {n:s.snapshot() for n,s in states.items()}, sort_keys=True).encode()).hexdigest())


def cell_result(path, cell, mode, cap, source, expected_sources, protocol, prepared):
    issues=[]; report=read(path/'shared_pool.json'); pager=read(path/'pager_summary.json'); raw=read(path/'raw.json')
    workload=read(path/'workload.json'); config=read(path/'config.json'); res=read(path/'measurement_resources.json')
    check(cell['status']=='COMPLETE' and cell['returncode']==0 and read(path/'status.json')['status']=='COMPLETE' and
          report['status']=='QUALIFIED_LIFECYCLE_ONLY' and report['qualification'] and report['mode']==mode, 'engine qualification completion', issues)
    check(report['sources']==expected_sources and all(expected_sources.get(n)==h for n,h in read(path/'environment.json')['sources'].items()), 'frozen11 sources', issues)
    check([r['document_id'] for r in workload['source_requests']]==protocol['workload']['document_ids'] and
          workload['source_requests']==prepared['source_requests'] and workload['actual_prompt_token_ids']==[ids[:64] for ids in prepared['actual_prompt_token_ids']], 'frozen4 input identities', issues)
    check(config['verify_kernel'] and config['requests']==4 and config['prompt_tokens']==64 and config['output_tokens']==8 and
          config['token_budget']==160 and config['kv_bytes']==1073741824, 'qualification config', issues)
    req, errors=requests(raw,workload); issues.extend(errors)
    check(raw['status']=='COMPLETE' and len(req)==4 and all(r['status']=='completed' and len(r['output_token_ids'])==8 for r in req.values()), '4 complete requests/32 tokens', issues)
    caps={r['layer_name']:r['cap'] for r in pager['layers']}; sizes={r['layer_name']:r['pinned_bytes']//r['num_experts'] for r in pager['layers']}
    check(caps=={n:cap for n in LAYERS} and len(pager['layers'])==16 and set(sizes.values())=={12582912}, 'layer allocation', issues)
    for allocation in (pager, res, report['allocation']):
        storages=allocation['expert_scratch_unique_storages']
        check(allocation['layer_caps']==caps and allocation['expert_slots_total']==384 and allocation['private_slots_total']==16*cap and
              allocation['shared_slots']==384-16*cap and allocation['shared_pool_mode']==mode and
              allocation['expert_scratch_bytes']==sum(s['bytes'] for s in storages)==4831838208 and
              len(storages)==len({(s['device'],s['pointer']) for s in storages})==2, 'unique384 storage', issues)
    check(res['actual_unique_kv_storage_bytes']==1073741824 and res['scheduler_requests']==0 and
          read(path/'engine_args.json')['kv_cache_memory_bytes']==1073741824, 'actual/requested1GiB KV', issues)
    checks=[json.loads(s) for s in (path/'gpu_checks.jsonl').read_text().splitlines()]
    check([c['stage'] for c in checks]==['before_initialization','before_model_load','after_run'] and
          all(c['decision']=='PASS' and not c['foreign_pids'] and not c['query_errors'] for c in checks), 'GPU3 boundaries', issues)
    records=[json.loads(s) for s in (path/'pager/calls.jsonl').read_text().splitlines()]
    checked=check_calls(records,mode,cap,sizes,raw,report['validation']); issues.extend(checked.pop('issues'))
    check(len(records)==pager['all_calls'] and pager['failed_calls']==pager['measurement_calls']==0 and pager['validation_run'], 'pager qualification counts', issues)
    check(all(sum(r[k] for r in records)==pager['all'][k] for k in ('miss','evict','weight_copy_bytes','group_count')), 'pager all-call totals', issues)
    return dict(mode=mode,issues=issues,**checked,request_count=len(req),output_tokens=sum(len(r['output_token_ids']) for r in req.values()),
                sources=report['sources'],gpu_checks=checks,resources=dict(private_slots=16*cap,stage_slots=384-16*cap,expert_bytes=4831838208,kv_bytes=res['actual_unique_kv_storage_bytes']))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--results',type=Path,required=True); p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    if args.out.exists(): p.error('--out must name a new file')
    issues=[]; cells=[]; execution=None; source=args.source.resolve(); provenance={}
    try:
        sys.path.insert(0,str(source)); protocol=read(source.parent/'protocol.json'); expected_sources=read(source.parent/'sources.json')
        actual_sources={n:sha(source/n) for n in expected_sources}; check(len(actual_sources)==11 and actual_sources==expected_sources, 'sealed11 source content mismatch', issues)
        prepared_path=source.parent/'prepared/workload.json'; prepared=read(prepared_path)
        check(sha(prepared_path)==protocol['workload']['prepared_sha256'], 'sealed preparation hash', issues)
        execution=read(args.results/'execution.json'); check(execution['status']=='COMPLETE' and Counter(c['label'] for c in execution['cells'])==Counter(MODES.keys()), 'two-engine completion/coverage', issues)
        for cell in execution['cells']:
            try:
                mode,cap=MODES[cell['label']]; result=cell_result(args.results/cell['label'],cell,mode,cap,source,expected_sources,protocol,prepared)
                cells.append(dict(label=cell['label'],**result)); issues.extend(cell['label']+': '+s for s in result['issues'])
            except ERRORS as exc: issues.append(cell['label']+': '+repr(exc)); cells.append(dict(label=cell['label'],status='UNVERIFIED',error=repr(exc)))
        provenance=dict(source_sha256=actual_sources,helper_sha256=sha(HELPERS/'analyze_native_pager_transfer.py'),checker_sha256=sha(Path(__file__)))
    except ERRORS as exc: issues.append('campaign: '+repr(exc))
    result=dict(status='ISSUES' if issues else 'QUALIFIED_NEW_HOST_LIFECYCLE_ONLY',issues=issues,cells=cells,execution=execution,provenance=provenance,
        scope='Each engine follows its own current rows from16 empty private LRUs. Qualification references/readback add memory/time; no performance, quality, cross-arm trajectory equality or continuous isolation claim.')
    with args.out.open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    if issues: raise SystemExit('qualification issues retained in output')


if __name__=='__main__': main()

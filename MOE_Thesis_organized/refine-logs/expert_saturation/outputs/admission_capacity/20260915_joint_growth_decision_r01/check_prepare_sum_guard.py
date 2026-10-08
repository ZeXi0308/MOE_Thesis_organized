"""One counterexample using pinned native schedule, CPU allocator/transfer fixtures.

Accepted I files remain read-only. The comparison removes one prefilter from an
in-memory copy of I's actual closures; it is not a proposed scheduler or GPU run.
"""
import argparse
import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS, MethodType
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
REPO = HERE.parents[4]
I = OUT / '20260915_natural_ltr_style_component_r01'
sys.path[:0] = [str(I / 'pkg'), str(REPO / 'refine-logs/expert_saturation/experiments/admission_capacity')]
import ltr_style_native as adapter
import verify_native_recovery_execution as harness
from recovery_service_components import LTRRequestState
from staged_save_contract import RequestState, StoreEvidence, prepare, commit_reason


def install_closures(scheduler, native, cs, remove_sum_guard):
    tree = ast.parse(Path(adapter.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install')
    executable = next(n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == 'executable')
    guard = [n for n in executable.body if isinstance(n, ast.If)
             and ast.unparse(n.test) == 'sum((r.remaining_blocks for r in rows)) > pool.get_num_free_blocks()']
    assert len(guard) == 1, 'This fixture targets exactly the accepted sum guard'
    if remove_sum_guard:
        executable.body.remove(guard[0])
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'step' for t in n.targets))
    build = ast.parse('def build(scheduler,native,owned,pool,cs,threshold,quantum):\n diagnostic=True\n oldcalc=cs._calc_num_offloadable_tokens\n hadcalc=True\n').body[0]
    build.body += deepcopy(fn.body[start:])
    env = dict(vars(adapter))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[build], type_ignores=[])),
                 '<actual-I-closures-one-guard-comparison>', 'exec'), env)
    owned = scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
    data, undo = env['build'](scheduler, native, owned, scheduler.kv_cache_manager.block_pool, cs, 30, 10)
    cells = dict(zip(scheduler.schedule.__code__.co_freevars, scheduler.schedule.__closure__))
    selector = cells['selector'].cell_contents
    selector.counters.states = {'target': LTRRequestState(idle=30), 'V': LTRRequestState(), 'peer': LTRRequestState()}
    return data, undo


def view(scheduler, rid):
    r = scheduler.requests[rid]
    return RequestState(rid, r.num_computed_tokens, r.num_prompt_tokens,
                        r.num_output_tokens, r.max_tokens, r.status.name,
                        tuple(b.block_id for b in scheduler.owned.get(rid, [])))


def snapshot(scheduler):
    return dict(free=len(scheduler.available), running=[r.request_id for r in scheduler.running],
                waiting=[r.request_id for r in scheduler.waiting],
                requests={rid: asdict(view(scheduler, rid)) for rid in scheduler.requests})


def run(source, remove_sum_guard):
    initial = dict(pool=dict(total_blocks=4, free_blocks=0), running_ids=['V', 'peer'], waiting_count=1,
        requests={
            'V': dict(computed_tokens=31, prompt_tokens=16, output_tokens=16, num_preemptions=0, block_counts=[2]),
            'peer': dict(computed_tokens=16, prompt_tokens=8, output_tokens=9, num_preemptions=0, block_counts=[1]),
            'target': dict(computed_tokens=0, prompt_tokens=16, output_tokens=1, num_preemptions=1, block_counts=[0]),
        })
    with patch.object(harness.adapter, 'install', return_value=([], lambda: None)):
        s, _, _ = harness.create(source, initial, ['target'], False)
    for rid, arrival in {'target': 0., 'V': 1., 'peer': 2.}.items():
        s.requests[rid].arrival_time = arrival
    namespace = dict(s.schedule.__func__.__globals__)
    exec(compile(adapter.patched_schedule_tree(source.read_text()), str(source), 'exec'), namespace)
    native_method = MethodType(namespace['schedule'], s)
    victim_rs = NS(req=s.requests['V'], transfer_jobs=set(),
                   group_states=[NS(block_ids=[1, 2], offload_keys=['V-prefix-0', 'V-prefix-1'])])
    cs = NS(_calc_num_offloadable_tokens=lambda rs, n: n, _req_status={'V': victim_rs}, _jobs={})
    native_calls = []

    def native():
        result = native_method()  # actual native running/preemption/waiting/allocation loop
        meta = NS(store_jobs={}, load_jobs={}, jobs_to_flush=set())
        if result.num_scheduled_tokens.get('V'):
            count = cs._calc_num_offloadable_tokens(victim_rs, s.requests['V'].num_computed_tokens)
            if count:
                prefix = tuple(b.block_id for b in s.owned['V'][:count // 16])
                job_id = 7
                cs._jobs[job_id] = NS(req_id='V', is_store=True,
                    keys=set(victim_rs.group_states[0].offload_keys[:len(prefix)]))
                victim_rs.transfer_jobs.add(job_id)
                meta.store_jobs[job_id] = NS(req_id='V', src_spec=NS(block_ids=list(prefix)))
        if 'V' in (result.preempted_req_ids or ()):
            meta.jobs_to_flush = set(victim_rs.transfer_jobs)
        result.kv_connector_metadata = meta  # explicit CPU registration/flush substitute
        native_calls.append(dict(scheduled=dict(result.num_scheduled_tokens),
            preempted=sorted(result.preempted_req_ids or ()), free_after_schedule=len(s.available),
            store_jobs=sorted(meta.store_jobs), flush_jobs=sorted(meta.jobs_to_flush)))
        return result

    before = snapshot(s)
    plan = prepare(0, view(s, 'V'), view(s, 'target'), len(s.available))
    all_growth = sum(view(s, r.request_id).remaining_blocks for r in s.running)
    data, undo = install_closures(s, native, cs, remove_sum_guard)
    trace = []
    first = s.schedule()
    outputs, completed = harness.returned(s, first)  # CPU synchronous output substitute
    trace.append(dict(call=0, after_return=snapshot(s), outputs=outputs, completed=completed))
    evidence = None
    if victim_rs.transfer_jobs:
        evidence = StoreEvidence('V', plan.saved_tokens, plan.source_blocks, tuple(sorted(victim_rs.transfer_jobs)))
    reason = commit_reason(plan, 1, view(s, 'V'), view(s, 'target'), len(s.available), evidence)
    second = s.schedule()
    outputs, completed = harness.returned(s, second)
    trace.append(dict(call=1, after_return=snapshot(s), outputs=outputs, completed=completed))
    undo()
    assert native_calls[0]['scheduled'] == {'V': 1}
    assert native_calls[0]['preempted'] == ['peer']
    assert native_calls[0]['free_after_schedule'] == 1
    assert s.discarded[0] == dict(request_id='peer', computed=16)
    if remove_sum_guard:
        assert reason == 'READY' and data['applied_rotations'] == 1
        assert native_calls[1]['preempted'] == ['V'] and native_calls[1]['flush_jobs'] == [7]
        assert native_calls[1]['scheduled'] == {'target': 17}
    else:
        assert reason == 'CANCEL_STORE_NOT_REGISTERED' and data['applied_rotations'] == 0
        assert any(e['event'] == 'intent' and any(
            x['reason'] == 'prepare growth not funded' for x in e['skipped']) for e in data['events'])
    return dict(remove_only_sum_guard=remove_sum_guard, initial=before,
        prefilter=dict(all_running_remaining_blocks=all_growth, free=before['free'], rejects=all_growth > before['free']),
        staged_plan=asdict(plan), store_evidence=asdict(evidence) if evidence else None,
        shared_contract_commit_reason_after_first_return=reason,
        native_calls=native_calls, trace=trace, discarded=s.discarded,
        failed_allocations=s.failed_allocations, adapter_events=data['events'], applied_rotations=data['applied_rotations'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    captures = json.loads((OUT / '20260914_load_ready_contract_r01/native_source.json').read_text())
    entry = captures['v1/core/sched/scheduler.py']
    text = entry['source'] if isinstance(entry, dict) else entry
    assert hashlib.sha256(text.encode()).hexdigest() == adapter.SCHEDULER_SHA256
    with tempfile.TemporaryDirectory(prefix='.sum-guard-native-', dir=HERE) as temp:
        source = Path(temp) / 'scheduler.py'
        source.write_text(text)
        result = dict(status='COUNTEREXAMPLE_CONFIRMED_CPU_NATIVE_LOOP',
            evidence='Pinned native scheduler/preempt/cache/post-schedule methods plus accepted I closures; fake allocator, outputs, connector job registration/flush; no GPU/native transfer/load result',
            scope='Legality of staged prepare/commit only, not absence of peer cost or service advantage',
            sources=dict(adapter=str(Path(adapter.__file__)), native_sha256=adapter.SCHEDULER_SHA256,
                         harness=str(Path(harness.__file__))),
            arms=[run(source, False), run(source, True)])
    with args.output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'evidence', 'scope')}))


if __name__ == '__main__':
    main()

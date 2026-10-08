"""Full eligible-running victim baselines; install BEFORE native pressure observer.
Adapted from A's full-running path: scheduled prefix plus current/unprocessed suffix.
Keep native self-preempt break and PRIORITY rollback. Fixed48 restoration stays separate.
BidKV score/tie: vLLM-HUST/vllm-ascend-hust-bidkv commit5ee80256d263d58b1e512d9d436d47e9bac564ba (Apache-2.0).
"""
import ast
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
from pathlib import Path
import time
from types import MethodType

SCHEDULER_SHA256 = '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def bidkv_score(request):
    computed = max(int(getattr(request, 'num_computed_tokens', 0) or 0), 0)
    preemptions = max(int(getattr(request, 'num_preemptions', 0) or 0), 0)
    ids = getattr(request, 'output_token_ids', None)
    try: outputs = len(ids) if ids is not None else int(getattr(request, 'num_output_tokens', 0) or 0)
    except TypeError: outputs = int(getattr(request, 'num_output_tokens', 0) or 0)
    maximum = getattr(request, 'max_tokens', None)
    if not isinstance(maximum, (int, float)) or maximum <= 0: maximum = 1024
    completion = min(max(float(outputs)/float(maximum), 0.0), 1.0)
    utility = max(float(computed), 0.0)/max(1.0+0.5*completion+0.3*max(float(preemptions), 0.0)+1e-6, 1e-6)
    return dict(utility=utility, computed_tokens=computed, completion=completion, num_preemptions=preemptions)


def releasable_blocks(single, request_id):
    """Count physical blocks whose complete live reference count is held here."""
    occurrences, objects = Counter(), {}
    for b in single.req_to_blocks.get(request_id, ()):
        if b.is_null: continue
        require(b.block_id not in objects or objects[b.block_id] is b, 'block ID aliases different objects')
        objects[b.block_id] = b; occurrences[b.block_id] += 1
    require(all(objects[k].ref_cnt >= n for k,n in occurrences.items()), 'owned block refcount underflow')
    return sum(objects[k].ref_cnt == n for k,n in occurrences.items())


def patched_schedule_tree(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
    matches = [n for n in ast.walk(fn) if isinstance(n, ast.If) and
               ast.unparse(n.test) == 'self.policy == SchedulingPolicy.PRIORITY']
    require(len(matches) == 1 and not fn.decorator_list, 'native victim branch ambiguous')
    branch = matches[0]
    require(len(branch.body) == 3 and ast.unparse(branch.orelse[0]) == 'preempted_req = self.running.pop()'
            and len(branch.orelse) == 1 and ast.unparse(branch.body[1]) == 'self.running.remove(preempted_req)'
            and isinstance(branch.body[2], ast.If)
            and ast.unparse(branch.body[2].test) == 'preempted_req in scheduled_running_reqs', 'native rollback branch changed')
    prefix = ast.parse('preempted_req = self._victim_choice_pick(request, req_index, scheduled_running_reqs, num_scheduled_tokens, token_budget, num_new_tokens, new_blocks)').body
    suffix = ast.parse('self._victim_choice_after_remove(preempted_req, req_index, token_budget, scheduled_running_reqs, num_scheduled_tokens, req_to_new_blocks)').body
    branch.orelse = prefix + copy.deepcopy(branch.body[1:]) + suffix
    return ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))


@contextmanager
def victim_choice(engine, output_path, mode):
    require(mode in ('bidkv-full', 'max-free-full'), 'unsupported victim mode')
    s = engine.engine_core.engine_core.scheduler
    m, pool = s.kv_cache_manager, s.kv_cache_manager.block_pool
    singles = m.coordinator.single_type_managers
    require(len(singles) == 1 and type(singles[0]).__name__ == 'FullAttentionManager' and
            type(singles[0].kv_cache_spec).__name__ == 'FullAttentionSpec', 'requires single FullAttention group')
    single = singles[0]
    require(getattr(s.policy, 'name', str(s.policy)).lower() == 'fcfs' and
            not s.scheduler_config.async_scheduling and not s.use_eagle and not s.num_lookahead_tokens
            and s.num_sampled_tokens_per_step == 1 and not s.is_encoder_decoder and s.connector is None
            and s.lora_config is None and not s.running and not s.waiting and not s.skipped_waiting,
            'install on drained synchronous FCFS decoder without connector/speculation/LoRA')
    original_fn = type(s).schedule
    require(getattr(s.schedule, '__func__', None) is original_fn, 'install victim choice before schedule wrappers')
    source_path = Path(inspect.getsourcefile(original_fn)); source_bytes = source_path.read_bytes()
    require(hashlib.sha256(source_bytes).hexdigest() == SCHEDULER_SHA256, 'native scheduler source hash differs')
    namespace = dict(original_fn.__globals__)
    exec(compile(patched_schedule_tree(source_bytes.decode()), str(source_path)+':victim-choice', 'exec'), namespace)
    attrs = ('schedule', '_preempt_request', '_victim_choice_pick', '_victim_choice_after_remove')
    require(not any(hasattr(s,k) for k in attrs[2:]), 'victim hooks already installed')
    saved = {k: (k in vars(s), vars(s).get(k)) for k in attrs}
    original_preempt = s._preempt_request
    events, pending, hook_wall = [], None, 0.0
    report = dict(schema='c-victim-choice-policy-v1', status='INCOMPLETE', mode=mode,
        initial_scheduler_step=s.current_step, scheduler_source_sha256=SCHEDULER_SHA256, decisions=events,
        candidate_scope='All scheduled running prefix plus current and unprocessed suffix; skipped unscheduled prefix excluded explicitly.',
        self_preempt_rule='Native break, no current guard or continuation.',
        apc_domain='Selected victim must have computed>=prompt, known=computed+1 and scheduled tokens0/1. Cell additionally qualifies prefix-free unique prompts; failure is unsupported domain, not a policy result.',
        cost_scope='Hook wall includes scoring, physical queries and event payload construction; excludes native preempt, context setup and exit JSON serialization. Analysis episode ends at completion; cell measurement phase includes context exit writes.')

    def pick(current, req_index, scheduled, scheduled_tokens, budget, num_new_tokens, allocation_result):
        nonlocal pending, hook_wall
        started = time.perf_counter()
        try:
            require(pending is None and allocation_result is None and s.running[req_index] is current
                    and getattr(current.status, 'name', str(current.status)) == 'RUNNING', 'not a native RUNNING None trigger')
            scheduled_ids = {id(r) for r in scheduled}
            rows = []
            for index, r in enumerate(s.running):
                if index < req_index and id(r) not in scheduled_ids: continue
                require(getattr(r.status, 'name', str(r.status)) == 'RUNNING', 'nonrunning victim candidate')
                row = dict(request_id=r.request_id, running_index=index, arrival_time=r.arrival_time,
                    kind='scheduled-prefix' if index < req_index else 'current' if index == req_index else 'suffix',
                    num_prompt_tokens=r.num_prompt_tokens, num_tokens=r.num_tokens,
                    num_computed_tokens=r.num_computed_tokens,
                    single_token_decode=r.num_computed_tokens >= r.num_prompt_tokens and
                        r.num_tokens == r.num_computed_tokens+1)
                if mode == 'bidkv-full': row['bidkv'] = bidkv_score(r)
                else: row['releasable_blocks'] = releasable_blocks(single, r.request_id)
                rows.append(row)
            chosen = (min(rows, key=lambda x: (-x['bidkv']['utility'], x['arrival_time'], str(x['request_id'])))
                      if mode == 'bidkv-full' else max(rows, key=lambda x: (x['releasable_blocks'], x['running_index'])))
            selected = s.running[chosen['running_index']]
            if 'releasable_blocks' not in chosen:
                chosen['releasable_blocks'] = releasable_blocks(single, selected.request_id)
            tail = rows[-1]
            if 'releasable_blocks' not in tail:
                tail['releasable_blocks'] = releasable_blocks(single, tail['request_id'])
            in_step = events[-1]['trigger_index_in_call']+1 if events and events[-1]['scheduler_step']==s.current_step else 0
            event = dict(decision=len(events), scheduler_step=s.current_step, trigger_index_in_call=in_step,
                trigger_current_id=current.request_id, trigger_num_new_tokens=num_new_tokens,
                free_blocks_at_trigger=pool.get_num_free_blocks(), current_running_index=req_index,
                running_count=len(s.running), candidate_count=len(rows), excluded_unscheduled_prefix=len(s.running)-len(rows),
                candidate_non_single_decode_count=sum(not x['single_token_decode'] for x in rows),
                native_tail_id=s.running[-1].request_id, selected_id=selected.request_id,
                selected_kind=chosen['kind'], changed_from_native_tail=selected is not s.running[-1],
                selected_releasable_blocks=chosen['releasable_blocks'], selected_score=chosen.get('bidkv', chosen['releasable_blocks']),
                selected_candidate=dict(chosen), native_tail_candidate=dict(tail),
                budget_before=budget, selected_scheduled_tokens=scheduled_tokens.get(selected.request_id,0))
            events.append(event)
            require(chosen['single_token_decode'] and event['selected_scheduled_tokens'] in (0,1),
                    'UNSUPPORTED_SELECTED_APC_DOMAIN: victim must be single-token decode')
            pending=(selected,event)
            return selected
        finally: hook_wall += time.perf_counter()-started

    def after_remove(request, req_index, budget, scheduled, scheduled_tokens, new_blocks):
        nonlocal hook_wall
        started = time.perf_counter()
        try:
            require(pending is not None and pending[0] is request and request not in s.running, 'victim removal differs')
            e = pending[1]; prefix = e['selected_kind']=='scheduled-prefix'; refund=budget-e['budget_before']
            require(req_index == e['current_running_index']-int(prefix) and
                    refund == (e['selected_scheduled_tokens'] if prefix else 0), 'native index/token rollback differs')
            if prefix: require(request not in scheduled and request.request_id not in scheduled_tokens and
                               request.request_id not in new_blocks, 'native scheduled-prefix maps not rolled back')
            e['rollback'] = dict(scheduled_prefix=prefix, refunded_tokens=refund, req_index_after=req_index,
                                  budget_after=budget, removed_from_final_pending_maps=prefix)
        finally: hook_wall += time.perf_counter()-started

    def preempt(request, timestamp):
        nonlocal pending, hook_wall
        started, native_wall = time.perf_counter(), 0.0
        require(pending is not None and pending[0] is request and 'rollback' in pending[1], 'preempt without victim trigger')
        e = pending[1]
        receipt = dict(request_id=request.request_id, free_blocks_before=pool.get_num_free_blocks(), native_returned=False)
        e['preemption'] = receipt
        try:
            begin = time.perf_counter()
            try: result = original_preempt(request, timestamp)
            finally: native_wall = time.perf_counter()-begin
            receipt['native_returned'] = True
            require(getattr(request.status, 'name', str(request.status)) == 'PREEMPTED', 'native preempt status differs')
            require(request.request_id not in single.req_to_blocks, 'native preempt retained request KV list')
            return result
        except BaseException as exc:
            receipt['error'] = f'{type(exc).__name__}: {exc}'; raise
        finally:
            receipt['free_blocks_after'] = pool.get_num_free_blocks()
            receipt['actual_freed_blocks'] = receipt['free_blocks_after']-receipt['free_blocks_before']
            receipt['predicted_free_delta_matches'] = receipt['actual_freed_blocks']==e['selected_releasable_blocks']
            pending = None; hook_wall += time.perf_counter()-started-native_wall
            if receipt['native_returned']: require(receipt['predicted_free_delta_matches'], 'physical release prediction differs')

    with Path(output_path).open('x', encoding='utf-8') as stream:
        try:
            s.schedule = MethodType(namespace['schedule'], s)
            s._victim_choice_pick, s._victim_choice_after_remove, s._preempt_request = pick, after_remove, preempt
            yield report
            require(pending is None, 'unconsumed native victim selection')
            report['status'] = 'COMPLETE'
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'; raise
        finally:
            for key,(had,value) in saved.items():
                if had: setattr(s,key,value)
                elif key in vars(s): delattr(s,key)
            report.update(selection_decisions=len(events), actual_preemptions=sum(e.get('preemption',{}).get('native_returned',False) for e in events),
                          policy_hook_wall_s=hook_wall, hooks_restored=True)
            json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')

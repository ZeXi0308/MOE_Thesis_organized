"""Retain blocked decode KV and consume resident slack before native eviction.
Install before observe_native_pressure; fixed48 restoration remains independent.
No future lengths, KV mutation, sampling changes, or victim-score changes.
"""
import ast
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import time
from types import MethodType

SCHEDULER_SHA256 = '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def patched_schedule_tree(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
    require(not fn.decorator_list, 'decorated native schedule unsupported')
    loops = [n for n in fn.body if isinstance(n, ast.While) and
             ast.unparse(n.test) == 'req_index < len(self.running) and token_budget > 0']
    require(len(loops) == 1, 'native running loop ambiguous')
    outer = loops[0]; index = fn.body.index(outer)
    require(ast.unparse(fn.body[index-1]) == 'req_index = 0', 'native running index setup changed')
    inners = [n for n in ast.walk(outer) if isinstance(n, ast.While) and
              ast.unparse(n.test) == 'True' and n.body and
              ast.unparse(n.body[0]).startswith('new_blocks = self.kv_cache_manager.allocate_slots(')]
    require(len(inners) == 1, 'native allocation retry loop ambiguous')
    inner = inners[0]
    require(ast.unparse(inner.body[1]) == 'if new_blocks is not None:\n    break', 'native allocation success changed')
    inner.body[2:2] = ast.parse('''
if _slack_pass == 0 and self._resident_slack_skip(request, num_new_tokens, req_index, new_blocks):
    _slack_deferred_current = True
    _slack_skipped = True
    break
''').body
    exits = [n for n in outer.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'new_blocks is None']
    require(len(exits) == 1 and ast.unparse(exits[0]) == 'if new_blocks is None:\n    break', 'native None outer exit changed')
    exits[0].body[0:0] = ast.parse('''
if _slack_deferred_current:
    req_index += 1
    continue
''').body
    outer.body[0:0] = ast.parse('_slack_deferred_current = False').body
    envelope = ast.parse('''
_slack_skipped = False
for _slack_pass in range(2):
    req_index = 0
    pass
    if _slack_pass == 0 and _slack_skipped and not num_scheduled_tokens:
        self._resident_slack_fallback()
        continue
    break
''').body
    envelope[1].body[1] = outer
    fn.body[index-1:index+1] = envelope
    waiting = [n for n in fn.body if isinstance(n, ast.If) and
               ast.unparse(n.test) == 'not preempted_reqs and self._pause_state == PauseState.UNPAUSED']
    require(len(waiting) == 1, 'native waiting gate ambiguous')
    waiting[0].test = ast.BoolOp(op=ast.And(), values=[ast.UnaryOp(op=ast.Not(), operand=ast.Name(id='_slack_skipped', ctx=ast.Load())), waiting[0].test])
    returns = [n for n in fn.body if isinstance(n, ast.Return)]
    require(len(returns) == 1 and ast.unparse(returns[0]) == 'return scheduler_output', 'native final return changed')
    where = fn.body.index(returns[0])
    fn.body[where:where] = ast.parse('''
if _slack_skipped:
    self._resident_slack_finish(scheduler_output)
''').body
    return ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))


@contextmanager
def resident_slack(engine, output_path):
    s = engine.engine_core.engine_core.scheduler
    m = s.kv_cache_manager; pool = m.block_pool
    managers = m.coordinator.single_type_managers
    require(len(managers) == 1 and type(managers[0]).__name__ == 'FullAttentionManager' and
            type(managers[0].kv_cache_spec).__name__ == 'FullAttentionSpec', 'requires single FullAttention group')
    single = managers[0]
    require(getattr(s.policy, 'name', str(s.policy)).lower() == 'fcfs' and
            not s.scheduler_config.async_scheduling and not s.use_eagle and not s.num_lookahead_tokens and
            s.num_sampled_tokens_per_step == 1 and not s.is_encoder_decoder and s.connector is None and
            s.ec_connector is None and s.lora_config is None and not s.running and not s.waiting and not s.skipped_waiting,
            'install on drained synchronous FCFS decoder without connector/speculation/LoRA')
    original_fn = type(s).schedule
    require(getattr(s.schedule, '__func__', None) is original_fn, 'install resident slack before schedule wrappers')
    source_path = Path(inspect.getsourcefile(original_fn)); data = source_path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == SCHEDULER_SHA256, 'native scheduler source hash differs')
    namespace = dict(original_fn.__globals__)
    exec(compile(patched_schedule_tree(data.decode()), str(source_path)+':resident-slack', 'exec'), namespace)
    attrs = ('schedule', '_resident_slack_skip', '_resident_slack_fallback', '_resident_slack_finish')
    require(not any(hasattr(s, k) for k in attrs[1:]), 'resident-slack hooks already installed')
    saved = {k: (k in vars(s), vars(s).get(k)) for k in attrs}
    calls, by_step, hook_wall, non_decode_none = [], {}, 0.0, 0
    report = dict(schema='c-resident-slack-policy-v1', status='INCOMPLETE', initial_scheduler_step=s.current_step,
        scheduler_source_sha256=SCHEDULER_SHA256, calls=calls,
        scope='Defer native RUNNING None only for single-token decode, retain KV; close WAITING for the entire affected call. Empty first pass retries once with native eviction.',
        cost_scope='Hook wall covers additional decisions and receipt payload construction. AST control overhead, native work and all hook costs remain in measured latency; setup and exit JSON serialization are excluded from hook wall.')

    def skip(request, new_tokens, req_index, allocation_result):
        nonlocal hook_wall, non_decode_none
        started = time.perf_counter()
        try:
            require(allocation_result is None and s.running[req_index] is request and
                    getattr(request.status, 'name', str(request.status)) == 'RUNNING', 'not a native RUNNING None')
            if not (new_tokens == 1 and request.num_computed_tokens >= request.num_prompt_tokens and
                    request.num_tokens == request.num_computed_tokens+1):
                non_decode_none += 1
                return False  # Prefill/catchup keeps the complete native eviction path.
            event = by_step.get(s.current_step)
            if event is None:
                event = dict(scheduler_step=s.current_step, skips=[], fallback=False, waiting_closed=True, completed=False)
                by_step[s.current_step] = event; calls.append(event)
            held = len(single.req_to_blocks.get(request.request_id, ()))
            require(held > 0, 'blocked resident has no owned KV block slots')
            event['skips'].append(dict(request_id=request.request_id, running_index=req_index,
                num_prompt_tokens=request.num_prompt_tokens, num_tokens=request.num_tokens,
                num_computed_tokens=request.num_computed_tokens, num_new_tokens=new_tokens,
                held_blocks=held, free_blocks=pool.get_num_free_blocks()))
            return True
        finally: hook_wall += time.perf_counter()-started

    def fallback():
        nonlocal hook_wall
        started = time.perf_counter()
        try:
            event = by_step[s.current_step]
            require(not event['fallback'], 'more than one empty-pass fallback')
            event['fallback'] = True
        finally: hook_wall += time.perf_counter()-started

    def finish(result):
        nonlocal hook_wall
        started = time.perf_counter()
        try:
            event = by_step[s.current_step]
            require(not event['completed'], 'duplicate affected-call receipt')
            event.update(completed=True, scheduled_tokens=dict(result.num_scheduled_tokens),
                scheduled_token_total=sum(result.num_scheduled_tokens.values()),
                actual_preempted_ids=list(result.preempted_req_ids or []),
                free_blocks_at_return=pool.get_num_free_blocks())
        finally: hook_wall += time.perf_counter()-started

    with Path(output_path).open('x', encoding='utf-8') as stream:
        try:
            s.schedule = MethodType(namespace['schedule'], s)
            s._resident_slack_skip, s._resident_slack_fallback, s._resident_slack_finish = skip, fallback, finish
            yield report
            require(all(e['completed'] for e in calls), 'unfinished affected schedule call')
            report['status'] = 'COMPLETE'
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'; raise
        finally:
            for key, (had, value) in saved.items():
                if had: setattr(s, key, value)
                elif key in vars(s): delattr(s, key)
            report.update(affected_calls=len(calls), skip_count=sum(len(e['skips']) for e in calls),
                fallback_calls=sum(e['fallback'] for e in calls), native_non_decode_none=non_decode_none,
                policy_hook_wall_s=hook_wall, hooks_restored=True)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')

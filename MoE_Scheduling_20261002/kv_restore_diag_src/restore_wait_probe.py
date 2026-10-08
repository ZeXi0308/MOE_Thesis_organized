"""Host-only observation of request0044 at measurement schedule steps49..57."""
import inspect
import time

def install(scheduler):
    if scheduler.requests:
        raise ValueError('install only on the drained engine before measurement')
    manager, active_step = scheduler.kv_cache_manager, None
    data = dict(target='gsm8k-test-0044', step_window=[49, 57], schedule_calls=0, steps=[], allocations=[], remote_updates=[])
    target = lambda rid: rid == 'measurement/gsm8k-test-0044' or rid.startswith('measurement/gsm8k-test-0044-')
    originals = [(scheduler, 'schedule'), (manager, 'allocate_slots'), (scheduler, '_update_waiting_for_remote_kv')]
    saved = [(obj, name, name in vars(obj), getattr(obj, name)) for obj, name in originals]
    original_schedule, original_allocate, original_update = [r[3] for r in saved]
    signature = inspect.signature(original_allocate)
    blocks = lambda value: [[b.block_id for b in group] for group in value.blocks] if value is not None else None
    def snapshot():
        matches = [r for rid, r in scheduler.requests.items() if target(rid)]
        if len(matches) > 1:
            raise ValueError('target request is not unique')
        request = matches[0] if matches else None
        rid = request.request_id if request else None
        held = [[b.block_id for b in m.req_to_blocks.get(rid, ())]
                for m in manager.coordinator.single_type_managers]
        return dict(at_perf_s=time.perf_counter(), request_id=rid,
            status=request.status.name if request else None, computed=request.num_computed_tokens if request else None,
            finished_recving=rid in scheduler.finished_recving_kv_req_ids,
            free_blocks=manager.block_pool.get_num_free_blocks(), held_block_ids=held,
            held_blocks=sum(map(len, held)), running=len(scheduler.running),
            waiting=len(scheduler.waiting), skipped_waiting=len(scheduler.skipped_waiting))
    def enabled(request=None):
        return active_step is not None and 49 <= active_step <= 57 and (request is None or target(request.request_id))
    def allocate(*args, **kwargs):
        request = args[0] if args else kwargs['request']
        if not enabled(request):
            return original_allocate(*args, **kwargs)
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        params = dict(bound.arguments); params['request'] = request.request_id
        params['new_computed_blocks'] = blocks(params.get('new_computed_blocks'))
        event = dict(step=active_step, parameters=params, before=snapshot())
        data['allocations'].append(event)
        result = original_allocate(*args, **kwargs)
        event.update(success=result is not None, returned_block_ids=blocks(result), after=snapshot())
        return result
    def update(request):
        if not enabled(request):
            return original_update(request)
        event = dict(step=active_step, before=snapshot()); data['remote_updates'].append(event)
        result = original_update(request)
        event.update(return_value=result, after=snapshot())
        return result
    def schedule(*args, **kwargs):
        nonlocal active_step
        active_step = data['schedule_calls']; data['schedule_calls'] += 1
        event = dict(step=active_step, before=snapshot()) if enabled() else None
        if event is not None:
            data['steps'].append(event)
        try:
            result = original_schedule(*args, **kwargs)
            if event is not None:
                event.update(after=snapshot(), total_scheduled_tokens=sum(result.num_scheduled_tokens.values()),
                    target_scheduled_tokens=sum(n for rid, n in result.num_scheduled_tokens.items() if target(rid)),
                    target_preempted=any(target(rid) for rid in (result.preempted_req_ids or [])))
            return result
        finally:
            active_step = None
    scheduler.schedule, manager.allocate_slots, scheduler._update_waiting_for_remote_kv = schedule, allocate, update
    def uninstall(raw=None):
        for obj, name, existed, original in reversed(saved):
            setattr(obj, name, original) if existed else delattr(obj, name)
        if raw is None:
            data['capture_alignment'] = dict(verified=False, reason='capture did not return')
            return
        rows, origin = raw['scheduler_steps'], raw['measurement_origin_perf_counter_s']
        aligned = data['schedule_calls'] == len(rows) and [s['step'] for s in data['steps']] == list(range(49, min(58, len(rows)))) and len(rows) > 49
        for event in data['steps']:
            row = rows[event['step']]
            aligned &= (row['step'] == event['step'] and event.get('total_scheduled_tokens') == row['total_scheduled_tokens']
                and event.get('target_scheduled_tokens') == sum(r['scheduled_tokens'] for r in row['scheduled'] if r['request_id'] == data['target'])
                and event.get('target_preempted') == (data['target'] in row['preempted_request_ids'])
                and origin + row['start_s'] <= event['before']['at_perf_s'] <= event['after']['at_perf_s'] <= origin + row['end_s'])
        data['capture_alignment'] = dict(verified=bool(aligned), capture_steps=len(rows), wrapper_steps=data['schedule_calls'])
    return data, uninstall

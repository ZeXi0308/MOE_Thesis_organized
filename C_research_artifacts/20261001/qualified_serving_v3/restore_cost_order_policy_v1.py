"""Bounded restore-prefix first-fit/min-recompute action baseline, with fixed48 gate.
Only the native queue selector may reorder. No future length or progress guarantee.
Install inside native-pressure observation; model execution and allocation stay native.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import restore_fixed_margin_policy_v1 as fixed

require, name = fixed.require, fixed.name
MAX_RESTORE_CANDIDATES = 16


def demand(manager, request, margin_blocks, has_running):
    require(name(request.status) == 'PREEMPTED' and request.num_computed_tokens == 0
            and not request.has_encoder_inputs, 'expected uncomputed local restoration')
    require(not request.skip_reading_prefix_cache and manager.enable_caching,
            'fixed APC query contract differs')
    blocks, hit, uncached = manager.coordinator.find_longest_cache_hit(
        request.block_hashes, request.num_tokens - 1)
    require(uncached == 0 and 0 <= hit < request.num_tokens, 'APC query contract differs')
    values = dict(request=request, num_new_tokens=request.num_tokens-hit,
        num_new_computed_tokens=hit, new_computed_blocks=manager.create_kv_cache_blocks(blocks),
        has_scheduled_reqs=has_running, reserved_blocks=0)
    return dict(known_recompute_tokens=request.num_tokens-hit,
                **fixed.fixed_gate(manager, values, margin_blocks))


@contextmanager
def restore_cost_order(engine, output_path, mode, margin_blocks=48):
    require(mode in ('first-fit', 'min-recompute') and type(margin_blocks) is int
            and margin_blocks == 48, 'mode must be first-fit/min-recompute with margin48')
    s = engine.engine_core.engine_core.scheduler
    m = s.kv_cache_manager
    require(name(s.policy).lower() == 'fcfs' and
            type(s.waiting).__name__ == 'FCFSRequestQueue' and not s.skipped_waiting,
            'requires native FCFS regular waiting queue')
    require(s.lora_config is None and s.connector is None and
            s.scheduler_config.enable_chunked_prefill and
            s.scheduler_config.long_prefill_token_threshold == 0 and
            s.num_waiting_for_streaming_input == 0, 'unsupported waiting-admission backend')
    output_path = Path(output_path)
    gate_path = output_path.with_name('restore-cost-order-fixed-gate.json')
    original_select, entry_allocate = s._select_waiting_queue_for_scheduling, m.allocate_slots
    pending, events, receipts = None, [], []
    selector_calls, selector_wall, started = 0, 0.0, time.perf_counter()
    report = dict(schema='c-restore-cost-order-policy-v1', status='INCOMPLETE', mode=mode,
        fixed_margin_blocks=margin_blocks, maximum_restore_candidates=MAX_RESTORE_CANDIDATES,
        initial_scheduler_step=s.current_step,
        decisions=events, allocation_receipts=receipts, fixed_gate_receipt=gate_path.name,
        scope='At most16 consecutive PREEMPTED requests, never crossing fresh WAITING. First-fit stops querying at first fit; min-recompute queries the bounded prefix. No future EOS/remaining-length inputs or progress guarantee.',
        cost_scope='Selector wall includes readonly APC/physical queries and reorder. Allocation passthrough overhead remains in whole-episode timing.')

    def rollback(item):
        q, request, position, event = item
        if position == 0:
            return
        require(request in q, 'selected restore disappeared before rollback')
        q.remove_request(request); q.insert(position, request)
        event['queue_rollback'] = True

    with output_path.open('x', encoding='utf-8') as stream:
        try:
            with fixed.restore_fixed_margin(engine, gate_path, margin_blocks) as gate_report:
                original_allocate = m.allocate_slots

                def select():
                    nonlocal pending, selector_calls, selector_wall
                    begin = time.perf_counter(); selector_calls += 1
                    try:
                        require(pending is None, 'previous selected restore was never allocated')
                        q = original_select()
                        if q is not s.waiting or not q or s.skipped_waiting:
                            return q
                        if name(q.peek_request().status) != 'PREEMPTED':
                            return q
                        prefix, boundary, visited = [], 'QUEUE_END', 0
                        for request in q:
                            visited += 1
                            if name(request.status) != 'PREEMPTED':
                                boundary = 'NON_PREEMPTED'; break
                            prefix.append(request)
                            if len(prefix) == MAX_RESTORE_CANDIDATES:
                                boundary = 'LIMIT16'; break
                        if len(prefix) == 1:
                            return q  # No alternative: leave admission to the fixed gate.
                        event = dict(decision=len(events), scheduler_step=s.current_step,
                            host_s=time.perf_counter()-started, restore_head_id=prefix[0].request_id,
                            running=len(s.running), waiting=len(q), prefix_boundary=boundary,
                            prefix_entries_visited=visited, prefix_request_ids=[r.request_id for r in prefix],
                            queried_candidates=0, candidates=[], selected_id=None, selected_original_position=None,
                            action='NO_RESTORE_CANDIDATE_FITS', queue_rollback=False,
                            bypasses_unfit_head=False, reorders_fit_head_for_lower_cost=False)
                        events.append(event)
                        for position, request in enumerate(prefix):
                            event['candidates'].append(dict(request_id=request.request_id,
                                original_queue_position=position,
                                **demand(m, request, margin_blocks, bool(s.running))))
                            event['queried_candidates'] += 1
                            if mode == 'first-fit' and not event['candidates'][-1]['policy_deferred']:
                                break
                        fits = [c for c in event['candidates'] if not c['policy_deferred']]
                        if not fits:
                            return q
                        chosen = fits[0] if mode == 'first-fit' else min(fits,
                            key=lambda c: (c['known_recompute_tokens'], c['original_queue_position']))
                        position = chosen['original_queue_position']; request = prefix[position]
                        head = event['candidates'][0]
                        event.update(selected_id=request.request_id, selected_original_position=position,
                            action='KEEP_RESTORE_HEAD' if position == 0 else 'REORDER_RESTORE_PREFIX',
                            bypasses_unfit_head=position > 0 and head['policy_deferred'],
                            reorders_fit_head_for_lower_cost=position > 0 and not head['policy_deferred'])
                        if position:
                            q.remove_request(request); q.prepend_request(request)
                        pending = (q, request, position, event)
                        return q
                    finally:
                        selector_wall += time.perf_counter()-begin

                def allocate(*args, **kwargs):
                    nonlocal pending
                    if pending is None:
                        return original_allocate(*args, **kwargs)
                    item = pending; request = args[0] if args else kwargs['request']
                    require(request is item[1] and item[0].peek_request() is request and
                            name(request.status) == 'PREEMPTED', 'selected restore/allocation identity differs')
                    receipt = dict(decision=item[3]['decision'], request_id=request.request_id,
                        scheduler_step=s.current_step, status_before=name(request.status),
                        num_new_tokens=args[1] if len(args)>1 else kwargs['num_new_tokens'],
                        free_blocks_before=m.block_pool.get_num_free_blocks(),
                        native_called=None, native_succeeded=None, native_returned_none=None)
                    receipts.append(receipt); gate_before = len(gate_report['decisions'])
                    try:
                        result = original_allocate(*args, **kwargs)
                        receipt['allocation_succeeded'] = result is not None
                        if result is None:
                            rollback(item)
                        return result
                    except BaseException as exc:
                        receipt['error'] = f'{type(exc).__name__}: {exc}'
                        rollback(item); raise
                    finally:
                        pending = None
                        receipt['free_blocks_after'] = m.block_pool.get_num_free_blocks()
                        gate_events = gate_report['decisions'][gate_before:]
                        require(len(gate_events) == 1 and gate_events[0]['request_id'] == request.request_id
                                and gate_events[0]['scheduler_step'] == receipt['scheduler_step'],
                                'fixed-gate selected allocation receipt differs')
                        g = gate_events[0]
                        receipt.update(fixed_gate_decision=g['decision'], native_called=g.get('native_called'),
                            policy_deferred=g['policy_deferred'])
                        if g.get('native_called') and 'native_returned_none' in g:
                            receipt['native_returned_none'] = g['native_returned_none']
                            receipt['native_succeeded'] = not g['native_returned_none']

                try:
                    s._select_waiting_queue_for_scheduling, m.allocate_slots = select, allocate
                    yield report
                    require(pending is None, 'unconsumed selected restore at measurement end')
                finally:
                    s._select_waiting_queue_for_scheduling, m.allocate_slots = original_select, original_allocate
                    if pending is not None:
                        rollback(pending); pending = None
            report['status'] = 'COMPLETE'
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'; raise
        finally:
            s._select_waiting_queue_for_scheduling = original_select
            report.update(selector_calls=selector_calls, selector_wall_s=selector_wall,
                selection_decisions=len(events),
                applied_reorders=sum(e['action'] == 'REORDER_RESTORE_PREFIX' for e in events),
                native_successful_selected_allocations=sum(r.get('native_succeeded') is True for r in receipts),
                native_successful_reordered_allocations=sum(r.get('native_succeeded') is True and
                    events[r['decision']]['action'] == 'REORDER_RESTORE_PREFIX' for r in receipts),
                hooks_restored=s._select_waiting_queue_for_scheduling is original_select and
                    m.allocate_slots == entry_allocate, observation_end_s=time.perf_counter()-started)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')

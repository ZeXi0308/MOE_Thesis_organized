"""Bounded new-request bypass at a fixed-margin-gated restoration queue head.

Same fixed-margin restoration gate as the control. Only the real native waiting
queue-selection hook can reorder. All model execution and allocation remain native.
This is a simple backfill action probe, not a novelty or progress-guarantee claim.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import restore_fixed_margin_policy_v1 as fixed

require, name = fixed.require, fixed.name
MAX_FRESH_CANDIDATES = 16


def demand(manager, request, margin_blocks, has_running):
    """Read-only APC/physical allocation query; never record an APC hit or touch KV."""
    require(request.num_computed_tokens == 0 and not request.has_encoder_inputs,
            'only local, uncomputed text waiting requests supported')
    require(not request.skip_reading_prefix_cache and manager.enable_caching,
            'fixed APC query contract differs')
    blocks, hit, uncached = manager.coordinator.find_longest_cache_hit(
        request.block_hashes, request.num_tokens - 1)
    require(uncached == 0 and 0 <= hit < request.num_tokens,
            'single FullAttention cache-query contract differs')
    values = dict(request=request, num_new_tokens=request.num_tokens-hit,
        num_new_computed_tokens=hit,
        new_computed_blocks=manager.create_kv_cache_blocks(blocks),
        has_scheduled_reqs=has_running, reserved_blocks=0)
    return fixed.fixed_gate(manager, values, margin_blocks)


@contextmanager
def restore_new_bypass(engine, output_path, margin_blocks=48):
    require(margin_blocks == 48, 'this action probe fixes the margin at48 blocks')
    s = engine.engine_core.engine_core.scheduler
    m = s.kv_cache_manager
    require(name(s.policy).lower() == 'fcfs' and
            type(s.waiting).__name__ == 'FCFSRequestQueue' and not s.skipped_waiting,
            'probe requires the native FCFS regular waiting queue')
    require(s.lora_config is None and s.connector is None and
            s.scheduler_config.enable_chunked_prefill and
            s.scheduler_config.long_prefill_token_threshold == 0 and
            s.num_waiting_for_streaming_input == 0,
            'unsupported waiting-admission backend')
    output_path = Path(output_path)
    gate_path = output_path.with_name('restore-new-bypass-fixed-gate.json')
    original_select = s._select_waiting_queue_for_scheduling
    pending = None
    events, allocation_receipts = [], []
    selector_calls, selector_wall = 0, 0.0
    started = time.perf_counter()
    report = dict(schema='c-restore-new-bypass-policy-v1', status='INCOMPLETE',
        fixed_margin_blocks=margin_blocks, maximum_fresh_candidates=MAX_FRESH_CANDIDATES,
        scope='First eligible fresh request may bypass a restoration head that fails the same48-block full-known+1 gate; finite original arrivals all retained. No EOS/remaining-length inputs.',
        decisions=events, allocation_receipts=allocation_receipts,
        fixed_gate_receipt=gate_path.name,
        cost_scope='Measured selector wall includes its readonly queries and reorder. Allocation passthrough overhead is included in whole-episode timing, not separately isolated.')

    def rollback(item):
        q, request, position, event = item
        require(request in q, 'pending bypass candidate disappeared before rollback')
        q.remove_request(request)
        q.insert(position, request)
        event['queue_rollback'] = True

    with output_path.open('x', encoding='utf-8') as stream:
        try:
            with fixed.restore_fixed_margin(engine, gate_path, margin_blocks) as gate_report:
                original_allocate = m.allocate_slots

                def select():
                    nonlocal pending, selector_calls, selector_wall
                    begin = time.perf_counter()
                    selector_calls += 1
                    try:
                        require(pending is None, 'previous bypass candidate was never allocated')
                        q = original_select()
                        if q is not s.waiting or not q or s.skipped_waiting:
                            return q
                        head = q.peek_request()
                        if name(head.status) != 'PREEMPTED':
                            return q
                        fresh, visited = [], 0
                        for index, request in enumerate(q):
                            visited += 1
                            if name(request.status) == 'WAITING':
                                fresh.append((index, request))
                                if len(fresh) == MAX_FRESH_CANDIDATES:
                                    break
                        if not fresh:
                            return q
                        head_need = demand(m, head, margin_blocks, bool(s.running))
                        event = dict(decision=len(events), scheduler_step=s.current_step,
                            host_s=time.perf_counter()-started, restore_head_id=head.request_id,
                            running=len(s.running), waiting=len(q), queue_entries_visited=visited,
                            head_plus_one_fits_without_margin=(head_need['candidate_full_plus_one_blocks']
                                + head_need['watermark'] <= head_need['free_blocks']),
                            restore_head=head_need, candidates=[], action='KEEP_RESTORE_HEAD')
                        events.append(event)
                        if not head_need['policy_deferred']:
                            return q
                        event['action'] = 'NO_FRESH_CANDIDATE_FITS'
                        for position, request in fresh:
                            candidate = demand(m, request, margin_blocks, bool(s.running))
                            event['candidates'].append(dict(request_id=request.request_id,
                                original_queue_position=position, **candidate))
                            if candidate['policy_deferred']:
                                continue
                            q.remove_request(request)
                            q.prepend_request(request)
                            event.update(action='BYPASS_WITH_FRESH', selected_id=request.request_id,
                                queue_rollback=False)
                            pending = (q, request, position, event)
                            break
                        return q
                    finally:
                        selector_wall += time.perf_counter()-begin

                def allocate(*args, **kwargs):
                    nonlocal pending
                    # Do not bind/inspect/time normal RUNNING allocations.
                    if pending is None:
                        return original_allocate(*args, **kwargs)
                    item = pending
                    request = args[0] if args else kwargs['request']
                    require(request is item[1] and item[0].peek_request() is request,
                            'selected candidate/allocation identity differs')
                    receipt = dict(decision=item[3]['decision'],
                        request_id=request.request_id, scheduler_step=s.current_step,
                        status_before=name(request.status),
                        num_new_tokens=args[1] if len(args)>1 else kwargs['num_new_tokens'],
                        free_blocks_before=m.block_pool.get_num_free_blocks())
                    allocation_receipts.append(receipt)
                    try:
                        result = original_allocate(*args, **kwargs)
                        receipt['native_succeeded'] = result is not None
                        if result is None:
                            rollback(item)
                        return result
                    except BaseException as exc:
                        receipt['error'] = f'{type(exc).__name__}: {exc}'
                        rollback(item)
                        raise
                    finally:
                        receipt['free_blocks_after'] = m.block_pool.get_num_free_blocks()
                        pending = None

                try:
                    s._select_waiting_queue_for_scheduling = select
                    m.allocate_slots = allocate
                    yield report
                    require(pending is None, 'unconsumed bypass at measurement end')
                finally:
                    s._select_waiting_queue_for_scheduling = original_select
                    m.allocate_slots = original_allocate
                    if pending is not None:
                        rollback(pending)
                        pending = None
            report['status'] = 'COMPLETE'
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            s._select_waiting_queue_for_scheduling = original_select
            report.update(selector_calls=selector_calls, selector_wall_s=selector_wall,
                competition_decisions=len(events),
                applied_reorders=sum(e['action']=='BYPASS_WITH_FRESH' for e in events),
                native_successful_bypass_allocations=sum(e.get('native_succeeded',False)
                    for e in allocation_receipts),
                hooks_restored=True, observation_end_s=time.perf_counter()-started)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')

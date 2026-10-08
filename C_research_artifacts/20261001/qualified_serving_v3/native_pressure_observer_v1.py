"""CPU-prepared, GPU-UNRUN observation helper for the pinned native runtime.

Use ``with observe_native_pressure(engine, path): common.generate(...)``.
No requests, allocation probes, scheduler decisions or ownership graph are added.
None means native KV admission/allocation rejection, not CUDA OOM. Full-ISL
checks the current sequence, not its future output cap. Host times are not ITL.
"""
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import time


ALLOCATION_PARAMETERS = (
    'request', 'num_new_tokens', 'num_new_computed_tokens', 'new_computed_blocks',
    'num_lookahead_tokens', 'num_external_computed_tokens', 'delay_cache_blocks',
    'num_encoder_tokens', 'full_sequence_must_fit', 'reserved_blocks',
    'has_scheduled_reqs')


def _name(value):
    return getattr(value, 'name', str(value))


@contextmanager
def observe_native_pressure(engine, output_path):
    """Observe a synchronous native call scope; always restore both methods.

    Output is exclusively reserved before installing wrappers, then written in
    finally, including failures. The caller owns runtime qualification and drain.
    The signature is the pinned v1 KVCacheManager.allocate_slots bound signature.
    """
    s = engine.engine_core.engine_core.scheduler
    manager = s.kv_cache_manager
    pool = manager.block_pool
    original_schedule, original_allocate = s.schedule, manager.allocate_slots
    signature = inspect.signature(original_allocate)
    if tuple(signature.parameters) != ALLOCATION_PARAMETERS:
        raise ValueError('native allocate_slots signature differs from pinned runtime')
    origin, calls, attempts, active = time.perf_counter(), [], [], None
    report = dict(schema='c-native-pressure-observer-v1', status='INCOMPLETE',
        scope='Observed native execution only; no OOM, counterfactual or policy-benefit claim.',
        allocation_signature=str(signature),
        configuration=dict(max_model_len=s.max_model_len,
            scheduler_reserve_full_isl=s.scheduler_reserve_full_isl,
            watermark_blocks=manager.watermark_blocks),
        scheduler_calls=calls, allocation_attempts=attempts)

    def head(queue):
        if not len(queue):
            return None
        request = queue.peek_request()
        return dict(request_id=request.request_id, status=_name(request.status))

    def state():
        return dict(total_blocks=pool.num_gpu_blocks,
            usable_blocks=pool.num_gpu_blocks - 1,
            free_blocks=pool.get_num_free_blocks(), running=len(s.running),
            waiting=len(s.waiting), skipped_waiting=len(s.skipped_waiting),
            waiting_head=head(s.waiting), skipped_waiting_head=head(s.skipped_waiting),
            effective_running=len(s.running) + s.num_waiting_for_streaming_input,
            seq_limit=s.max_num_running_reqs,
            token_budget_initial=s.max_num_scheduled_tokens,
            pause_state=_name(s._pause_state))

    def allocate(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        request = values['request']
        event = dict(attempt_id=len(attempts), call_id=None if active is None else active['call_id'],
            host_s=time.perf_counter() - origin, request_id=request.request_id,
            status_before=_name(request.status), num_prompt_tokens=request.num_prompt_tokens,
            num_tokens=request.num_tokens, num_computed_tokens=request.num_computed_tokens,
            arguments={name: values[name] for name in ALLOCATION_PARAMETERS
                       if name not in ('request', 'new_computed_blocks')},
            free_blocks_before=pool.get_num_free_blocks(), returned_none=None, succeeded=None)
        attempts.append(event)
        if active is not None:
            active['attempt_ids'].append(event['attempt_id'])
        try:
            result = original_allocate(*args, **kwargs)
            event.update(returned_none=result is None, succeeded=result is not None)
            return result
        except BaseException as exc:
            event['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            event['free_blocks_after'] = pool.get_num_free_blocks()

    def schedule(*args, **kwargs):
        nonlocal active
        previous = active
        call = dict(call_id=len(calls), start_s=time.perf_counter() - origin,
                    before=state(), attempt_ids=[], classifications=['OTHER'])
        calls.append(call)
        active = call
        try:
            result = original_schedule(*args, **kwargs)
            mapping = dict(result.num_scheduled_tokens)
            preempted = list(result.preempted_req_ids or [])
            call.update(scheduled_tokens=mapping, preempted_request_ids=preempted,
                remaining_token_budget=call['before']['token_budget_initial'] - sum(mapping.values()))
            return result
        except BaseException as exc:
            call['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            active = previous
            call.update(after=state(), return_s=time.perf_counter() - origin)
            events = [attempts[i] for i in call['attempt_ids']]
            rejected = [e for e in events if e['returned_none'] is True]
            waiting_rejected = [e['attempt_id'] for e in rejected
                                if e['status_before'] in ('WAITING', 'PREEMPTED')]
            call['waiting_rejection_attempt_ids'] = waiting_rejected
            facts = ['KV_ALLOCATION_REJECTED'] if rejected else []
            after = call['after']
            waiting = after['waiting'] + after['skipped_waiting'] > 0
            preempted = call.get('preempted_request_ids', [])
            if waiting and preempted:
                facts.append('PREEMPTION_GATE')
            remaining = call.get('remaining_token_budget')
            call['budget_exhausted_at_return'] = None if remaining is None else remaining == 0
            call['seq_limit_reached_at_return'] = after['effective_running'] >= after['seq_limit']
            # These are exit-boundary facts, not diagnoses for every queued request.
            # Waiting allocation None stops FIFO; later requests were not tested.
            if (waiting and not preempted and not waiting_rejected and 'error' not in call
                    and after['pause_state'] == 'UNPAUSED'):
                if remaining == 0:
                    facts.append('TOKEN_BUDGET_GATE')
                elif remaining > 0 and call['seq_limit_reached_at_return']:
                    facts.append('SEQ_LIMIT_GATE')
            call['classifications'] = facts or ['OTHER']

    with Path(output_path).open('x', encoding='utf-8') as stream:
        try:
            s.schedule, manager.allocate_slots = schedule, allocate
            yield report
            report['status'] = 'COMPLETE'
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            s.schedule, manager.allocate_slots = original_schedule, original_allocate
            report['observation_end_s'] = time.perf_counter() - origin
            report['schedule_calls'] = len(calls)
            report['allocation_calls'] = len(attempts)
            report['allocation_none_returns'] = sum(e['returned_none'] is True for e in attempts)
            report['limitations'] = [
                'KV None includes full-current-sequence admission and watermark checks; it is not OOM.',
                'Positive free blocks do not prove the waiting request could be admitted.',
                'Budget/sequence gates describe the schedule exit, not all requests independently.',
                'Unattempted FIFO successors are not independent KV rejection observations.',
                'Snapshots miss internal transients; no owner graph or required-block counterfactual is inferred.',
            ]
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')

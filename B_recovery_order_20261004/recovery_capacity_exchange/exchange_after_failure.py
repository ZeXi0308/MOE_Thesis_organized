"""Controlled development diagnostic: first 0070571 async failure, next begin.

Not an online target-selection method. Only trigger/target qualification changes
from the frozen exchange; donor arithmetic, native preempt/flush, Q1 and the
16-entry release stay intact. The original observer supplies the failure fact;
a pass-through wrapper captures object/episode identity at that same return.
"""
import hashlib
import importlib.util
import inspect
import math
from pathlib import Path
from types import FunctionType

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
RUNTIME_SHA = 'fa1decbde3afbcbdf51d479b234cae832aeb25ad3084939d144d4267d2ac7ac3'
OBSERVER_SHA = 'b0e59962645ba95f76bb380bec2f9c3675ceac62d7b57debb24a184cbb9c5c8f'
CONTROLLED_SOURCE_REQUEST = 'b-normal-0070571-long'
path = ROOT/'exchange_guarded_runtime.py'
if hashlib.sha256(path.read_bytes()).hexdigest() != RUNTIME_SHA:
    raise RuntimeError('Frozen runtime guard changed')
spec = importlib.util.spec_from_file_location('failure_exchange_runtime', path)
RUNTIME = importlib.util.module_from_spec(spec); spec.loader.exec_module(RUNTIME)
FROZEN = RUNTIME.GUARD.FROZEN
make_capture, MODES = RUNTIME.make_capture, RUNTIME.MODES


class _Trigger:
    def __init__(self, scheduler, receipts, observation):
        self.scheduler, self.receipts, self.observation = scheduler, receipts, observation
        self.step, self.seen, self.pending = 0, False, None
        self.data = dict(controlled_source_request=CONTROLLED_SOURCE_REQUEST,
            scope='First real async allocation failure only; inspect once at the next native begin. Explicit development ID, not deployable selection. No stale donor snapshot, new lookup or second allocation.',
            first_failure=None, events=[], status='INSTALLED')

    def allocate(self, old, *args, **kwargs):
        request = args[0] if args else kwargs.get('request')
        prefix = 'measured/'+CONTROLLED_SOURCE_REQUEST+'-'
        if self.seen or request is None or not request.request_id.startswith(prefix):
            return old(*args, **kwargs)
        count = len(self.observation['events'])
        episode, receipt = request.num_preemptions, self.receipts.get(request.request_id)
        result = old(*args, **kwargs)  # Native exceptions propagate, no latch/retry.
        added = self.observation['events'][count:]
        if result is not None or len(added) != 1:
            return result
        row = added[0]
        arguments = row.get('arguments', {})
        if (row.get('kind') != 'allocate' or row.get('request') != request.request_id
                or row.get('success') is not False or 'exception' in row
                or arguments.get('num_new_tokens') != 0
                or arguments.get('num_external_computed_tokens', 0) <= 0
                or arguments.get('delay_cache_blocks') is not True):
            return result
        self.seen = True
        record = dict(request=request.request_id, num_preemptions=episode,
            failure_step=self.step, last_receipt_host_perf_s=receipt,
            allocation_observation=row)
        self.data['first_failure'] = record
        before, after, descriptor = row['before'], row['after'], row['descriptor']
        valid = (self.step > 0 and episode > 0 and request.num_preemptions == episode
            and self.scheduler.requests.get(request.request_id) is request
            and type(receipt) in (int, float) and math.isfinite(receipt)
            and self.receipts.get(request.request_id) == receipt
            and descriptor.get('exact') is True
            and descriptor.get('full_fit_extra_blocks', 0) > before['free_gpu_blocks']
            and arguments.get('full_sequence_must_fit') is True
            and all(state['status'] == 'PREEMPTED' and state['num_computed_tokens'] == 0
                and state['num_in_flight_tokens'] == 0 and state['held_gpu_blocks'] == [0]
                and state['num_tokens'] == request.num_tokens for state in (before, after)))
        record['latched'] = valid
        if valid:
            self.pending = dict(ref=request, episode=episode, history=request.num_tokens,
                                receipt=receipt, record=record)
        else:
            self.data['events'].append(dict(kind='failure_rejected', step=self.step,
                reason='UNKNOWN_OR_CHANGED_FAILURE_BOUNDARY'))
        return result

    def next_begin(self, step):
        self.step = step
        pending, self.pending = self.pending, None
        if pending is None: return None
        request, scheduler = pending['ref'], self.scheduler
        reason = None
        if step != pending['record']['failure_step'] + 1: reason = 'NOT_NEXT_BEGIN'
        elif (scheduler.requests.get(request.request_id) is not request
              or request.num_preemptions != pending['episode']): reason = 'OBJECT_OR_EPISODE_CHANGED'
        elif self.receipts.get(request.request_id) != pending['receipt']: reason = 'NEW_OR_CHANGED_RECEIPT'
        elif (request.status.name != 'PREEMPTED' or request.num_tokens != pending['history']
              or request.num_computed_tokens or request.num_in_flight_tokens): reason = 'TARGET_STATE_CHANGED'
        self.data['events'].append(dict(kind='next_begin_check', step=step,
            request=request.request_id, eligible_for_fresh_checks=reason is None,
            reason=reason, host_perf_s=FROZEN.time.perf_counter()))
        return pending if reason is None else None


def _attach_for_trigger(trigger):
    source = inspect.getsource(FROZEN._attach)
    edits = (
        ('        if consumed: return',
         '        if consumed: return\n        failure = _failure_trigger(step)\n        if failure is None: return'),
        ("                if request.status.name != 'PREEMPTED' or request.priority != head.priority: break",
         "                if request.status.name != 'PREEMPTED' or request.priority != head.priority: break\n                if request is not failure['ref']: continue"),
        ("        if mode == 'stall8': return",
         "        decision['trigger_failure'] = failure['record']\n        if mode == 'stall8': return"),
    )
    for before, after in edits:
        if source.count(before) != 1: raise RuntimeError('Frozen failure-trigger boundary changed')
        source = source.replace(before, after)
    namespace = dict(FROZEN.__dict__, _failure_trigger=trigger.next_begin)
    exec(compile(source, str(__file__)+'[frozen-exchange-attach]', 'exec'), namespace)
    return namespace['_attach']


def install(scheduler, mode='stall8', *, last_receipts, selective, allocation_observation):
    if (hashlib.sha256((BASE/'capacity_handoff/observe_tail.py').read_bytes()).hexdigest() != OBSERVER_SHA
            or allocation_observation.get('status') != 'INSTALLED'
            or allocation_observation.get('qualification', {}).get('exact_layout') is not True
            or type(allocation_observation.get('events')) is not list):
        raise RuntimeError('Requires the existing exact recovering-allocation observer')
    trigger = _Trigger(scheduler, last_receipts, allocation_observation)
    parent_install = FunctionType(FROZEN.install.__code__,
        dict(FROZEN.__dict__, _attach=_attach_for_trigger(trigger)), 'install', FROZEN.install.__defaults__)
    parent_install.__kwdefaults__ = FROZEN.install.__kwdefaults__
    data, undo_exchange = parent_install(scheduler, mode, last_receipts=last_receipts, selective=selective)
    try: guard_data, undo_guard = RUNTIME._attach_guard(scheduler, data)
    except BaseException: undo_exchange(); raise
    manager = scheduler.kv_cache_manager
    old_allocate, own = manager.allocate_slots, 'allocate_slots' in vars(manager)
    def allocate(*args, **kwargs): return trigger.allocate(old_allocate, *args, **kwargs)
    manager.allocate_slots = allocate
    data.update(guard_adapter=guard_data, failure_trigger=trigger.data,
        after_failure_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    installed = True
    def uninstall():
        nonlocal installed
        if installed:
            if manager.allocate_slots is not allocate: raise RuntimeError('Failure trigger allocation slot changed')
            undo_guard(); undo_exchange()
            if own: manager.allocate_slots = old_allocate
            else: delattr(manager, 'allocate_slots')
            trigger.data['status'] = 'UNINSTALLED'
            trigger.data['pending_at_uninstall'] = trigger.pending is not None
            trigger.pending = None; installed = False
        return data
    return data, uninstall

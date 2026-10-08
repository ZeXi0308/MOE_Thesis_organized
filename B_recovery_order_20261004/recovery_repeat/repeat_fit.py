"""Same native recovery-fit predicate; total actual bypass budget one or eight."""
import hashlib
import importlib.util
from pathlib import Path
import time
from types import FunctionType

FIT_SHA = '7d2965b21f60516879b3a14a9a92533e882c9193890f4cf3cf3f77004bf8cbd6'
FIT_PATH = Path(__file__).resolve().parents[1]/'recovery_fit/fit_once.py'
if hashlib.sha256(FIT_PATH.read_bytes()).hexdigest() != FIT_SHA:
    raise RuntimeError('Frozen recovery-fit source changed')
_spec = importlib.util.spec_from_file_location('repeat_frozen_fit_once', FIT_PATH)
FROZEN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FROZEN)
LIMITS = {'once': 1, 'repeat8': 8}


def _decision(scheduler, manager, single, cs, mode):
    # Native mode performs the exact frozen qualification and records a shadow;
    # its one-event scratch list is reset, while public observations accumulate.
    shadow, observe = FROZEN._decision(scheduler, manager, single, cs, 'native')
    data = dict(shadow, mode=mode, events=[], action_limit=LIMITS[mode],
        action_count=0, shadow_count=0, bypassed_episodes=[],
        frozen_fit_sha256=FIT_SHA,
        scope='Same frozen recovery-fit predicate at every native waiting decision; '
              'one actual bypass per request/preemption episode; total budget 1 or 8')
    bypassed = set()

    def callback(current, budget, preempted, queue):
        shadow['events'].clear()
        observe(current, budget, preempted, queue)
        data['callback_calls'] = shadow['callback_calls']
        if not shadow['events']:
            return
        event = shadow['events'][0]
        chosen = next(r for r in queue if r.request_id == event['candidate_head'])
        episode = (chosen.request_id, chosen.num_preemptions)
        reasons = []
        if episode in bypassed:
            reasons.append('EPISODE_ALREADY_BYPASSED')
        if data['action_count'] >= data['action_limit']:
            reasons.append('BUDGET_EXHAUSTED')
        event.update(kind='fit_opportunity', candidate_num_preemptions=episode[1],
            requested_action='BYPASS', action='SHADOW_ONLY' if reasons else 'REORDER_PENDING',
            action_skip_reason=reasons[0] if reasons else None, action_skip_reasons=reasons,
            action_limit=data['action_limit'], action_count_before=data['action_count'],
            action_count_after=data['action_count'])
        data['events'].append(event)
        data['shadow_count'] += 1
        if reasons:
            for reason in reasons:
                data['skip_counts'][reason] = data['skip_counts'].get(reason, 0)+1
            event['return_host_perf_s'] = time.perf_counter()
            return
        event['action_begin_host_perf_s'] = time.perf_counter()
        try:
            queue.remove_request(chosen)
            queue.prepend_request(chosen)
            bypassed.add(episode)
            data['bypassed_episodes'].append(dict(request=episode[0], num_preemptions=episode[1]))
            data['action_count'] += 1
            data['outcome'] = 'REORDERED'
            event.update(waiting_after=[r.request_id for r in queue],
                final_head=next(iter(queue)).request_id, queue_changed=True,
                action='REORDERED', action_count_after=data['action_count'])
        except BaseException as error:
            after = list(queue)
            event.update(waiting_after=[r.request_id for r in after],
                final_head=after[0].request_id if after else None,
                action='ERROR', error=repr(error))
            data.update(status='ERROR', outcome='ERROR')
            raise
        finally:
            event['return_host_perf_s'] = time.perf_counter()
    return data, callback


def _attach(scheduler, manager, single, cs, mode, tree):
    if mode not in LIMITS:
        raise ValueError(mode)
    # Reuse the frozen native AST insertion, code identity check and uninstall.
    namespace = dict(FROZEN._attach.__globals__)
    namespace['_decision'] = lambda s, m, single, cs, unused: _decision(s, m, single, cs, mode)
    attach = FunctionType(FROZEN._attach.__code__, namespace)
    return attach(scheduler, manager, single, cs, 'fit_once', tree)


def install(scheduler, mode='once'):
    if mode not in LIMITS:
        raise ValueError(mode)
    # All original model/layout/native source guards execute without changes.
    namespace = dict(FROZEN.install.__globals__, _attach=_attach)
    return FunctionType(FROZEN.install.__code__, namespace)(scheduler, mode)

"""Eight recovery bypasses, comparing per-episode with per-request use limits.

Ordinary action-value probe only: no protection, allocation, or service changes.
Both modes observe the same frozen legal set and both selection suggestions.
"""
import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import FunctionType

BASE = Path(__file__).resolve().parents[1]
REPEAT_PATH = BASE/'recovery_repeat/repeat_fit.py'
REPEAT_SHA = '1ff383057d9d3c657efad27ef6a7d48520c8af513a1639006ff703f9ca489c40'
if hashlib.sha256(REPEAT_PATH.read_bytes()).hexdigest() != REPEAT_SHA:
    raise RuntimeError('Frozen repeat-fit source changed')
_spec = importlib.util.spec_from_file_location('unique_frozen_repeat', REPEAT_PATH)
REPEAT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REPEAT)
LIMITS = {'repeat8': 8, 'unique8': 8}


def _build_decision():
    source = inspect.getsource(REPEAT._decision)

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError('Frozen repeat selection boundary changed: '+old[:80])
        source = source.replace(old, new)

    replace("        frozen_fit_sha256=FIT_SHA,",
            "        used_request_ids=[], frozen_repeat_sha256=REPEAT_SHA,\n"
            "        frozen_fit_sha256=FIT_SHA,")
    replace("              'one actual bypass per request/preemption episode; total budget 1 or 8')",
            "              'repeat8 retains per-episode eligibility; unique8 permits each request ID once; '\n"
            "              'both total budgets eight, with common legal-set/suggestion observation')")
    replace('    bypassed = set()', '    bypassed = set()\n    used_ids = set()')
    replace("        chosen = next(r for r in queue if r.request_id == event['candidate_head'])\n"
            "        episode = (chosen.request_id, chosen.num_preemptions)\n"
            "        reasons = []",
            "        repeat_choice = next(r for r in queue if r.request_id == event['candidate_head'])\n"
            "        eligible_ids = {row['request'] for row in event['candidates'] if row['eligible']}\n"
            "        unused = [r for r in queue if r.request_id in eligible_ids and r.request_id not in used_ids]\n"
            "        unique_choice = min(unused, key=lambda r: (r.arrival_time, r.request_id), default=None)\n"
            "        chosen = repeat_choice if mode == 'repeat8' else unique_choice\n"
            "        episode = (chosen.request_id, chosen.num_preemptions) if chosen is not None else None\n"
            "        event.update(repeat8_suggestion=repeat_choice.request_id,\n"
            "            repeat8_suggestion_num_preemptions=repeat_choice.num_preemptions,\n"
            "            repeat8_would_execute=data['action_count'] < data['action_limit'] and (repeat_choice.request_id, repeat_choice.num_preemptions) not in bypassed,\n"
            "            unique8_suggestion=unique_choice.request_id if unique_choice is not None else None,\n"
            "            candidate_head=chosen.request_id if chosen is not None else None,\n"
            "            used_request_ids_before=sorted(used_ids), used_request_ids_after=sorted(used_ids))\n"
            "        reasons = []\n"
            "        if chosen is None:\n"
            "            reasons.append('NO_UNUSED_REQUEST_IN_LEGAL_SET')")
    replace("candidate_num_preemptions=episode[1]", "candidate_num_preemptions=episode[1] if episode else None")
    replace('            bypassed.add(episode)',
            "            bypassed.add(episode)\n"
            "            used_ids.add(chosen.request_id)\n"
            "            data['used_request_ids'] = sorted(used_ids)\n"
            "            event['used_request_ids_after'] = sorted(used_ids)")
    namespace = dict(REPEAT.__dict__, LIMITS=LIMITS, REPEAT_SHA=REPEAT_SHA)
    exec(compile(source, str(__file__)+'[pinned-repeat-selection]', 'exec'), namespace)
    return namespace['_decision']


_decision = _build_decision()


def _attach(scheduler, manager, single, cs, mode, tree):
    namespace = dict(REPEAT._attach.__globals__, _decision=_decision, LIMITS=LIMITS)
    return FunctionType(REPEAT._attach.__code__, namespace)(scheduler, manager, single, cs, mode, tree)


def install(scheduler, mode='repeat8'):
    """Return (observations, uninstall), using the unchanged native source guards."""
    if mode not in LIMITS:
        raise ValueError(mode)
    namespace = dict(REPEAT.install.__globals__, _attach=_attach, LIMITS=LIMITS)
    return FunctionType(REPEAT.install.__code__, namespace)(scheduler, mode)

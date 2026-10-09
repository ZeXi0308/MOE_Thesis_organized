"""Ordinary arrival-age versus client-output-stall recovery ordering, budget8.

The frozen legal set, select-then-episode guard, native mutation and refs stay
unchanged. Receipts are host synchronous client returns, never GPU completion.
"""
import hashlib
import importlib.util
import inspect
import math
from pathlib import Path
from types import FunctionType

BASE = Path(__file__).resolve().parents[1]
REPEAT_SHA = '1ff383057d9d3c657efad27ef6a7d48520c8af513a1639006ff703f9ca489c40'
COMPACT_SHA = '778a62f5b6325e33541489c815c6a324c9d6c7f5723a9b26b93956b319a17683'
path = BASE/'recovery_repeat/repeat_fit.py'
if hashlib.sha256(path.read_bytes()).hexdigest() != REPEAT_SHA:
    raise RuntimeError('Frozen repeat source changed')
spec = importlib.util.spec_from_file_location('service_age_frozen_repeat', path)
REPEAT = importlib.util.module_from_spec(spec); spec.loader.exec_module(REPEAT)
LIMITS = {'age8':8, 'stall8':8}


def _select(event, queue, receipts, bypassed, count, limit, mode):
    eligible = {row['request'] for row in event['candidates'] if row['eligible']}
    requests = [r for r in queue if r.request_id in eligible]
    age = min(requests, key=lambda r:(r.arrival_time,r.request_id))
    rows, failures = [], set()
    for request in requests:
        stamp = receipts.get(request.request_id)
        reason = ('MISSING_RECEIPT' if stamp is None else
            'NONFINITE_RECEIPT' if type(stamp) not in (int,float) or not math.isfinite(stamp) else
            'NOT_STRICTLY_PAST_RECEIPT' if stamp >= event['host_perf_s'] else None)
        if reason: failures.add(reason)
        rows.append(dict(request=request.request_id, arrival_time=request.arrival_time,
            num_preemptions=request.num_preemptions,
            last_receipt_host_perf_s=stamp if type(stamp) in (int,float) and math.isfinite(stamp) else None,
            stall_s=event['host_perf_s']-stamp if reason is None else None,
            known=reason is None, unknown_reason=reason))
    stall = age if failures else min(requests,
        key=lambda r:(receipts[r.request_id],r.arrival_time,r.request_id))
    def executable(request):
        return count < limit and (request.request_id,request.num_preemptions) not in bypassed
    chosen = age if mode=='age8' else stall
    event.update(receipt_candidates=rows, receipt_fallback_reasons=sorted(failures),
        stall8_fallback_reason='UNKNOWN_CANDIDATE_RECEIPT' if failures else None,
        receipt_observation_host_perf_s=event['host_perf_s'],
        age8_suggestion=age.request_id, age8_suggestion_num_preemptions=age.num_preemptions,
        age8_would_execute=executable(age), stall8_suggestion=stall.request_id,
        stall8_suggestion_num_preemptions=stall.num_preemptions,
        stall8_would_execute=executable(stall), candidate_head=chosen.request_id)
    return chosen


def _build_decision():
    source = inspect.getsource(REPEAT._decision)
    def replace(old,new):
        nonlocal source
        if source.count(old)!=1: raise RuntimeError('Service-age source boundary changed: '+old[:80])
        source=source.replace(old,new)
    replace('def _decision(scheduler, manager, single, cs, mode):',
            'def _decision(scheduler, manager, single, cs, mode, receipts):')
    replace("        frozen_fit_sha256=FIT_SHA,", "        frozen_repeat_sha256=REPEAT_SHA, frozen_fit_sha256=FIT_SHA,")
    replace("              'one actual bypass per request/preemption episode; total budget 1 or 8')",
            "              'arrival-age or longest client-output stall selects before the same episode guard; '\n"
            "              'unknown receipt falls back to arrival for the whole legal set; total budget8')")
    replace("        chosen = next(r for r in queue if r.request_id == event['candidate_head'])",
            "        chosen = _select(event, queue, receipts, bypassed, data['action_count'], data['action_limit'], mode)")
    namespace=dict(REPEAT.__dict__,LIMITS=LIMITS,REPEAT_SHA=REPEAT_SHA,_select=_select)
    exec(compile(source,str(__file__)+'[frozen-repeat-selection]','exec'),namespace)
    return namespace['_decision']


_decision = _build_decision()


def _attach(scheduler,manager,single,cs,mode,tree,last_receipts):
    namespace=dict(REPEAT._attach.__globals__,LIMITS=LIMITS,
        _decision=lambda s,m,g,c,selected:_decision(s,m,g,c,selected,last_receipts))
    return FunctionType(REPEAT._attach.__code__,namespace)(scheduler,manager,single,cs,mode,tree)


def install(scheduler,mode='age8',*,last_receipts):
    if mode not in LIMITS or type(last_receipts) is not dict: raise ValueError('Requires age8/stall8 and a receipt dict')
    namespace=dict(REPEAT.install.__globals__,LIMITS=LIMITS,
        _attach=lambda s,m,g,c,selected,t:_attach(s,m,g,c,selected,t,last_receipts))
    return FunctionType(REPEAT.install.__code__,namespace)(scheduler,mode)


RECEIPT_ANCHOR = '                row["token_times_s"].extend([received] * len(added))'
RECEIPT_INSERT = ('\n                if added:\n'
    '                    _service_age_last_receipts[row["internal_request_id"]] = origin + received')


def make_capture(compact,last_receipts):
    """Insert after successful prefix validation; reuse origin/received clocks."""
    if hashlib.sha256(Path(compact.__file__).read_bytes()).hexdigest()!=COMPACT_SHA:
        raise RuntimeError('Frozen compact capture changed')
    source=compact.COMPACT_SOURCE
    if source.count(RECEIPT_ANCHOR)!=1: raise RuntimeError('Expected one validated output update')
    source=source.replace(RECEIPT_ANCHOR,RECEIPT_ANCHOR+RECEIPT_INSERT)
    original=compact._compile(source,'compact')
    namespace=dict(original.__globals__,_service_age_last_receipts=last_receipts)
    capture=FunctionType(original.__code__,namespace,argdefs=original.__defaults__)
    capture.__kwdefaults__=original.__kwdefaults__
    capture.service_age_capture_sha256=hashlib.sha256(source.encode()).hexdigest()
    return capture

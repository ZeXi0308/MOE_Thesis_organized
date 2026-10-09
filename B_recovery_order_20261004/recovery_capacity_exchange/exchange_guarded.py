"""Narrow return-guard adapter for the frozen one-donor exchange.

The original native-only guard remains authoritative except for this adapter's
single, already executed and result-matched donor preemption in the same call.
No native allocation, transfer, reference, counter or error body is changed.
"""
import ast
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import CodeType, FunctionType, MethodType
import time

ROOT = Path(__file__).resolve().parent
FROZEN_SHA = 'ce54ed804c757d936c945741b1f19b8e8906053836a03630d978e7e4d771ca97'
STAGED_SHA = '11b986fe0a28fc8931e6785229935cd98027cbe8475a3d61c2ac792f29c428cb'
ERROR = 'Native-full ordinary arm attempted a forced rotation'
CALLBACK = '_b_authorized_exchange_return'
path = ROOT/'exchange_once.py'
if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_SHA:
    raise RuntimeError('Frozen failed-run exchange source changed')
spec = importlib.util.spec_from_file_location('guarded_frozen_exchange', path)
FROZEN = importlib.util.module_from_spec(spec); spec.loader.exec_module(FROZEN)
make_capture = FROZEN.make_capture
MODES = FROZEN.MODES


def _instrument(tree):
    tree = copy.deepcopy(tree)
    matches = 0
    expected = "not allow_forced_rotations and (data['applied_rotations'] or scheduler._rotation_forced_count or plan is not None)"
    for node in ast.walk(tree):
        if not isinstance(node,ast.If) or ast.unparse(node.test) != expected:
            continue
        if (len(node.body) != 1 or not isinstance(node.body[0],ast.Raise)
                or ast.unparse(node.body[0]) != f'raise RuntimeError({ERROR!r})' or node.orelse):
            raise RuntimeError('Original native-only guard body changed')
        extra = ast.parse(f'not {CALLBACK}(result, data, plan, now, step)',mode='eval').body
        node.test = ast.copy_location(ast.BoolOp(op=ast.And(),values=[node.test,extra]),node.test)
        matches += 1
    if matches != 1: raise RuntimeError('Expected one frozen native-only return guard')
    return ast.fix_missing_locations(tree)


def _nested_schedule(code):
    install = next(c for c in code.co_consts if isinstance(c,CodeType) and c.co_name=='install')
    return next(c for c in install.co_consts if isinstance(c,CodeType) and c.co_name=='schedule')


def _locate(scheduler):
    found, seen = [], set()
    def visit(value, owner, depth=0):
        if isinstance(value,MethodType): value = value.__func__
        if not isinstance(value,FunctionType) or id(value) in seen or depth>16: return
        seen.add(id(value))
        if (value.__name__=='schedule' and {'allow_forced_rotations','plan','native','data','scheduler'}
                <= set(value.__code__.co_freevars)):
            found.append((value,owner)); return
        for cell in value.__closure__ or ():
            try: child=cell.cell_contents
            except ValueError: continue
            if isinstance(child,(FunctionType,MethodType)): visit(child,cell,depth+1)
    visit(scheduler.schedule,None)
    if len(found)!=1: raise RuntimeError('Expected one existing staged schedule wrapper')
    return found[0]


def _authorize(scheduler, exchange, receipt, result, original_data, plan, entry_time, native_step):
    """Only authenticate this call's already logged native preemption."""
    reasons=[]
    actions=[e for e in exchange['events'] if e.get('kind')=='action']
    action=actions[0] if len(actions)==1 else None
    if exchange['mode']!='exchange_once' or exchange['action_count']!=1 or action is None:
        reasons.append('NOT_ONE_EXCHANGE_ACTION')
    if original_data['applied_rotations'] or plan is not None:
        reasons.append('OTHER_STAGED_ACTION_OR_PLAN')
    if scheduler._rotation_forced_count!=1: reasons.append('FORCED_COUNT_NOT_ONE')
    now=time.perf_counter()
    ids=list(result.preempted_req_ids or ())
    if action is not None:
        if not (action.get('native_preempt_called') is True and action.get('forced_count')==1
                and entry_time<=action['host_perf_s']<=now):
            reasons.append('ACTION_NOT_FROM_CURRENT_NATIVE_CALL')
        if ids!=[action['donor']]: reasons.append('RESULT_PREEMPT_IDS_MISMATCH')
        donor=scheduler.requests.get(action['donor'])
        cs=scheduler.connector.connector_scheduler
        state=cs._req_status.get(action['donor'])
        if (donor is None or state is None or state.req is not donor
                or donor.status.name!='PREEMPTED'
                or donor.num_preemptions!=action['donor_num_preemptions']):
            reasons.append('DONOR_IDENTITY_OR_PREEMPTION_MISMATCH')
        if not set(action['pending_store_jobs']) <= set(result.kv_connector_metadata.jobs_to_flush):
            reasons.append('PENDING_STORE_NOT_FLUSHED')
    row=dict(native_step=native_step,native_entry_host_perf_s=entry_time,host_perf_s=now,
        allowed=not reasons,reasons=reasons,result_preempted_req_ids=ids,
        forced_count=scheduler._rotation_forced_count,donor=action['donor'] if action else None,
        action_host_perf_s=action['host_perf_s'] if action else None)
    receipt['checks'].append(row)
    return not reasons


def _attach_guard(scheduler, exchange):
    old, owner = _locate(scheduler)
    filename = inspect.getsourcefile(old)
    if filename is None: raise RuntimeError('Missing staged source')
    source = Path(filename).read_bytes()
    if hashlib.sha256(source).hexdigest()!=STAGED_SHA: raise RuntimeError('Staged source changed')
    original = _nested_schedule(compile(source,filename,'exec',dont_inherit=True))
    if not FROZEN.HELPERS._same_code(original,old.__code__):
        raise RuntimeError('Live staged schedule differs from exact frozen reconstruction')
    modified = _instrument(ast.parse(source))
    code = _nested_schedule(compile(modified,filename,'exec',dont_inherit=True))
    if code.co_freevars!=old.__code__.co_freevars:
        raise RuntimeError('Staged closure layout changed')
    receipt=dict(status='INSTALLED', frozen_exchange_sha256=FROZEN_SHA,
        staged_source_sha256=STAGED_SHA,
        adapted_ast_sha256=hashlib.sha256(ast.dump(modified,include_attributes=False).encode()).hexdigest(),
        original_code_verified=True,checks=[],
        scope='Only the native-only return guard predicate is extended for the same-call single logged donor; allow_forced_rotations and all native counters remain unchanged.')
    def callback(result,data,plan,now,step):
        return _authorize(scheduler,exchange,receipt,result,data,plan,now,step)
    namespace=dict(old.__globals__)
    if CALLBACK in namespace: raise RuntimeError('Duplicate exchange guard')
    namespace[CALLBACK]=callback
    new=FunctionType(code,namespace,old.__name__,old.__defaults__,old.__closure__)
    new.__kwdefaults__=old.__kwdefaults__
    if owner is None:scheduler.schedule=new
    else:owner.cell_contents=new
    installed=True
    def undo():
        nonlocal installed
        if installed:
            current=scheduler.schedule if owner is None else owner.cell_contents
            if current is not new:raise RuntimeError('Staged guard slot changed before uninstall')
            if owner is None:scheduler.schedule=old
            else:owner.cell_contents=old
            installed=False;receipt['status']='UNINSTALLED'
        return receipt
    return receipt,undo


def install(scheduler,mode='stall8',*,last_receipts,selective):
    data,undo_exchange=FROZEN.install(scheduler,mode,last_receipts=last_receipts,selective=selective)
    try:receipt,undo_guard=_attach_guard(scheduler,data)
    except BaseException:
        undo_exchange();raise
    data['guard_adapter']=receipt
    installed=True
    def uninstall():
        nonlocal installed
        if installed:
            undo_guard();undo_exchange();installed=False
        return data
    return data,uninstall

"""Native observation versus frozen stall8; no new selection or receipt rule."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import FunctionType

BASE=Path(__file__).resolve().parents[1]
FROZEN_SHA='fd9c358d7aae9dfb475e5e06930d66c556ab2fed2cff74e82be1a481e5349b2f'
path=BASE/'recovery_service_age/service_age.py'
if hashlib.sha256(path.read_bytes()).hexdigest()!=FROZEN_SHA:
    raise RuntimeError('Frozen service-age policy changed')
spec=importlib.util.spec_from_file_location('native_frozen_service_age',path)
FROZEN=importlib.util.module_from_spec(spec);spec.loader.exec_module(FROZEN)
LIMITS={'native':8,'stall8':8}
make_capture=FROZEN.make_capture


def _build_decision():
    source=inspect.getsource(FROZEN._build_decision)
    anchor='    namespace=dict(REPEAT.__dict__,LIMITS=LIMITS,REPEAT_SHA=REPEAT_SHA,_select=_select)'
    if source.count(anchor)!=1:raise RuntimeError('Frozen decision builder boundary changed')
    extra=("    replace('        reasons = []', \"        reasons = []\\n"
           "        if mode == 'native':\\n            reasons.append('NATIVE_OBSERVATION_ONLY')\")\n")
    source=source.replace(anchor,extra+anchor)
    namespace=dict(FROZEN.__dict__,LIMITS=LIMITS)
    exec(compile(source,str(__file__)+'[native-guard-builder]','exec'),namespace)
    return namespace['_build_decision']()


_decision=_build_decision()


def _attach(scheduler,manager,single,cs,mode,tree,last_receipts):
    namespace=dict(FROZEN._attach.__globals__,LIMITS=LIMITS,_decision=_decision)
    return FunctionType(FROZEN._attach.__code__,namespace)(scheduler,manager,single,cs,mode,tree,last_receipts)


def install(scheduler,mode='native',*,last_receipts):
    if mode not in LIMITS:raise ValueError(mode)
    namespace=dict(FROZEN.install.__globals__,LIMITS=LIMITS,_attach=_attach)
    data,uninstall=FunctionType(FROZEN.install.__code__,namespace)(scheduler,mode,last_receipts=last_receipts)
    data['native_service_age_scope']='native only observes stall/age suggestions with explicit NATIVE_OBSERVATION_ONLY; stall8 is the frozen rule. Suggestion would_execute fields describe current guard eligibility, not a native action.'
    data['frozen_service_age_sha256']=FROZEN_SHA
    return data,uninstall

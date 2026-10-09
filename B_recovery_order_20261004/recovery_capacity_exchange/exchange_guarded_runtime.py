"""Exact normal-capacity runtime reconstruction for the frozen narrow guard.

The child executes staged_store_rotation in a synthetic, unregistered namespace;
its '[normal_capacity]' filename is not a source file. Rebuild that pinned
construction, require exact live code/closure compatibility, then retain the
previous single-donor authorization predicate and all original native paths.
"""
import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import CodeType, FunctionType

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PINS = {
    'recovery_capacity_exchange/exchange_guarded.py': '6132bfacffcfc63d09e218213d3c6577fc749a7088eabe58d69d329da99dc689',
    'normal_capacity/run_cell.py': '9f92617d44bbf8cda6992261129c35ff844a14a569dc8f259ed2bdedaa2388bb',
}


def _load(relative, name):
    path = BASE / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[relative]:
        raise RuntimeError('Frozen runtime guard dependency changed: ' + relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


GUARD = _load('recovery_capacity_exchange/exchange_guarded.py', 'exchange_runtime_frozen_guard')
NORMAL = _load('normal_capacity/run_cell.py', 'exchange_runtime_normal_constructor')
make_capture, MODES = GUARD.make_capture, GUARD.MODES


def _runtime_source(old, scheduler):
    path = NORMAL.PKG / 'staged_store_rotation.py'
    filename = str(path) + '[normal_capacity]'
    namespace = old.__globals__
    cap = namespace.get('NORMAL_FIXED_CAP')
    if (namespace.get('__name__') != 'normal_capacity_native_observation'
            or namespace.get('__file__') != str(path)
            or old.__code__.co_filename != filename
            or type(cap) is not int or not 1 <= cap <= 1024
            or cap != scheduler.max_num_running_reqs):
        raise RuntimeError('Unknown staged runtime construction or signed capacity')
    # This existing constructor checks the frozen staged source SHA itself.
    source = NORMAL.adapted_rotation_source().encode()
    compiled = compile(source, filename, 'exec', dont_inherit=True)
    expected_install = next(c for c in compiled.co_consts
                            if isinstance(c, CodeType) and c.co_name == 'install')
    live_install = namespace.get('install')
    if (not isinstance(live_install, FunctionType)
            or live_install.__globals__ is not namespace
            or not GUARD.FROZEN.HELPERS._same_code(expected_install, live_install.__code__)):
        raise RuntimeError('Live staged installer differs from pinned runtime construction')
    return filename, source, dict(
        normal_constructor_sha256=PINS['normal_capacity/run_cell.py'],
        staged_original_sha256=GUARD.STAGED_SHA,
        staged_runtime_source_sha256=hashlib.sha256(source).hexdigest(),
        namespace=namespace['__name__'], filename=filename, fixed_cap=cap,
        installer_code_verified=True,
        construction='normal_capacity.run_cell.normal_install_rotation -> adapted_rotation_source -> exec; only fixed-cap qualification differs from staged source')


# Reuse the frozen attach/install bodies. Only source acquisition and provenance
# change; _instrument, _authorize, closure replacement and teardown are original.
_source = inspect.getsource(GUARD._attach_guard)
_old = """    filename = inspect.getsourcefile(old)
    if filename is None: raise RuntimeError('Missing staged source')
    source = Path(filename).read_bytes()
    if hashlib.sha256(source).hexdigest()!=STAGED_SHA: raise RuntimeError('Staged source changed')"""
if _source.count(_old) != 1 or _source.count('    def callback(result,data,plan,now,step):') != 1:
    raise RuntimeError('Frozen narrow guard attach boundary changed')
_source = _source.replace(_old, '    filename, source, runtime = _runtime_source(old, scheduler)')
_source = _source.replace('    def callback(result,data,plan,now,step):',
    "    receipt['runtime_source_construction'] = runtime\n    def callback(result,data,plan,now,step):")
_namespace = dict(GUARD.__dict__, _runtime_source=_runtime_source)
exec(compile(_source, str(__file__) + '[frozen-attach]', 'exec'), _namespace)
_attach_guard = _namespace['_attach_guard']
install = FunctionType(GUARD.install.__code__, dict(GUARD.__dict__, _attach_guard=_attach_guard),
                       'install', GUARD.install.__defaults__, GUARD.install.__closure__)
install.__kwdefaults__ = GUARD.install.__kwdefaults__

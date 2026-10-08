"""Unrun interface correction: earlier native queue heads do not cancel the gate."""
import hashlib
from pathlib import Path

FROZEN_SHA256 = '81e1e3d1a1a4d13336188242876c6c29ab2e298da62c9ef3735af7104c0917f0'
FROZEN = Path(__file__).resolve().with_name('retry_defer.py')
source = FROZEN.read_text()
if hashlib.sha256(source.encode()).hexdigest() != FROZEN_SHA256:
    raise RuntimeError('Frozen retry-gate policy changed')
OLD = '''            if queue is not current.waiting or request is not active['target'] or current.skipped_waiting:
                release('NATIVE_HEAD_OR_QUEUE_CHANGED', now())
                return False'''
NEW = '''            if request is not active['target']:
                data['events'].append(dict(kind='native_queue_pass_through', step=step,
                    host_perf_s=now(), target=active['target'].request_id,
                    actual_head=request.request_id,
                    actual_queue='waiting' if queue is current.waiting else
                        'skipped_waiting' if queue is current.skipped_waiting else 'unknown',
                    skipped_order=[r.request_id for r in current.skipped_waiting]))
                return False'''
if source.count(OLD) != 1:
    raise RuntimeError('Expected exactly one frozen active queue cancellation branch')
adapted_source = source.replace(OLD, NEW)
COMPILED_SOURCE_SHA256 = hashlib.sha256(adapted_source.encode()).hexdigest()
namespace = dict(__name__='retry_defer_persistent_compiled', __file__=str(FROZEN))
exec(compile(adapted_source, str(FROZEN)+'[persistent_head]', 'exec'), namespace)
H = namespace['H']
_instrument = namespace['_instrument']
_decision = namespace['_decision']
_attach = namespace['_attach']


def install(scheduler, mode='native', *, selective):
    data, uninstall = namespace['install'](scheduler, mode, selective=selective)
    data['interface_adapter'] = dict(name='persistent_target_gate',
        frozen_source_sha256=FROZEN_SHA256, compiled_source_sha256=COMPILED_SOURCE_SHA256,
        scope='Only earlier native heads pass through while the target gate remains active; all original safety releases and 0.5s deadline retained.',
        timing_semantics='extra_gate_observed_s is observed gate lifetime, including native queue/gate-closed periods and overlapping native pass-through progress. It is not measured counterfactual added latency, saved latency, or an additive cost. The 0.5s deadline removes only this gate at the next observation.')
    return data, uninstall

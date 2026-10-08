"""Move one native connector finalization after existing sampling synchronization."""
from contextlib import contextmanager
import hashlib
import inspect
import sys
import time

RUNNER_SHA = '81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0'
MIXIN_SHA = 'd140e79e93b8670a2100b8b365606d3af402430a59c990ef65820da987ef4974'


def _guard(runner):
    from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    import vllm
    hashes = {}
    for cls, expected in ((type(runner), RUNNER_SHA), (KVConnectorModelRunnerMixin, MIXIN_SHA)):
        with open(inspect.getsourcefile(cls), 'rb') as source:
            observed = hashlib.sha256(source.read()).hexdigest()
        if observed != expected:
            raise RuntimeError(f'Unqualified native source: {cls.__name__}')
        hashes[cls.__name__] = observed
    model, parallel = runner.model_config, runner.parallel_config
    connector = get_kv_transfer_group()
    if (vllm.__version__ != '0.26.0' or type(runner).__name__ != 'GPUModelRunner'
            or runner.use_async_scheduling or runner.scheduler_config.async_scheduling is not False
            or runner.speculative_config is not None or runner.is_pooling_model
            or runner.broadcast_pp_output or runner.supports_mm_inputs
            or any(getattr(parallel, name) != 1 for name in
                   ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size'))
            or model.model not in ('allenai/OLMoE-1B-7B-0924', '/root/autodl-tmp/moe-a-20261002/hf/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5')
            or model.revision != '6d84c48581ece794365f2b8e9cfb043c68ade9c5'
            or str(runner.dtype) != 'torch.bfloat16'
            or type(connector).__name__ != 'OffloadingConnector'
            or type(connector.connector_worker.spec).__name__ != 'CPUOffloadingSpec'
            or type(connector.connector_worker.worker).__name__ != 'CPUOffloadingWorker'):
        raise RuntimeError('Requires pinned OLMoE BF16, single GPU, synchronous non-speculative generation and native CPU offload')
    return hashes


def install(engine, mode):
    runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    return _install_runner(runner, mode, _guard(runner))


def _install_runner(runner, mode, source_hashes):
    if mode not in ('native', 'after_sample'):
        raise ValueError('B_COMPLETION_FINALIZE must be native or after_sample')
    names = ('maybe_get_kv_connector_output', '_bookkeeping_sync', 'execute_model', 'sample_tokens')
    if any(name in vars(runner) for name in names):
        raise RuntimeError('Completion handoff requires unwrapped runner methods')
    originals = {name: getattr(runner, name) for name in names}
    pending = None
    active_row = None
    iteration = 0
    closed = False
    data = dict(mode=mode, status='INSTALLED', events=[], source_sha256=source_hashes,
        instrumented_context_enters=0, instrumented_native_finalizations=0,
        clock='host time.perf_counter; bookkeeping return is not a GPU transfer completion timestamp',
        extra_cuda_queries=0, extra_cuda_synchronizations=0,
        scope='Ordinary forward contexts only; no-forward keeps its original path; each instrumented native finalization runs once')

    def finish(cm, row, reason, error=(None, None, None)):
        data['instrumented_native_finalizations'] += 1
        row.update(finalize_reason=reason, finalize_begin_perf_s=time.perf_counter())
        try:
            return cm.__exit__(*error)
        finally:
            row['finalize_end_perf_s'] = time.perf_counter()

    def flush(reason, error=(None, None, None)):
        nonlocal pending
        if pending is not None:
            cm, row = pending
            pending = None  # Native-finalizer failure must never cause a second exit.
            return finish(cm, row, reason, error)

    @contextmanager
    def context(scheduler_output, defer_finalize=False):
        nonlocal pending, active_row
        if pending is not None:
            flush('reject_overlapping_context')
            raise RuntimeError('Pending connector context crossed a forward boundary')
        cm = originals['maybe_get_kv_connector_output'](scheduler_output, defer_finalize=defer_finalize)
        output = cm.__enter__()
        data['instrumented_context_enters'] += 1
        row = dict(iteration=iteration, enter_perf_s=time.perf_counter(), deferred=False)
        data['events'].append(row)
        active_row = row
        try:
            yield output
        except BaseException:
            if not finish(cm, row, 'forward_exception', sys.exc_info()):
                raise
        else:
            row['forward_exit_perf_s'] = time.perf_counter()
            if mode == 'after_sample' and not defer_finalize and scheduler_output.total_num_scheduled_tokens > 0:
                row['deferred'] = True
                pending = (cm, row)
            else:
                finish(cm, row, 'native_exit')

    def bookkeeping(*args, **kwargs):
        result = originals['_bookkeeping_sync'](*args, **kwargs)
        if active_row is not None:
            active_row['bookkeeping_return_perf_s'] = time.perf_counter()
        flush('after_bookkeeping')
        return result

    def guarded(name, *args, **kwargs):
        nonlocal iteration, active_row
        if name == 'execute_model':
            if pending is not None:
                flush('reject_next_iteration')
                raise RuntimeError('Unconsumed connector context before next iteration')
            iteration += 1
            active_row = None
        try:
            result = originals[name](*args, **kwargs)
            if pending is not None and (name == 'sample_tokens' or result is not None):
                flush('unexpected_runner_return')
                raise RuntimeError('Runner returned without the required bookkeeping handoff')
            return result
        except BaseException:
            flush(name + '_exception', sys.exc_info())
            raise

    runner.maybe_get_kv_connector_output = context
    runner._bookkeeping_sync = bookkeeping
    runner.execute_model = lambda *a, **k: guarded('execute_model', *a, **k)
    runner.sample_tokens = lambda *a, **k: guarded('sample_tokens', *a, **k)

    def uninstall():
        nonlocal closed
        if closed:
            return data
        try:
            flush('uninstall_cleanup')
        finally:
            for name in names:
                delattr(runner, name)
            closed = True
            data['status'] = 'UNINSTALLED'
        return data
    return data, uninstall

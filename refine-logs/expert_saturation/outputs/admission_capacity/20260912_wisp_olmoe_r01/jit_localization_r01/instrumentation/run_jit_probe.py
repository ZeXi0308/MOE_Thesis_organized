"""Observe Triton compiler/cache and pager host intervals; forward the native runner CLI."""
import argparse
import functools
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import runpy
import sys
import threading
import time
import traceback

EXPECTED = {
    'triton/__init__.py': '96e68192cab5dd11fa30d5dc95d56c29ceecda53477fc3af8f37bee026b7a021',
    'triton/runtime/jit.py': '0b70358d901afd70937a344a7811e4a2bcd10edca8a7c241e47842dc9ac660c8',
    'triton/compiler/compiler.py': '2a063bb7909777d3659945244277e900d9f7cfc35de9a758578469ee4b90fce9',
    'triton/runtime/cache.py': '0355c3e3452fa4500122244b8fff8864f22bc05c417a3d6f5dada9f2e92f8040',
    'vllm/utils/jit_monitor.py': 'cadf567b611e42424e3a2badd2d07da5d2047f14942ca139011b9fe4b707409d',
}


def plain(value):
    """Never stringify a tensor or read device contents."""
    if type(value) is float and not math.isfinite(value): return {'nonfinite_float': str(value)}
    if value is None or type(value) in (str, int, float, bool): return value
    if isinstance(value, (tuple, list)): return [plain(x) for x in value]
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    return {'unserialized_type': type(value).__name__}


def source_info(src):
    names = getattr(getattr(src, 'fn', None), 'arg_names', ())
    constants = {str(names[k[0]]) + str(k[1:]): plain(v)
                 for k, v in getattr(src, 'constants', {}).items()
                 if isinstance(k, tuple) and k and isinstance(k[0], int) and k[0] < len(names)}
    return dict(kernel=getattr(src, 'name', type(src).__name__),
                signature=plain(getattr(src, 'signature', None)), constexprs=constants)


class Probe:
    def __init__(self, pager):
        self.pager, self.events, self.undo = pager, [], []
        self.local, self.ids = threading.local(), itertools.count()

    def event(self, kind, **fields):
        active = getattr(self.local, 'apply', None)
        runtime = self.pager._runtime
        context = dict(active or {})
        if not active and runtime is not None:
            context.update(phase=runtime.context.get('phase'), step_id=runtime.context.get('step_id'))
        if active and runtime.records and runtime.records[-1]['call_id'] == active['call_id']:
            context['group_index'] = len(runtime.records[-1]['groups']) - 1
        row = dict(event_id=next(self.ids), kind=kind, thread_id=threading.get_ident(),
                   attribution='same_thread_apply' if active else 'no_active_apply', context=context, **fields)
        self.events.append(row)
        return row

    def timed(self, row, fn, *args, **kwargs):
        row['start_perf_ns'] = time.perf_counter_ns()
        try:
            result = fn(*args, **kwargs)
            row['status'] = 'complete'
            return result
        except BaseException as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            row['end_perf_ns'] = time.perf_counter_ns()

    def patch(self, obj, name, value):
        self.undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def install(self, triton, compiler, groups):
        knobs = triton.knobs
        prior_listener = knobs.compilation.listener

        def listener(**kw):
            row = getattr(self.local, 'compiler', None)
            if row is None: row = self.event('unwrapped_compiler_listener', **source_info(kw['src']))
            times = kw['times']
            row.update(cache_hit=kw['cache_hit'],
                       result_kind='disk_cache_load' if kw['cache_hit'] else 'compiler_pipeline',
                       cache_key=kw['metadata'].get('hash'),
                       native_times_us={k: plain(getattr(times, k)) for k in
                                        ('ir_initialization', 'lowering_stages', 'store_results')})
            if prior_listener is not None: return prior_listener(**kw)

        original_compile = compiler.compile

        @functools.wraps(original_compile)
        def compile_call(src, *a, **kw):
            row = self.event('compiler_call', cache_hit=None, result_kind='unclassified', **source_info(src))
            previous = getattr(self.local, 'compiler', None)
            self.local.compiler = row
            try: return self.timed(row, original_compile, src, *a, **kw)
            finally: self.local.compiler = previous

        original_handles = compiler.CompiledKernel._init_handles

        @functools.wraps(original_handles)
        def handles(kernel):
            if kernel.module is not None: return original_handles(kernel)
            row = self.event('launcher_and_driver_load', cache_key=kernel.hash, **source_info(kernel.src))
            return self.timed(row, original_handles, kernel)

        original_apply = groups._apply

        @functools.wraps(original_apply)
        def apply(runtime, method, layer, x, weights, ids):
            previous = getattr(self.local, 'apply', None)
            context = dict(call_id=runtime.next_call_id, layer=layer.layer_name,
                           phase=runtime.context.get('phase'), step_id=runtime.context.get('step_id'),
                           x_shape=list(x.shape), topk_shape=list(ids.shape),
                           private_cap=runtime.layers[id(layer)]['state'].cap_experts)
            self.local.apply = context
            row = self.event('pager_apply')
            context['apply_event_id'] = row['event_id']
            try: return self.timed(row, original_apply, runtime, method, layer, x, weights, ids)
            finally: self.local.apply = previous

        self.patch(knobs.compilation, 'listener', listener)
        self.patch(compiler, 'compile', compile_call)
        self.patch(importlib.import_module('triton.compiler'), 'compile', compile_call)
        self.patch(triton, 'compile', compile_call)
        self.patch(compiler.CompiledKernel, '_init_handles', handles)
        self.patch(groups, '_apply', apply)

    def close(self):
        for obj, name, old in reversed(self.undo): setattr(obj, name, old)
        self.undo.clear()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    destination = args.output / 'jit_probe.json'
    if destination.exists(): raise FileExistsError(destination)
    source = Path(__file__).resolve().parent.parent / 'source'
    sys.path.insert(0, str(source))
    report = dict(schema='triton_host_localization_v1', status='STARTED', events=[], installed_sources={},
                  invocation=[sys.executable, *sys.argv],
                  observer_start_perf_ns=time.perf_counter_ns(), observer_start_unix_ns=time.time_ns(),
                  cache_environment={k: v for k, v in os.environ.items() if k.startswith('TRITON_')},
                  instrumentation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Host intervals only. Compiler listener distinguishes actual pipeline from disk-cache load. '
                        'Pipeline includes cache writes; native_times_us uses wall time and includes intermediate I/O. '
                        'Handle initialization includes launcher construction and driver binary load, not GPU kernel duration. '
                        'Intervals nest inside pager_apply; never add them or deduct instrumentation as a speedup. '
                        'No-active-apply events have no reliable layer attribution, including any other-thread compilation. '
                        'No extra device copies, synchronization, threads, GC changes, or cache policy changes.')
    probe = None
    try:
        for name, expected in EXPECTED.items():
            package, relative = name.split('/', 1)
            path = Path(importlib.util.find_spec(package).origin).parent / relative
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            report['installed_sources'][name] = dict(path=str(path), sha256=actual)
            if actual != expected: raise RuntimeError('unreviewed installed source: ' + name)
        triton = importlib.import_module('triton')
        compiler = importlib.import_module('triton.compiler.compiler')
        pager = importlib.import_module('wisp_v026_adapter')
        groups = importlib.import_module('wisp_expert_groups')
        probe = Probe(pager)
        probe.install(triton, compiler, groups)
        runpy.run_path(str(source / 'run_native_pager.py'), run_name='__main__')
        report['status'] = 'COMPLETE'
    except BaseException:
        report.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        if probe is not None:
            probe.close()
            report['events'] = sorted(probe.events, key=lambda r: r['event_id'])
        report['observer_end_perf_ns'] = time.perf_counter_ns()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__': main()

"""Measurement-only host CPU/GC localization; forward the original covered CLI."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time
import traceback


def environment():
    result = dict(perf_ns=time.perf_counter_ns(), pid=os.getpid(),
                  native_thread_id=threading.get_native_id())
    for key, query in (
        ('sched_schedstats', lambda: Path('/proc/sys/kernel/sched_schedstats').read_text().strip()),
        ('cpu_affinity', lambda: sorted(os.sched_getaffinity(0))),
    ):
        try:
            result[key] = dict(status='observed', value=query())
        except (OSError, AttributeError) as exc:
            result[key] = dict(status='unknown', value=None, error=str(exc))
    return result


class MeasurementHooks:
    def __init__(self, runtime, engine, root, report):
        self.runtime, self.engine, self.root, self.report = runtime, engine, root, report
        self.restores, self.observer, self.engine_call = [], None, -1
        self.engine_event = None

    def span(self, kind, function, *args, **kwargs):
        if kind == 'engine':
            self.engine_call += 1
            record = {}
        else:
            record = self.runtime.records[-1]
            if self.engine_event is None or record['context']['phase'] != 'measurement':
                raise RuntimeError('layer span outside active measurement engine call')
            parent = self.report['spans'][self.engine_event]
            if not parent['context']:
                parent.update(context=dict(record['context']), rows=record['rows'])
        event = dict(kind=kind, event_id=len(self.report['spans']), engine_call=self.engine_call,
            parent_event_id=self.engine_event, call_id=record.get('call_id'),
            layer=record.get('layer_name'), rows=record.get('rows'),
            context=dict(record.get('context', {})), thread_id=threading.get_ident(),
            native_thread_id=threading.get_native_id(), status='started')
        self.report['spans'].append(event)
        if kind == 'engine':
            self.engine_event = event['event_id']
        event['start_perf_ns'] = time.perf_counter_ns()
        event['start_thread_cpu_ns'] = time.thread_time_ns()
        event['start_process_cpu_ns'] = time.process_time_ns()
        event['cpu_start_sample_end_perf_ns'] = time.perf_counter_ns()
        try:
            value = function(*args, **kwargs)
        except BaseException as exc:
            event.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        else:
            event['status'] = 'complete'
            return value
        finally:
            event['cpu_end_sample_start_perf_ns'] = time.perf_counter_ns()
            event['end_process_cpu_ns'] = time.process_time_ns()
            event['end_thread_cpu_ns'] = time.thread_time_ns()
            event['end_perf_ns'] = time.perf_counter_ns()
            if kind == 'engine':
                event['context_scope'] = 'first observed layer; empty if step made no layer call'
                self.engine_event = None

    def install(self):
        from runtime_variation_observer import RuntimeVariationObserver
        if Path(sys.modules[RuntimeVariationObserver.__module__].__file__).resolve() != self.root / 'source' / 'runtime_variation_observer.py':
            raise RuntimeError('runtime observer resolved outside frozen source')
        runtime, engine = self.runtime, self.engine
        namespace = runtime.apply.__func__.__globals__
        source = self.root / 'source' / 'run_shared_pool_pager.py'
        planner = namespace['plan_shared_pool']
        if (Path(namespace['__file__']).resolve() != source.resolve()
            or Path(planner.__code__.co_filename).resolve() != self.root / 'source' / 'shared_pool_plan.py'
            or runtime.shared_mode != 'oneshot'):
            raise RuntimeError('unexpected live runpy namespace, planner source or pool mode')
        self.report['live_namespace'] = dict(file=str(source), dispatch=runtime.apply.__func__.__qualname__,
            planner_file=planner.__code__.co_filename, planner=planner.__qualname__)
        self.report['environment_before'] = environment()
        self.observer = RuntimeVariationObserver()
        self.report['observer_start_perf_ns'] = time.perf_counter_ns()
        namespace['plan_shared_pool'] = lambda *a, **kw: self.span('planner', planner, *a, **kw)
        self.restores.append(('planner', lambda: namespace.__setitem__('plan_shared_pool', planner)))
        kernel = runtime.kernel
        runtime.kernel = lambda *a, **kw: self.span('kernel', kernel, *a, **kw)
        self.restores.append(('kernel', lambda: setattr(runtime, 'kernel', kernel)))
        step = engine.step
        own_step = getattr(engine, '__dict__', {}).get('step')
        had_step = 'step' in getattr(engine, '__dict__', {})
        engine.step = lambda *a, **kw: self.span('engine', step, *a, **kw)
        self.restores.append(('engine', lambda: setattr(engine, 'step', own_step) if had_step
                              else delattr(engine, 'step')))

    def close(self):
        for name, restore in reversed(self.restores):
            try:
                restore()
                self.report['restored_hooks'].append(name)
            except BaseException:
                self.report['restoration_errors'].append(dict(hook=name, error=traceback.format_exc()))
        if self.observer is not None:
            try:
                self.observer.close()
                self.report['observer_stop_perf_ns'] = time.perf_counter_ns()
                self.report['final_snapshot'] = self.observer.snapshot()
            except BaseException:
                self.report['restoration_errors'].append(dict(hook='gc', error=traceback.format_exc()))
            finally:
                self.report['gc_events'] = self.observer.events
        self.report['environment_after'] = environment()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--host-cost-observer', choices=['off', 'on'], required=True)
    args, rest = parser.parse_known_args()
    original = argparse.ArgumentParser(add_help=False)
    original.add_argument('--output', type=Path, required=True)
    output, _ = original.parse_known_args(rest)
    root = Path(__file__).resolve().parent.parent
    destination = output.output / 'host_cost_observation.json'
    if destination.exists():
        raise FileExistsError(destination)
    files = ['instrumentation/run_host_cost.py', 'instrumentation/run_covered.py',
             'source/runtime_variation_observer.py', 'source/native_capture.py',
             'source/run_shared_pool_pager.py', 'source/shared_pool_plan.py', 'source/wisp_v026_adapter.py']
    report = dict(schema='host_cost_observation_v1', mode=args.host_cost_observer, status='STARTED',
        started_perf_ns=time.perf_counter_ns(), spans=[], gc_events=[], measurement_captures=0,
        source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files},
        restored_hooks=[], restoration_errors=[], hooks_restored=False,
        environment_before=dict(status='unobserved', reason='observer off or measurement not reached'),
        environment_after=dict(status='unobserved', reason='observer off or measurement not reached'),
        scope='Only run_id=measurement is observed. Native engine cpu_delta includes before/after observer '
              'envelopes; narrow CPU sample endpoints lie inside recorded perf envelopes. Thread/process/rusage '
              'and GC/kernel/engine intervals overlap and must not be added or deducted as savings. '
              'Kernel spans are host calls, not GPU execution. GC callback timing is not pure removable overhead. '
              'Observer starts after compile setup/warmup and closes after capture, before export/shutdown. '
              'All wrapper allocations and recording remain in capture/process cost; off forwards unchanged arguments.')
    saved_argv, saved_path, native, capture = sys.argv, list(sys.path), None, None
    try:
        sys.path.insert(0, str(root / 'source'))
        native = importlib.import_module('native_capture')
        if Path(native.__file__).resolve() != root / 'source' / 'native_capture.py':
            raise RuntimeError('native_capture resolved outside frozen source')
        capture = native.capture_episode

        def observed_capture(engine, *a, **kw):
            if kw.get('run_id') != 'measurement':
                return capture(engine, *a, **kw)
            report['measurement_captures'] += 1
            if report['measurement_captures'] != 1:
                raise RuntimeError('expected one ordinary measurement capture')
            if args.host_cost_observer == 'off':
                raw = capture(engine, *a, **kw)
            else:
                pager = importlib.import_module('wisp_v026_adapter')
                if Path(pager.__file__).resolve() != root / 'source' / 'wisp_v026_adapter.py':
                    raise RuntimeError('pager resolved outside frozen source')
                hooks = MeasurementHooks(pager._runtime, engine, root, report)
                try:
                    hooks.install()
                    raw = capture(engine, *a, **dict(kw, cpu_diagnostics=True, runtime_observer=hooks.observer))
                finally:
                    hooks.close()
            report['capture_status'] = raw['status']
            return raw

        native.capture_episode = observed_capture
        sys.argv = [str(root / 'instrumentation' / 'run_covered.py'), *rest]
        runpy.run_path(sys.argv[0], run_name='__main__')
        if report['measurement_captures'] != 1 or report.get('capture_status') != 'COMPLETE' or report['restoration_errors']:
            raise RuntimeError('measurement incomplete or hook restoration failed')
        report['status'] = 'COMPLETE'
    except BaseException:
        report.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        if capture is not None:
            native.capture_episode = capture
        sys.argv, sys.path[:] = saved_argv, saved_path
        report['hooks_restored'] = not report['restoration_errors']
        report['finished_perf_ns'] = time.perf_counter_ns()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()

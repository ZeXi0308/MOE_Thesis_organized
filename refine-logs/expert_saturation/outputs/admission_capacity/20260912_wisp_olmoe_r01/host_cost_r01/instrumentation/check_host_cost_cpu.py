"""Preserve the focused host-hook CPU fixture; no model, torch or CUDA import."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'source'))
spec = importlib.util.spec_from_file_location('host_cost_cpu_check', ROOT / 'instrumentation/run_host_cost.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
import native_capture
import wisp_v026_adapter
from shared_pool_plan import plan_shared_pool
from analyze_layer_budget import LRU

NAMESPACE = {'__file__': str(ROOT / 'source/run_shared_pool_pager.py'), 'plan_shared_pool': plan_shared_pool}
exec('def dispatch(self): pass', NAMESPACE)


class Engine:
    def __init__(self, runtime, fail=False, layers=(13, 14)):
        self.runtime, self.fail, self.layers, self.count = runtime, fail, layers, 0

    def step(self):
        runtime = self.runtime
        runtime.context = dict(phase='measurement', step_id=self.count, valid_row_stop=3)
        self.count += 1
        for layer in self.layers:
            runtime.records.append(dict(call_id=len(runtime.records), context=dict(runtime.context),
                                        layer_name=f'model.layers.{layer}.mlp.experts', rows=3))
            NAMESPACE['plan_shared_pool']([0, 1], LRU(21).snapshot(), -1 if self.fail else layer)
            assert runtime.kernel() == 7
        return ['ok']


def runtime():
    result = SimpleNamespace(shared_mode='oneshot', kernel=lambda: 7, records=[], context={})
    result.apply = MethodType(NAMESPACE['dispatch'], result)
    return result


def check_spans(failed):
    state = runtime()
    engine, kernel, callbacks = Engine(state, failed), state.kernel, list(gc.callbacks)
    report = dict(spans=[], gc_events=[], restored_hooks=[], restoration_errors=[])
    hooks = probe.MeasurementHooks(state, engine, ROOT, report)
    try:
        hooks.install()
        if failed:
            try:
                engine.step()
            except ValueError as error:
                assert 'invalid layer' in str(error)
            else:
                raise AssertionError('real planner failure not propagated')
        else:
            assert engine.step() == ['ok'] and engine.step() == ['ok']
    finally:
        hooks.close()
    assert NAMESPACE['plan_shared_pool'] is plan_shared_pool
    assert state.kernel is kernel and 'step' not in engine.__dict__
    assert gc.callbacks == callbacks and not report['restoration_errors']
    spans = report['spans']
    assert len(spans) == (2 if failed else 10)
    assert [s['event_id'] for s in spans] == list(range(len(spans)))
    for s in spans:
        assert s['start_perf_ns'] <= s['cpu_start_sample_end_perf_ns'] <= s['cpu_end_sample_start_perf_ns'] <= s['end_perf_ns']
        assert s['end_thread_cpu_ns'] >= s['start_thread_cpu_ns'] and s['end_process_cpu_ns'] >= s['start_process_cpu_ns']
        assert s['status'] == ('failed' if failed else 'complete')
    assert [s['engine_call'] for s in spans if s['kind'] == 'engine'] == ([0] if failed else [0, 1])
    return dict(failure=failed, status='PASS', spans=len(spans), actual_planner=True, restored=True)


def check_cli(base, mode, failed):
    output, state = base / (mode + str(failed)), runtime()
    engine, callbacks = Engine(state, failed, (13,)), list(gc.callbacks)
    wisp_v026_adapter._runtime = state

    def fake_capture(engine, *a, **kw):
        if kw['run_id'] == 'warmup':
            assert 'step' not in engine.__dict__ and gc.callbacks == callbacks
            assert kw == dict(run_id='warmup', cpu_diagnostics=False, runtime_observer=None)
        else:
            assert ('step' in engine.__dict__) == (mode == 'on')
            assert kw.get('cpu_diagnostics', False) == (mode == 'on')
            assert (kw.get('runtime_observer') is not None) == (mode == 'on')
            engine.step()
        return dict(status='COMPLETE')

    native_capture.capture_episode = fake_capture
    rest = ['--pool-execution', 'oneshot', '--output', str(output), '--model', 'literal spaces and $value']

    def fake_covered(path, run_name):
        assert path == str(ROOT / 'instrumentation/run_covered.py') and run_name == '__main__'
        assert sys.argv == [path, *rest]
        native_capture.capture_episode(engine, run_id='warmup', cpu_diagnostics=False, runtime_observer=None)
        native_capture.capture_episode(engine, run_id='measurement', runtime_observer=None)

    probe.runpy.run_path = fake_covered
    sys.argv = ['run_host_cost.py', '--host-cost-observer', mode, *rest]
    argv = sys.argv
    try:
        probe.main()
    except ValueError:
        assert failed
    else:
        assert not failed
    assert sys.argv is argv and native_capture.capture_episode is fake_capture
    assert gc.callbacks == callbacks and 'step' not in engine.__dict__
    path = output / 'host_cost_observation.json'
    before = path.read_bytes()
    report = json.loads(before)
    assert report['status'] == ('FAILED' if failed else 'COMPLETE') and report['hooks_restored']
    assert len(report['spans']) == (0 if mode == 'off' else 2 if failed else 3)
    try:
        probe.main()
    except FileExistsError:
        pass
    else:
        raise AssertionError('exclusive output not enforced')
    assert path.read_bytes() == before
    return dict(mode=mode, failure=failed, status='PASS', json=str(path), exclusive_unchanged=True)


if __name__ == '__main__':
    saved = sys.argv, native_capture.capture_episode, wisp_v026_adapter._runtime, probe.runpy.run_path
    base = Path(tempfile.mkdtemp(prefix='host-cost-cli-cpu-'))
    try:
        spans = [check_spans(failed) for failed in (False, True)]
        cli = [check_cli(base, mode, failed) for mode, failed in [('off', False), ('on', False), ('on', True)]]
        compile((ROOT / 'instrumentation/run_host_cost.py').read_text(), 'run_host_cost.py', 'exec')
        files = ['instrumentation/run_host_cost.py', 'instrumentation/check_host_cost_cpu.py',
                 'source/shared_pool_plan.py', 'source/analyze_layer_budget.py', 'source/runtime_variation_observer.py',
                 'source/native_capture.py', 'source/wisp_v026_adapter.py', 'source/run_shared_pool_pager.py']
        assert not any(name in sys.modules for name in ('torch', 'vllm'))
        print(json.dumps(dict(status='PASS', syntax='PASS', spans=spans, cli=cli,
            source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
            scope='Only preserved real CPU planner/hook and stub CLI checks; no GPU or compile-domain rerun.'), indent=2))
    finally:
        sys.argv, native_capture.capture_episode, wisp_v026_adapter._runtime, probe.runpy.run_path = saved

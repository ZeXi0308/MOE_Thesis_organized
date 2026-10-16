"""CPU protocol mocks for graph ownership and cache/timing boundaries.

These checks do not execute CUDA or establish graph-capture compatibility of
the installed kernel. That remains an explicitly separate GPU diagnostic.
"""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('bench_under_test',
    Path(__file__).resolve().parents[1]/'code/bench_moe_routes.py')
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def fake_torch(events, match=True):
    class Tensor:
        def clone(self):
            events.append('clone_reference')
            return Tensor()
    class Stream:
        def __init__(self,device=None):pass
        def wait_stream(self,other):events.append('wait_stream')
    class Graph:
        def __init__(self):events.append('create_graph')
        def replay(self):events.append('replay')
        def reset(self):events.append('reset_graph')
    class Event:
        def __init__(self,enable_timing):pass
        def record(self):events.append('event_record')
        def synchronize(self):events.append('event_sync')
        def elapsed_time(self,other):return 1.0
    @contextlib.contextmanager
    def stream_context(stream):
        events.append('side_stream_begin')
        yield
        events.append('side_stream_end')
    @contextlib.contextmanager
    def graph_context(graph,stream):
        events.append('capture_begin')
        yield
        events.append('capture_end')
    class Profiler:
        def __enter__(self):
            events.append('profile_begin')
            return self
        def __exit__(self,*args):events.append('profile_end')
        def export_chrome_trace(self,path):
            Path(path).write_text(json.dumps({'traceEvents':[{'cat':'kernel','name':'same_kernel','dur':5}]}))
    torch=NS(cuda=NS(Stream=Stream,current_stream=lambda:Stream(),stream=stream_context,
        CUDAGraph=Graph,graph=graph_context,synchronize=lambda:events.append('sync'),Event=Event),
        allclose=lambda *a,**kw:match,
        profiler=NS(profile=lambda **kw:Profiler(),ProfilerActivity=NS(CPU='cpu',CUDA='cuda')))
    def native():
        events.append('native_call')
        return Tensor()
    cache=NS(add_=lambda value:events.append('turnover'),numel=lambda:64,element_size=lambda:4)
    return torch,native,cache


class BenchExecutionTests(unittest.TestCase):
    def test_graph_warms_then_captures_and_replay_has_no_native_python_call(self):
        events=[]
        torch,native,_=fake_torch(events)
        with patch.dict(sys.modules,{'torch':torch}):
            execution=bench.CaseExecution(native,'cudagraph','cuda:0')
            execution.prepare(3)
            self.assertEqual(events.count('native_call'),4)
            self.assertLess(events.index('side_stream_end'),events.index('capture_begin'))
            self.assertLess(events.index('clone_reference'),events.index('capture_begin'))
            self.assertLess(events.index('capture_end'),events.index('replay'))
            execution.call()
            self.assertEqual(events.count('native_call'),4)
            self.assertEqual(events.count('replay'),2)
            self.assertTrue(execution.metadata['matches_eager_warmup'])
            self.assertTrue(execution.metadata['capture_succeeded'])
            self.assertGreaterEqual(execution.metadata['capture_wall_ms'],0)
            execution.close()
            self.assertEqual(events[-1],'reset_graph')
            self.assertIsNone(execution.graph)
            self.assertIsNone(execution.output)
            self.assertIsNone(execution.call)

    def test_eager_is_native_and_failed_graph_comparison_can_be_released(self):
        events=[]
        torch,native,_=fake_torch(events,match=False)
        with patch.dict(sys.modules,{'torch':torch}):
            execution=bench.CaseExecution(native,'eager','cuda:0')
            execution.prepare(2)
            self.assertIs(execution.call,native)
            self.assertNotIn('create_graph',events)
            execution.close()
            execution=bench.CaseExecution(native,'cudagraph','cuda:0')
            try:
                with self.assertRaisesRegex(RuntimeError,'disagrees'):
                    execution.prepare(2)
            finally:
                execution.close()
            self.assertEqual(events[-1],'reset_graph')

    def test_cache_turnover_is_outside_timers_and_profile(self):
        events=[]
        torch,native,cache=fake_torch(events)
        with patch.dict(sys.modules,{'torch':torch}),tempfile.TemporaryDirectory() as directory:
            repeats=bench.time_repeat(native,2,cache)
            self.assertEqual(events,['turnover','sync','event_record','native_call','event_record','event_sync']*2)
            self.assertEqual(repeats['cuda_event_ms_calls'],[1.,1.])
            events.clear()
            profile=bench.profile_once(native,Path(directory)/'profile.json',cache,'cudagraph')
            self.assertEqual(events,['turnover','sync','profile_begin','native_call','sync','profile_end'])
            self.assertEqual(profile['execution_mode'],'cudagraph')
            self.assertEqual(profile['cache_policy'],'turnover')
            self.assertTrue(profile['cache_turnover_before_profile'])
            self.assertEqual(profile['cache_turnover_bytes'],256)
            self.assertEqual(profile['kernel_duration_sum_us'],5)


if __name__=='__main__':unittest.main()

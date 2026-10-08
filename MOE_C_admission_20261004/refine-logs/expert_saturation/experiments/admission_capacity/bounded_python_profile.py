"""One bounded cProfile + GC window; diagnostic timing, no CUDA profiler."""
import cProfile
import gc
import json
import pstats
import sys
import time
from pathlib import Path


class Window:
    def __init__(self, engine, output, start=500, count=32):
        if sys.getprofile() is not None:
            raise RuntimeError('Refuse to replace an existing Python profiler')
        self.engine, self.output = engine, Path(output)
        self.start, self.end = start, start+count
        self.original, self.was_local = engine.step, 'step' in vars(engine)
        self.index = 0
        self.active = self.exported = False
        self.profiler = cProfile.Profile()
        self.calls, self.gc_events, self.spans = [], [], []
        engine.step = self.step

    def gc_callback(self, phase, info):
        self.gc_events.append(dict(phase=phase, info=dict(info),
                                   call=self.index, time_ns=time.perf_counter_ns()))

    def close(self):
        if not self.active:
            return
        self.profiler.disable()
        self.active = False
        if self.gc_callback in gc.callbacks:
            gc.callbacks.remove(self.gc_callback)
        self.profiler.dump_stats(str(self.output.with_suffix('.pstats')))
        stats = pstats.Stats(self.profiler)
        rows = []
        for (filename, line, name), (primitive, total, self_s, cumulative_s, callers) in stats.stats.items():
            rows.append(dict(file=filename, line=line, name=name,
                             primitive_calls=primitive, total_calls=total,
                             self_s=self_s, cumulative_s=cumulative_s,
                             callers=[dict(file=k[0], line=k[1], name=k[2], values=v)
                                      for k,v in callers.items()]))
        self.output.write_text(json.dumps(dict(kind='CPROFILE_GC_DIAGNOSTIC',
            calls=self.calls, spans=self.spans, gc_events=self.gc_events,
            functions=rows, gc_enabled=gc.isenabled(), gc_threshold=gc.get_threshold()), indent=2)+'\n')
        self.exported = True

    def step(self, *args, **kwargs):
        n = self.index
        if n == self.start:
            gc.callbacks.append(self.gc_callback)
            self.active = True
            self.profiler.enable()
        started = time.perf_counter_ns() if self.active else None
        try:
            result = self.original(*args, **kwargs)
            if self.active:
                self.calls.append(n)
            return result
        finally:
            if self.active:
                self.spans.append(dict(call=n, start_ns=started,
                                       end_ns=time.perf_counter_ns()))
            self.index += 1
            if self.active and (n+1 == self.end or sys.exc_info()[0] is not None):
                self.close()

    def uninstall(self):
        try:
            self.close()
        finally:
            if self.was_local:
                self.engine.step = self.original
            else:
                delattr(self.engine, 'step')
        return dict(start=self.start, end_exclusive=self.end,
                    recorded_calls=self.calls, exported=self.exported,
                    complete=self.calls==list(range(self.start,self.end)),
                    scope='Python call/GC diagnostic. No CUDA activity trace. Profiled time is not a performance result.')


def install(engine, output):
    return Window(engine, output)

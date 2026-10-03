"""One fixed CPU/CUDA activity window. All request timing is diagnostic."""
import sys
from pathlib import Path


class Window:
    def __init__(self, engine, output, start, count, factory, scope):
        self.engine, self.output = engine, Path(output)
        self.start, self.end = start, start+count
        self.factory, self.scope = factory, scope
        self.original = engine.step
        self.was_local = 'step' in vars(engine)
        self.index = 0
        self.profiler = None
        self.active = False
        self.exported = False
        self.calls = []
        engine.step = self.step

    def close(self, exc=(None, None, None)):
        if self.active:
            self.active = False
            self.profiler.__exit__(*exc)
            self.profiler.export_chrome_trace(str(self.output))
            self.exported = True

    def step(self, *args, **kwargs):
        n = self.index
        if n == self.start:
            self.profiler = self.factory()
            self.profiler.__enter__()
            self.active = True
        try:
            if self.active:
                with self.scope('decode_call_'+str(n)):
                    result = self.original(*args, **kwargs)
                self.calls.append(n)
                return result
            return self.original(*args, **kwargs)
        finally:
            self.index += 1
            if self.active and (n+1 == self.end or sys.exc_info()[0] is not None):
                self.close(sys.exc_info())

    def uninstall(self):
        try:
            self.close()
        finally:
            if self.was_local:
                self.engine.step = self.original
            else:
                delattr(self.engine, 'step')
        return dict(start=self.start, end_exclusive=self.end, recorded_calls=self.calls,
                    exported=self.exported, complete=self.calls==list(range(self.start,self.end)),
                    scope='Profiler activation/export can synchronize and perturb all timing. Diagnostic only.')


def install(engine, output):
    import torch
    return Window(engine, output, 500, 32,
                  lambda: torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                      torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                      profile_memory=False, with_stack=False), torch.profiler.record_function)

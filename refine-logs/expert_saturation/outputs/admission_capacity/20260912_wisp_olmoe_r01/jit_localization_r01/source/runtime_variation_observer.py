"""Passive host observations; their work remains inside the measured wall time.

GC durations are callback start/stop envelopes, not subtractable pure GC cost.
scaling_cur_freq is a driver-exported sample, not an entire step's CPU frequency.
"""
import ctypes
import gc
import os
from pathlib import Path
import sys
import threading
import time


class RuntimeVariationObserver:
    def __init__(self):
        self.events, self._active = [], {}
        self._totals = [dict(starts=0, completed=0, duration_ns=0, collected=0,
                             uncollectable=0, unmatched_stops=0) for _ in range(3)]
        self._sched_getcpu = None
        if sys.platform.startswith("linux"):
            try:
                self._sched_getcpu = ctypes.CDLL(None).sched_getcpu
                self._sched_getcpu.argtypes, self._sched_getcpu.restype = [], ctypes.c_int
            except (OSError, AttributeError):
                pass
        self._callback = self._on_gc
        gc.callbacks.append(self._callback)

    def _on_gc(self, phase, info):
        stamp, thread_id = time.perf_counter_ns(), threading.get_ident()
        generation = info["generation"]
        key, totals = (thread_id, generation), self._totals[generation]
        if phase == "start":
            event = dict(generation=generation, thread_id=thread_id,
                native_thread_id=threading.get_native_id(), start_perf_ns=stamp,
                stop_perf_ns=None, duration_ns=None, collected=None, uncollectable=None)
            self.events.append(event)
            self._active[key] = event
            totals["starts"] += 1
        elif phase == "stop":
            event = self._active.pop(key, None)
            if event is None:
                event = dict(generation=generation, thread_id=thread_id,
                    native_thread_id=threading.get_native_id(), start_perf_ns=None, duration_ns=None)
                self.events.append(event)
                totals["unmatched_stops"] += 1
            else:
                event["duration_ns"] = stamp - event["start_perf_ns"]
                totals["duration_ns"] += event["duration_ns"]
                totals["completed"] += 1
            event.update(stop_perf_ns=stamp, collected=info["collected"],
                         uncollectable=info["uncollectable"])
            totals["collected"] += info["collected"]
            totals["uncollectable"] += info["uncollectable"]

    def _core(self):
        try:
            value = self._sched_getcpu() if self._sched_getcpu is not None else -1
            return value if value >= 0 else None
        except (OSError, ValueError):
            return None

    def _cpu_sample(self):
        before, frequency, error = self._core(), None, None
        source = (f"/sys/devices/system/cpu/cpu{before}/cpufreq/scaling_cur_freq"
                  if before is not None else None)
        if source:
            try:
                frequency = int(Path(source).read_text().strip())
                if frequency <= 0:
                    raise ValueError("nonpositive frequency")
            except (OSError, ValueError) as exc:
                frequency, error = None, type(exc).__name__
        after = self._core()
        known = before is not None and after is not None
        migrated = before != after if known else None
        return dict(core_before=before, core_after=after, migrated=migrated,
            scaling_cur_freq_khz=frequency, source=source, error=error,
            status="unknown" if not known or frequency is None else "migrated" if migrated else "observed",
            meaning="driver-exported sample for core_before; not whole-step actual frequency")

    def snapshot(self):
        started = time.perf_counter_ns()
        cpu, rss, rss_error = self._cpu_sample(), None, None
        try:
            rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError) as exc:
            rss_error = type(exc).__name__
        result = dict(start_perf_ns=started, cpu=cpu,
            rss=dict(bytes=rss, status="observed" if rss is not None else "unknown",
                     source="/proc/self/statm resident pages * SC_PAGE_SIZE", error=rss_error),
            gc=dict(enabled=gc.isenabled(), count=list(gc.get_count()), threshold=list(gc.get_threshold()),
                totals_by_generation=[dict(v) for v in self._totals], events_count=len(self.events),
                active=[dict(e) for e in self._active.values()],
                timing_scope="callback start/stop envelope; not subtractable pure overhead"))
        result["end_perf_ns"] = time.perf_counter_ns()
        return result

    def close(self):
        """Detach only this observer; keep complete and in-progress event records."""
        for index, callback in enumerate(gc.callbacks):
            if callback is self._callback:
                del gc.callbacks[index]
                break
        return self.events

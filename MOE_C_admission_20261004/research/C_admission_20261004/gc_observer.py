"""Passive CPython GC callback timing; never changes collection policy."""
import gc
import time


class GCObserver:
    def __init__(self):
        self.phase = "ingress"
        self.events = []
        self.errors = []
        self.callback_count = 0
        self.callback_wall_s = 0.0
        self.callback_cpu_s = 0.0
        self._pending = {}
        self._callback = None
        self.installed_perf_s = self.removed_perf_s = None
        self.enabled_at_install = self.thresholds_at_install = None

    def install(self):
        if self._callback is not None:
            raise RuntimeError("GC observer already installed")
        self.enabled_at_install = gc.isenabled()
        self.thresholds_at_install = list(gc.get_threshold())
        self.installed_perf_s = time.perf_counter()
        self._callback = self._on_gc
        gc.callbacks.append(self._callback)

    def _on_gc(self, action, info):
        entered = time.perf_counter()
        cpu_entered = time.thread_time()
        self.callback_count += 1
        try:
            generation = info["generation"]
            if action == "start":
                event = dict(generation=generation, start_perf_s=entered,
                    end_perf_s=None, phase=self.phase, stop_phase=None,
                    collected=None, uncollectable=None)
                self.events.append(event)
                # A repeated start leaves the earlier unpaired event visible.
                self._pending[generation] = event
            elif action == "stop":
                event = self._pending.pop(generation, None)
                if event is None:
                    event = dict(generation=generation, start_perf_s=None,
                        end_perf_s=None, phase=None, stop_phase=None,
                        collected=None, uncollectable=None)
                    self.events.append(event)
                event.update(end_perf_s=entered, stop_phase=self.phase,
                    collected=info["collected"], uncollectable=info["uncollectable"])
            else:
                raise ValueError("unexpected GC callback action: " + str(action))
        except Exception as exc:
            self.errors.append(dict(perf_s=entered, action=action, phase=self.phase,
                error=f"{type(exc).__name__}: {exc}"))
        finally:
            self.callback_cpu_s += time.thread_time() - cpu_entered
            self.callback_wall_s += time.perf_counter() - entered

    def close(self):
        if self._callback is None:
            return
        try:
            gc.callbacks.remove(self._callback)
        except ValueError:
            self.errors.append(dict(perf_s=time.perf_counter(), action="remove",
                phase=self.phase, error="GC observer callback was already absent"))
        finally:
            self._callback = None
            self.removed_perf_s = time.perf_counter()

    def report(self, origin_perf_s):
        return dict(schema_version=1, origin_perf_s=origin_perf_s,
            installed_perf_s=self.installed_perf_s, removed_perf_s=self.removed_perf_s,
            enabled_at_install=self.enabled_at_install,
            thresholds_at_install=self.thresholds_at_install, events=self.events,
            unpaired_start_count=sum(e["end_perf_s"] is None for e in self.events),
            unpaired_stop_count=sum(e["start_perf_s"] is None for e in self.events),
            error_count=len(self.errors), errors=self.errors,
            callback_count=self.callback_count, callback_wall_s=self.callback_wall_s,
            callback_cpu_s=self.callback_cpu_s, cpu_clock="thread_time",
            semantics="Absolute perf_counter callback boundaries, not pure GC CPU time. "
                "Phase is the host measurement phase at callback entry. Callback overhead "
                "is observer work only; no time is subtracted from service measurements. "
                "No GC policy is changed; other callbacks are preserved.")

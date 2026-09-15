"""Exclusive main-thread wall/CPU spans; no CUDA events or policy changes."""
import threading
import time


class Spans:
    def __init__(self):
        self.rows = []
        self.stack = []
        self.originals = []
        self.owner = threading.get_ident()
        self.step = -1

    def wrap(self, obj, name, label):
        original = getattr(obj, name)
        was_local = name in vars(obj)
        def call(*args, **kwargs):
            if threading.get_ident() != self.owner:
                return original(*args, **kwargs)
            if label == 'engine.step':
                self.step += 1
            frame = [time.perf_counter(), time.thread_time(), 0., 0.]
            self.stack.append(frame)
            try:
                return original(*args, **kwargs)
            finally:
                wall, cpu = time.perf_counter()-frame[0], time.thread_time()-frame[1]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][2] += wall
                    self.stack[-1][3] += cpu
                self.rows.append(dict(step=self.step, label=label, wall_s=wall, thread_s=cpu,
                                      exclusive_wall_s=wall-frame[2], exclusive_thread_s=cpu-frame[3]))
        setattr(obj, name, call)
        self.originals.append((obj, name, original, was_local))

    def uninstall(self):
        for obj, name, original, was_local in reversed(self.originals):
            if was_local:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        self.originals.clear()


def install(engine):
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    scheduler = engine.engine_core.engine_core.scheduler.connector.connector_scheduler
    worker = get_kv_transfer_group().connector_worker
    bindings = [(engine, 'step', 'engine.step')]
    for name in ['build_connector_meta', '_build_store_jobs', '_update_req_states',
                 'get_num_new_matched_tokens', 'update_connector_output']:
        bindings.append((scheduler, name, 'scheduler.'+name))
    for name in ['handle_preemptions', 'start_kv_transfers', 'prepare_store_kv', 'get_finished']:
        bindings.append((worker, name, 'worker.'+name))
    for obj, name, _ in bindings:
        if not callable(getattr(obj, name)):
            raise ValueError('missing native method '+name)
    spans = Spans()
    try:
        for obj, name, label in bindings:
            spans.wrap(obj, name, label)
    except Exception:
        spans.uninstall()
        raise
    return spans

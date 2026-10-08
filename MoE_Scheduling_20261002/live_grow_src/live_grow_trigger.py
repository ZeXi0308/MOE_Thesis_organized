"""One forced diagnostic action after an observed native allocation failure.

No future lengths, replayed routes or learned threshold are used. This is an
interface/cost probe, not a performance-qualified online allocation policy.
"""
import json
import time


class FirstAllocationFailureGrow:
    def __init__(self, engine, runner, runtime, path):
        self.engine, self.runner, self.runtime, self.path = engine, runner, runtime, path
        self.manager = engine.engine_core.engine_core.scheduler.kv_cache_manager
        self.original = self.manager.allocate_slots
        self.had_instance_method = "allocate_slots" in vars(self.manager)
        self.pending = None
        self.attempted = False
        self.failure = None  # Retain a failed transaction's CPU snapshot until exit.
        self.data = dict(status="ARMED", trigger="first_actual_allocate_slots_failure",
                         source_cap=24, target_cap=23, target_kv_bytes=1275068416,
                         failures=[], action=None)

        def observed_allocate(*args, **kwargs):
            value = self.original(*args, **kwargs)
            if value is None and not self.attempted:
                req = args[0] if args else kwargs.get("request")
                entry = dict(request_id=getattr(req, "request_id", None),
                             computed_tokens=getattr(req, "num_computed_tokens", None),
                             free_blocks=self.manager.block_pool.get_num_free_blocks(),
                             num_new_tokens=args[1] if len(args) > 1 else kwargs.get("num_new_tokens"))
                self.data["failures"].append(entry)
                if self.pending is None:
                    self.pending = entry
            return value

        self.manager.allocate_slots = observed_allocate

    def after_step(self, call, now):
        if self.pending is None or self.attempted:
            return
        self.attempted = True
        from live_kv_grow import grow
        action = self.data["action"] = dict(after_engine_call=call["index"],
                                            start_s=now(), trigger=self.pending)
        start = time.perf_counter()
        self.data["status"] = "RUNNING"
        try:
            receipt = grow(self.engine, self.runner, self.runtime,
                           target_cap=23, target_kv_bytes=1275068416)
            action["transaction"] = receipt
            self.data["status"] = "COMPLETE"
        except Exception as exc:
            self.failure = exc
            action["error"] = f"{type(exc).__name__}: {exc}"
            action["transaction"] = getattr(exc, "receipt", None)
            self.data["status"] = "FAILED"
            raise
        finally:
            action.update(end_s=now(), transaction_call_s=time.perf_counter()-start)
            self.save()

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")

    def close(self):
        if self.had_instance_method:
            self.manager.allocate_slots = self.original
        else:
            del self.manager.allocate_slots
        if not self.attempted:
            self.data["status"] = "NOT_TRIGGERED"
        self.save()

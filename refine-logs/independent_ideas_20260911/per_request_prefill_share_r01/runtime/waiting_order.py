"""Reorder only native FCFS requests that have never started execution."""


class WaitingBypassLedger:
    """Count later arrivals first scheduled before an older waiting request."""

    def __init__(self):
        self.rank, self.bypasses = {}, {}

    def register(self, request_id):
        if request_id in self.rank:
            raise ValueError("duplicate arrival in bypass ledger")
        self.rank[request_id] = len(self.rank)
        self.bypasses[request_id] = 0

    def order(self, requests):
        pending = sorted(requests, key=lambda r: self.rank[r.request_id])
        virtual = self.bypasses.copy()  # Proposal only: no service has occurred yet.
        ordered = []
        while pending:
            barrier = next((i for i, r in enumerate(pending)
                            if virtual[r.request_id] >= 1), len(pending) - 1)
            chosen = min(pending[:barrier + 1],
                         key=lambda r: (r.num_prompt_tokens, self.rank[r.request_id]))
            index = pending.index(chosen)
            for older in pending[:index]:
                virtual[older.request_id] += 1
            ordered.append(pending.pop(index))
        return ordered

    def record(self, scheduled_tokens, ever_scheduled, mode, event):
        pending = {r["request_id"] for r in event["before"]}
        event.update(first_admitted=[], bypass_charges=[],
                     bypasses_before={rid: self.bypasses[rid] for rid in sorted(pending)})
        try:
            for rid, amount in scheduled_tokens.items():
                if amount <= 0 or rid in ever_scheduled:
                    continue
                if rid not in pending:
                    raise ValueError("first admission absent from captured waiting queue")
                pending.remove(rid)
                event["first_admitted"].append(rid)
                for older in sorted(pending, key=self.rank.__getitem__):
                    if self.rank[older] < self.rank[rid]:
                        self.bypasses[older] += 1
                        event["bypass_charges"].append(dict(older=older, admitted=rid))
            if mode == "bounded_bypass_once" and any(n > 1 for n in self.bypasses.values()):
                raise ValueError("native first-admission order exceeded one bypass")
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event["bypasses_after"] = {rid: self.bypasses[rid] for rid in event["bypasses_before"]}


def reorder_waiting(scheduler, mode, ever_scheduled, event, ledger=None):
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
    from vllm.v1.request import RequestStatus

    waiting, running = scheduler.waiting, scheduler.running
    before = list(waiting)
    snapshot = lambda: [(id(r), r.request_id, r.status, r.num_prompt_tokens,
                         r.num_computed_tokens, r.num_output_tokens) for r in scheduler.running]
    running_state = snapshot()
    describe = lambda requests: [dict(request_id=r.request_id, prompt_tokens=r.num_prompt_tokens,
        computed_tokens=r.num_computed_tokens, output_tokens=r.num_output_tokens, status=r.status.name) for r in requests]
    event.update(mode=mode, before=describe(before), after=None, running_before=describe(running),
                 action_applied=False, order_changed=False, error=None)
    try:
        if mode not in ("fcfs", "short_prompt_first", "bounded_bypass_once"):
            raise ValueError("unsupported waiting order")
        if type(waiting) is not FCFSRequestQueue or scheduler.policy is not SchedulingPolicy.FCFS:
            raise ValueError("waiting action requires the native FCFS deque and policy")
        if scheduler.skipped_waiting:
            raise ValueError("skipped_waiting is outside the frozen waiting-order regime")
        if len({r.request_id for r in before}) != len(before) or {id(r) for r in before} & {id(r) for r in running}:
            raise ValueError("waiting identity duplicates or overlaps running")
        if any(r.status is not RequestStatus.WAITING or r.request_id in ever_scheduled
               or r.num_computed_tokens != 0 or r.num_output_tokens != 0
               or getattr(r, "num_preemptions", 0) or r.num_prompt_tokens <= 0 for r in before):
            raise ValueError("waiting action requires never-started, unblocked requests")
        ordered = sorted(before, key=lambda r: r.num_prompt_tokens) if mode == "short_prompt_first" else before
        if mode == "bounded_bypass_once":
            if ledger is None:
                raise ValueError("bounded bypass requires a first-admission ledger")
            ordered = ledger.order(before)
        waiting.clear()  # Identical deque mutation in both arms; request objects remain untouched.
        waiting.extend(ordered)
        event.update(action_applied=True, order_changed=[id(r) for r in before] != [id(r) for r in ordered])
        if scheduler.waiting is not waiting or [id(r) for r in waiting] != [id(r) for r in ordered]:
            raise ValueError("native waiting deque identity/order changed unexpectedly")
        if scheduler.running is not running or snapshot() != running_state:
            raise ValueError("waiting action modified running state")
    except Exception as exc:
        event["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        event.update(after=describe(scheduler.waiting), running_after=describe(scheduler.running),
                     running_unchanged=scheduler.running is running and snapshot() == running_state)

"""CPU-only admission accounting checks against the recorded native queue."""
import enum
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime"))
from waiting_order import WaitingBypassLedger, reorder_waiting

reference = ROOT / "runtime_reference.json"
if not reference.exists():
    reference = ROOT.parents[1] / "independent_ideas_20260908/host_timing_boundary_r01/waiting_order_r01/runtime_reference.json"
Status = enum.IntEnum("RequestStatus", "WAITING WAITING_FOR_REMOTE_KVS RUNNING PREEMPTED")
request_module = NS(Request=NS, RequestStatus=Status)
queue_module = ModuleType("vllm.v1.core.sched.request_queue")
with patch.dict(sys.modules, {"vllm.v1.request": request_module}):
    exec(compile(json.loads(reference.read_text())["v1/core/sched/request_queue.py"]["text"],
                 "recorded_request_queue.py", "exec"), vars(queue_module))
MODULES = {"vllm.v1.core.sched.request_queue": queue_module, "vllm.v1.request": request_module}


def setup(lengths):
    rows = [NS(request_id=str(i), num_prompt_tokens=n, num_computed_tokens=0,
               num_output_tokens=0, num_preemptions=0, status=Status.WAITING)
            for i, n in enumerate(lengths)]
    ledger = WaitingBypassLedger()
    for r in rows:
        ledger.register(r.request_id)
    scheduler = NS(waiting=queue_module.FCFSRequestQueue(rows), running=[],
                   skipped_waiting=queue_module.FCFSRequestQueue(), policy=queue_module.SchedulingPolicy.FCFS)
    return scheduler, ledger


class BypassSemantics(unittest.TestCase):
    def setUp(self):
        mock = patch.dict(sys.modules, MODULES)
        mock.start()
        self.addCleanup(mock.stop)

    def test_full_cap_reorders_do_not_charge_and_first_admission_does(self):
        s, ledger = setup([2048, 128, 128])
        s.running = [NS(request_id=f"running-{i}", status=Status.RUNNING,
                        num_prompt_tokens=128, num_computed_tokens=140, num_output_tokens=13) for i in range(8)]
        seen = {r.request_id for r in s.running}
        running_state = [vars(r).copy() for r in s.running]
        for _ in range(3):
            event = {}
            reorder_waiting(s, "bounded_bypass_once", seen, event, ledger)
            ledger.record({rid: 1 for rid in seen}, seen, "bounded_bypass_once", event)
            self.assertEqual(ledger.bypasses, {"0": 0, "1": 0, "2": 0})
        self.assertEqual([vars(r) for r in s.running], running_state)
        self.assertEqual([r.request_id for r in s.waiting], ["1", "0", "2"])
        event = {}
        reorder_waiting(s, "bounded_bypass_once", seen, event, ledger)
        s.running.pop()  # A completion has made room for a new request.
        admitted = s.waiting.popleft()  # Native scheduler really consumes one free slot.
        ledger.record({admitted.request_id: 128}, seen, "bounded_bypass_once", event)
        self.assertEqual(event["first_admitted"], ["1"])
        self.assertEqual(event["bypass_charges"], [{"older": "0", "admitted": "1"}])
        self.assertEqual(ledger.bypasses["0"], 1)
        event = {}
        reorder_waiting(s, "bounded_bypass_once", seen | {"1"}, event, ledger)
        ledger.record({"1": 1}, seen | {"1"}, "bounded_bypass_once", event)
        self.assertEqual([r.request_id for r in s.waiting], ["0", "2"])
        self.assertEqual(event["first_admitted"], [])
        self.assertEqual(ledger.bypasses["0"], 1)

    def test_multiple_first_admissions_and_same_length_fcfs(self):
        for lengths in ([2048, 128, 128, 2048, 128], [128] * 5):
            s, ledger = setup(lengths)
            event = {}
            reorder_waiting(s, "bounded_bypass_once", set(), event, ledger)
            order = [r.request_id for r in s.waiting]
            expected = ["1", "0", "2", "4", "3"] if lengths[0] == 2048 else list(map(str, range(5)))
            self.assertEqual(order, expected)
            ledger.record(dict.fromkeys(order, 1), set(), "bounded_bypass_once", event)
            self.assertLessEqual(max(ledger.bypasses.values()), 1)
            if lengths[0] == 128:
                self.assertFalse(event["order_changed"])
                self.assertEqual(sum(ledger.bypasses.values()), 0)

    def test_started_and_preempted_waiters_fail_before_queue_mutation(self):
        for fault in ("ever", "computed", "output", "preempted", "status"):
            with self.subTest(fault=fault):
                s, ledger = setup([2048, 128]); r = s.waiting[0]
                seen = {r.request_id} if fault == "ever" else set()
                if fault == "computed": r.num_computed_tokens = 1
                if fault == "output": r.num_output_tokens = 1
                if fault == "preempted": r.num_preemptions = 1
                if fault == "status": r.status = Status.PREEMPTED
                before = list(s.waiting); event = {}
                with self.assertRaisesRegex(ValueError, "never-started"):
                    reorder_waiting(s, "bounded_bypass_once", seen, event, ledger)
                self.assertEqual(list(s.waiting), before)
                self.assertFalse(event["action_applied"])
                self.assertEqual(sum(ledger.bypasses.values()), 0)


if __name__ == "__main__":
    unittest.main()

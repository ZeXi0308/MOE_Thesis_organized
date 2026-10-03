"""Five targeted CPU checks using the recorded native v0.26 FCFS queue source."""
import enum
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime"))
from native_capture import capture_episode
from waiting_order import reorder_waiting

Status = enum.IntEnum("RequestStatus", "WAITING WAITING_FOR_REMOTE_KVS RUNNING PREEMPTED")
request_module = NS(Request=NS, RequestStatus=Status)
queue_module = ModuleType("vllm.v1.core.sched.request_queue")
reference = json.loads((ROOT / "runtime_reference.json").read_text())
with patch.dict(sys.modules, {"vllm.v1.request": request_module}):
    exec(compile(reference["v1/core/sched/request_queue.py"]["text"], "recorded_request_queue.py", "exec"), vars(queue_module))
MODULES = {"vllm.v1.core.sched.request_queue": queue_module, "vllm.v1.request": request_module,
           "vllm": NS(SamplingParams=lambda **kw: NS(**kw)),
           "vllm.sampling_params": NS(RequestOutputKind=NS(CUMULATIVE="cumulative"))}


def request(rid, length):
    return NS(request_id=rid, num_prompt_tokens=length, num_computed_tokens=0,
              num_output_tokens=0, num_preemptions=0, status=Status.WAITING)


def scheduler(rows):
    return NS(waiting=queue_module.FCFSRequestQueue(rows), running=[],
              skipped_waiting=queue_module.FCFSRequestQueue(), policy=queue_module.SchedulingPolicy.FCFS)


class WaitingOrderTest(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, MODULES)
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_reorder_is_stable_and_keeps_objects_and_deque(self):
        rows = [request("long-a", 2048), request("short-a", 128), request("short-b", 128), request("long-b", 2048)]
        s = scheduler(rows); queue = s.waiting; event = {}
        reorder_waiting(s, "short_prompt_first", set(), event)
        self.assertIs(s.waiting, queue)
        self.assertEqual([id(r) for r in s.waiting], [id(rows[i]) for i in (1, 2, 0, 3)])
        self.assertTrue(event["order_changed"])

    def test_fcfs_records_same_queue_rewrite_without_reordering(self):
        rows = [request("long", 2048), request("short", 128)]; s = scheduler(rows); event = {}
        reorder_waiting(s, "fcfs", set(), event)
        self.assertEqual(event["before"], event["after"])
        self.assertTrue(event["action_applied"])
        self.assertFalse(event["order_changed"])
        self.assertEqual([id(r) for r in s.waiting], list(map(id, rows)))

    def test_started_blocked_restarted_and_nonfcfs_fail_before_mutation(self):
        for fault in ("ever", "computed", "output", "status", "skipped", "preempted", "priority"):
            with self.subTest(fault=fault):
                r = request("r", 128); s = scheduler([r]); event = {}; seen = {"r"} if fault == "ever" else set()
                if fault == "computed": r.num_computed_tokens = 1
                if fault == "output": r.num_output_tokens = 1
                if fault == "status": r.status = Status.WAITING_FOR_REMOTE_KVS
                if fault == "skipped": s.skipped_waiting.append(request("blocked", 128))
                if fault == "preempted": r.num_preemptions = 1
                if fault == "priority": s.policy = queue_module.SchedulingPolicy.PRIORITY
                with self.assertRaises(ValueError): reorder_waiting(s, "short_prompt_first", seen, event)
                self.assertFalse(event["action_applied"])
                self.assertEqual(event["before"], event["after"])
                self.assertIs(s.waiting[0], r)

    def test_running_partial_prefill_and_decode_unchanged(self):
        s = scheduler([request("long", 2048), request("short", 128)])
        s.running = [request("partial", 2048), request("decode", 128)]
        for r, computed in zip(s.running, (512, 140)):
            r.status = Status.RUNNING; r.num_computed_tokens = computed
        running = s.running; state = [vars(r).copy() for r in running]; event = {}
        reorder_waiting(s, "short_prompt_first", {r.request_id for r in running}, event)
        self.assertIs(s.running, running)
        self.assertEqual([vars(r) for r in running], state)
        self.assertTrue(event["running_unchanged"])

    def test_capture_preserves_failed_action_and_requests_before_schedule(self):
        s = scheduler([]); s.requests = {}; s.get_request_counts = lambda: (0, len(s.waiting))
        s.schedule = lambda: self.fail("native schedule must not execute after blocked waiting guard")
        def add_request(*args, **kwargs):
            r = request("native", 4); r.status = Status.WAITING_FOR_REMOTE_KVS
            s.requests[r.request_id] = r; s.waiting.append(r); return r.request_id
        engine = NS(engine_core=NS(engine_core=NS(scheduler=s)),
            vllm_config=NS(scheduler_config=NS(max_num_seqs=8, async_scheduling=False, stream_interval=1)),
            has_unfinished_requests=lambda: bool(s.requests), add_request=add_request, step=lambda: s.schedule())
        workload = dict(source_requests=[dict(request_id="r", document_id="d")],
            actual_prompt_token_ids=[[1, 2, 3, 4]], arrival_traces_s={"steady": [0]})
        raw = capture_episode(engine, workload, dict(cap=8, output_tokens=2, waiting_order="short_prompt_first"),
                              regime="steady", arrival_scale=1, run_id="test")
        self.assertEqual(raw["status"], "INCOMPLETE")
        self.assertEqual(raw["requests"][0]["status"], "failed")
        self.assertEqual(raw["scheduler_steps"], [])
        self.assertEqual(len(raw["waiting_order_actions"]), 1)
        action = raw["waiting_order_actions"][0]
        self.assertIn("never-started", action["error"])
        self.assertEqual(action["before"], action["after"])
        self.assertGreaterEqual(action["end_s"], action["start_s"])


if __name__ == "__main__":
    unittest.main()

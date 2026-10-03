"""CPU lifecycle checks for the selected-save LTR-style native seam.

The allocator, connector jobs and returned outputs are fixtures.  These tests
exercise the actual adapter closures; GPU store/load and the pinned installed
vLLM source still require a separate runtime qualification.
"""
from __future__ import annotations

from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import ltr_fair_native
from ltr_fair_native import _attach
from ltr_fair_policy import LTRFairPolicy, LTRRequestState


def status(name):
    return NS(name=name)


class Queue(list):
    def remove_request(self, request):
        self.remove(request)

    def prepend_request(self, request):
        self.insert(0, request)


class FakeScheduler:
    def __init__(self, *, free, target_blocks=0, slot_cap=2, victim_max=100):
        blocks = [NS(block_id=i, is_null=False) for i in range(20)]
        self.pool = NS(blocks=blocks, get_num_free_blocks=lambda: self.free)
        self.free = free
        self.owned = {"v": blocks[1:3], "t": blocks[3:3+target_blocks]}
        self.kv_cache_manager = NS(
            block_pool=self.pool,
            coordinator=NS(single_type_managers=[NS(req_to_blocks=self.owned)]))
        self.requests = {
            "v": NS(request_id="v", num_prompt_tokens=16,
                    num_output_tokens=16, num_computed_tokens=31,
                    max_tokens=victim_max, status=status("RUNNING"), arrival_time=1.0),
            "t": NS(request_id="t", num_prompt_tokens=32,
                    num_output_tokens=1, num_computed_tokens=0,
                    max_tokens=100, status=status("PREEMPTED"), arrival_time=0.0),
        }
        self.running = [self.requests["v"]]
        self.waiting = Queue([self.requests["t"]])
        self.skipped_waiting = Queue()
        self.max_num_running_reqs = slot_cap
        self.natural_preempt_id = None
        self.pending_load_next = False
        self.store_enabled = True
        self.omit_flush = False
        self._job_id = 6
        self.calls = []

    def schedule(self):
        raise AssertionError("native closure must replace this method")

    def _preempt_request(self, request, _timestamp):
        if request in self.running:
            self.running.remove(request)
        self.free += len(self.owned.get(request.request_id, ()))
        self.owned[request.request_id] = []
        request.status = status("PREEMPTED")
        request.num_computed_tokens = 0
        if request not in self.waiting:
            self.waiting.prepend_request(request)


class Harness:
    def __init__(self, *, free=1, target_blocks=0, slot_cap=2,
                 victim_max=100, threshold=2, quantum=3):
        self.s = FakeScheduler(free=free, target_blocks=target_blocks,
                               slot_cap=slot_cap, victim_max=victim_max)
        self.policy = LTRFairPolicy(threshold, quantum)
        self.policy.counters.states["t"] = LTRRequestState(idle=threshold)
        group = NS(block_ids=[1, 2], offload_keys=["k1", "k2"])
        self.rs = NS(req=self.s.requests["v"], transfer_jobs=set(),
                     group_states=[group])
        self.cs = NS(_calc_num_offloadable_tokens=lambda _rs, n: n,
                     _req_status={"v": self.rs}, _jobs={})
        self.data, self.uninstall = _attach(
            self.s, self.native, self.cs, block_size=16, policy=self.policy)

    def native(self):
        s = self.s
        preempted = []
        s._rotation_begin(preempted, 0.0)
        meta = NS(store_jobs={}, load_jobs={}, jobs_to_flush=set())
        scheduled = {}
        if s.natural_preempt_id is not None:
            r = s.requests[s.natural_preempt_id]
            s._preempt_request(r, 0.0)
            preempted.append(r)
            s.natural_preempt_id = None
        for r in list(s.running):
            scheduled[r.request_id] = 1
            if r.request_id == "v" and s.store_enabled:
                count = self.cs._calc_num_offloadable_tokens(
                    self.rs, r.num_computed_tokens)
                if count:
                    ids = [b.block_id for b in s.owned["v"][:count // 16]]
                    job = NS(req_id="v", is_store=True,
                             keys=set(self.rs.group_states[0].offload_keys[:len(ids)]))
                    self.cs._jobs[self.s._job_id] = job
                    self.rs.transfer_jobs.add(self.s._job_id)
                    meta.store_jobs[self.s._job_id] = NS(
                        req_id="v", src_spec=NS(block_ids=ids))
            r.num_computed_tokens += 1
            r.num_output_tokens += 1
        if preempted and not s.omit_flush:
            meta.jobs_to_flush = set(self.rs.transfer_jobs)
        if s.waiting and len(s.running) < s.max_num_running_reqs:
            r = s.waiting[0]
            need = max(0, (r.num_prompt_tokens+r.num_output_tokens+15)//16
                       - len(s.owned.get(r.request_id, ())))
            if need <= s.free:
                s.waiting.pop(0)
                if s.pending_load_next:
                    r.status = status("WAITING_FOR_REMOTE_KVS")
                    s.skipped_waiting.append(r)
                    meta.load_jobs[8] = NS(req_id=r.request_id)
                    s.pending_load_next = False
                else:
                    s.free -= need
                    s.owned[r.request_id] = s.pool.blocks[4:4+need]
                    r.status = status("RUNNING")
                    s.running.append(r)
                    scheduled[r.request_id] = 1
                    r.num_computed_tokens += 1
        result = NS(num_scheduled_tokens=scheduled,
                    preempted_req_ids={r.request_id for r in preempted},
                    kv_connector_metadata=meta)
        s.calls.append(result)
        return result

    def load_ready(self):
        r = self.s.requests["t"]
        self.s.skipped_waiting.remove(r)
        r.status = status("PREEMPTED")
        self.s.waiting.prepend_request(r)


class NativeLifecycleTest(unittest.TestCase):
    def test_engine_entry_uses_existing_scheduler_and_cache_block_size(self):
        scheduler = object()
        config = NS(cache_config=NS(block_size=16))
        engine = NS(engine_core=NS(engine_core=NS(scheduler=scheduler)),
                    vllm_config=config)
        with patch.object(ltr_fair_native, "install", return_value=("data", "undo")) as install:
            self.assertEqual(ltr_fair_native.install_on_engine(
                engine, threshold=30, quantum=1), ("data", "undo"))
        install.assert_called_once_with(scheduler, vllm_config=config, block_size=16,
                                        threshold=30, quantum=1)

    def test_prepare_commit_store_and_flush(self):
        h = Harness(free=1, slot_cap=1)
        h.s.schedule()
        self.assertEqual(h.policy.active_target, "t")
        self.assertEqual(h.policy.counters.states["t"].quantum_remaining, 3)
        self.assertEqual([e["event"] for e in h.data["events"] if e["event"] == "store_delta"],
                         ["store_delta"])
        h.s.schedule()
        self.assertEqual(h.data["applied_rotations"], 1)
        self.assertIn("v", h.s.calls[-1].preempted_req_ids)
        self.assertEqual(h.s.calls[-1].kv_connector_metadata.jobs_to_flush, {6})
        self.assertEqual(h.policy.counters.states["t"].quantum_remaining, 2)
        self.assertEqual(h.policy.last_forced_preempted, ("v",))
        self.assertEqual(h.policy.last_natural_preempted, ())
        h.uninstall()

    def test_direct_free_no_victim_and_pending_load_spends_no_quantum(self):
        h = Harness(free=2, target_blocks=1, slot_cap=2, quantum=2)
        h.s.pending_load_next = True
        h.s.schedule()
        self.assertEqual(h.data["free_priorities"], 1)
        self.assertEqual(h.data["applied_rotations"], 0)
        self.assertEqual(h.policy.active_target, "t")
        self.assertEqual(h.policy.counters.states["t"].quantum_remaining, 2)
        h.load_ready()
        h.s.schedule()
        self.assertEqual(h.policy.counters.states["t"].quantum_remaining, 1)
        # The first new output does not end the quantum.
        self.assertEqual(h.policy.active_target, "t")
        h.s.schedule()
        self.assertEqual(h.policy.counters.states["t"].quantum_remaining, 0)
        h.s.schedule()
        self.assertIsNone(h.policy.active_target)
        h.uninstall()

    def test_prepare_stale_target_cancels_without_preemption(self):
        h = Harness(free=1, slot_cap=1)
        h.s.schedule()
        h.s.requests["t"].num_output_tokens += 1
        h.s.schedule()
        self.assertEqual(h.data["applied_rotations"], 0)
        self.assertIsNone(h.policy.active_target)
        self.assertEqual(h.s.calls[-1].preempted_req_ids, set())
        self.assertEqual([e["reason"] for e in h.data["events"]
                          if e["event"] == "commit_check"][-1], "CANCEL_TARGET_CHANGED")
        h.uninstall()

    def test_invalid_ownership_is_rejected_before_action(self):
        h = Harness(free=1, slot_cap=1)
        h.s.owned["v"][0] = NS(block_id=1, is_null=False)
        with self.assertRaisesRegex(RuntimeError, "physical block ownership"):
            h.s.schedule()
        self.assertEqual(h.data["applied_rotations"], 0)

    def test_native_preemption_during_prepare_cancels_plan(self):
        h = Harness(free=1, slot_cap=1)
        h.s.natural_preempt_id = "v"
        h.s.schedule()
        self.assertIsNone(h.policy.active_target)
        self.assertEqual(h.data["applied_rotations"], 0)
        self.assertTrue(any(e.get("reason") == "prepare_victim_preempted"
                            for e in h.data["events"]))
        h.uninstall()

    def test_peer_growth_does_not_prefilter_legal_prepare(self):
        # free=0 and peer next-block growth=1.  Native can preempt peer during
        # this call; the selected victim still decodes and registers its store.
        h = Harness(free=0, target_blocks=1, slot_cap=2)
        peer = NS(request_id="peer", num_prompt_tokens=8, num_output_tokens=9,
                  num_computed_tokens=16, max_tokens=100,
                  status=status("RUNNING"), arrival_time=2.0)
        h.s.requests["peer"] = peer
        h.s.running.append(peer)
        h.s.owned["peer"] = [h.s.pool.blocks[6]]
        h.s.natural_preempt_id = "peer"
        self.assertEqual((h.s.free, 3-len(h.s.owned["t"])), (0, 2))
        self.assertEqual((peer.num_prompt_tokens+peer.num_output_tokens+15)//16
                         - len(h.s.owned["peer"]), 1)
        h.s.schedule()
        self.assertEqual(h.policy.active_target, "t")
        self.assertIn("peer", h.s.calls[-1].preempted_req_ids)
        self.assertTrue(any(e["event"] == "prepare" for e in h.data["events"]))
        self.assertTrue(any(e["event"] == "store_delta" and e["new_store_blocks"] == 1
                            for e in h.data["events"]))
        self.assertEqual(h.data["applied_rotations"], 0)
        h.uninstall()

    def test_natural_preemption_of_boosted_target_is_not_forbidden(self):
        h = Harness(free=3, slot_cap=2)
        h.s.schedule()
        self.assertEqual(h.policy.active_target, "t")
        h.s.natural_preempt_id = "t"
        h.s.schedule()
        self.assertIsNone(h.policy.active_target)
        self.assertIn("t", h.s.calls[-1].preempted_req_ids)
        self.assertEqual(h.policy.last_natural_preempted, ("t",))
        self.assertEqual(h.policy.last_forced_preempted, ())
        h.uninstall()

    def test_commit_block_change_and_resource_loss_cancel(self):
        for change, expected in (("ownership", "CANCEL_BLOCK_OWNERSHIP_CHANGED"),
                                 ("resource", "CANCEL_TARGET_UNFUNDED")):
            with self.subTest(change=change):
                h = Harness(free=1, slot_cap=1)
                h.s.schedule()
                if change == "ownership":
                    h.s.owned["v"][0] = h.s.pool.blocks[5]
                else:
                    h.s.free = 0
                h.s.schedule()
                self.assertEqual(h.data["applied_rotations"], 0)
                self.assertEqual([e["reason"] for e in h.data["events"]
                                  if e["event"] == "commit_check"][-1], expected)
                self.assertEqual(h.policy.last_forced_preempted, ())
                h.uninstall()

    def test_terminal_before_commit_cancels(self):
        h = Harness(free=1, slot_cap=1)
        h.s.schedule()
        h.s.waiting.clear()
        h.s.requests.pop("t")
        h.s.schedule()
        self.assertEqual(h.data["applied_rotations"], 0)
        self.assertIsNone(h.policy.active_target)
        self.assertEqual([e["reason"] for e in h.data["events"]
                          if e["event"] == "commit_check"][-1],
                         "CANCEL_REQUEST_FINISHED")
        h.uninstall()

    def test_stale_commit_scans_another_boosted_target(self):
        h = Harness(free=1, slot_cap=1)
        h.s.schedule()
        h.s.requests["t"].num_output_tokens += 1
        u = NS(request_id="u", num_prompt_tokens=32, num_output_tokens=1,
               num_computed_tokens=0, max_tokens=100,
               status=status("PREEMPTED"), arrival_time=0.5)
        h.s.requests["u"] = u
        h.s.waiting.append(u)
        h.policy.counters.states["u"] = LTRRequestState(idle=2)
        h.s.schedule()
        self.assertEqual(h.data["applied_rotations"], 0)
        self.assertEqual(h.policy.active_target, "u")
        self.assertEqual([e["target"] for e in h.data["events"]
                          if e["event"] == "prepare"], ["t", "u"])
        h.uninstall()

    def test_unfunded_and_terminal_noop(self):
        h = Harness(free=0, slot_cap=1, target_blocks=0)
        h.s.owned["v"] = h.s.pool.blocks[1:2]
        h.rs.group_states[0].block_ids = [1]
        h.s.schedule()
        self.assertIsNone(h.policy.active_target)
        self.assertEqual(h.data["applied_rotations"], 0)
        h.s.requests.pop("t")
        h.s.waiting.clear()
        h.s.schedule()
        self.assertEqual(h.data["applied_rotations"], 0)
        report = h.uninstall()
        self.assertEqual(report["censor_counts"]["UNFUNDED_TOTAL"], 1)
        self.assertEqual(report["no_op_counts"]["NO_BOOSTED_PREEMPTED"], 1)

    def test_backend_reject_is_reported_and_falls_back(self):
        h = Harness(free=1, slot_cap=1)
        with patch.object(ltr_fair_native, "prepare",
                          side_effect=ValueError("backend rejected pair")):
            result = h.s.schedule()
        self.assertEqual(result.num_scheduled_tokens, {"v": 1})
        self.assertIsNone(h.policy.active_target)
        report = h.uninstall()
        self.assertEqual(report["censor_counts"]["backend rejected pair"], 1)
        self.assertEqual(report["no_op_counts"]["NO_ELIGIBLE_VICTIM"], 1)

    def test_missing_flush_fails_loudly(self):
        h = Harness(free=1, slot_cap=1)
        h.s.schedule()
        h.s.omit_flush = True
        with self.assertRaisesRegex(RuntimeError, "flush omitted"):
            h.s.schedule()
        self.assertEqual(h.data["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()

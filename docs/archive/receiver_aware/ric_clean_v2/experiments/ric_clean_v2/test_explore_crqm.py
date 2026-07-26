from __future__ import annotations

import unittest

try:
    from .explore_crqm import Task, evaluate_order, exact_r0, exhaustive, materialize, milp_b
except ImportError:  # pragma: no cover
    from explore_crqm import Task, evaluate_order, exact_r0, exhaustive, materialize, milp_b  # type: ignore


class CRQMCoreTest(unittest.TestCase):
    def window(self, depths=(2, 8)):
        tasks = tuple(
            Task(
                task_id=f"t{i}", join_id=f"j{i}", request_id=f"r{i}", sender_rank=0,
                receiver_rank=i % 3, layer_id=1, expert_id=i, topk_slot=0,
                cut_service_us=1.0, receiver_service_us=2.0,
            )
            for i in range(6)
        )
        return materialize("olmoe", 0, tasks, depths, {0: 0.0, 1: 2.0, 2: 3.0, 4: 5.0, 8: 9.0, 16: 17.0})

    def test_matched_multiset_and_identity(self):
        window = self.window()
        self.assertEqual(sorted(window.queue_maps[0].values()), sorted(window.queue_maps[1].values()))
        self.assertEqual([task.task_id for task in window.tasks], [f"t{i}" for i in range(6)])

    def test_negative_control(self):
        window = self.window((0, 0))
        b = exhaustive(window, (0, 1))
        r0 = exact_r0(window)
        self.assertAlmostEqual(b["cvar99"], r0["cvar99"])
        self.assertEqual(r0["first_action_sets"][0], r0["first_action_sets"][1])

    def test_flow_accounting(self):
        window = self.window()
        cvar, mean_value, flows = evaluate_order(window, range(6), (0, 1))
        self.assertEqual(len(flows), 12)
        self.assertGreaterEqual(cvar, mean_value)

    def test_backlog_drains_from_t0_and_receiver_serializes(self):
        window = self.window()
        _cvar, _mean, flows = evaluate_order(window, range(6), (0,))
        # Receiver 0 handles t0 then t3. Initial work drains from t=0 and is
        # applied once; the second candidate serializes behind the first.
        first = max(1.0, window.queue_maps[0][0]) + 2.0
        second = max(4.0, first) + 2.0
        self.assertAlmostEqual(flows[0], first)
        self.assertAlmostEqual(flows[3], second)
        self.assertNotAlmostEqual(flows[3], 4.0 + window.queue_maps[0][0] + 2.0)

    def test_histories_replay_queue_work(self):
        window = self.window()
        for world in (0, 1):
            for receiver, events in window.queue_histories[world].items():
                self.assertAlmostEqual(
                    sum(event["service_us"] for event in events),
                    window.queue_maps[world][receiver],
                )

    def test_milp_matches_enumeration(self):
        window = self.window()
        milp = milp_b(window)
        exact = exhaustive(window, (0, 1))
        self.assertAlmostEqual(milp["cvar99"], exact["cvar99"], places=6)
        self.assertAlmostEqual(milp["mean"], exact["mean"], places=6)


if __name__ == "__main__":
    unittest.main()

"""CPU reservation observation/probe checks using pinned native helper methods."""
import unittest
from unittest.mock import patch

import admission_probe as probe
import reservation_observer as observer
from test_admission_probe import (NS_NATIVE, Queue, Request, Status,
                                  allocation_state, manager, native_class)


NativeScheduler = native_class('scheduler.py', 'Scheduler',
    ('_request_remaining_blocks', '_inflight_prefill_reserved_blocks'), NS_NATIVE)


class ReservationObserverTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        for name, value in [('perf_counter', lambda: self.now), ('time', lambda: 1000.)]:
            mocked = patch.object(probe.time, name, side_effect=value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def make_gate(self, enabled=False, free=3, **observer_kwargs):
        old = Request('old', Status.RUNNING, preemptions=1, computed=16, tokens=48)
        req = Request('new', tokens=32)
        scheduler = NativeScheduler()
        scheduler.requests = {'old': old, 'new': req}
        scheduler.running = [old]
        scheduler.finished_recving_kv_req_ids = set()
        scheduler._inflight_prefills = {old}
        scheduler.max_model_len = 64
        scheduler.kv_cache_manager = manager(free=free, held=1, rid='old', cached=True)
        gate = probe.Gate(scheduler, cap=4, kv_floor=0, probe_enabled=False)
        gate.admitted_ids.add('old')
        observer.install(gate, probe_enabled=enabled, **observer_kwargs)
        return gate, req, old

    @staticmethod
    def before(gate, req):
        return gate.before_allocate(req, token_budget=16, num_new_tokens=16,
            num_new_computed_tokens=0, num_external_computed_tokens=0,
            num_lookahead_tokens=0, num_encoder_tokens=0, delay_cache_blocks=False,
            full_sequence_must_fit=True, reserved_blocks=0, has_scheduled_reqs=True)

    def test_default_observer_uses_native_read_only_geometry_and_preserves_decision(self):
        gate, req, old = self.make_gate()
        before = allocation_state(gate.s.kv_cache_manager)
        self.assertEqual(gate.s._request_remaining_blocks(old), 2)
        self.assertEqual(gate.s._inflight_prefill_reserved_blocks(), 2)
        check = self.before(gate, req)
        row = check['row']
        self.assertEqual(allocation_state(gate.s.kv_cache_manager), before)
        self.assertEqual(row['known_prefill_unallocated_blocks'], 2)
        self.assertEqual(row['full_plus_existing_unallocated_blocks'], 4)
        self.assertEqual(row['chunk_plus_existing_unallocated_blocks'], 3)
        self.assertTrue(row['full_commit_opportunity'])
        self.assertFalse(row['chunk_commit_opportunity'])
        self.assertEqual(row['reason'], 'allow')
        self.assertFalse(row['denied'])
        self.assertFalse(row['changed_by_reservation'])
        self.assertFalse(row['changed_by_recovery'])
        self.assertIsNone(gate.event)  # Running replay is not old pending-recovery trigger.
        report = gate.report()
        self.assertFalse(report['reservation_probe']['probe_enabled'])
        self.assertEqual(report['reservation_probe']['direct_action_evaluations'], 0)
        first = report['reservation_observation']['first_full_commit_opportunity']
        self.assertIs(first['decision'], row)
        actual = gate.s.kv_cache_manager.allocate_slots(req, num_new_tokens=16,
                                                       full_sequence_must_fit=True)
        gate.after_allocate(req, check, actual)
        self.assertTrue(first['decision']['native_allocation_result'])

    def test_one_frozen_target_releases_on_zero_reservation_or_total_equality(self):
        for release in ('zero_reservation', 'total_equals_free'):
            with self.subTest(release=release):
                self.now = 100.
                gate, req, old = self.make_gate(enabled=True)
                first_row = self.before(gate, req)['row']
                self.assertTrue(first_row['changed_by_reservation'])
                self.assertFalse(first_row['changed_by_recovery'])
                self.assertEqual(first_row['reason'], 'reservation')
                self.assertEqual(gate.barrier, 'reservation')
                report = gate.report()['reservation_probe']
                first = report['first_opportunity']
                self.assertIs(first['decision'], first_row)
                self.assertEqual(report['direct_action_evaluations'], 1)
                later = Request('later')
                gate.s.requests['later'] = later
                self.assertFalse(self.before(gate, later)['row']['changed_by_reservation'])
                self.now = 100.02
                self.assertTrue(self.before(gate, req)['row']['denied'])
                self.assertEqual(gate.report()['reservation_probe']['direct_action_evaluations'], 2)
                if release == 'zero_reservation':
                    gate.s._inflight_prefills.clear()
                else:
                    gate.s.kv_cache_manager.block_pool.free = 4
                self.now = 100.03
                row = self.before(gate, req)['row']
                self.assertFalse(row['denied'])
                self.assertFalse(row['changed_by_reservation'])
                self.assertIsNotNone(gate.report()['reservation_probe']['first_release'])
                gate.s._inflight_prefills.add(old)
                gate.s.kv_cache_manager.block_pool.free = 3
                self.now = 100.04
                self.assertFalse(self.before(gate, req)['row']['denied'])
                self.assertIs(gate.report()['reservation_probe']['first_opportunity'], first)
                self.assertEqual(gate.report()['reservation_probe']['direct_action_evaluations'], 2)

    def test_deadline_restores_fifo_and_never_holds_old_recovery(self):
        gate, req, old = self.make_gate(enabled=True)
        later = Request('later')
        gate.s.requests['later'] = later
        queue = Queue([req, later, old])
        self.assertTrue(self.before(gate, req)['row']['denied'])
        gate.hold(queue, queue.pop_request())
        self.now = 100.249999
        self.assertIsNone(gate.early(old))
        self.assertIsNone(self.before(gate, old)['row'])
        self.assertEqual(gate.early(later), 'hold')
        fifo = gate.decisions[-1]
        self.assertTrue(fifo['reservation_probe_fifo_held'])
        self.assertIsNone(fifo['native_fit'])
        self.assertFalse(fifo['changed_by_recovery'])
        gate.hold(queue, queue.pop_request())
        report = gate.report()['reservation_probe']
        self.assertEqual(report['fifo_action_evaluations'], 1)
        self.assertEqual(set(report['fifo_held_request_ids']), {'later'})
        self.now = 100.25
        self.assertEqual(gate.early(old), 'restart')
        self.assertEqual(queue.items, [req, later, old])
        self.assertEqual(gate.held, [])
        self.assertIsNone(gate.barrier)
        self.assertFalse(self.before(gate, req)['row']['denied'])
        self.assertIsNotNone(gate.report()['reservation_probe']['first_release'])

        # Sampling the state can itself cross the wall-clock deadline: that
        # diagnostic row must not become an extra FIFO action.
        self.now = 200.
        gate, req, old = self.make_gate(enabled=True)
        later = Request('later')
        gate.s.requests['later'] = later
        queue = Queue([req, later, old])
        self.before(gate, req)
        gate.hold(queue, queue.pop_request())
        self.now = 200.249999
        original_state = gate.state
        def crossing_state():
            self.now = 200.250001
            return original_state()
        with patch.object(gate, 'state', side_effect=crossing_state):
            self.assertEqual(gate.early(later), 'restart')
        self.assertEqual(queue.items, [req, later, old])
        self.assertEqual(gate.held, [])
        self.assertFalse(gate.decisions[-1]['denied'])
        self.assertEqual(gate.decisions[-1]['reason'], 'reservation_deadline_restore')
        self.assertFalse(gate.decisions[-1]['reservation_probe_fifo_held'])
        self.assertEqual(gate.report()['reservation_probe']['fifo_action_evaluations'], 0)

    def test_native_capacity_failure_is_preserved_before_and_after_probe(self):
        gate, req, _ = self.make_gate(enabled=True, free=1)
        row = self.before(gate, req)['row']
        self.assertEqual(row['reason'], 'native_capacity')
        self.assertFalse(row['denied'])
        self.assertFalse(row['changed_by_reservation'])
        self.assertIsNone(gate.report()['reservation_probe']['first_opportunity'])
        gate.s.kv_cache_manager.block_pool.free = 3
        self.assertTrue(self.before(gate, req)['row']['changed_by_reservation'])
        self.now = 100.1
        gate.s.kv_cache_manager.block_pool.free = 1
        row = self.before(gate, req)['row']
        self.assertEqual(row['reason'], 'native_capacity')
        self.assertFalse(row['denied'])
        self.assertFalse(row['changed_by_reservation'])
        self.assertIsNone(gate.report()['reservation_probe']['first_release'])
        self.assertEqual(gate.report()['reservation_probe']['direct_action_evaluations'], 1)
        self.now = 100.25
        row = self.before(gate, req)['row']
        self.assertEqual(row['reason'], 'native_capacity')
        self.assertFalse(row['denied'])
        self.assertIsNotNone(gate.report()['reservation_probe']['first_release'])
        with self.assertRaises(AssertionError):
            observer.install(probe.Gate(gate.s, probe_enabled=False),
                             probe_enabled=True, max_extra_s=.250001)

    def test_fixed_headroom_triggers_with_zero_reservation_but_shadow_keeps_allow(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                gate, req, _ = self.make_gate(enabled=enabled, fixed_headroom_blocks=32)
                gate.s._inflight_prefills.clear()
                gate.s.kv_cache_manager.watermark_blocks = 1
                before = allocation_state(gate.s.kv_cache_manager)
                row = self.before(gate, req)['row']
                self.assertEqual(allocation_state(gate.s.kv_cache_manager), before)
                self.assertEqual(row['known_prefill_unallocated_blocks'], 0)
                self.assertEqual(row['full_fit_margin_blocks'], 0)  # 3 free - 2 full - 1 watermark.
                self.assertTrue(row['fixed_headroom_opportunity'])
                self.assertFalse(row['full_commit_opportunity'])
                self.assertEqual(row['denied'], enabled)
                self.assertEqual(row['reason'], 'headroom' if enabled else 'allow')
                self.assertEqual(row['changed_by_headroom'], enabled)
                self.assertFalse(row['changed_by_reservation'])
                self.assertFalse(row['changed_by_recovery'])
                report = gate.report()['reservation_probe']
                self.assertEqual(report['signal_kind'], 'fixed_headroom')
                self.assertEqual(report['fixed_headroom_blocks'], 32)
                self.assertEqual(report['direct_action_evaluations'], int(enabled))
                self.assertTrue(report['first_opportunity']['state']['fixed_headroom_opportunity'])
                self.assertFalse(gate.probe_enabled)

    def test_fixed_headroom_ignores_reservation_changes_and_releases_at_margin32_once(self):
        gate, req, old = self.make_gate(enabled=True, fixed_headroom_blocks=32)
        self.assertTrue(self.before(gate, req)['row']['changed_by_headroom'])
        first = gate.report()['reservation_probe']['first_opportunity']
        gate.s._inflight_prefills.clear()
        self.now = 100.001
        row = self.before(gate, req)['row']
        self.assertEqual(row['known_prefill_unallocated_blocks'], 0)
        self.assertEqual(row['full_fit_margin_blocks'], 1)
        self.assertTrue(row['denied'])
        self.assertIsNone(gate.report()['reservation_probe']['first_release'])
        gate.s._inflight_prefills.add(old)
        gate.s.kv_cache_manager.block_pool.free = 34
        row = self.before(gate, req)['row']
        self.assertEqual(row['full_fit_margin_blocks'], 32)
        self.assertFalse(row['fixed_headroom_opportunity'])
        self.assertFalse(row['denied'])
        release = gate.report()['reservation_probe']['first_release']
        self.assertEqual(release['reason'], 'headroom_sufficient')
        self.assertEqual(release['perf_s'], 100.001)  # No 100 ms minimum wait.
        gate.s.kv_cache_manager.block_pool.free = 3
        self.assertFalse(self.before(gate, req)['row']['denied'])
        later = Request('later')
        gate.s.requests['later'] = later
        self.assertFalse(self.before(gate, later)['row']['denied'])
        self.assertIs(gate.report()['reservation_probe']['first_opportunity'], first)
        self.assertEqual(gate.report()['reservation_probe']['direct_action_evaluations'], 2)

    def test_fixed_headroom_preserves_native_failure_and_fifo_hard_deadline(self):
        gate, req, old = self.make_gate(enabled=True, free=1, fixed_headroom_blocks=32)
        row = self.before(gate, req)['row']
        self.assertEqual(row['reason'], 'native_capacity')
        self.assertFalse(row['denied'])
        self.assertFalse(row['fixed_headroom_opportunity'])
        self.assertFalse(row['changed_by_headroom'])
        self.assertIsNone(gate.report()['reservation_probe']['first_opportunity'])
        gate.s.kv_cache_manager.block_pool.free = 3
        self.assertTrue(self.before(gate, req)['row']['changed_by_headroom'])
        self.assertEqual(gate.barrier, 'headroom')
        self.now = 100.1
        gate.s.kv_cache_manager.block_pool.free = 1
        row = self.before(gate, req)['row']
        self.assertEqual(row['reason'], 'native_capacity')
        self.assertFalse(row['denied'])
        self.assertFalse(row['changed_by_headroom'])
        self.assertIsNone(gate.report()['reservation_probe']['first_release'])
        gate.begin()
        gate.s.kv_cache_manager.block_pool.free = 3
        self.before(gate, req)
        later = Request('later')
        gate.s.requests['later'] = later
        queue = Queue([req, later, old])
        gate.hold(queue, queue.pop_request())
        self.now = 100.249999
        self.assertIsNone(gate.early(old))
        self.assertIsNone(self.before(gate, old)['row'])
        self.assertEqual(gate.early(later), 'hold')
        row = gate.decisions[-1]
        self.assertEqual(row['reason'], 'fifo_headroom')
        self.assertTrue(row['headroom_probe_fifo_held'])
        self.assertFalse(row['reservation_probe_fifo_held'])
        self.assertIsNone(row['native_fit'])
        gate.hold(queue, queue.pop_request())
        self.now = 100.25
        self.assertEqual(gate.early(old), 'restart')
        self.assertEqual(queue.items, [req, later, old])
        self.assertEqual(gate.held, [])
        self.assertIsNone(gate.barrier)
        self.assertFalse(self.before(gate, req)['row']['denied'])
        report = gate.report()['reservation_probe']
        self.assertEqual(report['first_release']['reason'], 'hard_deadline')
        self.assertEqual(report['first_release']['perf_s'], 100.25)
        self.assertEqual(report['fifo_action_evaluations'], 1)


if __name__ == '__main__':
    unittest.main()

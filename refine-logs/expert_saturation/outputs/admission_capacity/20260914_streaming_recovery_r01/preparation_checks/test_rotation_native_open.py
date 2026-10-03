"""Dynamic membership CPU fixtures; no native worker, EOS, or performance claim."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import rotation_native as adapter
from verify_headroom_fast import FakeScheduler
from verify_rotation_native import Queue, check

SOURCE = Path('/private/tmp/moe-native-v026-recovery-source/scheduler.py')


class Fixture(FakeScheduler):
    ec_connector = None
    policy = NS(name='FCFS')
    is_encoder_decoder = False
    scheduler_reserve_full_isl = True
    max_num_running_reqs = 32
    natural_next = None

    def _preempt_request(self, request, timestamp):
        request.status = NS(name='PREEMPTED')
        request.num_computed_tokens = 0
        request.num_preemptions += 1
        self.waiting.prepend_request(request)
        for block in self.owned.pop(request.request_id, []):
            block.ref_cnt = 0
            self.available.append(block.block_id)

    def fixture_schedule(self):
        forced, tokens, resumed = [], {}, set()
        self._rotation_begin(forced, 0.0)
        if self.natural_next:
            request = self.requests[self.natural_next]
            self.running.remove(request)
            self._preempt_request(request, 0.0)
            forced.append(request)
            self.natural_next = None
        if self._rotation_target is not None:
            request = self.requests[self._rotation_target]
            if request in self.waiting:
                self.waiting.remove_request(request)
                self.running.append(request)
                resumed.add(request.request_id)
            blocks = self.owned.setdefault(request.request_id, [])
            while len(blocks) < (request.num_tokens + 3) // 4:
                block = self.kv_cache_manager.block_pool.blocks[self.available.pop()]
                block.ref_cnt = 1
                blocks.append(block)
            request.status = NS(name='RUNNING')
            request.num_computed_tokens += 1
            tokens[request.request_id] = 1  # Internal work only; no fake output.
        return NS(num_scheduled_tokens=tokens,
                  preempted_req_ids={r.request_id for r in forced},
                  scheduled_cached_reqs=NS(resumed_req_ids=resumed))


def fixture(population='open', expected=32):
    scheduler = Fixture(12, {'output_tokens': 128})
    scheduler.owned = scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
    cfg = NS(scheduler_config=NS(async_scheduling=False), speculative_config=None)
    replacement = ast.parse('def schedule(self):\n return self.fixture_schedule()')
    kwargs = {} if population is None else {'population_mode': population}
    with patch.object(adapter.inspect, 'getsourcefile', return_value=str(SOURCE)), \
            patch.object(adapter, 'patched_schedule_tree', return_value=replacement):
        decisions, uninstall = adapter.install(scheduler, vllm_config=cfg, block_size=4,
            expected_requests=expected, victim_order='most_output', **kwargs)
    scheduler.waiting, scheduler.skipped_waiting = Queue(), Queue()
    return scheduler, decisions, uninstall


def add(scheduler, rid, prompt=4, outputs=4, blocks=2, computed=None):
    request = NS(request_id=rid, num_computed_tokens=prompt + outputs - 1 if computed is None else computed,
                 num_prompt_tokens=prompt, num_output_tokens=outputs, num_tokens=prompt + outputs,
                 max_tokens=128, num_preemptions=0, num_output_placeholders=0,
                 num_in_flight_tokens=0, spec_token_ids=[], has_encoder_inputs=False,
                 status=NS(name='RUNNING'))
    request.is_finished = lambda: request.status.name.startswith('FINISHED_')
    scheduler.requests[rid] = request
    scheduler.running.append(request)
    scheduler.owned[rid] = []
    for _ in range(blocks):
        block = scheduler.kv_cache_manager.block_pool.blocks[scheduler.available.pop()]
        block.ref_cnt = 1
        scheduler.owned[rid].append(block)
    return request


def protected_fixture():
    scheduler, decisions, uninstall = fixture()
    add(scheduler, 'A')
    add(scheduler, 'B', prompt=8, outputs=8, blocks=4)
    target = add(scheduler, 'T', prompt=12, outputs=9, blocks=5)
    scheduler.natural_next = 'T'
    for _ in range(31):
        scheduler.schedule()
    assert decisions[-1]['forced_preempted'] == ['B']
    assert decisions[-1]['recovery_target'] == 'T'
    return scheduler, decisions, target, uninstall


class OpenPopulation(unittest.TestCase):
    def test_late_arrival_qualified_once_without_closed_size(self):
        scheduler, decisions, _ = fixture(expected=10000)
        add(scheduler, 'A')
        scheduler.schedule()
        calls = scheduler.kv_cache_manager.calls
        scheduler.schedule()
        self.assertEqual(scheduler.kv_cache_manager.calls, calls)
        add(scheduler, 'late', prompt=8, outputs=1, blocks=2)
        scheduler.schedule()
        self.assertEqual(scheduler.kv_cache_manager.calls, calls + 1)
        self.assertTrue(decisions[-1]['active'])

    def test_partial_prefill_is_not_a_victim(self):
        scheduler, decisions, _ = fixture()
        add(scheduler, 'partial', prompt=8, outputs=0, blocks=1, computed=4)
        scheduler.schedule()
        self.assertIsNone(decisions[-1]['proposal'])
        self.assertEqual(decisions[-1]['not_applied_reason'], 'native recovery already underway')
        self.assertEqual(scheduler.kv_cache_manager.calls, 0)

    def test_shared_block_rejected_on_first_pure_and_selected_victim(self):
        scheduler, _, _ = fixture()
        add(scheduler, 'A')
        scheduler.owned['A'][0].ref_cnt = 2
        with self.assertRaisesRegex(RuntimeError, 'ownership'):
            scheduler.schedule()
        scheduler, _, _ = fixture()
        add(scheduler, 'A')
        add(scheduler, 'B', prompt=8, outputs=8, blocks=4)
        add(scheduler, 'T', prompt=12, outputs=9, blocks=5)
        scheduler.natural_next = 'T'
        scheduler.schedule()
        scheduler.owned['B'][0].ref_cnt = 2
        for _ in range(29):
            scheduler.schedule()
        with self.assertRaisesRegex(RuntimeError, 'ownership'):
            scheduler.schedule()

    def test_terminal_without_output_releases_separately(self):
        scheduler, decisions, target, _ = protected_fixture()
        target.status = NS(name='FINISHED_STOPPED')
        scheduler.running.remove(target)
        del scheduler.requests['T']
        scheduler.schedule()
        self.assertNotIn('recovery_completed', decisions[-1])
        self.assertEqual(decisions[-1]['recovery_released'], dict(request_id='T',
            reason='terminal', status='FINISHED_STOPPED', new_output_tokens=0))
        self.assertIsNone(scheduler._rotation_target)

    def test_actual_new_output_and_disappearance_are_distinct(self):
        scheduler, decisions, target, _ = protected_fixture()
        target.num_output_tokens += 1
        scheduler.schedule()
        self.assertEqual(decisions[-1]['recovery_completed'], 'T')
        self.assertEqual(decisions[-1]['recovery_released']['reason'], 'new_output')
        scheduler, _, target, _ = protected_fixture()
        del scheduler.requests['T']
        scheduler.running.remove(target)
        with self.assertRaisesRegex(RuntimeError, 'without terminal status'):
            scheduler.schedule()

    def test_default_closed_activation_and_late_arrival_rejection(self):
        scheduler, decisions, _ = fixture(population=None, expected=2)
        add(scheduler, 'A')
        scheduler.schedule()
        self.assertFalse(decisions[-1]['active'])
        add(scheduler, 'B')
        scheduler.schedule()
        self.assertTrue(decisions[-1]['active'])
        add(scheduler, 'late')
        with self.assertRaisesRegex(RuntimeError, 'closed synchronous'):
            scheduler.schedule()

    def test_existing_closed_allocation_and_native_payload_cases(self):
        for fundable, order in ((False, 'least_progress'), (True, 'least_progress'),
                                (True, 'most_output'), (True, 'first_most_then_least')):
            with self.subTest(fundable=fundable, order=order):
                self.assertEqual(check(SOURCE, fundable, order)['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()

"""CPU-only lifecycle checks for the actual measure_episode observer scope."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

PKG = Path(__file__).parent / 'pkg'
sys.path.insert(0, str(PKG))
from request_measurement import measure_episode, _install_flush_wait_observer

MODULE = 'vllm.distributed.kv_transfer.kv_transfer_state'


class CPUOffloadingWorker:
    def __init__(self, fail=False):
        self.calls, self.fail, self.last_ids = 0, fail, None

    def wait(self, job_ids):
        self.calls += 1
        self.last_ids = job_ids
        if self.fail:
            raise RuntimeError('intentional native wait failure')
        return 'native-wait-return'


class OffloadingConnectorWorker:
    def __init__(self, worker):
        self.worker, self.calls = worker, 0

    def handle_preemptions(self, metadata):
        self.calls += 1
        if metadata.jobs_to_flush:
            return self.worker.wait(metadata.jobs_to_flush)
        return 'native-empty-return'


class OffloadingConnector:
    def __init__(self, wrapper):
        self.connector_worker = wrapper


class Scheduler:
    def __init__(self):
        self.calls = 0
        self.kv_cache_manager = NS(block_pool=NS(get_num_free_blocks=lambda: self.calls * 84))
        self.connector = NS(connector_scheduler=NS(_req_status={}, _jobs={}))

    def _preempt_request(self, request, timestamp):
        self.calls += 1


class Engine:
    def __init__(self, wrapper):
        self.wrapper, self.pending, self.ids = wrapper, None, {7}
        self.scheduler = Scheduler()
        self.vllm_config = NS(scheduler_config=NS(async_scheduling=False, stream_interval=1))
        self.engine_core = NS(engine_core=NS(scheduler=self.scheduler))

    def add_request(self, rid, prompt, params, arrival_time):
        self.pending = rid
        internal = rid + '-internal'
        cs = self.scheduler.connector.connector_scheduler
        cs._req_status[internal] = NS(transfer_jobs={7})
        cs._jobs[7] = NS(req_id=internal, is_store=True, pending_count=1)
        return internal

    def has_unfinished_requests(self):
        return self.pending is not None

    def step(self):
        rid = self.pending
        self.scheduler._preempt_request(NS(request_id=rid+'-internal', num_output_tokens=0), 0.)
        self.empty_return = self.wrapper.handle_preemptions(NS(jobs_to_flush=set()))
        self.wait_return = self.wrapper.handle_preemptions(NS(jobs_to_flush=self.ids))
        self.pending = None
        return [NS(request_id=rid, finished=True, outputs=[NS(token_ids=[42], finish_reason='stop')])]


def execute(worker, missing=False):
    wrapper = OffloadingConnectorWorker(worker)
    engine = Engine(wrapper)
    vllm = ModuleType('vllm'); vllm.SamplingParams = lambda **kwargs: NS(**kwargs)
    sampling = ModuleType('vllm.sampling_params'); sampling.RequestOutputKind = NS(CUMULATIVE='cumulative')
    module = ModuleType(MODULE)
    if not missing:
        module._KV_CONNECTOR_AGENT = OffloadingConnector(wrapper)
    with patch.dict(sys.modules, {'vllm': vllm, 'vllm.sampling_params': sampling, MODULE: module}):
        result = measure_episode(engine,
            dict(source_requests=[dict(request_id='short', document_id='short')],
                 actual_prompt_token_ids=[[11, 12]], arrival_traces_s={'steady': [0.]}),
            dict(output_tokens=128, ignore_eos=False, min_tokens=0), 'steady', 1., 'cpu',
            record_preemptions=True, record_flush_wait=True)
    return result, engine, wrapper


class FlushWaitTests(unittest.TestCase):
    def test_success_exactly_once_and_original_instance_or_class_state_restored(self):
        worker = CPUOffloadingWorker()
        original_wait = worker.wait
        worker.wait = original_wait  # Existing instance method must survive uninstall exactly.
        raw, engine, wrapper = execute(worker)
        self.assertEqual(raw['status'], 'COMPLETE')
        self.assertEqual((engine.scheduler.calls, wrapper.calls, worker.calls), (1, 2, 1))
        self.assertIs(worker.last_ids, engine.ids)
        self.assertEqual(engine.empty_return, 'native-empty-return')
        self.assertEqual(engine.wait_return, 'native-wait-return')
        observed = raw['flush_wait_observation']
        self.assertEqual(observed['status'], 'INSTALLED_AND_RESTORED')
        self.assertTrue(observed['methods_restored'])
        self.assertEqual(len(observed['events']), 1)
        event = observed['events'][0]
        self.assertEqual((event['engine_call_index'], event['job_ids']), (0, [7]))
        self.assertTrue(event['handle_completed'])
        self.assertEqual(len(event['wait_calls']), 1)
        call = event['wait_calls'][0]
        self.assertTrue(call['completed'])
        self.assertLessEqual(event['handle_entered_s'], call['entered_s'])
        self.assertLessEqual(call['entered_s'], call['returned_s'])
        self.assertLessEqual(call['returned_s'], event['handle_returned_s'])
        self.assertEqual(raw['preemption_events'][0]['pending_native_store_jobs'], dict(status='KNOWN', job_ids=[7]))
        self.assertIs(vars(worker)['wait'], original_wait)
        self.assertNotIn('handle_preemptions', vars(wrapper))
        self.assertNotIn('_preempt_request', vars(engine.scheduler))
        wrapper.handle_preemptions(NS(jobs_to_flush={8}))  # Later drain cannot extend this observation.
        self.assertEqual(len(observed['events']), 1)
        tree = ast.parse((PKG/'run_recovery_cadence.py').read_text())
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'measure_episode')
        self.assertIs(ast.literal_eval(next(k.value for k in call.keywords if k.arg == 'record_flush_wait')), True)

    def test_native_exception_missing_object_and_partial_installation_restore(self):
        worker = CPUOffloadingWorker(fail=True)
        raw, engine, wrapper = execute(worker)
        self.assertEqual(raw['status'], 'INCOMPLETE')
        self.assertIn('intentional native wait failure', raw['error'])
        self.assertEqual((engine.scheduler.calls, wrapper.calls, worker.calls), (1, 2, 1))
        event = raw['flush_wait_observation']['events'][0]
        self.assertFalse(event['handle_completed'])
        self.assertFalse(event['wait_calls'][0]['completed'])
        self.assertIn('intentional native wait failure', event['wait_calls'][0]['error'])
        self.assertNotIn('wait', vars(worker))
        self.assertNotIn('handle_preemptions', vars(wrapper))
        self.assertNotIn('_preempt_request', vars(engine.scheduler))
        raw, _, wrapper = execute(CPUOffloadingWorker(), missing=True)
        self.assertEqual(raw['status'], 'COMPLETE')
        self.assertEqual(raw['flush_wait_observation']['status'], 'UNKNOWN')
        self.assertFalse(raw['flush_wait_observation']['installed'])
        self.assertTrue(raw['flush_wait_observation']['errors'])
        # The second setattr fails: installation must undo the first wrapper.
        worker = CPUOffloadingWorker(); wrapper = OffloadingConnectorWorker(worker)
        module = ModuleType(MODULE); module._KV_CONNECTOR_AGENT = OffloadingConnector(wrapper)
        original_setattr = object.__setattr__
        def reject_wait(obj, name, value):
            if name == 'wait':
                raise TypeError('intentional observer installation failure')
            original_setattr(obj, name, value)
        with patch.dict(sys.modules, {MODULE: module}), patch.object(CPUOffloadingWorker, '__setattr__', reject_wait):
            data, uninstall = _install_flush_wait_observer(lambda: 0., lambda: 0)
        self.assertEqual(data['status'], 'UNKNOWN')
        self.assertFalse(data['installed'])
        self.assertTrue(data['methods_restored'])
        self.assertNotIn('handle_preemptions', vars(wrapper))
        self.assertNotIn('wait', vars(worker))
        uninstall()


if __name__ == '__main__':
    unittest.main()

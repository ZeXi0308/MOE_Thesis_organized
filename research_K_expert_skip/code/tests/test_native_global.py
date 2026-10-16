"""CPU-only checks of temporary native-k mutation and honest accounting."""
import importlib.util
from contextlib import contextmanager
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
stub=types.ModuleType('vllm.v1.worker.gpu_worker')
stub.Worker=object
with mock.patch.dict(sys.modules,{'vllm.v1.worker.gpu_worker':stub}):
    spec=importlib.util.spec_from_file_location('_native_worker_test',CODE/'skip_worker.py')
    worker_module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_module)
with mock.patch.dict(sys.modules,{'vllm.v1.worker.gpu_worker':stub,'skip_worker':worker_module}):
    graph_spec=importlib.util.spec_from_file_location('_graph_worker_test',CODE/'skip_graph_worker.py')
    graph_module=importlib.util.module_from_spec(graph_spec)
    graph_spec.loader.exec_module(graph_module)


class NativeGlobalPolicy(unittest.TestCase):
    def test_graph_scalar_fill_enters_inference_mode_before_base_execute(self):
        active = False
        @contextmanager
        def inference_mode():
            nonlocal active
            active = True
            try:
                yield
            finally:
                active = False
        class Scalar:
            def fill_(scalar, value):
                self.assertTrue(active)
                scalar.value = value
        worker = graph_module.ExpertSkipGraphWorker()
        worker._skip_policy = object()
        scalar = worker._graph_actual_tokens = Scalar()
        output = types.SimpleNamespace(total_num_scheduled_tokens=3)
        def base_execute(instance, supplied):
            self.assertIs(instance, worker)
            self.assertIs(supplied, output)
            self.assertFalse(active)
            return "executed"
        torch_stub = types.SimpleNamespace(inference_mode=inference_mode)
        with mock.patch.dict(sys.modules, {"torch": torch_stub}), mock.patch.object(
                worker_module.ExpertSkipWorker, "execute_model", base_execute):
            self.assertEqual(worker.execute_model(output), "executed")
        self.assertIs(worker._graph_actual_tokens, scalar)
        self.assertEqual(scalar.value, 3)
        self.assertFalse(active)

    def test_graph_rpc_cannot_replace_captured_policy(self):
        worker=graph_module.ExpertSkipGraphWorker()
        worker._skip_spec={'mode':'marginal','backend':'triton'}
        worker._skip_policy=object()
        identity=worker._skip_policy
        worker._graph_count_routes=True
        resets=[]
        worker.skip_snapshot=lambda reset=False:resets.append(reset)
        worker.skip_execution_metadata=lambda:{}
        worker.skip_set_policy(dict(worker._skip_spec))
        self.assertIs(worker._skip_policy,identity)
        self.assertEqual(resets,[True])
        for spec,scope,trace in [(None,'decode',0),(worker._skip_spec,'all',0),
                                 (worker._skip_spec,'decode',1)]:
            with self.assertRaises(ValueError):
                worker.skip_set_policy(spec,scope,trace)
        self.assertIs(worker._skip_policy,identity)

    def test_calls_native_once_with_real_width_then_restores(self):
        router=types.SimpleNamespace(top_k=8,_routing_replay_out=None)
        calls=[]
        def route(value,*,marker):
            calls.append((router.top_k,value,marker))
            return 'weights',types.SimpleNamespace(shape=(2,router.top_k))
        result=worker_module.call_native_reduced_topk(router,route,('hidden',),{'marker':17},4)
        self.assertEqual(calls,[(4,'hidden',17)])
        self.assertEqual(result[1].shape,(2,4))
        self.assertEqual(router.top_k,8)

    def test_restores_on_error_and_never_reduces_below_one(self):
        router=types.SimpleNamespace(top_k=8,_routing_replay_out=None)
        def fail():
            self.assertEqual(router.top_k,1)
            raise RuntimeError('native failure')
        with self.assertRaisesRegex(RuntimeError,'native failure'):
            worker_module.call_native_reduced_topk(router,fail,(),{},99)
        self.assertEqual(router.top_k,8)

    def test_alias_and_reject_unsupported_quality_budget(self):
        self.assertEqual(worker_module.native_drop_count({'mode':'global_native','j':2}),2)
        for spec in ({'drop_j':2,'j':4},{'drop_j':-1},{'drop_j':True},{'mass_budget':.1}):
            with self.assertRaises(ValueError):
                worker_module.native_drop_count(spec)

    def test_native_spec_metadata_and_unmeasured_fields_survive_snapshot(self):
        worker=worker_module.ExpertSkipWorker()
        worker._skip_alignment=None
        worker._skip_install={'layer':{'top_k':8,'native_dynamic_k_supported':True}}
        spec={'mode':'global_native','j':4,'backend':'torch'}
        installed=worker.skip_set_policy(spec,scope='decode')
        self.assertEqual(installed['layers']['layer']['eligible_top_k'],4)
        self.assertEqual(installed['layers']['layer']['original_top_k'],8)
        self.assertEqual(worker._skip_policy.config.mode,'dense')
        torch_stub=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:None))
        with mock.patch.dict(sys.modules,{'torch':torch_stub}):
            snapshot=worker.skip_snapshot()
        self.assertEqual(snapshot['spec'],spec)
        self.assertIsNone(snapshot['layers']['totals']['removed_gate_mass_sum'])
        self.assertIsNone(snapshot['layers']['totals']['skipped_slots'])
        self.assertFalse(snapshot['layers']['removed_gate_mass_measured'])
        worker.skip_set_policy({**spec,'count_routes':False})
        self.assertIsNone(worker._skip_policy)
        with mock.patch.dict(sys.modules,{'torch':torch_stub}):
            uncounted=worker.skip_snapshot()
        self.assertEqual(uncounted['layers']['totals'],{})
        self.assertFalse(uncounted['layers']['logical_route_counters_enabled'])
        worker.skip_set_policy(None)
        self.assertIsNone(worker._skip_native_j)
        self.assertEqual(worker.skip_execution_metadata()['layer']['eligible_top_k'],8)


if __name__=='__main__':
    unittest.main()

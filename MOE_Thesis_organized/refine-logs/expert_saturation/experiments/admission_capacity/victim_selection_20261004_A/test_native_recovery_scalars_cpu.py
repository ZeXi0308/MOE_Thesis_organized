"""Targeted risk: scalar observer must forward exact native calls and return values."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

P=Path(__file__).parent/'candidate_native_equal_release_host_once_r01/pkg/native_offload_observer.py'
spec=importlib.util.spec_from_file_location('native_scalars_probe',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Target:
    def __init__(self):
        self.calls=[];self.answer=(None,False);self._jobs={}
        self._req_status={'r':NS(num_locally_computed_tokens=32,transfer_jobs=set())}
    def get_num_new_matched_tokens(self,r,n):
        self.calls.append(('lookup',r,n));return self.answer
    def update_state_after_alloc(self,r,b,n):
        self.calls.append(('allocated',r,b,n))
        if n:
            self._jobs[7]=NS(pending_count=1,is_store=False)
            self._req_status['r'].transfer_jobs.add(7)
        return 'native allocation return'
    def update_connector_output(self,o):
        self.calls.append(('update',o))
        for jid in o.kv_connector_worker_meta.completed_jobs:self._jobs.pop(jid,None)
        return 'native update return'
class OffloadingConnector:
    def __init__(self,target):self.connector_scheduler=target

class TestNativeScalars(unittest.TestCase):
    def setUp(self):
        self.t=Target();self.s=NS(connector=OffloadingConnector(self.t),
            kv_cache_manager=NS(block_pool=NS(get_num_free_blocks=lambda:5)))
        self.r=NS(request_id='r',num_preemptions=1,num_output_tokens=12,num_tokens=64,status=NS(name='PREEMPTED'))
        self.d,self.undo=m.install_recovery_lookup(self.s)
    def tearDown(self):self.undo()
    def test_native_lookup_return_identity_none_and_gpu_prefix(self):
        x=self.t.get_num_new_matched_tokens(self.r,32)
        self.assertIs(x,self.t.answer);self.assertEqual(self.d['lookup'][0]['local_computed_tokens'],32)
        self.assertIsNone(self.d['lookup'][0]['offered_external_tokens'])
        self.assertEqual(self.t.calls,[('lookup',self.r,32)])
    def test_alloc_requested_is_separate_from_acknowledged(self):
        b=NS(blocks=[[1,2,3,4]])
        self.assertEqual(self.t.update_state_after_alloc(self.r,b,16),'native allocation return')
        self.assertEqual(self.d['allocated'][0]['requested_external_tokens'],16)
        self.assertEqual(self.d['load_acknowledgements'],[])
        o=NS(kv_connector_worker_meta=NS(completed_jobs={7:1}))
        self.assertEqual(self.t.update_connector_output(o),'native update return')
        self.assertTrue(self.d['load_acknowledgements'][0]['native_job_removed'])
        self.assertEqual(self.d['load_acknowledgements'][0]['preemptions'],1)
        self.assertEqual(len(self.t.calls),2)
    def test_no_initial_request_logging_or_residual_wrapper(self):
        self.r.num_preemptions=0
        self.t.get_num_new_matched_tokens(self.r,0)
        self.t.update_state_after_alloc(self.r,NS(blocks=[[]]),0)
        self.assertEqual(self.d['lookup'],[]);self.assertEqual(self.d['allocated'],[])
        self.undo()
        for name in ('get_num_new_matched_tokens','update_state_after_alloc','update_connector_output'):
            self.assertNotIn(name,vars(self.t))
        self.undo=lambda:None

if __name__=='__main__':unittest.main()

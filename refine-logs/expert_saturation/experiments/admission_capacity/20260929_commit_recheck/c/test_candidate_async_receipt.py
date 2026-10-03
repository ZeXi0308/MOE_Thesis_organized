"""Focused CPU receipt checks against the exact pre-fix H1 candidate bytes."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest


HERE = Path(__file__).resolve().parent
GH_BASE = Path(os.environ['H1_ACCEPTED_GH_BASE'])
GH_PACKAGE_ROOT = Path(os.environ['H1_ACCEPTED_GH_PACKAGE_ROOT'])
BASE_SHA = '24629c0bbd8aa2c121310a053826fdb3bafd7159bf6f2c9444f514ec9a56e31c'
CANDIDATE_SHA = '8aa05c3ce0d5fa8c0971a43a24256409984248be6eb014c21624c79610da850d'
sys.path.insert(0, str(GH_PACKAGE_ROOT))
sys.dont_write_bytecode = True


class Queue(list):
    def remove_request(self, request):
        self.remove(request)

    def prepend_request(self, request):
        self.insert(0, request)


def load_patched():
    base = GH_BASE.read_bytes()
    if hashlib.sha256(base).hexdigest() != BASE_SHA:
        raise RuntimeError('G/H adapter base drift')
    temp = tempfile.TemporaryDirectory()
    path = Path(temp.name) / 'pkg/staged_store_rotation.py'
    path.parent.mkdir(parents=True)
    path.write_bytes(base)
    subprocess.run(['git', 'apply', str(HERE / 'accepted_gh_commit_recheck.patch')],
                   cwd=temp.name, check=True, capture_output=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != CANDIDATE_SHA:
        raise RuntimeError('Pre-fix H1 candidate drift')
    subprocess.run(['git', 'apply', str(HERE / 'candidate_h1_async_receipt.patch')],
                   cwd=temp.name, check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location('h1_async_receipt_test_adapter', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    tree = ast.parse(path.read_text())
    install = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install')
    start = next(i for i, n in enumerate(install.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'step' for t in n.targets))
    fn = ast.parse('def fixture(scheduler, native, owned, pool, cs):\n'
                   ' manager=scheduler.kv_cache_manager\n'
                   ' expected_requests=32\n open_population=True\n'
                   " diagnostic=False\n store_scope='selected'\n"
                   " global_cooldown_steps=20\n population_mode='open'\n"
                   ' oldcalc=cs._calc_num_offloadable_tokens\n hadcalc=True\n'
                   ' save=False\n commit_recheck=True\n').body[0]
    fn.body += deepcopy(install.body[start:])
    env = dict(vars(adapter))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 '<candidate-actual-closures>', 'exec'), env)
    return temp, adapter, env['fixture']


class CandidateAsyncReceiptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp, cls.adapter, cls.fixture = load_patched()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def case(self, outcome):
        blocks = {i: NS(block_id=i, is_null=False, ref_cnt=int(i <= 2)) for i in range(1, 6)}
        owned = {'v': [blocks[1], blocks[2]], 't': []}

        def request(rid, prompt, output, computed, status):
            req = NS(request_id=rid, num_prompt_tokens=prompt, num_output_tokens=output,
                     num_computed_tokens=computed, max_tokens=100, status=NS(name=status))
            req.is_finished = lambda: req.status.name.startswith('FINISHED')
            return req

        victim = request('v', 16, 17, 32, 'RUNNING')
        target = request('t', 32, 1, 0, 'PREEMPTED')
        pool = NS(blocks=blocks,
                  get_num_free_blocks=lambda: sum(block.ref_cnt == 0 for block in blocks.values()))
        manager = NS(block_pool=pool, get_blocks=lambda rid: NS(
            get_block_ids=lambda: [[block.block_id for block in owned[rid]]]))
        scheduler = NS(requests={'v': victim, 't': target}, running=[victim],
                       waiting=Queue([target]), skipped_waiting=Queue(),
                       max_num_running_reqs=2, num_waiting_for_streaming_input=0,
                       kv_cache_manager=manager)
        cs = NS(_calc_num_offloadable_tokens=lambda rs, n: n,
                _req_status={'v': NS(transfer_jobs=set()), 't': NS(transfer_jobs=set())},
                _jobs={})
        plan = self.adapter.prepare(1,
            self.adapter.RequestState('v', 31, 16, 16, 100, 'RUNNING', (1, 2)),
            self.adapter.RequestState('t', 0, 32, 1, 100, 'PREEMPTED', ()), 3)

        def preempt(req, timestamp):
            raise AssertionError('direct receipt must not preempt victim')

        scheduler._preempt_request = preempt

        def native():
            preempted = []
            scheduler._rotation_begin(preempted, 0.0)
            self.assertEqual(preempted, [])
            self.assertEqual(scheduler._rotation_target, 't')
            if outcome != 'missing_blocks':
                take = [blocks[3], blocks[4], blocks[5]]
                for block in take:
                    block.ref_cnt = 1
                owned['t'] = take
            scheduler.waiting.remove_request(target)
            scheduled = {'v': 1}
            resumed = []
            load_jobs = {}
            if outcome == 'compute':
                target.status = NS(name='RUNNING')
                scheduler.running.append(target)
                scheduled['t'] = 1
                resumed.append('t')
            else:
                target.status = NS(name='WAITING_FOR_REMOTE_KVS')
                scheduler.skipped_waiting.append(target)
                cs._req_status['t'].transfer_jobs.add(11)
                cs._jobs[11] = NS(req_id='t', is_store=False)
                if outcome in ('load', 'missing_blocks'):
                    load_jobs[11] = NS(req_id='t')
            meta = NS(store_jobs={}, load_jobs=load_jobs, jobs_to_flush=set())
            return NS(num_scheduled_tokens=scheduled, preempted_req_ids=set(),
                      scheduled_cached_reqs=NS(resumed_req_ids=resumed),
                      kv_connector_metadata=meta)

        data, undo = type(self).fixture(scheduler, native, owned, pool, cs)
        cells = dict(zip(scheduler._rotation_begin.__code__.co_freevars,
                         scheduler._rotation_begin.__closure__))
        cells['step'].cell_contents = 2
        cells['plan'].cell_contents = plan
        cells['plan_request_refs'].cell_contents = (victim, target)
        return scheduler, data, undo

    def test_compute_receipt(self):
        scheduler, data, undo = self.case('compute')
        try:
            scheduler.schedule()
            self.assertEqual(data['direct_commits'], 1)
            event = next(e for e in data['events'] if e['event'] == 'direct_commit')
            self.assertEqual((event['native_admission'], event['scheduled_tokens'],
                              event['load_job_ids']), ('SCHEDULED_TOKENS', 1, []))
        finally:
            undo()

    def test_async_load_admission_receipt(self):
        scheduler, data, undo = self.case('load')
        try:
            scheduler.schedule()
            self.assertEqual(data['direct_commits'], 1)
            event = next(e for e in data['events'] if e['event'] == 'direct_commit')
            self.assertEqual((event['native_admission'], event['scheduled_tokens'],
                              event['load_job_ids']), ('ASYNC_LOAD_ADMITTED', 0, [11]))
            self.assertFalse(any(e['event'] == 'target_new_output' for e in data['events']))
        finally:
            undo()

    def test_async_status_without_load_metadata_fails(self):
        scheduler, data, undo = self.case('missing_job')
        try:
            with self.assertRaisesRegex(RuntimeError, 'lacks native load admission'):
                scheduler.schedule()
            self.assertEqual(data['direct_commits'], 0)
            self.assertEqual(data['status'], 'ERROR')
            self.assertFalse(any(e['event'] == 'direct_commit' for e in data['events']))
        finally:
            undo()

    def test_async_status_without_physical_blocks_fails(self):
        scheduler, data, undo = self.case('missing_blocks')
        try:
            with self.assertRaisesRegex(RuntimeError, 'lacks native load admission'):
                scheduler.schedule()
            self.assertEqual(data['direct_commits'], 0)
            self.assertEqual(data['status'], 'ERROR')
        finally:
            undo()


if __name__ == '__main__':
    unittest.main()

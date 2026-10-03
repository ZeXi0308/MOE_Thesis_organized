"""CPU checks of the patched adapter closures; no native transfer or GPU claim."""
import ast
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
SOURCE = HERE.parents[1] / 'staged_store_rotation.py'
MODE = os.environ.get('H1_TEST_MODE', 'shared')
if MODE == 'gh':
    RELATIVE = Path('pkg/staged_store_rotation.py')
    PATCH = HERE / 'accepted_gh_commit_recheck.patch'
    GH_BASE = Path(os.environ['H1_ACCEPTED_GH_BASE'])
    BASE_BYTES = GH_BASE.read_bytes()
    assert __import__('hashlib').sha256(BASE_BYTES).hexdigest() == '24629c0bbd8aa2c121310a053826fdb3bafd7159bf6f2c9444f514ec9a56e31c'
    sys.path.insert(0, os.environ['H1_ACCEPTED_GH_PACKAGE_ROOT'])
elif MODE == 'shared':
    RELATIVE = Path('refine-logs/expert_saturation/experiments/admission_capacity/staged_store_rotation.py')
    PATCH = HERE / 'commit_recheck.patch'
    BASE_BYTES = subprocess.run(['git', 'show', 'HEAD:' + str(RELATIVE)], cwd=REPO,
                                check=True, capture_output=True).stdout
else:
    raise ValueError('H1_TEST_MODE must be shared or gh')
sys.path.insert(0, str(SOURCE.parent))
sys.dont_write_bytecode = True


class Queue(list):
    def remove_request(self, request):
        self.remove(request)

    def prepend_request(self, request):
        self.insert(0, request)


def load_patched():
    temp = tempfile.TemporaryDirectory()
    path = Path(temp.name) / RELATIVE
    path.parent.mkdir(parents=True)
    path.write_bytes(BASE_BYTES)
    if 'def _direct_resume_reason(' not in path.read_text():
        subprocess.run(['git', 'apply', str(PATCH)], cwd=temp.name, check=True,
                       capture_output=True, text=True)
    spec = importlib.util.spec_from_file_location('staged_store_rotation_recheck_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(path.read_text())
    install = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install')
    start = next(i for i, n in enumerate(install.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'step' for t in n.targets))
    fn = ast.parse('def fixture(scheduler, native, owned, pool, cs, save, commit_recheck):\n'
                   ' manager=scheduler.kv_cache_manager\n'
                   ' expected_requests=32\n'
                   " open_population=%s\n" % (MODE == 'gh') +
                   " diagnostic=False\n store_scope='selected'\n"
                   " global_cooldown_steps=20\n population_mode='open'\n" +
                   ' oldcalc=cs._calc_num_offloadable_tokens\n'
                   ' hadcalc=True\n').body[0]
    fn.body += deepcopy(install.body[start:])
    env = dict(vars(module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 '<patched-actual-closures>', 'exec'), env)
    return temp, module, env['fixture']


class CommitRecheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp, cls.adapter, cls.fixture = load_patched()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def case(self, *, enabled=True, free=3, slots=2, streaming=0, mutate=None, run=True,
             finish_during_native=False):
        blocks = {i: NS(block_id=i, is_null=False, ref_cnt=int(i <= 2)) for i in range(1, 3 + free)}
        owned = {'v': [blocks[1], blocks[2]], 't': []}
        def request(rid, p, o, c, status):
            req = NS(request_id=rid, num_prompt_tokens=p, num_output_tokens=o,
                num_computed_tokens=c, max_tokens=100, status=NS(name=status))
            req.is_finished = lambda: req.status.name.startswith('FINISHED')
            return req
        victim = request('v', 16, 16, 31, 'RUNNING')
        target = request('t', 32, 1, 0, 'PREEMPTED')
        pool = NS(blocks=blocks,
            get_num_free_blocks=lambda: sum(b.ref_cnt == 0 for b in blocks.values()))
        manager = NS(block_pool=pool, get_blocks=lambda rid: NS(
            get_block_ids=lambda: [[b.block_id for b in owned[rid]]]))
        scheduler = NS(requests={'v': victim, 't': target}, running=[victim],
            waiting=Queue([target]), skipped_waiting=Queue(),
            max_num_running_reqs=slots, num_waiting_for_streaming_input=streaming,
            kv_cache_manager=manager)
        cs = NS(_calc_num_offloadable_tokens=lambda rs, n: n,
                _req_status={'v': NS(transfer_jobs=set()), 't': NS(transfer_jobs=set())},
                _jobs={})
        plan = self.adapter.prepare(1, self.adapter.RequestState('v', 31, 16, 16, 100, 'RUNNING', (1, 2)),
            self.adapter.RequestState('t', 0, 32, 1, 100, 'PREEMPTED', ()), free)
        victim.num_computed_tokens += 1
        victim.num_output_tokens += 1

        def preempt(req, timestamp):
            for block in owned[req.request_id]:
                block.ref_cnt -= 1
            owned[req.request_id] = []
            req.status = NS(name='PREEMPTED')
            scheduler.waiting.prepend_request(req)
        scheduler._preempt_request = preempt

        def native():
            preempted = []
            scheduler._rotation_begin(preempted, 0.0)
            scheduled = {r.request_id: 1 for r in scheduler.running}
            resumed = []
            if scheduler._rotation_target is not None:
                req = scheduler.requests[scheduler._rotation_target]
                need = (req.num_prompt_tokens + req.num_output_tokens + 15) // 16
                if (req in scheduler.waiting
                        and len(scheduler.running) + scheduler.num_waiting_for_streaming_input
                            < scheduler.max_num_running_reqs
                        and pool.get_num_free_blocks() >= need):
                    take = [b for b in blocks.values() if b.ref_cnt == 0][:need]
                    for block in take:
                        block.ref_cnt = 1
                    owned[req.request_id] = take
                    scheduler.waiting.remove_request(req)
                    scheduler.running.append(req)
                    req.status = NS(name='RUNNING')
                    scheduled[req.request_id] = 1
                    resumed.append(req.request_id)
                    if finish_during_native:
                        req.status = NS(name='FINISHED_STOPPED')
                        scheduler.running.remove(req)
                        del scheduler.requests[req.request_id]
                        for block in owned[req.request_id]:
                            block.ref_cnt = 0
                        owned[req.request_id] = []
            meta = NS(store_jobs={}, load_jobs={},
                jobs_to_flush=set(cs._req_status['v'].transfer_jobs) if preempted else set())
            return NS(num_scheduled_tokens=scheduled,
                preempted_req_ids={r.request_id for r in preempted},
                scheduled_cached_reqs=NS(resumed_req_ids=resumed),
                kv_connector_metadata=meta)

        data, undo = type(self).fixture(scheduler, native, owned, pool, cs, False, enabled)
        cells = dict(zip(scheduler._rotation_begin.__code__.co_freevars,
                         scheduler._rotation_begin.__closure__))
        cells['step'].cell_contents = 2
        cells['plan'].cell_contents = plan
        cells['plan_request_refs'].cell_contents = (victim, target)
        if mutate:
            mutate(scheduler, owned, blocks, cs, cells)
        result = scheduler.schedule() if run else None
        return NS(scheduler=scheduler, target=target, victim=victim, owned=owned,
                  blocks=blocks, cs=cs, cells=cells, result=result, data=data, undo=undo)

    def test_direct_and_default_off(self):
        direct = self.case()
        self.assertEqual(direct.data['direct_commits'], 1)
        self.assertEqual(direct.data['applied_rotations'], 0)
        self.assertEqual(direct.result.preempted_req_ids, set())
        self.assertIn(direct.victim, direct.scheduler.running)
        self.assertIn(direct.target, direct.scheduler.running)
        self.assertIsNone(direct.cells['plan'].cell_contents)
        direct.undo()
        off = self.case(enabled=False)
        self.assertEqual(off.data['direct_commits'], 0)
        self.assertEqual(off.result.preempted_req_ids, {'v'})
        off.undo()
        def pending_store(s, owned, blocks, cs, cells):
            cs._req_status['v'].transfer_jobs.add(7)
            cs._jobs[7] = NS(is_store=True)
        direct = self.case(mutate=pending_store)
        self.assertEqual(direct.data['direct_commits'], 1)
        self.assertFalse(direct.result.kv_connector_metadata.jobs_to_flush)
        self.assertEqual(direct.cs._req_status['v'].transfer_jobs, {7})
        direct.undo()

    def test_slot_and_block_fallback(self):
        for kwargs, reason in [({'slots': 1}, 'KEEP_NO_SEQUENCE_SLOT'),
                               ({'slots': 2, 'streaming': 1}, 'KEEP_NO_SEQUENCE_SLOT'),
                               ({'free': 2}, 'KEEP_INSUFFICIENT_FREE_BLOCKS')]:
            with self.subTest(reason=reason):
                case = self.case(**kwargs)
                self.assertEqual(case.result.preempted_req_ids, {'v'})
                self.assertEqual(case.data['direct_commits'], 0)
                self.assertIn(reason, [e['reason'] for e in case.data['events']
                                       if e['event'] == 'commit_recheck'])
                case.undo()
        def unknown_slot(s, *_):
            del s.max_num_running_reqs
        # Native fixture needs a slot value to run, so check the read-only gate directly.
        case = self.case(run=False)
        unknown_slot(case.scheduler)
        self.assertEqual(self.adapter._direct_resume_reason(case.scheduler,
            case.scheduler.kv_cache_manager, case.scheduler.kv_cache_manager.block_pool,
            case.owned, case.cs, case.target), 'KEEP_NO_SEQUENCE_SLOT')
        case.undo()
        # Unknown streaming occupancy must not be treated as a free native slot.
        case = self.case(run=False)
        del case.scheduler.num_waiting_for_streaming_input
        self.assertEqual(self.adapter._direct_resume_reason(case.scheduler,
            case.scheduler.kv_cache_manager, case.scheduler.kv_cache_manager.block_pool,
            case.owned, case.cs, case.target), 'KEEP_NO_SEQUENCE_SLOT')
        case.undo()

    def test_ownership_pending_and_partial_load(self):
        def shared(s, owned, blocks, cs, cells):
            blocks[1].ref_cnt = 2
        case = self.case(mutate=shared)
        self.assertEqual(case.data['direct_commits'], 0)
        self.assertIn('KEEP_SHARED_OR_INVALID_BLOCK', [e['reason'] for e in case.data['events']
            if e['event'] == 'commit_recheck'])
        case.undo()
        def pending_victim(s, owned, blocks, cs, cells):
            cs._req_status['v'].transfer_jobs.add(9)
            cs._jobs[9] = NS(is_store=False)
        case = self.case(mutate=pending_victim)
        self.assertFalse(case.result.preempted_req_ids)
        self.assertEqual(case.data['direct_commits'], 0)
        self.assertEqual(next(e['reason'] for e in case.data['events']
            if e['event'] == 'commit_check'), 'CANCEL_VICTIM_PENDING_OR_UNKNOWN_TRANSFER')
        case.undo()
        def foreign_pending(s, owned, blocks, cs, cells):
            s.skipped_waiting.append(NS(request_id='other', status=NS(name='WAITING_FOR_REMOTE_KVS')))
        case = self.case(mutate=foreign_pending)
        self.assertFalse(case.result.preempted_req_ids)
        self.assertEqual(case.data['direct_commits'], 0)
        case.undo()
        def partial(s, owned, blocks, cs, cells):
            blocks[3].ref_cnt = 1
            owned['t'] = [blocks[3]]
        case = self.case(mutate=partial)
        self.assertEqual(case.data['direct_commits'], 0)
        self.assertFalse(case.result.preempted_req_ids)
        case.undo()
        case = self.case(run=False)
        case.cs._req_status['t'].transfer_jobs.add(8)
        self.assertEqual(self.adapter._direct_resume_reason(case.scheduler,
            case.scheduler.kv_cache_manager, case.scheduler.kv_cache_manager.block_pool,
            case.owned, case.cs, case.target), 'KEEP_PENDING_TARGET_TRANSFER')
        case.undo()

    def test_stale_identity_and_terminal_release(self):
        def stale(s, owned, blocks, cs, cells):
            cells['step'].cell_contents = 3
        case = self.case(mutate=stale)
        self.assertEqual(case.data['direct_commits'], 0)
        self.assertFalse(case.result.preempted_req_ids)
        case.undo()
        def changed_identity(s, owned, blocks, cs, cells):
            old = s.requests['t']
            new = NS(**vars(old))
            s.requests['t'] = new
            s.waiting[0] = new
        case = self.case(mutate=changed_identity)
        self.assertEqual(next(e['reason'] for e in case.data['events']
            if e['event'] == 'commit_check'), 'CANCEL_REQUEST_IDENTITY_CHANGED')
        case.undo()
        def target_finished_before_commit(s, owned, blocks, cs, cells):
            s.requests['t'].status = NS(name='FINISHED_STOPPED')
        case = self.case(mutate=target_finished_before_commit)
        self.assertEqual(case.data['direct_commits'], 0)
        self.assertFalse(case.result.preempted_req_ids)
        case.undo()
        def pending_load_before_commit(s, owned, blocks, cs, cells):
            s.requests['t'].status = NS(name='WAITING_FOR_REMOTE_KVS')
        case = self.case(mutate=pending_load_before_commit)
        self.assertEqual(case.data['direct_commits'], 0)
        self.assertFalse(case.result.preempted_req_ids)
        case.undo()
        case = self.case(finish_during_native=True)
        self.assertEqual(case.data['direct_commits'], 1)
        if MODE == 'gh':
            case.scheduler.schedule()
        self.assertIsNone(case.cells['protected'].cell_contents)
        case.undo()
        case = self.case()
        case.target.status = NS(name='FINISHED_STOPPED')
        case.scheduler.running.remove(case.target)
        del case.scheduler.requests['t']
        case.scheduler.schedule()
        self.assertIsNone(case.cells['protected'].cell_contents)
        case.undo()


if __name__ == '__main__':
    unittest.main()

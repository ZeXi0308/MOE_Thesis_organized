"""Three targeted checks using the observed step-545 funding boundary; no GPU claim."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'candidate_fit_first_r01' / 'pkg'))
sys.path.insert(0, str(HERE))
from absence_rotation import AbsenceRotation, RequestView, RotationConfig

spec = importlib.util.spec_from_file_location('capacity_adapter', HERE / 'staged_store_rotation.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def tracker_and_rows():
    tracker = AbsenceRotation(config=RotationConfig(min_steps_between_swaps=0), victim_order='most_output')
    tracker.absent_since['t'] = 515
    rows = [RequestView('old', 1950, 1480, 2504, 471), RequestView('alt', 2727, 2260, 3284, 468)]
    return tracker, rows


class Queue(list):
    def remove_request(self, request):
        self.remove(request)

    def prepend_request(self, request):
        self.insert(0, request)


def closures():
    tree = ast.parse((HERE / 'staged_store_rotation.py').read_text())
    install = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install')
    start = next(i for i, n in enumerate(install.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'step' for t in n.targets))
    seed = ast.parse('''def fixture(scheduler, native, owned, pool, cs):
 manager=scheduler.kv_cache_manager
 expected_requests=32
 open_population=True
 diagnostic=False
 store_scope='selected'
 global_cooldown_steps=0
 population_mode='open'
 save=True
 commit_recheck=False
 fit_first_resume=False
 capacity_victim=True
 oldcalc=cs._calc_num_offloadable_tokens
 hadcalc=True
''').body[0]
    seed.body += deepcopy(install.body[start:])
    namespace = vars(adapter).copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[seed], type_ignores=[])),
                 '<actual-capacity-closures>', 'exec'), namespace)
    return namespace['fixture']


def native_case(omit_preemption_receipt=False):
    def request(rid, p, o, c, status):
        req = NS(request_id=rid, num_prompt_tokens=p, num_output_tokens=o,
                 num_computed_tokens=c, max_tokens=1024, status=NS(name=status))
        req.is_finished = lambda: req.status.name.startswith('FINISHED')
        return req
    reqs = {'old': request('old', 1480, 471, 1950, 'RUNNING'),
            'alt': request('alt', 2260, 468, 2727, 'RUNNING'),
            't': request('t', 3041, 444, 0, 'PREEMPTED')}
    blocks = {i: NS(block_id=i, is_null=False, ref_cnt=int(i < 293)) for i in range(385)}
    owned = {'old': [blocks[i] for i in range(122)],
             'alt': [blocks[i] for i in range(122, 293)], 't': []}
    pool = NS(blocks=blocks, get_num_free_blocks=lambda: sum(b.ref_cnt == 0 for b in blocks.values()))
    manager = NS(block_pool=pool, get_blocks=lambda rid: NS(
        get_block_ids=lambda: [[b.block_id for b in owned[rid]]]))
    scheduler = NS(requests=reqs, running=[reqs['old'], reqs['alt']], waiting=Queue([reqs['t']]),
        skipped_waiting=Queue(), max_num_running_reqs=32, num_waiting_for_streaming_input=0,
        kv_cache_manager=manager)
    cs = NS(_calc_num_offloadable_tokens=lambda rs, n: n,
            _req_status={rid: NS(transfer_jobs=set()) for rid in reqs}, _jobs={})
    def preempt(req, timestamp):
        for block in owned[req.request_id]:
            block.ref_cnt = 0
        owned[req.request_id] = []
        req.status = NS(name='PREEMPTED')
        scheduler.waiting.prepend_request(req)
    scheduler._preempt_request = preempt
    def native():
        preempted = []
        scheduler._rotation_begin(preempted, 0.0)
        scheduled = {r.request_id: 1 for r in scheduler.running if not scheduler._rotation_hold(r)}
        resumed = []
        if scheduler._rotation_target is not None:
            req = reqs[scheduler._rotation_target]
            if req in scheduler.waiting:
                need = (req.num_prompt_tokens + req.num_output_tokens + 15) // 16
                take = [b for b in blocks.values() if b.ref_cnt == 0][:need]
                assert len(take) == need
                for block in take:
                    block.ref_cnt = 1
                owned[req.request_id] = take
                scheduler.waiting.remove_request(req)
                scheduler.running.append(req)
                req.status = NS(name='RUNNING')
                scheduled[req.request_id] = 1
                resumed.append(req.request_id)
        return NS(num_scheduled_tokens=scheduled,
            preempted_req_ids=set() if omit_preemption_receipt else {r.request_id for r in preempted},
            scheduled_cached_reqs=NS(resumed_req_ids=resumed),
            kv_connector_metadata=NS(store_jobs={}, load_jobs={}, jobs_to_flush=set()))
    data, undo = closures()(scheduler, native, owned, pool, cs)
    cells = dict(zip(scheduler._rotation_begin.__code__.co_freevars, scheduler._rotation_begin.__closure__))
    cells['step'].cell_contents = 545
    cells['tracker'].cell_contents.absent_since['t'] = 515
    return NS(s=scheduler, reqs=reqs, data=data, cells=cells, undo=undo)


class CapacityVictimTests(unittest.TestCase):
    def test_step545_funding_filter_preserves_other_eligibility(self):
        tracker, rows = tracker_and_rows()
        rows.extend([RequestView('progress_protected', 1929, 1000, 2024, 930),
                     RequestView('absence_limited', 1899, 1000, 2024, 900),
                     RequestView('recently_resumed', 1898, 1000, 2024, 899)])
        tracker.absence_count['absence_limited'] = 8
        tracker.resident_since['recently_resumed'] = 530
        held = dict(old=122, alt=171, progress_protected=300, absence_limited=300, recently_resumed=300)
        d = tracker.decide(545, rows, ['t'], 92, {'t': 218}, held)
        self.assertEqual((d.action, d.resume_id, d.original_victim_id, d.victim_id), ('rotate', 't', 'old', 'alt'))
        self.assertEqual(d.absence_steps, 30)

    def test_off_is_original_and_no_fundable_victim_is_noop(self):
        tracker, rows = tracker_and_rows()
        off = tracker.decide(545, rows, ['t'], 92, {'t': 218})
        self.assertEqual((off.action, off.victim_id, off.original_victim_id), ('rotate', 'old', None))
        tracker, rows = tracker_and_rows()
        previous_swap = tracker.last_swap_step
        unfunded = tracker.decide(545, rows, ['t'], 92, {'t': 218}, {'old': 122, 'alt': 125})
        self.assertEqual(unfunded.action, 'noop')
        self.assertEqual(unfunded.original_victim_id, 'old')
        self.assertEqual(tracker.last_swap_step, previous_swap)
        for other in ('fit_first_resume', 'commit_recheck'):
            with self.assertRaisesRegex(ValueError, 'Capacity-victim'):
                adapter.install(None, vllm_config=None, save=True, block_size=16,
                                capacity_victim=True, **{other: True})

    def test_commit_count_requires_native_checks_and_cancel_never_counts(self):
        for outcome in ('commit', 'cancel', 'missing_native_receipt'):
            with self.subTest(outcome=outcome):
                c = native_case(omit_preemption_receipt=outcome == 'missing_native_receipt')
                c.s.schedule()
                self.assertEqual(c.data['capacity_victim_commits'], 0)
                choice = next(e for e in c.data['events'] if e['event'] == 'capacity_victim_choice')
                self.assertEqual((choice['old_held_blocks'], choice['new_held_blocks']), (122, 171))
                for rid in ('old', 'alt'):
                    c.reqs[rid].num_output_tokens += 1
                    c.reqs[rid].num_computed_tokens += 1
                if outcome == 'cancel':
                    c.cells['step'].cell_contents = 548
                if outcome == 'missing_native_receipt':
                    with self.assertRaisesRegex(RuntimeError, 'Native preemption notification absent'):
                        c.s.schedule()
                else:
                    c.s.schedule()
                self.assertEqual(c.data['capacity_victim_commits'], int(outcome == 'commit'))
                commits = [e for e in c.data['events'] if e['event'] == 'capacity_victim_commit']
                self.assertEqual(len(commits), int(outcome == 'commit'))
                if commits:
                    self.assertEqual((commits[0]['choice_step'], commits[0]['step'], commits[0]['new_victim']), (545, 546, 'alt'))
                if outcome == 'cancel':
                    self.assertIsNone(c.cells['capacity_plan_choice'].cell_contents)
                c.undo()


if __name__ == '__main__':
    unittest.main()

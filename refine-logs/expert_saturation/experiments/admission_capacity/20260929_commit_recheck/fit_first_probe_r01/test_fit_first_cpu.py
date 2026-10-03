"""Three focused closure fixtures; mock admission is not native GPU evidence."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'candidate_h1_perf_b1_r01' / 'pkg'))
spec = importlib.util.spec_from_file_location('fit_first_adapter', HERE / 'staged_store_rotation.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
tree = ast.parse((HERE / 'staged_store_rotation.py').read_text())
install = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install')
start = next(i for i, n in enumerate(install.body) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'step' for t in n.targets))
seed = ast.parse('''def fixture(scheduler, native, owned, pool, cs, fit_first_resume):
 manager=scheduler.kv_cache_manager
 expected_requests=32
 open_population=True
 diagnostic=False
 store_scope='selected'
 global_cooldown_steps=0
 population_mode='open'
 save=True
 commit_recheck=False
 oldcalc=cs._calc_num_offloadable_tokens
 hadcalc=True
''').body[0]
seed.body += deepcopy(install.body[start:])
namespace = vars(adapter).copy()
exec(compile(ast.fix_missing_locations(ast.Module(body=[seed], type_ignores=[])),
             '<actual-fit-first-closures>', 'exec'), namespace)
fixture = namespace['fixture']


class Queue(list):
    def remove_request(self, request):
        self.remove(request)

    def prepend_request(self, request):
        self.insert(0, request)


def request(rid, prompt, output, computed, status):
    req = NS(request_id=rid, num_prompt_tokens=prompt, num_output_tokens=output,
             num_computed_tokens=computed, max_tokens=100, status=NS(name=status))
    req.is_finished = lambda: req.status.name.startswith('FINISHED')
    return req


def case(kind='scheduled', enabled=True):
    blocks = {i: NS(block_id=i, is_null=False, ref_cnt=int(i < 3)) for i in range(5)}
    reqs = {'v': request('v', 16, 17, 32, 'RUNNING'),
            't': request('t', 48, 1, 0, 'PREEMPTED'),
            'a': request('a', 16, 1, 0, 'PREEMPTED'),
            'b': request('b', 16, 1, 0, 'PREEMPTED')}
    owned = {rid: [] for rid in reqs}
    owned['v'] = [blocks[i] for i in range(3)]
    pool = NS(blocks=blocks, get_num_free_blocks=lambda: sum(b.ref_cnt == 0 for b in blocks.values()))
    manager = NS(block_pool=pool, get_blocks=lambda rid: NS(
        get_block_ids=lambda: [[b.block_id for b in owned[rid]]]))
    scheduler = NS(requests=reqs, running=[reqs['v']], waiting=Queue([reqs['t'], reqs['a'], reqs['b']]),
        skipped_waiting=Queue(), max_num_running_reqs=32, num_waiting_for_streaming_input=0,
        kv_cache_manager=manager, _inflight_prefill_reserved_blocks=lambda: 0)
    cs = NS(_calc_num_offloadable_tokens=lambda rs, n: n,
        _req_status={rid: NS(transfer_jobs=set()) for rid in reqs}, _jobs={})

    def native():
        preempted = []
        scheduler._rotation_begin(preempted, 0.0)
        scheduled = {r.request_id: 1 for r in scheduler.running if not scheduler._rotation_hold(r)}
        resumed, loads = [], {}
        if scheduler._rotation_target is not None:
            req = reqs[scheduler._rotation_target]
            if req in scheduler.waiting and kind != 'rejected':
                need = (req.num_prompt_tokens + req.num_output_tokens + 15) // 16
                take = [b for b in blocks.values() if b.ref_cnt == 0][:need]
                assert len(take) == need
                for block in take:
                    block.ref_cnt = 1
                owned[req.request_id] = take
                scheduler.waiting.remove_request(req)
                if kind.startswith('async'):
                    req.status = NS(name='WAITING_FOR_REMOTE_KVS')
                    scheduler.skipped_waiting.append(req)
                    if kind == 'async':
                        loads[100] = NS(req_id=req.request_id)
                        cs._jobs[100] = NS(req_id=req.request_id, is_store=False)
                else:
                    req.status = NS(name='RUNNING')
                    scheduler.running.append(req)
                    scheduled[req.request_id] = 1
                    resumed.append(req.request_id)
        return NS(num_scheduled_tokens=scheduled, preempted_req_ids=set(),
            scheduled_cached_reqs=NS(resumed_req_ids=resumed),
            kv_connector_metadata=NS(store_jobs={}, load_jobs=loads, jobs_to_flush=set()))

    data, undo = fixture(scheduler, native, owned, pool, cs, enabled)
    cells = dict(zip(scheduler._rotation_begin.__code__.co_freevars, scheduler._rotation_begin.__closure__))
    cells['step'].cell_contents = 40
    tracker = cells['tracker'].cell_contents
    tracker.absent_since.update(t=0, a=5, b=5)
    return NS(s=scheduler, reqs=reqs, owned=owned, blocks=blocks, pool=pool,
              cs=cs, data=data, cells=cells, tracker=tracker, undo=undo)


class FitFirstTests(unittest.TestCase):
    def test_longest_eligible_fit_and_native_receipt_then_new_output(self):
        for kind, admission in [('scheduled', 'SCHEDULED_TOKENS'), ('async', 'ASYNC_LOAD_ADMITTED')]:
            with self.subTest(kind=kind):
                c = case(kind)
                c.s.schedule()
                choice = next(e for e in c.data['events'] if e['event'] == 'fit_first_choice')
                receipt = next(e for e in c.data['events'] if e['event'] == 'fit_first_admitted')
                self.assertEqual((choice['target'], choice['original_target'], choice['planned_victim']), ('b', 't', 'v'))
                self.assertEqual((choice['free_blocks'], choice['required_blocks'], choice['absence_steps']), (2, 2, 35))
                self.assertEqual(receipt['native_admission'], admission)
                self.assertGreaterEqual(receipt['host_perf_counter_s'], choice['host_perf_counter_s'])
                self.assertEqual(c.data['fit_first_resumes'], 1)
                self.assertEqual(c.data['applied_rotations'], 0)
                self.assertEqual(c.data['direct_commits'], 0)
                self.assertFalse(any(e['event'] == 'prepare' for e in c.data['events']))
                self.assertIsNone(c.cells['plan'].cell_contents)
                self.assertIs(c.cells['protected'].cell_contents, c.reqs['b'])
                # This fixture supplies a subsequent output; the GPU run must observe it.
                target = c.reqs['b']
                target.num_output_tokens += 1
                target.num_computed_tokens = target.num_prompt_tokens + target.num_output_tokens - 1
                target.status = NS(name='RUNNING')
                c.s.skipped_waiting.clear()
                if target not in c.s.running:
                    c.s.running.append(target)
                c.s.schedule()
                output = next(e for e in c.data['events'] if e['event'] == 'target_new_output')
                self.assertEqual(output['request'], 'b')
                self.assertGreaterEqual(output['host_perf_counter_s'], receipt['host_perf_counter_s'])
                self.assertIsNone(c.cells['protected'].cell_contents)
                c.undo()

    def test_unsafe_or_too_young_candidates_fall_back_to_original_prepare(self):
        def pending(c):
            for rid in ('a', 'b'):
                c.cs._req_status[rid].transfer_jobs.add(10)
        def partial(c):
            c.owned['a'] = [c.blocks[3]]
            c.blocks[3].ref_cnt = 1
        conditions = {
            'off': lambda c: None,
            'pending_transfer': pending,
            'partial_owned_kv': partial,
            'no_slot': lambda c: setattr(c.s, 'max_num_running_reqs', 1),
            'shared_ownership': lambda c: setattr(c.blocks[0], 'ref_cnt', 2),
            'native_reservation': lambda c: setattr(c.s, '_inflight_prefill_reserved_blocks', lambda: 1),
            'unknown_reservation': lambda c: setattr(c.s, '_inflight_prefill_reserved_blocks', lambda: None),
            'too_young': lambda c: c.tracker.absent_since.update(a=11, b=11),
        }
        for name, mutate in conditions.items():
            with self.subTest(name=name):
                c = case(enabled=name != 'off')
                mutate(c)
                c.s.schedule()
                self.assertFalse(any(e['event'] == 'fit_first_choice' for e in c.data['events']))
                self.assertEqual(c.data['fit_first_resumes'], 0)
                self.assertTrue(any(e['event'] == 'prepare' for e in c.data['events']))
                self.assertEqual(c.cells['plan'].cell_contents.target.request_id, 't')
                c.undo()
        c = case()
        c.tracker.absent_since.update(t=20, a=20, b=20)
        c.s.schedule()
        self.assertEqual(c.data['fit_first_resumes'], 0)
        self.assertFalse(any(e['event'] in ('prepare', 'fit_first_choice') for e in c.data['events']))
        c.undo()

    def test_choice_without_native_admission_never_counts_as_resume(self):
        for kind in ('rejected', 'async_missing_receipt'):
            with self.subTest(kind=kind):
                c = case(kind)
                with self.assertRaises(RuntimeError):
                    c.s.schedule()
                self.assertTrue(any(e['event'] == 'fit_first_choice' for e in c.data['events']))
                self.assertFalse(any(e['event'] == 'fit_first_admitted' for e in c.data['events']))
                self.assertEqual(c.data['fit_first_resumes'], 0)
                self.assertEqual(c.data['status'], 'ERROR')
                c.undo()
        with self.assertRaisesRegex(ValueError, 'commit_recheck=False'):
            adapter.install(None, vllm_config=None, save=True, block_size=16,
                            commit_recheck=True, fit_first_resume=True)


if __name__ == '__main__':
    unittest.main()

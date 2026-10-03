"""CPU-only q10 release-condition experiment; intentionally no runnable adapter candidate.

The actual adapter closures are extracted and only their output release comparison
is changed inside this fixture. The third check documents an unsafe resource boundary.
"""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'capacity_victim_probe_r01/test_capacity_victim_cpu.py'
spec = importlib.util.spec_from_file_location('capacity_cpu_fixture', BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SOURCE = HERE.parent/'candidate_capacity_victim_r01/pkg/staged_store_rotation.py'


def fixture_for_quantity(quantity):
    tree = ast.parse(SOURCE.read_text())
    install = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install')
    start = next(i for i,n in enumerate(install.body) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='step' for t in n.targets))
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
    matches = 0
    for node in ast.walk(seed):
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'protected is not None and protected.num_output_tokens > output_start':
            node.test = ast.parse(f'protected is not None and protected.num_output_tokens-output_start >= {quantity}', mode='eval').body
            matches += 1
    assert matches == 1
    namespace = vars(base.adapter).copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[seed],type_ignores=[])),
                 '<CPU-only-output-quantity-comparison>', 'exec'), namespace)
    return namespace['fixture']


def recovered_case(quantity):
    base.closures = lambda: fixture_for_quantity(quantity)
    c = base.native_case()
    c.s.schedule()  # Prepare.
    for rid in ('old','alt'):
        c.reqs[rid].num_output_tokens += 1
        c.reqs[rid].num_computed_tokens += 1
    c.s.schedule()  # Successful forced recovery commit, before first new output.
    c.target = c.reqs['t']
    c.output_start = c.target.num_output_tokens
    c.target.num_computed_tokens = c.target.num_prompt_tokens+c.target.num_output_tokens-1
    c.owned = c.cells['owned'].cell_contents
    c.pool = c.s.kv_cache_manager.block_pool
    c.s._inflight_prefill_reserved_blocks = lambda: 0
    return c


def supplied_output(c, delta, fund_current_growth=True):
    c.target.num_output_tokens = c.output_start+delta
    c.target.num_computed_tokens = c.target.num_prompt_tokens+c.target.num_output_tokens-1
    if fund_current_growth:
        need = (c.target.num_prompt_tokens+c.target.num_output_tokens+15)//16-len(c.owned['t'])
        free = [b for b in c.pool.blocks.values() if b.ref_cnt==0]
        assert len(free)>=need
        for block in free[:need]:
            block.ref_cnt = 1
            c.owned['t'].append(block)


class ProtectionQuantityBoundaryTests(unittest.TestCase):
    def test_q1_releases_at_one_and_funded_q10_releases_at_ten(self):
        for quantity in (1,10):
            with self.subTest(quantity=quantity):
                c = recovered_case(quantity)
                for delta in range(1,quantity):
                    supplied_output(c,delta)
                    c.s.schedule()
                    self.assertIs(c.cells['protected'].cell_contents,c.target)
                    self.assertFalse(any(e['event']=='target_new_output' for e in c.data['events']))
                supplied_output(c,quantity)
                c.s.schedule()
                self.assertIsNone(c.cells['protected'].cell_contents)
                self.assertEqual(sum(e['event']=='target_new_output' for e in c.data['events']),1)
                c.undo()

    def test_real_terminal_releases_before_q10(self):
        c = recovered_case(10)
        supplied_output(c,2)
        c.target.status = NS(name='FINISHED_STOPPED')
        c.s.running.remove(c.target)
        c.s.requests.pop(c.target.request_id)
        c.s.schedule()
        event = next(e for e in c.data['events'] if e['event']=='target_terminal')
        self.assertEqual((event['new_output_tokens'],event['status']),(2,'FINISHED_STOPPED'))
        self.assertIsNone(c.cells['protected'].cell_contents)
        c.undo()

    def test_q10_cross_block_exhaustion_triggers_existing_native_guard(self):
        c = recovered_case(10)
        # The target has 218 blocks. Four supplied outputs move its history to
        # 3489 tokens, so the next decode needs block 219 while q10 remains active.
        supplied_output(c,4,fund_current_growth=False)
        free = [b for b in c.pool.blocks.values() if b.ref_cnt==0]
        peer = NS(request_id='peer',num_prompt_tokens=len(free)*16-1,num_output_tokens=1,
                  num_computed_tokens=len(free)*16-1,max_tokens=1024,status=NS(name='RUNNING'))
        peer.is_finished = lambda: False
        c.s.requests['peer'] = peer
        c.s.running.append(peer)
        c.owned['peer'] = free
        for block in free:
            block.ref_cnt = 1
        self.assertEqual(c.pool.get_num_free_blocks(),0)
        self.assertEqual(c.s._inflight_prefill_reserved_blocks(),0)
        self.assertEqual((c.target.num_prompt_tokens+c.target.num_output_tokens+15)//16-len(c.owned['t']),1)
        native_calls = []
        def native_allocate_failure():
            native_calls.append(1)
            preempted = []
            c.s._rotation_begin(preempted,0.0)
            held = {r.request_id for r in c.s.running if c.s._rotation_hold(r)}
            self.assertEqual(held,{'old','peer'})
            # The pinned FCFS allocator preempts running.pop() when allocation
            # fails. This is a finite mock of that path, not a native GPU run.
            victim = c.s.running.pop()
            c.s._preempt_request(victim,0.0)
            preempted.append(victim)
            block = next(b for b in c.pool.blocks.values() if b.ref_cnt==0)
            block.ref_cnt = 1
            c.owned['t'].append(block)
            return NS(num_scheduled_tokens={'t':1},preempted_req_ids={'peer'},
                scheduled_cached_reqs=NS(resumed_req_ids=[]),
                kv_connector_metadata=NS(store_jobs={},load_jobs={},jobs_to_flush=set()))
        cells = dict(zip(c.s.schedule.__code__.co_freevars,c.s.schedule.__closure__))
        cells['native'].cell_contents = native_allocate_failure
        with self.assertRaisesRegex(RuntimeError,'Unexpected natural preemption during protection'):
            c.s.schedule()
        self.assertEqual(len(native_calls),1)
        self.assertEqual(c.data['status'],'ERROR')
        c.undo()


if __name__=='__main__':
    unittest.main()

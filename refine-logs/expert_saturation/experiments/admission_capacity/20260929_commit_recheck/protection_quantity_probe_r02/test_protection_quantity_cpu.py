"""Three resource/termination checks of actual r02 closures; no GPU performance claim."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('capacity_fixture', HERE.parent/'capacity_victim_probe_r01/test_capacity_victim_cpu.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SOURCE = HERE/'staged_store_rotation.py'
spec = importlib.util.spec_from_file_location('protection_quantity_adapter', SOURCE)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def fixture(quantity):
    tree = ast.parse(SOURCE.read_text())
    install = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
    start = next(i for i,n in enumerate(install.body) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='step' for t in n.targets))
    seed = ast.parse(f'''def fixture(scheduler, native, owned, pool, cs):
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
 recovery_min_outputs={quantity}
 oldcalc=cs._calc_num_offloadable_tokens
 hadcalc=True
''').body[0]
    seed.body += deepcopy(install.body[start:])
    namespace = vars(adapter).copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[seed],type_ignores=[])),
                 '<actual-r02-protection-closures>', 'exec'),namespace)
    return namespace['fixture']


def recovered_case(quantity=10):
    base.closures = lambda: fixture(quantity)
    c = base.native_case()
    c.s._inflight_prefill_reserved_blocks = lambda:0
    c.s.schedule()
    for rid in ('old','alt'):
        c.reqs[rid].num_output_tokens += 1
        c.reqs[rid].num_computed_tokens += 1
    c.s.schedule()
    c.target = c.reqs['t']
    c.output_start = c.target.num_output_tokens
    c.owned = c.cells['owned'].cell_contents
    c.pool = c.s.kv_cache_manager.block_pool
    c.schedule_cells = dict(zip(c.s.schedule.__code__.co_freevars,c.s.schedule.__closure__))
    return c


def output(c,delta):
    c.target.num_output_tokens = c.output_start+delta
    c.target.num_computed_tokens = c.target.num_prompt_tokens+c.target.num_output_tokens-1


def peer_consumes_free(c,leave=0,needs_next_block=False):
    free = [b for b in c.pool.blocks.values() if b.ref_cnt==0]
    blocks = free[:len(free)-leave]
    prompt = len(blocks)*16 if needs_next_block else len(blocks)*16-1
    peer = NS(request_id='peer',num_prompt_tokens=prompt,num_output_tokens=1,
              num_computed_tokens=prompt,max_tokens=1024,status=NS(name='RUNNING'))
    peer.is_finished = lambda:False
    c.s.requests['peer'] = peer
    c.s.running.insert(0,peer)
    c.owned['peer'] = blocks
    for block in blocks:
        block.ref_cnt = 1
    return peer


def decode_native(c,trace):
    def native():
        preempted = []
        c.s._rotation_begin(preempted,0.0)
        scheduled,held = {},[]
        for req in c.s.running:
            if c.s._rotation_hold(req):
                held.append(req.request_id)
                continue
            need=max(0,(req.num_prompt_tokens+req.num_output_tokens+15)//16-len(c.owned[req.request_id]))
            free=[b for b in c.pool.blocks.values() if b.ref_cnt==0]
            if len(free)<need:
                raise AssertionError('fixture native allocation would require preemption')
            for block in free[:need]:
                block.ref_cnt=1
                c.owned[req.request_id].append(block)
            scheduled[req.request_id]=1
        trace.append(dict(held=held,scheduled=scheduled,free=c.pool.get_num_free_blocks()))
        return NS(num_scheduled_tokens=scheduled,preempted_req_ids=set(),
            scheduled_cached_reqs=NS(resumed_req_ids=[]),
            kv_connector_metadata=NS(store_jobs={},load_jobs={},jobs_to_flush=set()))
    return native


def empty_following_step(c):
    c.s._rotation_begin([],0.0)
    return NS(num_scheduled_tokens={},preempted_req_ids=set(),
        scheduled_cached_reqs=NS(resumed_req_ids=[]),
        kv_connector_metadata=NS(store_jobs={},load_jobs={},jobs_to_flush=set()))


class ProtectionQuantityTests(unittest.TestCase):
    def test_unfunded_or_unknown_q10_falls_back_and_q1_keeps_first_output_semantics(self):
        for quantity,reserved,expected in ((10,0,'KEEP_FUTURE_GROWTH_UNFUNDED'),
                                            (10,None,'KEEP_UNKNOWN_NATIVE_RESERVATION'),
                                            (10,1,'KEEP_NATIVE_INFLIGHT_RESERVATION'),
                                            (1,0,'OUTPUT_GOAL_REACHED')):
            with self.subTest(quantity=quantity,reserved=reserved):
                c=recovered_case(quantity)
                peer_consumes_free(c)
                c.s._inflight_prefill_reserved_blocks=lambda:reserved
                output(c,1)
                trace=[]
                c.schedule_cells['native'].cell_contents=decode_native(c,trace)
                c.s.schedule()
                release=next(e for e in c.data['events'] if e['event']=='protection_release')
                self.assertEqual(release['reason'],expected)
                self.assertEqual((release['free_blocks'],release['held_blocks']),(0,218))
                self.assertEqual(release['future_growth_blocks'],int(quantity==10))
                self.assertIsNone(c.cells['protected'].cell_contents)
                self.assertFalse(trace[0]['held'])
                self.assertEqual(sum(e['event']=='target_new_output' for e in c.data['events']),1)
                self.assertFalse(any(e['event']=='protection_extend' for e in c.data['events']))
                c.undo()
        for kwargs in ({'recovery_min_outputs':2}, {'recovery_min_outputs':10,'capacity_victim':False}):
            with self.assertRaises(ValueError):
                adapter.install(None,vllm_config=None,save=True,block_size=16,**kwargs)

    def test_future_block_is_reserved_from_peer_growth_until_ten_outputs(self):
        c=recovered_case()
        peer_consumes_free(c,leave=1,needs_next_block=True)
        trace=[]
        c.schedule_cells['native'].cell_contents=decode_native(c,trace)
        for delta in range(1,10):
            output(c,delta)
            c.s.schedule()
            self.assertIs(c.cells['protected'].cell_contents,c.target)
            self.assertIn('peer',trace[-1]['held'])
            self.assertEqual(trace[-1]['scheduled']['t'],1)
        extension=next(e for e in c.data['events'] if e['event']=='protection_extend')
        self.assertEqual((extension['free_blocks'],extension['held_blocks'],extension['future_required_blocks']),
                         (1,218,219))
        self.assertEqual(extension['future_compute_tokens'],3041+444+10-1)
        self.assertEqual(extension['native_inflight_reserved_blocks'],0)
        self.assertEqual(len(c.owned['t']),219)
        self.assertEqual(sum(e['event']=='target_new_output' for e in c.data['events']),1)
        self.assertEqual(sum(e['event']=='protection_extend' for e in c.data['events']),1)
        admitted=next(e for e in c.data['events'] if e['event']=='recovery_commit_admitted')
        self.assertEqual((admitted['native_admission'],admitted['scheduled_tokens']),('SCHEDULED_TOKENS',1))
        output(c,10)
        c.schedule_cells['native'].cell_contents=lambda:empty_following_step(c)
        c.s.schedule()
        release=next(e for e in c.data['events'] if e['event']=='protection_release')
        self.assertEqual((release['reason'],release['new_output_tokens'],release['extended']),
                         ('OUTPUT_GOAL_REACHED',10,True))
        self.assertIsNone(c.cells['protected'].cell_contents)
        c.undo()

    def test_terminal_releases_extended_protection_before_quantity(self):
        c=recovered_case()
        trace=[]
        c.schedule_cells['native'].cell_contents=decode_native(c,trace)
        output(c,1)
        c.s.schedule()
        self.assertTrue(any(e['event']=='protection_extend' for e in c.data['events']))
        output(c,2)
        c.target.status=NS(name='FINISHED_STOPPED')
        c.s.running.remove(c.target)
        c.s.requests.pop('t')
        for block in c.owned['t']:
            block.ref_cnt=0
        c.owned['t']=[]
        c.schedule_cells['native'].cell_contents=lambda:empty_following_step(c)
        c.s.schedule()
        release=next(e for e in c.data['events'] if e['event']=='protection_release')
        self.assertEqual((release['reason'],release['new_output_tokens'],release['extended']),
                         ('TERMINAL_FINISHED_STOPPED',2,True))
        self.assertEqual(sum(e['event']=='target_new_output' for e in c.data['events']),1)
        self.assertEqual(sum(e['event']=='target_terminal' for e in c.data['events']),1)
        self.assertIsNone(c.cells['protected'].cell_contents)
        c.undo()


if __name__=='__main__':
    unittest.main()

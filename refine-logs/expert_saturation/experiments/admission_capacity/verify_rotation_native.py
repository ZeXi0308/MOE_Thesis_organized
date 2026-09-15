"""CPU allocator fixture with native preempt/cache-output methods; no GPU claim.

The schedule loop and allocations are fixtures. The pinned native AST insertion,
preemption method and cached-request payload are checked; no worker is executed.
"""
import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import rotation_native as adapter
from absence_rotation import RotationConfig
from verify_headroom_fast import FakeScheduler


class Queue(list):
    remove_request = list.remove
    def prepend_request(self, r): self.insert(0, r)


class Blocks:
    def __init__(self, blocks): self.blocks = blocks
    def get_block_ids(self, allow_none=False): return ([b.block_id for b in self.blocks],)


def check(source, fundable, victim_order="least_progress"):
    if not fundable and victim_order != "least_progress":
        raise ValueError("the unfunded fixture targets the smaller least-progress victim")
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    names = ('_preempt_request', '_make_cached_request_data')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    states = NS(RUNNING=NS(name='RUNNING'), PREEMPTED=NS(name='PREEMPTED'))
    ns = dict(RequestStatus=states, itertools=itertools, CachedRequestData=NS)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + methods, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), ns)
    class Scheduler(FakeScheduler):
        ec_connector = None
        policy = NS(name='FCFS')
        is_encoder_decoder = use_pp = False
        scheduler_reserve_full_isl = True
        scheduler_config = NS(async_scheduling=False)
        def fixture_schedule(self):
            forced, scheduled, resumed, running = [], {}, [], []
            self._rotation_begin(forced, 0.0)
            self.last_forced = list(forced)
            budget = 7
            for r in list(self.running):
                if r not in self.running: continue
                if self._rotation_hold(r): continue
                amount = min(r.num_tokens-r.num_computed_tokens, budget)
                needed = (r.num_computed_tokens+amount+3)//4-len(self.owned[r.request_id])
                if needed > len(self.available):
                    victim = self.running.pop()
                    self._preempt_request(victim, 0.0)
                    forced.append(victim)
                    if victim is r: break
                self.allocate(r, amount); scheduled[r.request_id] = amount
                running.append(r); budget -= amount
            if len(forced) == self._rotation_forced_count and self.waiting and budget:
                r = self.waiting[0]
                if (self._rotation_target is None or r.request_id == self._rotation_target) and (r.num_tokens+3)//4 <= len(self.available):
                    self.waiting.pop(0); self.running.append(r); resumed.append(r)
                    self.allocate(r, min(r.num_tokens, budget)); scheduled[r.request_id] = min(r.num_tokens, budget)
            cached = self._make_cached_request_data(running, resumed, scheduled, {},
                {rid: Blocks(self.owned[rid]) for rid in scheduled})
            result = NS(num_scheduled_tokens=scheduled, preempted_req_ids=self.reset_preempted_req_ids,
                        scheduled_cached_reqs=cached)
            self.reset_preempted_req_ids = set(); self.prev_step_scheduled_req_ids = set(scheduled)
            for rid, amount in scheduled.items():
                r = self.requests[rid]; r.status = states.RUNNING; r.num_computed_tokens += amount
            return result
        def allocate(self, r, amount):
            blocks = self.owned.setdefault(r.request_id, [])
            while len(blocks) < (r.num_computed_tokens+amount+3)//4:
                blocks.append(self.kv_cache_manager.block_pool.blocks[self.available.pop(0)])
        def _free_request_blocks(self, r):
            self.available.extend(b.block_id for b in self.owned.pop(r.request_id))
    for name in names: setattr(Scheduler, name, ns[name])
    s = Scheduler(12 if fundable else 11, {'output_tokens':128})
    s.owned = s.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
    s._inflight_prefills = NS(discard=lambda r: None)
    s.encoder_cache_manager = NS(free=lambda r: None)
    s.log_stats = False; s.reset_preempted_req_ids = set(); s.prev_step_scheduled_req_ids = set()
    cfg = NS(scheduler_config=NS(async_scheduling=False), speculative_config=None)
    fixture = ast.parse('def schedule(self):\n    return self.fixture_schedule()')
    with patch.object(adapter.inspect, 'getsourcefile', return_value=str(source)), patch.object(adapter, 'patched_schedule_tree', return_value=fixture):
        decisions, uninstall = adapter.install(s, vllm_config=cfg, block_size=4, expected_requests=3,
            rotation_config=RotationConfig(min_absence_steps=1), victim_order=victim_order)
    positions = {'A': (7,4) if fundable else (3,2), 'B':(15,8), 'C':(19,12)}
    state = {'pool':{'free_blocks':0}, 'running_ids':list(positions), 'waiting_count':0,
        'requests':{rid:dict(computed_tokens=n, prompt_tokens=p, output_tokens=n+1-p,
                           num_preemptions=0, block_counts=[(n+3)//4]) for rid,(n,p) in positions.items()}}
    s.load(state); s.waiting = Queue(); s.skipped_waiting = Queue()
    for r in s.requests.values():
        r.status=states.RUNNING; r.has_encoder_inputs=False; r.all_token_ids=list(range(r.num_tokens))
    outputs = []
    for i in range(7 if fundable else 3):
        out = s.schedule(); outputs.append(out)
        for rid in out.num_scheduled_tokens:
            r=s.requests[rid]
            if r.num_computed_tokens == r.num_tokens:
                r.num_output_tokens+=1; r.num_tokens+=1; r.all_token_ids.append(0)
    assert decisions[1]['natural_preempted']==['C']
    applied = 0
    for d in decisions:
        expected = ('most_output' if applied == 0 else 'least_progress') if victim_order == 'first_most_then_least' else victim_order
        assert d['effective_victim_order'] == expected and d['applied_rotations_before'] == applied
        applied += bool(d['forced_preempted'])
    if fundable:
        victim = 'A' if victim_order == 'least_progress' else 'B'
        assert decisions[2]['forced_preempted']==[victim] and decisions[2]['resumed']==['C']
        assert outputs[2].preempted_req_ids=={victim} and outputs[2].scheduled_cached_reqs.resumed_req_ids=={'C'}
        c=outputs[2].scheduled_cached_reqs; assert len(c.new_block_ids[c.req_ids.index('C')][0])==2
        assert all(d['recovery_target']=='C' for d in decisions[2:6]) and decisions[6]['recovery_completed']=='C'
        assert victim_order != 'least_progress' or any(d['held'] for d in decisions[2:6])
        assert all(d['recovery_remaining_blocks_after']<=d['free_after'] for d in decisions[2:6])
    else:
        assert not decisions[2]['forced_preempted'] and 'cannot fund' in decisions[2]['not_applied_reason']
    uninstall()
    return dict(fundable=fundable, victim_order=victim_order, steps=len(decisions), status='PASS')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--scheduler-source', type=Path, default=Path('/private/tmp/moe-native-v026-recovery-source/scheduler.py'))
    a=p.parse_args(); assert hashlib.sha256(a.scheduler_source.read_bytes()).hexdigest()==adapter.SCHEDULER_SHA256
    compile(adapter.patched_schedule_tree(a.scheduler_source.read_text()), str(a.scheduler_source), 'exec')
    print(json.dumps(dict(status='PASS', evidence_type='CPU_ALLOCATOR_FIXTURE_NOT_GPU', rows=[check(a.scheduler_source, True), check(a.scheduler_source, False), check(a.scheduler_source, True, 'most_output'), check(a.scheduler_source, True, 'first_most_then_least')])))

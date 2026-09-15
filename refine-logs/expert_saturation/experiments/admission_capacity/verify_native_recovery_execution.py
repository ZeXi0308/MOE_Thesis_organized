"""Execute the pinned native schedule on independent CPU allocator fixtures.

Only before1026 and earlier queue events initialize either arm. Native schedule,
preemption, cached payload and post-schedule methods execute unchanged apart from
the actual adapter AST hooks. Allocator, synchronous outputs and runtime objects
are fakes: this checks structural execution, not GPU timing, tensors or EOS.
"""
import argparse
import ast
from contextlib import nullcontext
from copy import deepcopy
from enum import Enum
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

import recovery_execution_share as share
import rotation_native as adapter
from verify_headroom_fast import FakeScheduler
from verify_rotation_native import Blocks, Queue as BaseQueue


class Queue(BaseQueue):
    def peek_request(self): return self[0]
    def pop_request(self): return self.pop(0)
    def prepend_requests(self, requests): self[:0] = list(requests)


class Request(NS):
    __hash__ = object.__hash__
    __eq__ = object.__eq__
    @property
    def num_tokens_with_spec(self): return self.num_tokens


Status = Enum('RequestStatus', 'RUNNING PREEMPTED WAITING FINISHED_STOPPED')
Policy = Enum('SchedulingPolicy', 'FCFS PRIORITY')
Pause = Enum('PauseState', 'UNPAUSED PAUSED_ALL')


def native_class(source):
    names = {'schedule', '_preempt_request', '_make_cached_request_data',
             '_update_after_schedule', '_get_new_block_ids_to_zero'}
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {m.name for m in methods} == names
    ns = dict(RequestStatus=Status, SchedulingPolicy=Policy, PauseState=Pause,
              time=time, itertools=itertools, CachedRequestData=NS,
              SchedulerOutput=lambda **kw: NS(has_structured_output_requests=False, **kw),
              NewRequestData=NS(from_request=lambda r, *args: NS(req_id=r.request_id)),
              create_request_queue=lambda _: Queue(), record_function_or_nullcontext=lambda _: nullcontext())
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0)] + methods, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), ns)

    class Scheduler(FakeScheduler):
        ec_connector = dynamic_sd_lookup = lora_config = None
        policy, _pause_state = Policy.FCFS, Pause.UNPAUSED
        is_encoder_decoder = use_pp = need_mamba_block_aligned_split = False
        scheduler_reserve_full_isl = True
        num_sampled_tokens_per_step = 1
        max_num_encoder_input_tokens = num_waiting_for_streaming_input = 0
        max_num_running_reqs = 32
        prefill_capacity_bound = needs_kv_cache_zeroing = False
        enable_return_routed_experts = log_stats = False
        scheduler_config = NS(async_scheduling=False, long_prefill_token_threshold=0,
                              enable_chunked_prefill=True)
        kv_cache_config = NS(kv_cache_groups=[None])

        def _select_waiting_queue_for_scheduling(self): return self.waiting
        def _is_blocked_waiting_status(self, status): return False

        def _free_request_blocks(self, request):
            self.discarded.append(dict(request_id=request.request_id,
                                       computed=request.num_computed_tokens))
            self.available.extend(b.block_id for b in self.owned.pop(request.request_id, []))

        def allocate_slots(self, request, amount, **kw):
            owned = self.owned.setdefault(request.request_id, [])
            end = request.num_computed_tokens + kw.get('num_new_computed_tokens', 0) + amount
            need = max(0, (end + 15) // 16 - len(owned))
            required = (max(0, (request.num_tokens + 15) // 16 - len(owned))
                        if kw.get('full_sequence_must_fit') else need)
            if request.request_id in self.fail_ids or required > len(self.available):
                self.failed_allocations.append(dict(request_id=request.request_id,
                    required=required, free=len(self.available), computed=request.num_computed_tokens))
                return None
            new = [self.kv_cache_manager.block_pool.blocks[self.available.pop(0)]
                   for _ in range(need)]
            owned.extend(new)
            return Blocks(new)

    for name in names: setattr(Scheduler, name, ns[name])
    return Scheduler


def create(source, before, waiting, enabled, *, fail_ids=(), first_prefill=None):
    s = native_class(source)(before['pool']['total_blocks'], {'output_tokens': 1024})
    s.owned = s.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
    s.current_step = 1025
    s.finished_req_ids = set(); s.reset_preempted_req_ids = set()
    s.prev_step_scheduled_req_ids = set(); s._inflight_prefills = set()
    s.discarded = []; s.failed_allocations = []; s.fail_ids = set(fail_ids)
    s.encoder_cache_manager = NS(free=lambda r: None, get_freed_mm_hashes=lambda: [])
    m = s.kv_cache_manager
    m.new_step_starts = lambda: None
    m.allocate_slots = s.allocate_slots
    m.get_blocks = lambda rid: Blocks(s.owned.get(rid, []))
    m.get_computed_blocks = lambda r: (Blocks([]), 0, 0)
    m.get_num_common_prefix_blocks = lambda rid: [0]
    m.take_kv_cache_block_copies = lambda: ([], [])
    m.take_new_block_ids = lambda: []
    cfg = NS(scheduler_config=NS(async_scheduling=False), speculative_config=None)
    with patch.object(adapter.inspect, 'getsourcefile', return_value=str(source)):
        decisions, uninstall = adapter.install(s, vllm_config=cfg, block_size=16,
            mode='native', protect_native_recovery=enabled)
    s.load(deepcopy(before))
    s.requests = {rid: Request(**vars(r)) for rid, r in s.requests.items()}
    s.running = [s.requests[rid] for rid in before['running_ids']]
    s.waiting = Queue(s.requests[rid] for rid in waiting); s.skipped_waiting = Queue()
    for rid, r in s.requests.items():
        r.status = Status.RUNNING if rid in before['running_ids'] else Status.PREEMPTED
        r.has_encoder_inputs = r.use_structured_output = False
        r.next_decode_eligible_step = 0; r.prefill_stats = None
        r.is_prefill_chunk = r.num_computed_tokens < r.num_tokens - 1
        r.all_token_ids = [0] * r.num_tokens
    if first_prefill:
        r = s.requests[first_prefill]
        r.status = Status.WAITING; r.num_output_tokens = r.num_preemptions = 0
        r.num_tokens = r.num_prompt_tokens; r.all_token_ids = [0] * r.num_tokens
    return s, decisions, uninstall


def plan(s, target):
    def view(r):
        return share.Request(r.request_id, r.num_tokens, r.num_computed_tokens,
            len(s.owned.get(r.request_id, [])), r.num_output_tokens, r.max_tokens-r.num_output_tokens)
    return share.allocate(share.State(view(s.requests[target]),
        tuple(view(r) for r in s.running if r.request_id != target), len(s.available)))


def returned(s, result):
    outputs, completed = [], []
    for rid in result.num_scheduled_tokens:
        r = s.requests[rid]; r.num_in_flight_tokens = 0
        if r.num_computed_tokens == r.num_tokens:
            r.num_output_tokens += 1; r.num_tokens += 1; r.all_token_ids.append(0)
            outputs.append(rid)
            if r.num_output_tokens == r.max_tokens:
                s.running.remove(r); s._free_request_blocks(r)
                r.status = Status.FINISHED_STOPPED; completed.append(rid)
    for rid in completed: del s.requests[rid]
    return outputs, completed


def run(source, before, waiting, enabled, target):
    s, decisions, uninstall = create(source, before, waiting, enabled)
    trace = []
    for index in range(5):
        predicted = plan(s, target) if enabled and index < 4 else None
        before_output = s.requests[target].num_output_tokens
        result = s.schedule(); d = decisions[-1]
        if predicted:
            assert result.num_scheduled_tokens == predicted['scheduled']
            assert len(s.available) == predicted['free_after_allocations']
            assert set(d['held']) == set(predicted['held'])
        after_schedule = len(s.available)
        outputs, completed = returned(s, result)
        trace.append(dict(step=1026+index, scheduled=dict(result.num_scheduled_tokens),
            free_after_schedule=after_schedule, free_after_return=len(s.available),
            held=d['held'], native_recovery_started=d.get('native_recovery_started'),
            recovery_target=d['recovery_target'], recovery_completed=d.get('recovery_completed'),
            target_outputs_before=before_output, target_outputs_after=s.requests[target].num_output_tokens,
            target_computed_after=s.requests[target].num_computed_tokens,
            natural_preempted=d['natural_preempted'], outputs=outputs, completed=completed,
            predicted=predicted, allocation_failures=deepcopy(s.failed_allocations)))
    assert not any(row['recovery_completed'] for row in trace[:4])
    if enabled:
        assert trace[0]['native_recovery_started']['request_id'] == target
        assert sum(bool(row['native_recovery_started']) and
                   row['native_recovery_started']['request_id'] == target for row in trace) == 1
        assert trace[3]['held'] and trace[3]['target_outputs_after'] == before['requests'][target]['output_tokens']+1
        assert trace[4]['recovery_completed'] == target
    else:
        assert target in trace[3]['natural_preempted']
        assert any(d['request_id'] == target and d['computed'] == 2991 for d in s.discarded)
        assert all(row['native_recovery_started'] is None for row in trace)
    uninstall()
    return dict(enabled=enabled, trace=trace, discarded=s.discarded)


def main():
    global adapter
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--adapter-source', type=Path)
    p.add_argument('--scheduler-source', type=Path,
                   default=Path('/private/tmp/moe-native-v026-recovery-source/scheduler.py'))
    args = p.parse_args(); assert not args.output.exists()
    if args.adapter_source:
        spec = importlib.util.spec_from_file_location('qualified_native_recovery_adapter', args.adapter_source)
        adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    assert hashlib.sha256(args.scheduler_source.read_bytes()).hexdigest() == adapter.SCHEDULER_SHA256
    raw = json.loads((args.cell/'raw.json').read_bytes())
    before = deepcopy(raw['memory_trace'][1026]['before'])
    past = json.loads((args.cell/'headroom-decisions.json').read_bytes())[:1026]
    waiting = []
    for d in past:
        assert len(d['preempted']) <= 1, 'fixture requires unambiguous past preemption order'
        for rid in d['preempted']: waiting.insert(0, rid)
        for rid in d['resumed']:
            if rid in waiting: waiting.remove(rid)
    assert set(waiting) == set(before['requests'])-set(before['running_ids'])
    target = waiting[0]; assert '0020902' in target
    arms = [run(args.scheduler_source, before, waiting, enabled, target) for enabled in (False, True)]
    validation = []
    for row in arms[0]['trace'][:4]:
        i = row['step']; scheduled = {r['internal_request_id']: r['scheduled_tokens']
            for r in raw['scheduler_steps'][i]['scheduled']}
        actual_free = raw['memory_trace'][i]['after']['pool']['free_blocks']
        assert row['scheduled'] == scheduled and row['free_after_schedule'] == actual_free
        validation.append(dict(step=i, scheduled_and_free_match=True))
    negative = []
    for name, kw in [('allocation_failure', dict(fail_ids=[target])),
                     ('first_prefill', dict(first_prefill=target))]:
        s, decisions, uninstall = create(args.scheduler_source, before, waiting, True, **kw)
        result = s.schedule(); assert 'native_recovery_started' not in decisions[-1]
        assert (target not in result.num_scheduled_tokens) == (name == 'allocation_failure')
        negative.append(dict(case=name, status='PASS', scheduled=dict(result.num_scheduled_tokens)))
        uninstall()
    # A separate declared-cap fixture: the returned new token completes and
    # removes the target before the following schedule observes the release.
    s, decisions, uninstall = create(args.scheduler_source, before, waiting, True)
    s.requests[target].max_tokens = s.requests[target].num_output_tokens + 1
    completion_trace = []
    for index in range(4):
        result = s.schedule()
        assert 'recovery_completed' not in decisions[-1]
        outputs, completed = returned(s, result)
        completion_trace.append(dict(step=1026+index, outputs=outputs, completed=completed))
    assert target not in s.requests and target in completion_trace[-1]['completed']
    s.schedule(); assert decisions[-1]['recovery_completed'] == target
    negative.append(dict(case='new_output_and_completion_release', status='PASS',
                         trace=completion_trace, release_step=1030))
    uninstall()
    result = dict(status='PASS', evidence_type='CPU_PINNED_NATIVE_SCHEDULE_FAKE_ALLOCATOR_OUTPUT',
        scope=__doc__, causal_cutoff='memory_trace[1026].before plus pre1026 queue events',
        initial=before, initial_waiting_order=waiting, target=target, arms=arms,
        validation_only_original_future=validation, negative_checks=negative,
        sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path(__file__), Path(adapter.__file__), Path(share.__file__), args.scheduler_source,
             args.cell/'raw.json', args.cell/'headroom-decisions.json']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(dict(status='PASS', output=str(args.output), cells=2, calls_per_cell=5,
                         negative_checks=[dict(case=n['case'], status=n['status']) for n in negative])))


if __name__ == '__main__': main()

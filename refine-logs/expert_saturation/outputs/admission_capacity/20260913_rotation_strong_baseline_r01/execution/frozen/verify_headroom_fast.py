"""Recorded-state CPU adapter conformance; no native execution or timing claim."""
import argparse
import ast
import hashlib
import heapq
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import completion_headroom as adapter


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class FullAttentionManager:
    def __init__(self, pool):
        self.block_pool, self.req_to_blocks = pool, {}


class KVCacheCoordinatorNoPrefixCache:
    def __init__(self, pool):
        self.single_type_managers = (FullAttentionManager(pool),)


class FakeBlocks:
    def __init__(self, manager, blocks):
        self.manager, self.blocks = manager, blocks

    def get_block_ids(self):
        self.manager.calls += 1
        self.manager.entries += len(self.blocks)
        return ([b.block_id for b in self.blocks],)


class FakeManager:
    enable_caching = use_eagle = watermark_blocks = False
    num_kv_cache_groups = 1

    def __init__(self, pool):
        self.block_pool, self.coordinator = pool, KVCacheCoordinatorNoPrefixCache(pool)
        self.calls = self.entries = 0

    def get_blocks(self, rid):
        return FakeBlocks(self, self.coordinator.single_type_managers[0].req_to_blocks.get(rid, []))


class FakeScheduler:
    num_lookahead_tokens = num_spec_tokens = 0
    dcp_world_size = pcp_world_size = 1
    use_v2_model_runner = defer_block_free = False
    connector = None
    max_num_scheduled_tokens, max_model_len = 1024, 4096

    def __init__(self, total, config):
        self.config, self.requests = config, {}
        self.running = self.waiting = self.skipped_waiting = []
        pool = NS(blocks=[NS(block_id=i, is_null=i == 0) for i in range(total)])
        pool.null_block, pool.get_num_free_blocks = pool.blocks[0], lambda: len(self.available)
        self.kv_cache_manager, self.available = FakeManager(pool), list(range(1, total))

    def load(self, state):
        owned = self.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
        counts = {rid: row['block_counts'][0] for rid, row in state['requests'].items()}
        for rid in list(owned):
            blocks = owned[rid]
            while len(blocks) > counts.get(rid, 0):
                heapq.heappush(self.available, blocks.pop().block_id)
            if rid not in counts:
                del owned[rid]
        for rid, count in counts.items():
            blocks = owned.setdefault(rid, [])
            while len(blocks) < count:
                blocks.append(self.kv_cache_manager.block_pool.blocks[heapq.heappop(self.available)])
        require(len(self.available) == state['pool']['free_blocks'], 'recorded pool accounting')
        self.requests = {rid: NS(request_id=rid, num_computed_tokens=r['computed_tokens'],
            num_prompt_tokens=r['prompt_tokens'], num_output_tokens=r['output_tokens'],
            num_tokens=r['prompt_tokens'] + r['output_tokens'], max_tokens=self.config['output_tokens'],
            num_preemptions=r['num_preemptions'], num_output_placeholders=0,
            num_in_flight_tokens=0, spec_token_ids=[], status=NS(name='RUNNING'))
            for rid, r in state['requests'].items()}
        self.running = [self.requests[rid] for rid in state['running_ids']]
        self.waiting = [None] * state['waiting_count']

    def schedule(self):
        raise AssertionError('install must replace schedule')

    def recorded_native(self):
        # Only reached after the adapter has independently chosen its action.
        require(self._headroom_held == set(self.saved['held']), 'held action differs')
        scheduled = {r['internal_request_id']: r['scheduled_tokens'] for r in self.step['scheduled']}
        require(scheduled == self.saved['actual_scheduled'], 'two raw action ledgers disagree')
        require(not self.step['preempted_request_ids'], 'fixture requires no preemption')
        self.load(self.trace['after'])
        return NS(num_scheduled_tokens=scheduled, preempted_req_ids=[])


def replay(cell, observer, source):
    raw, saved, config = [json.loads((cell / n).read_text()) for n in
                         ('raw.json', 'headroom-decisions.json', 'config.json')]
    traces, steps = raw['memory_trace'], raw['scheduler_steps']
    require(len(traces) == len(steps) == len(saved), 'step count mismatch')
    scheduler = FakeScheduler(traces[0]['before']['pool']['total_blocks'], config)
    fixture = ast.parse('def schedule(self):\n    return self.recorded_native()\n')
    cfg = NS(scheduler_config=NS(async_scheduling=False), speculative_config=None)
    with patch.object(adapter.inspect, 'getsourcefile', return_value=str(source)), \
            patch.object(adapter, 'patched_schedule_tree', return_value=fixture):
        decisions, uninstall = adapter.install(scheduler, vllm_config=cfg, block_size=16,
            expected_requests=config['requests'], mode='headroom', observer=observer)
    for i, (trace, step, expected) in enumerate(zip(traces, steps, saved)):
        require(trace['attempted_step'] == step['step'] == expected['step'] == i, 'step identity')
        scheduler.load(trace['before'])
        scheduler.trace, scheduler.step, scheduler.saved = trace, step, expected
        scheduler.schedule()
        clean = lambda d: {k: v for k, v in d.items() if k not in ('observer', 'decision_seconds')}
        require(clean(decisions[-1]) == clean(expected), f'{observer} decision mismatch at {i}')
    uninstall()
    manager = scheduler.kv_cache_manager
    return dict(observer=observer, exact_steps=len(saved), active_steps=sum(d['active'] for d in saved),
        held_steps=sum(bool(d['held']) for d in saved), native_id_extraction_calls=manager.calls,
        native_id_extracted_entries=manager.entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scheduler-source', type=Path,
                        default=Path('/private/tmp/moe-native-v026-recovery-source/scheduler.py'))
    args = parser.parse_args()
    require(not args.output.exists(), 'refuse to overwrite an artifact')
    cells = [args.run_dir / 'gpu_results' / f'repeat{i}-headroom' for i in range(2)]
    result = dict(status='PASS', evidence_type='CPU_RECORDED_STATE_ADAPTER_CONFORMANCE',
        scope='Complete install wrappers checked/fast on recorded pre-action states; native scheduling is a recorded-after-state fixture. Constructed block IDs are not original GPU IDs. No performance counterfactual or GPU timing.',
        adapter_sha256=hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
        scheduler_sha256=hashlib.sha256(args.scheduler_source.read_bytes()).hexdigest(), rows=[])
    for cell in cells:
        rows = [replay(cell, observer, args.scheduler_source) for observer in ('checked', 'fast')]
        result['rows'].append(dict(cell=str(cell), observers=rows,
            input_sha256={n: hashlib.sha256((cell / n).read_bytes()).hexdigest() for n in
                          ('raw.json', 'headroom-decisions.json', 'config.json')}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

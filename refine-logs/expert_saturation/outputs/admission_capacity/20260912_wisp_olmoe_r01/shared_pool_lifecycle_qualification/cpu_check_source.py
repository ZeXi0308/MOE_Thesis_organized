"""Persisted original 8080-call CPU symbol check; no GPU or runtime execution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shlex
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir', type=Path,
                    default=Path('refine-logs/expert_saturation/experiments/admission_capacity'))
parser.add_argument('--out', type=Path, required=True, help='New output file; never overwrite')
args = parser.parse_args()
if args.out.exists():
    parser.error('--out must name a new file')
source_dir = args.source_dir.resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(source_dir))
from shared_pool_plan import plan_shared_pool
from analyze_layer_budget import LRU
from wisp_expert_groups import partition_experts

sources = [source_dir / name for name in
           ('shared_pool_plan.py', 'analyze_layer_budget.py', 'wisp_expert_groups.py')]
sources.append(Path(__file__).resolve())
def source_hashes():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
hashes_before = source_hashes()
rng = random.Random(913384)
states = [LRU(21) for _ in range(16)]
pool = [('poison', s) for s in range(384)]
counts = dict(calls=0, empty_entry=0, partial_entry=0, full_entry=0,
              all64_active=0, empty_active=0, d2d=0, h2d=0, max_shared=0,
              wrong_order_failures=0)
# Same sequence as the original ephemeral validation.
sequence = [(layer, active) for layer in range(16)
            for active in ([], [layer], list(range(64)), [], list(range(20)))]
sequence += [(rng.randrange(16), rng.sample(range(64), rng.randrange(65)))
             for _ in range(8000)]
for layer, active in sequence:
    state = states[layer]
    original = state.snapshot()
    snapshot = copy.deepcopy(original)
    if counts['calls'] % 2:
        snapshot['expert_to_slot'] = {int(e): s for e, s in snapshot['expert_to_slot'].items()}
    before_input = copy.deepcopy(snapshot)
    before_pool = list(pool)
    entry = set(state.mapping)
    count_key = 'empty_entry' if not entry else 'full_entry' if len(entry) == 21 else 'partial_entry'
    counts[count_key] += 1
    plan = plan_shared_pool(active, snapshot, layer)
    assert snapshot == before_input
    groups = partition_experts(active, entry, 21)
    loaded, evicted = [], []
    for group in groups:
        miss, victims = state.ensure(group)
        loaded += miss
        evicted += victims
    assert plan['final_state'] == state.snapshot()
    assert plan['canonical_groups'] == groups
    assert (plan['canonical_miss'], plan['canonical_evict']) == (len(loaded), len(evicted))
    assert plan['canonical_loaded_experts'] == loaded and plan['canonical_evicted_experts'] == evicted
    assert plan['entry_evicted_experts'] == sorted(entry - state.mapping.keys())
    assert plan['entry_evict'] == len(entry - state.mapping.keys())
    assert sorted(c['expert'] for c in plan['h2d']) == sorted(set(active) - entry)
    assert sorted(c['expert'] for c in plan['d2d']) == sorted(set(active) & entry - state.mapping.keys())
    logical_map = plan['expert_map_device']
    assert len(logical_map) == 384 and logical_map[64:] == [-1] * 320
    assert len(plan['final_state']['expert_map_device']) == 64
    wrong_order = list(pool)
    for c in plan['h2d']:
        wrong_order[c['dst']] = (layer, c['expert'])
    for c in plan['d2d']:
        wrong_order[c['dst']] = wrong_order[c['src']]
    if any(wrong_order[logical_map[e]] != (layer, e) for e in active):
        counts['wrong_order_failures'] += 1
    for c in plan['d2d']:
        assert pool[c['src']] == (layer, c['expert'])
        pool[c['dst']] = pool[c['src']]
    for c in plan['h2d']:
        pool[c['dst']] = (layer, c['expert'])
    for e in range(64):
        if e in active:
            assert pool[logical_map[e]] == (layer, e)
        else:
            assert logical_map[e] == -1
    for e, s in state.mapping.items():
        assert pool[layer * 21 + s] == (layer, e)
    assert pool[:layer*21] == before_pool[:layer*21]
    assert pool[(layer+1)*21:336] == before_pool[(layer+1)*21:336]
    assert len(plan['shared_experts']) <= 43
    counts['calls'] += 1
    counts['empty_active'] += not active
    counts['all64_active'] += len(active) == 64
    counts['d2d'] += len(plan['d2d'])
    counts['h2d'] += len(plan['h2d'])
    counts['max_shared'] = max(counts['max_shared'], len(plan['shared_experts']))
# Original tie-heavy, sparse-slot cases, followed by malformed inputs.
for entry_size in [0, 1, 7, 20, 21]:
    state = LRU(21)
    for slot, expert in zip(rng.sample(range(21), entry_size), rng.sample(range(64), entry_size)):
        state.slots[slot] = expert
        state.mapping[expert] = slot
        state.ticks[slot] = 4
    state.clock = 4
    for size in [0, 1, 21, 22, 43, 64]:
        plan_shared_pool(rng.sample(range(64), size), state.snapshot(), 15)
invalid = []
def reject(active, snap, layer):
    try:
        plan_shared_pool(active, snap, layer)
    except (ValueError, TypeError, KeyError):
        invalid.append(True)
    else:
        raise AssertionError('malformed input accepted')
blank = LRU(21).snapshot()
for active, layer in [([-1], 0), ([64], 0), ([True], 0), ([1.0], 0),
                      ([0], 16), ([0], -1), ([0], True)]:
    reject(active, blank, layer)
bad = copy.deepcopy(blank)
bad['expert_to_slot'] = {'0': 0, 0: 0}
reject([], bad, 0)
bad = copy.deepcopy(blank)
bad['slot_to_expert'][:2] = [0, 0]
bad['expert_to_slot'] = {'0': 1}
reject([], bad, 0)
bad = copy.deepcopy(blank)
bad['lru_tick'][0] = 1
reject([], bad, 0)
bad = copy.deepcopy(blank)
bad['expert_map_device'][0] = 0
reject([], bad, 0)
assert counts['wrong_order_failures'] > 0 and counts['max_shared'] == 43
assert source_hashes() == hashes_before, 'source changed during validation'
command = 'PYTHONDONTWRITEBYTECODE=1 ' + shlex.join([
    sys.executable, str(Path(__file__).resolve()), '--source-dir', str(source_dir), '--out', 'NEW_FILE'])
result = dict(validation='PASS', evidence_type='STRUCTURAL_CPU_SYMBOL_MODEL',
              seed=913384, **counts, adversarial_state_calls=30,
              invalid_inputs_rejected=len(invalid), source_sha256=hashes_before,
              reproducible_command=command,
              scope='Planner symbols and exact LRU metadata only; no GPU numerical or performance result')
with args.out.open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(json.dumps({k: v for k, v in result.items() if k not in
                  ('source_sha256', 'reproducible_command', 'scope')}, indent=2))

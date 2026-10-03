"""Exact observed LRU replay, then independent current-call ordinary branches."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import runpy
import sys

from analyze_layer_budget import LRU, ReplayMismatch, UPSTREAM_SHA, equal


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def require(condition, message):
    if not condition: raise ValueError(message)


def restore(snapshot):
    state = LRU(len(snapshot['slot_to_expert']))
    state.slots = list(snapshot['slot_to_expert'])
    state.mapping = {int(k): v for k, v in snapshot['expert_to_slot'].items()}
    state.ticks, state.clock = list(snapshot['lru_tick']), snapshot['lru_clock']
    require({e: s for s, e in enumerate(state.slots) if e >= 0} == state.mapping,
            'snapshot slot/map mismatch')
    return state


def anchor(state, snapshot, label, record):
    require(all(k in snapshot for k in ('slot_to_expert', 'expert_to_slot', 'lru_tick', 'lru_clock')),
            'missing full CPU LRU anchor: ' + label)
    for key, value in state.snapshot().items():
        if key in snapshot: equal(value, snapshot[key], label + '/' + key, record)


def verify_engine(root, engine, sequence, functions, hashes):
    cell = root / 'results' / engine
    def load(relative):
        path = cell / relative; hashes[str(path.relative_to(root))] = sha(path)
        return read(path)
    pager, episodes = load('pager_summary.json'), load('episodes.json')
    require(pager['status'] == 'complete' and pager['failed_calls'] == 0, 'incomplete pager')
    require(pager['upstream_sha256'] == UPSTREAM_SHA, 'unrecognized ensure implementation')
    layers = {x['layer_name']: x for x in pager['layers']}
    require(len(layers) == 16 and all(x['cap'] == 24 and x['num_experts'] == 64 for x in layers.values()),
            'expected sixteen cap24/64-expert layers')
    require(len(episodes) == len(sequence) == 6, 'expected six episodes per engine')
    states = {name: LRU(24) for name in layers}; ever = defaultdict(set)
    metadata, anchors, expected_phases = {}, {}, Counter(initialization=32)
    for i, (episode, mode) in enumerate(zip(episodes, sequence)):
        prefix = 'repeat_' + str(i) + '_retention_' + mode; phase = prefix + '/measurement'
        require(episode['phase'] == phase and episode['status'] == 'COMPLETE', 'episode identity/status')
        raw, reset = load(prefix + '/raw.json'), load(prefix + '/reset.json')
        require(raw['status'] == 'COMPLETE', 'raw incomplete')
        initial = load(prefix + '/measurement_initial_cache.json')
        require(set(initial) == set(reset['cache_before']) == set(reset['cache_after']) == set(layers),
                'anchor layer coverage')
        require(len(raw['prefill_release_actions']) == 1, 'missing unique old-completion boundary')
        release = raw['prefill_release_actions'][0]
        require(all(x['status'] == 'completed' and not x['present_in_scheduler']
                    for x in release['old_requests'].values()), 'old completion boundary not verified')
        calls = {}
        for call in raw['engine_calls']:
            for step in range(call['scheduler_step_start'], call['scheduler_step_stop']):
                require(step not in calls, 'overlapping scheduler step/engine call join'); calls[step] = call['index']
        metadata[prefix] = dict(mode=mode, repeat=i, reset=reset, calls=calls,
                                old_complete_before_engine_call=release['engine_call'])
        anchors[(phase, 0)] = initial
        for action in raw['event_actions']:
            step = raw['engine_calls'][action['engine_call']]['scheduler_step_start']
            anchors[(phase, step)] = action['before']['cache']
        for suffix, count in [('warmup', 32), ('warmup_injection_none_early_release8', 384),
                ('warmup_injection_frequency_early_release8', 384), ('warmup_injection_frequency_late_release8', 384),
                ('warmup_injection_decode_late_release8', 384), ('measurement', 384)]:
            expected_phases[prefix + '/' + suffix] = count
    observed, totals, measurement_totals = Counter(), Counter(), Counter()
    rows, resets, full_anchors, last_prefix = [], [], 0, None
    trace = cell / 'pager/calls.jsonl'; hashes[str(trace.relative_to(root))] = sha(trace)
    with trace.open() as stream:
        for index, line in enumerate(stream):
            r = json.loads(line); context = r['context']; phase = context['phase']; name = r['layer_name']
            equal(r['call_id'], index, 'consecutive_call_id', r)
            require(r['status'] == 'complete' and r['grouping_axis'] == 'expert', 'incomplete/nonexpert call')
            prefix = phase.split('/')[0]
            if prefix != 'initialization' and prefix != last_prefix:
                require(prefix in metadata and prefix not in resets, 'unknown or repeated reset boundary')
                reset = metadata[prefix]['reset']
                for layer in layers:
                    anchor(states[layer], reset['cache_before'][layer], 'reset_before/' + layer, r)
                    equal(reset['cache_after'][layer], LRU(24).snapshot(), 'empty_reset/' + layer, r)
                    states[layer] = restore(reset['cache_after'][layer])
                resets.append(prefix); last_prefix = prefix; full_anchors += len(layers)
            state = states[name]; entry = set(state.mapping)
            a = set(e for row in r['row_topk_experts'] for e in row)
            equal(len(r['row_topk_experts']), r['rows'], 'row_count', r)
            require(all(len(row) == len(set(row)) == 8 and all(0 <= e < 64 for e in row)
                        for row in r['row_topk_experts']), 'invalid top-k rows')
            equal(sorted(entry), r['entry_resident_experts'], 'entry_resident', r)
            equal(sorted(a), r['active_experts'], 'active_union', r)
            equal(sorted(a - entry), r['missing_experts'], 'entry_missing', r)
            saved_anchor = anchors.get((phase, context['step_id']), {}).get(name)
            if saved_anchor is not None:
                anchor(state, saved_anchor, 'phase_anchor', r); full_anchors += 1
            t = r['retention']; plan, info = functions['retention_plan'](
                r['row_topk_experts'], context, entry, 24, t['mode'],
                salt=t.get('hash_salt') or 'unused', order=t['order'], guard=t.get('guard', 'none'))
            equal(plan, [dict(execute_experts=g['required_experts'], ensure_experts=g['ensure_experts'],
                             protected_after=g['protected_after']) for g in r['groups']], 'exact_plan', r)
            for key, value in info.items(): equal(value, t[key], 'retention/' + key, r)
            equal(Counter(e for g in plan for e in g['execute_experts']), Counter(a), 'active_once', r)
            before = state.snapshot() if r['measurement'] and t['applied'] else None
            loaded, victims = [], []; size = layers[name]['pinned_bytes'] // 64
            for gi, group in enumerate(plan):
                require(len(group['ensure_experts']) <= 24, 'capacity exceeded')
                missing, evicted = state.ensure(group['ensure_experts'])
                for key, value in [('loaded_experts', missing), ('evicted_experts', evicted),
                        ('reloaded_experts', sorted(set(missing) & ever[name])), ('miss', len(missing)),
                        ('evict', len(evicted)), ('weight_copy_bytes', len(missing) * size)]:
                    equal(value, r['groups'][gi][key], key, r, gi)
                require(set(group['protected_after']) <= set(state.mapping), 'protected state lost')
                ever[name].update(missing); loaded.extend(missing); victims.extend(evicted)
            equal(Counter(loaded), Counter(a - entry), 'entry_miss_once', r)
            equal(loaded, r['loaded_experts'], 'loaded_order', r)
            equal(sorted(state.mapping), t['final_resident_experts'], 'final_resident', r)
            cost = dict(miss=len(loaded), evict=len(victims), group_count=len(plan), weight_copy_bytes=len(loaded)*size)
            for key, value in cost.items(): equal(value, r[key], 'call_total/' + key, r)
            totals.update(cost); observed[phase] += 1
            equal(r['measurement'], phase.endswith('/measurement'), 'measurement_phase', r)
            if r['measurement']:
                measurement_totals.update(cost); m = metadata[prefix]; step = context['step_id']
                rows.append(dict(engine=engine, repeat=m['repeat'], mode=m['mode'], phase=phase,
                    call_id=r['call_id'], layer=name, step=step, engine_call=m['calls'][step], rows=r['rows'],
                    old_complete_before_engine_call=m['old_complete_before_engine_call'],
                    active=sorted(a), entry=sorted(entry), final=sorted(state.mapping), miss=len(loaded),
                    groups=len(plan), applied=t['applied'], protected=t['chosen_protected_experts'], before=before))
    require(observed == expected_phases, 'missing/unexpected initialization, warmup or measurement calls')
    require(sum(observed.values()) == pager['all_calls'] and len(rows) == pager['measurement_calls'], 'pager count mismatch')
    for scope, computed in [('all', totals), ('measurement', measurement_totals)]:
        require(all(value == pager[scope][key] for key, value in computed.items()), 'pager aggregate mismatch/' + scope)
    return rows, dict(engine=engine, status='PASS', calls=sum(observed.values()), measurement_calls=len(rows),
        phase_counts=observed, full_layer_anchors=full_anchors, resets=len(resets), actual_totals=totals)


def displacements(rows, ordinary):
    following, next_by_call = {}, {}
    for r in reversed(rows):
        key = (r['engine'], r['phase'], r['layer'])
        next_by_call[(r['engine'], r['call_id'])] = following.get(key); following[key] = r
    events = []
    for r in rows:
        if not r['applied']: continue
        state = restore(r['before']); active, entry = set(r['active']), set(r['entry'])
        plan = ordinary(active, entry, 24); loaded = []
        require(Counter(e for g in plan for e in g) == Counter(active), 'ordinary active coverage')
        for group in plan:
            require(len(group) <= 24, 'ordinary capacity'); missing, _ = state.ensure(group); loaded.extend(missing)
        require(Counter(loaded) == Counter(active-entry) and len(loaded) == r['miss'], 'ordinary entry-miss-once')
        actual, plain = set(r['final']), set(state.mapping)
        beneficiaries, victims = actual-plain, plain-actual
        require(len(actual) == len(plain) and len(beneficiaries) == len(victims), 'resident conservation')
        nxt = next_by_call[(r['engine'], r['call_id'])]
        def period(row):
            return 'post_old' if row['engine_call'] >= row['old_complete_before_engine_call'] else 'old_active'
        e = {k: r[k] for k in ('engine', 'repeat', 'mode', 'phase', 'call_id', 'layer', 'step', 'engine_call', 'rows')}
        e.update(period=period(r), protected=r['protected'], entry_resident=r['entry'], actual_final=r['final'],
            ordinary_final=sorted(plain), beneficiaries=sorted(beneficiaries), victims=sorted(victims),
            actual_groups=r['groups'], ordinary_groups=len(plan), group_delta=r['groups']-len(plan),
            actual_current_miss=r['miss'], ordinary_current_miss=len(loaded),
            next_status='OBSERVED_SAME_POLICY' if nxt else 'NO_NEXT', next_call_id=None, next_step=None,
            next_engine_call=None, next_period=None, next_demand=None, beneficiaries_next_demand=None,
            victims_next_demand=None, conditional_actual_miss=None, conditional_ordinary_miss=None,
            conditional_net_miss_actual_minus_ordinary=None)
        if nxt:
            require(actual == set(nxt['entry']), 'next observed entry differs from current final')
            demand = set(nxt['active']); saved, added = beneficiaries & demand, victims & demand
            actual_miss, ordinary_miss = len(demand-actual), len(demand-plain)
            require(actual_miss == nxt['miss'] and actual_miss-ordinary_miss == len(added)-len(saved), 'conditional identity')
            e.update(next_call_id=nxt['call_id'], next_step=nxt['step'], next_engine_call=nxt['engine_call'],
                next_period=period(nxt), next_demand=sorted(demand), beneficiaries_next_demand=sorted(saved),
                victims_next_demand=sorted(added), conditional_actual_miss=actual_miss,
                conditional_ordinary_miss=ordinary_miss, conditional_net_miss_actual_minus_ordinary=actual_miss-ordinary_miss)
        events.append(e)
    return events


def summarize(events):
    result = dict(events=len(events), no_next=sum(e['next_status']=='NO_NEXT' for e in events),
        current_extra_groups=sum(e['group_delta'] for e in events),
        beneficiary_identities=sum(len(e['beneficiaries']) for e in events), victim_identities=sum(len(e['victims']) for e in events))
    known = [e for e in events if e['next_status'] != 'NO_NEXT']
    result.update(conditional_saved=sum(len(e['beneficiaries_next_demand']) for e in known),
        conditional_added=sum(len(e['victims_next_demand']) for e in known),
        conditional_net_miss_actual_minus_ordinary=sum(e['conditional_net_miss_actual_minus_ordinary'] for e in known),
        better=sum(e['conditional_net_miss_actual_minus_ordinary']<0 for e in known),
        equal=sum(e['conditional_net_miss_actual_minus_ordinary']==0 for e in known),
        worse=sum(e['conditional_net_miss_actual_minus_ordinary']>0 for e in known))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); result = dict(schema='conditional_retention_displacement_v1', status='UNRUN', issues=[])
    with args.out.open('x') as output:
        try:
            root = args.input_dir; protocol = read(root/'protocol.json'); source = root/'source/wisp_expert_groups.py'
            require(sha(source) == protocol['source_sha256']['wisp_expert_groups.py'], 'planner source mismatch')
            require(set(protocol['engine_sequences']) == {'0_forward', '1_rotated'}, 'unexpected engine matrix')
            functions = runpy.run_path(str(source)); hashes = {'protocol.json': sha(root/'protocol.json'), 'source/wisp_expert_groups.py': sha(source)}
            rows, verification = [], []
            for engine, sequence in protocol['engine_sequences'].items():
                records, checked = verify_engine(root, engine, sequence, functions, hashes)
                rows.extend(records); verification.append(checked)
            # No local branch is evaluated until both complete observed trajectories pass.
            events = displacements(rows, functions['partition_experts'])
            episodes = []
            for engine, sequence in protocol['engine_sequences'].items():
                for repeat, mode in enumerate(sequence):
                    subset = [e for e in events if (e['engine'],e['repeat']) == (engine,repeat)]
                    episodes.append(dict(engine=engine, repeat=repeat, mode=mode, **summarize(subset),
                        by_extra_groups={str(g):summarize([e for e in subset if e['group_delta']==g]) for g in sorted({e['group_delta'] for e in subset})},
                        by_next_period={p:summarize([e for e in subset if e['next_period']==p]) for p in ('old_active','post_old',None)}))
            result.update(status='CONDITIONAL_CURRENT_CALL_DISPLACEMENT_ONLY', verification=verification, events=events,
                episodes=episodes, totals=summarize(events),
                by_mode={m:summarize([e for e in events if e['mode']==m]) for m in sorted({r['mode'] for r in rows})},
                input_sha256=hashes, analyzer_sha256=sha(Path(__file__)), lru_helper_sha256=sha(Path(__file__).with_name('analyze_layer_budget.py')),
                upstream_ensure_sha256=UPSTREAM_SHA, python=sys.version,
                formulas=dict(beneficiaries='actual_final - ordinary_final', victims='ordinary_final - actual_final',
                    conditional_net='|next_demand - actual_final| - |next_demand - ordinary_final| = |victims & next_demand| - |beneficiaries & next_demand|',
                    sign='negative means fewer conditional next-entry misses; next execution is not branched'),
                scope=['All initialization/warmup/measurement calls reproduced before branching; reset and measured anchors include exact slots, map and absolute LRU ticks/clock where recorded.',
                    'All applied measurement calls retained, including zero displacement and NO_NEXT; no cross-episode next-demand join. Each ordinary branch lasts only its current call.',
                    'Future labels use each policy own actual demand, not a regenerated ordinary trajectory. Summed labels are descriptive and are not a policy miss, byte or latency saving.',
                    'old_active/post_old follows each raw completed-old release engine-call boundary, without changing arrivals or trajectories.',
                    'No kernel/time/load-envelope analysis; no subtraction of overlapping timings. Repeats and per-layer events are dependent, not independent samples.'])
        except ReplayMismatch as exc:
            result.update(status='REPLAY_MISMATCH_STOPPED', first_mismatch=exc.detail); result['issues'].append('Actual replay failed; no branch evaluated')
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            result.update(status='DIAGNOSTIC_FAILED'); result['issues'].append(repr(exc))
        json.dump(result, output, indent=2, allow_nan=False); output.write('\n')
    if result['issues']: raise SystemExit('diagnostic stopped; failure preserved')


if __name__ == '__main__': main()

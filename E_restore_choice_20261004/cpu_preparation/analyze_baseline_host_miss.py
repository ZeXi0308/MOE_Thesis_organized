"""Reproduce the two fixed-Host baseline miss-opportunity extracts, CPU only.

Run from any directory: python3 -B PATH/TO/analyze_baseline_host_miss.py
Writes E_restore_choice_20261004/baseline_host_miss_opportunity.json.
"""
import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from analyze_stop_order import Trace


def main():
    paths = [root/'execution/oct08_e246_history53005_r01/00_same_engine/00_host/raw.json',
             root/'execution/oct07_e223_r01/00_same_engine/00_host/raw.json']
    out = []
    for path in paths:
        t = Trace(path, 'measured/E243')
        cell = path.parent
        config = json.loads((cell/'config.json').read_text())
        resources = json.loads((cell/'resources.json').read_text())
        gpu = json.loads((cell/'gpu_before_measurement.json').read_text()) if (cell/'gpu_before_measurement.json').exists() else None
        engine = json.loads((cell/'engine_args.json').read_text())
        cs, ps, workrows, seen = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(set)
        for c in t.raw['commits']:
            cs[c['request_id']].append(c)
        for p in t.raw['preemptions']:
            ps[p['request_id']].append(p)
        for step, s in enumerate(t.raw['scheduler_steps']):
            for r in s['scheduled']:
                rid = r['request_id']
                positions = set(range(r['start_computed'], r['end_computed']))
                workrows[rid].append(dict(step=step, time_s=s['time_s'], start=r['start_computed'],
                    end=r['end_computed'], count=r['count'], repeated=len(positions & seen[rid])))
                seen[rid].update(positions)
        requests = []
        miss_repeated = miss_scheduled = second_repeated = second_scheduled = overlap_rebuilt = miss_total = 0
        for rid, commits in sorted(cs.items(), key=lambda pair: t.ids[pair[0]]):
            commits.sort(key=lambda c: c['allocation_s'])
            miss_indices = [i for i, c in enumerate(commits) if c['fallback'] == 'host_miss_native_recompute']
            if not miss_indices:
                continue
            first, external = miss_indices[0], t.ids[rid]
            request, seq, position_sets = t.requests[external], [], {}
            for i in range(first, len(commits)):
                c = commits[i]
                after = c['allocation_s']
                next_index = bisect.bisect_right(request['token_times_s'], after)
                candidates = []
                if next_index < len(request['token_times_s']):
                    candidates.append((request['token_times_s'][next_index], 'next_output'))
                pre = next((p for p in ps[rid] if p['time_s'] > after), None)
                if pre:
                    candidates.append((pre['time_s'], 'preemption'))
                if i+1 < len(commits):
                    candidates.append((commits[i+1]['allocation_s'], 'next_recovery'))
                end, kind = min(candidates)
                selected = [w for w in workrows[rid] if w['time_s'] >= after and
                            (w['time_s'] <= end if kind == 'next_output' else w['time_s'] < end)]
                positions = set().union(*(set(range(w['start'], w['end'])) for w in selected)) if selected else set()
                position_sets[i] = positions
                entry = dict(recovery_ordinal=i+1, event=c['event'], step=t.step(after),
                    known_tokens=c['known_tokens'], generated_tokens=c['generated_tokens'],
                    host_hit_tokens=c['host_hit_tokens'], external_tokens=c['external_tokens'],
                    actual_action=c['actual_action'], fallback=c['fallback'],
                    joint_capacity=c['joint_capacity'], eligible=c['eligible'], attempt_end=kind,
                    attempt_end_step=t.step(end),
                    next_strict_output_index=next_index if next_index < len(request['token_times_s']) else None,
                    next_strict_output_step=t.token_steps[external][next_index] if next_index < len(request['token_times_s']) else None,
                    scheduled_positions=sum(w['count'] for w in selected),
                    repeated_positions=sum(w['repeated'] for w in selected),
                    native_ranges=[{k: v for k, v in w.items() if k != 'time_s'} for w in selected])
                seq.append(entry)
                if i in miss_indices:
                    assert c['actual_action'] == 'recompute' and c['host_hit_tokens'] == c['external_tokens'] == 0
                    miss_total += 1
                    miss_scheduled += entry['scheduled_positions']
                    miss_repeated += entry['repeated_positions']
                    if i != first:
                        second_scheduled += entry['scheduled_positions']
                        second_repeated += entry['repeated_positions']
            pairs = []
            for i, j in zip(miss_indices, miss_indices[1:]):
                pos = [c for c in commits[i+1:j] if c['actual_action'] == 'host' and c['external_tokens'] > 0]
                overlap = len(position_sets[i] & position_sets[j])
                overlap_rebuilt += overlap
                pairs.append(dict(previous_miss_event=commits[i]['event'], next_miss_event=commits[j]['event'],
                    positive_Host_recovery_events_between=[c['event'] for c in pos],
                    next_miss_scheduled_positions=seq[j-first]['scheduled_positions'],
                    next_miss_repeated_positions=seq[j-first]['repeated_positions'],
                    positions_from_previous_miss_rebuilt_again=overlap))
            requests.append(dict(external_id=external, prompt_tokens=request['prompt_tokens'],
                output_tokens=len(request['output_token_ids']), finish_reason=request['finish_reason'],
                total_recovery_commits=len(commits),
                positive_Host_commits_before_first_miss=sum(c['actual_action'] == 'host' and c['external_tokens'] > 0 for c in commits[:first]),
                fallback_event_sequence=[c['event'] for c in commits if c['fallback'] == 'host_miss_native_recompute'],
                recoveries_from_first_miss=seq, adjacent_miss_pairs=pairs,
                whole_request_position_work=t.request_work[external]))
        whole = {k: sum(w[k] for w in t.request_work.values())
                 for k in ['scheduled', 'unique_positions', 'repeated_positions']}
        out.append(dict(group=path.parts[-4], cell=str(Path(root.name)/cell.relative_to(root)),
            raw_sha256=t.sha256, status=json.loads((cell/'status.json').read_text())['status'],
            configuration={**{k: config.get(k) for k in ['model', 'host_gib', 'batch_tokens', 'max_num_seqs', 'requests', 'natural', 'kv_bytes']},
                'gpu_blocks': resources['gpu_blocks'], 'gpu_bytes': resources.get('gpu_bytes'),
                'host_bytes': resources.get('host_bytes'), 'capacity_mode': resources.get('capacity_mode'),
                'max_model_len': engine.get('max_model_len'), 'gpu_before_measurement': gpu},
            summary=dict(requests=len(t.requests), fallback_commits=miss_total, distinct_miss_requests=len(requests),
                requests_with_multiple_misses=sum(len(r['fallback_event_sequence']) > 1 for r in requests),
                misses_with_any_later_recovery=sum(e['fallback'] == 'host_miss_native_recompute'
                    for r in requests for e in r['recoveries_from_first_miss'][:-1]),
                all_miss_attempt_scheduled_positions=miss_scheduled, all_miss_attempt_repeated_positions=miss_repeated,
                second_or_later_miss_scheduled_positions=second_scheduled, second_or_later_miss_repeated_positions=second_repeated,
                positions_from_previous_miss_rebuilt_again=overlap_rebuilt, whole_episode_position_work=whole,
                second_or_later_miss_repeated_fraction_of_episode=second_repeated/whole['repeated_positions']),
            requests_with_miss=requests))
    result = dict(schema='E.baseline_Host_miss_opportunity.v1',
        scope='Two completed fixed-Host baselines only; no GPU/wall-clock comparison or counterfactual execution.',
        definitions=[
            'Only committed fallback=host_miss_native_recompute with actual_action=recompute, external_tokens=0 and host_hit_tokens=0 counts as a miss execution; failed lookups are not executions.',
            'Recovery attempt work starts at allocation and ends at the earliest next strictly later output, next preemption, or next recovery; native scheduled intervals are counted once per disjoint attempt.',
            'Repeated positions are per-request positions already scheduled earlier in the complete episode; they are real work counts, not wall-clock costs or proven avoidable savings.',
            'Adjacent-miss overlap counts positions rebuilt in both miss attempts; decode work between misses can make total second-miss repeated work larger.',
            'No STORE-cursor, residency, eviction, or future cache retention is inferred from host misses.'],
        groups=out, decision=[
            'Latest field domain: every committed miss is the final recovery of a different request; no observed subsequent reuse opportunity for post-miss STORE replenishment. Stop that candidate in this measured domain.',
            'Historical E223 Host baseline may identify a repeated-miss deployment condition only; it does not establish this opportunity on the current card or show that replenishment would persist or improve service.'])
    output = root/'baseline_host_miss_opportunity.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(output)


if __name__ == '__main__':
    main()

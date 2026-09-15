"""Bounded short-service extraction from the canonical diagnostic-eager raw."""
import argparse
from collections import defaultdict
import json
from math import ceil
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
BUNDLE = Path('refine-logs/expert_saturation/outputs/admission_capacity/20260915_saved_recovery_start_r01')
SOURCE = BUNDLE / 'execution_weste_26862/readback/results/diagnostic-eager'


def extract():
    with (REPO / SOURCE / 'raw.json').open() as f:
        raw = json.load(f)
    with (REPO / SOURCE / 'selective-store.json').open() as f:
        selective = json.load(f)
    assert raw['status'] == 'COMPLETE'
    memory = {x['attempted_step']: x for x in raw['memory_trace']}
    schedules = {x['step']: x for x in raw['scheduler_steps']}
    snapshots = {x['step']: x for x in selective['eligibility_snapshots']}
    forced = {(x['step'], x['victim']) for x in selective['events']
              if x['event'] == 'commit_check' and x['reason'] == 'READY'}
    by_request = defaultdict(list)
    for event in raw['preemption_events']:
        by_request[event['victim_internal_request_id']].append(event)
    call_by_return = {x['returned_s']: x['call_index'] for x in raw['engine_steps']}
    outputs = defaultdict(list)
    for event in raw['output_events']:
        outputs[event['request_id']].append((call_by_return[event['received_s']], len(event['new_token_ids'])))
    bins = {'0': 0, '1-2': 0, '3-8': 0}
    segments = 0
    shortest = None
    short_rows = []
    for rid, preemptions in by_request.items():
        for prior, event in zip(preemptions, preemptions[1:]):
            start, end = prior['attempted_step'], event['attempted_step']
            count = event['victim_state']['output_tokens'] - prior['victim_state']['output_tokens']
            assert count >= 0
            segments += 1
            shortest = count if shortest is None else min(shortest, count)
            if count > 8:
                continue
            band = '0' if count == 0 else '1-2' if count <= 2 else '3-8'
            bins[band] += 1
            recovery = next(step for step in range(start, end)
                if sum(memory[step]['after']['requests'].get(rid, {}).get('block_counts', [])) > 0
                or any(x['internal_request_id'] == rid and x['scheduled_tokens'] > 0
                       for x in schedules[step]['scheduled']))
            source_id = raw['internal_to_source'][rid]
            service = [(step, n) for step, n in outputs[source_id] if recovery <= step < end]
            assert sum(n for _, n in service) == count
            snap = snapshots[end]
            resources = []
            for order, request in enumerate(snap['running_ids']):
                state = snap['requests'][request]
                growth = max(0, ceil((state['prompt'] + state['output']) / 16) - state['held_blocks'])
                if growth or request == rid:
                    resources.append(dict(request_id=raw['internal_to_source'][request],
                        internal_request_id=request, running_order=order, before=state,
                        next_output_extra_blocks=growth,
                        after=memory[end]['after']['requests'][request]))
            failures = memory[end]['allocation_failures']
            is_forced = (end, rid) in forced
            capacity = not is_forced and bool(failures) and event['pool']['free_blocks'] == 0
            kind = 'forced_rotation' if is_forced else 'native_capacity' if capacity else 'native_unlocalized'
            short_rows.append(dict(request_id=source_id, band=band,
                previous_preempt_step=start, recovery_start_step=recovery,
                first_new_output_step=service[0][0] if service else None,
                last_new_output_step=service[-1][0] if service else None,
                preempt_step=end, new_outputs=count, next_preempt_type=kind,
                protection_release_events=[x for x in selective['events']
                    if x['event'] == 'target_new_output' and x['request'] == rid and recovery <= x['step'] <= end],
                selector_decisions=[x for x in selective['selector_decisions'] if x['step'] == end],
                eligibility={k: snap[k] for k in ('step', 'free_blocks', 'plan_victim', 'plan_target',
                    'protected_id', 'protected_reserve')},
                request_resource_rows=resources, allocation_failures=failures,
                pool_after_schedule=memory[end]['after']['pool'],
                preemption_event={k: v for k, v in event.items()
                    if k not in ('output_token_ids_before', 'output_token_ids_after')}))
    return dict(status='OBSERVED_SHORT_SERVICE_LOCALIZATION',
        sources={'raw': str(SOURCE / 'raw.json'), 'selective_store': str(SOURCE / 'selective-store.json')},
        rotation_config=selective['rotation_config'],
        definition='One request between consecutive preemptions, after it re-acquires positive GPU blocks or executes. '
            'Recovery start may be allocation before async ready. Count actual returned new outputs from that start '
            'through the call before the next preemption, verified against the two preemption output counters. '
            'Initial residencies and residencies ending in completion are excluded. Bands 0, 1-2 and 3-8 are reported separately.',
        recovered_then_preempted_segments=segments, minimum_new_outputs=shortest, bins=bins,
        short_segments=sorted(short_rows, key=lambda x: x['preempt_step']),
        checks={'short_output_counts_match_engine_return_events': True,
                'all_0_to_8_segments_listed': sum(bins.values()) == len(short_rows)},
        boundary='Only this diagnostic trajectory and these requested bands. No main lifecycle/performance reanalysis, '
            'no causal time saving or benefit of a longer protection window. Empty 0/1-2 bands do not become a positive '
            'short-window result by changing the threshold to 3-8.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=REPO / BUNDLE / 'lifecycle_cost/short_service.json')
    output = parser.parse_args().output
    result = extract()
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(output=str(output), bins=result['bins'],
        rows=[{k: x[k] for k in ('request_id', 'recovery_start_step', 'first_new_output_step',
            'last_new_output_step', 'preempt_step', 'new_outputs', 'next_preempt_type')}
            for x in result['short_segments']], checks=result['checks']), indent=2))

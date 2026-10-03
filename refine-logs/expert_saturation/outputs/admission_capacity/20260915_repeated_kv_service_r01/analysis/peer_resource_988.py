"""Extract one observed native capacity fallback; shared raw remains read-only."""
import argparse
import json
from math import ceil
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
BUNDLE = Path('refine-logs/expert_saturation/outputs/admission_capacity/20260915_repeated_kv_service_r01')
SOURCE = BUNDLE / 'execution_weste_26862/readback/results/diag-on'


def one(rows):
    rows = list(rows)
    assert len(rows) == 1, f'Expected one matching source row, got {len(rows)}'
    return rows[0]


def extract():
    with (REPO / SOURCE / 'raw.json').open() as f:
        raw = json.load(f)
    with (REPO / SOURCE / 'selective-store.json').open() as f:
        selective = json.load(f)
    source_id = 'memory-train-article-0000799'
    rid = one(k for k, v in raw['internal_to_source'].items() if v == source_id)
    memory = one(x for x in raw['memory_trace'] if x['attempted_step'] == 988)
    schedule = one(x for x in raw['scheduler_steps'] if x['step'] == 988)
    snapshot = one(x for x in selective['eligibility_snapshots'] if x['step'] == 988)
    decision = one(x for x in selective['selector_decisions'] if x['step'] == 988)
    release = one(x for x in selective['events'] if x.get('step') == 986
                  and x['event'] == 'target_new_output' and x['request'] == rid)
    preempt = one(x for x in raw['preemption_events'] if x['attempted_step'] == 988)
    rows = []
    for order, request in enumerate(snapshot['running_ids']):
        before = memory['before']['requests'][request]
        after = memory['after']['requests'][request]
        history = before['prompt_tokens'] + before['output_tokens']
        need = max(0, ceil(history / 16) - before['block_counts'][0])
        if need or request == rid:
            rows.append(dict(internal_request_id=request,
                request_id=raw['internal_to_source'][request], running_order=order,
                before=before, after=after, history_tokens=history,
                next_output_extra_blocks=need,
                observed_block_delta=after['block_counts'][0] - before['block_counts'][0]))
    victim = one(x for x in rows if x['internal_request_id'] == rid)
    peers = [x for x in rows if x['internal_request_id'] != rid]
    failure = one(memory['allocation_failures'])
    checks = dict(
        protection_released_before_988=release['step'] == 986 and snapshot['protected_id'] is None,
        no_forced_plan=snapshot['plan_victim'] is None and snapshot['plan_target'] is None,
        selector_noop_cooldown=decision['action'] == 'noop' and decision['reason'] == 'swap cooldown',
        initial_free_one=memory['before']['pool']['free_blocks'] == snapshot['free_blocks'] == 1,
        victim_held244_growth0=victim['before']['block_counts'] == [244] and victim['next_output_extra_blocks'] == 0,
        four_peers_grow_one={x['request_id'] for x in peers} == {
            'memory-train-article-0002932', 'memory-train-article-0003475',
            'memory-train-article-0000480', 'memory-train-article-0002038'}
            and all(x['next_output_extra_blocks'] == x['observed_block_delta'] == 1 for x in peers),
        first_growth_is2932=peers[0]['request_id'] == 'memory-train-article-0002932',
        failure_is3475_at_free0=raw['internal_to_source'][failure['internal_request_id']] == 'memory-train-article-0003475'
            and failure['requested_tokens'] == 1 and failure['before']['free_blocks'] == 0,
        victim_native_preempt_record=preempt['victim_internal_request_id'] == rid
            and preempt['original_preemption_called'] and preempt['original_preemption_returned'],
        preempt_free0_to244=preempt['pool']['free_blocks'] == 0 and preempt['pool_after']['free_blocks'] == 244,
        victim_not_scheduled=all(x['internal_request_id'] != rid for x in schedule['scheduled']),
        block_conservation=1 + 244 - sum(x['observed_block_delta'] for x in peers)
            == memory['after']['pool']['free_blocks'] == 241)
    assert all(checks.values()), checks
    return dict(status='OBSERVED_SINGLE_STEP_RESOURCE_LOCALIZATION',
        sources={'raw': str(SOURCE / 'raw.json'), 'selective_store': str(SOURCE / 'selective-store.json')},
        step=988, request_id=source_id, protection_release_event=release,
        selector_decision=decision,
        eligibility={k: snapshot[k] for k in ('step', 'host_perf_counter_s', 'free_blocks',
            'running_ids', 'plan_victim', 'plan_target', 'protected_id', 'protected_reserve')},
        pool_before=memory['before']['pool'], pool_after=memory['after']['pool'],
        request_resource_rows=rows, allocation_failures=memory['allocation_failures'],
        preemption_event={k: v for k, v in preempt.items() if k not in
            ('output_token_ids_before', 'output_token_ids_after')},
        scheduled_peer_growth_rows=[x for x in schedule['scheduled']
            if x['internal_request_id'] in {p['internal_request_id'] for p in peers}],
        checks=checks,
        conclusion='With no forced plan and no protection, native allocation failure for peer3475 precedes eviction of0799. '
            'Peer2932 uses the initial free block; four peers grow by one block each. '
            'This localizes the observed resource transition, not a counterfactual benefit of extending protection.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=REPO / BUNDLE / 'analysis/peer_resource_988.json')
    output = parser.parse_args().output
    result = extract()
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(output=str(output), status=result['status'], checks=result['checks']), indent=2))

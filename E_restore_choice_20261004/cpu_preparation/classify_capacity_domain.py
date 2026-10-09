"""Classify existing lookup capacity failures; no engine or GPU execution."""
import argparse
import json
from pathlib import Path


def classify(path):
    raw = json.loads(path.read_text())
    decisions = raw['decisions']
    pending = [r for r in decisions if r['fallback'] == 'pending_native']
    rows = [r for r in decisions
            if r['fallback'] == 'full_capacity_not_jointly_available_native']
    pure, conservative = [], []
    for row in rows:
        free = row['free_blocks']
        full = row['full_required_blocks']
        watermark = row['watermark_blocks']
        reserved = row['reserved_blocks']
        assert row['host_hit_tokens'] > 0
        assert not row['joint_capacity'] and free < full + reserved + watermark
        (pure if free < full + watermark else conservative).append(row)
    return dict(
        raw=str(path),
        total_preempted_lookups=len(decisions),
        pending_native=dict(
            lookups=len(pending),
            requests=len({r['request_id'] for r in pending}),
            native_full_capacity_necessary_condition_passes=sum(
                r['full_required_blocks'] + r['watermark_blocks'] <= r['free_blocks']
                for r in pending),
            selector_joint_capacity_passes=sum(bool(r['joint_capacity']) for r in pending),
            timing='No samples' if not pending else 'Not classified by this counter; None is not a transfer-duration label'),
        ready_host_lookups=sum(r['host_hit_tokens'] is not None and
                               r['host_hit_tokens'] > 0 for r in decisions),
        eligible_lookups=sum(bool(r['eligible']) for r in decisions),
        capacity_fallback_lookups=len(rows),
        capacity_fallback_requests=len({r['request_id'] for r in rows}),
        full_sequence_native_gate_fails=dict(
            lookups=len(pure), requests=len({r['request_id'] for r in pure})),
        only_selector_full_plus_reservation_fails=dict(
            lookups=len(conservative),
            requests=len({r['request_id'] for r in conservative}),
            events=[r['event'] for r in conservative]),
        reserved_blocks_nonzero_lookups=sum(r['reserved_blocks'] > 0 for r in rows),
        reserved_blocks_max=max((r['reserved_blocks'] for r in rows), default=None),
        executed_capacity_fallback_commits=sum(
            r['fallback'] == 'full_capacity_not_jointly_available_native'
            for r in raw['commits']),
    )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('group', nargs='?', type=Path,
                        default=root / 'execution/oct08_two_burst160_20s53005_r01')
    args = parser.parse_args()
    paths = sorted((args.group / '00_same_engine').glob('*/raw.json'))
    assert len(paths) == 2, 'Require a completed two-arm group with both raw files'
    result = dict(
        schema='E.capacity_domain.v1',
        evidence='CPU classification of recorded current-state fields; no new GPU intervention',
        scope='APC disabled, one full-attention KV group, zero computed tokens at lookup, full-sequence admission enabled, no lookahead',
        predicates=dict(
            selector='full_required_blocks + reserved_blocks + watermark_blocks <= free_blocks',
            native_first_gate='full_required_blocks + watermark_blocks <= free_blocks',
            distinction='The native first full-sequence gate applies to both Host and recompute even when the executed recompute chunk is smaller.'),
        sources=[
            'selector.py:207-216',
            'run_cell.py:286',
            'native_sources/v1/core/sched/scheduler.py:928-946',
            'native_sources/v1/core/kv_cache_manager.py:411-427',
            'native_sources/v1/core/kv_cache_manager.py:449-466'],
        limits=[
            'Lookups can repeat for one waiting request; these counts are not independent interventions or capacity-wait duration.',
            'Passing the simplified full-sequence predicate is necessary, not proof of successful allocation; native per-action checks and callback still decide execution.',
            'Host has a further actual-prefix allocation check with in-flight reservations; recompute uses native chunk allocation without that reservation argument.',
            'No inference that disabling the full-sequence admission rule is safe, beneficial, or within an unchanged-admission experiment.'],
        arms=[classify(p) for p in paths],
    )
    out = args.group / 'capacity_domain.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(output=str(out), arms=result['arms']), indent=2))


if __name__ == '__main__':
    main()

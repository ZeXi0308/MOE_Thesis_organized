#!/usr/bin/env python3
"""Necessary native re-entry fit only; no time, allocation or service prediction.

Domain: single FullAttention, prefix cache off, lookahead zero, full-sequence
fit enabled, and zero owned pages after preemption. Host-ready KV does not
reduce this physical fit gate. Queue/phase/pending work and future reuse remain
unknown. Q = ceil(min(prompt + output, max_model_len) / block_size);
J = action-time refcount-derived releasable pages; actual release is validated
separately by the raw free-counter join for executed victims. D = Q - J is a
capacity offset holding other reservations and queue conditions fixed, not a
delay, victim score, or evidence of equal actual progress windows for candidates.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


PROBES = ('remaining_budget', 'max_release_shadow', 'cacheopt_cap_bucket')
SOURCE_DOMAIN = dict(
    status='ASSUMPTIONS_FROM_REVIEWED_NATIVE_SOURCE_NOT_EXTRA_RUNTIME_OBSERVATIONS',
    assumptions=['single FullAttention', 'lookahead=0', 'post-preempt owned pages=0'],
    source_sha256={
        'scheduler.py': '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941',
        'kv_cache_manager.py': '3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf',
        'single_type_kv_cache_manager.py': 'bcb27e38895332bf6a4c55608f2917eb9fd941ec5a629c14b88358f00aadeba8'})


def count(value):
    return type(value) is int and value >= 0


def reentry_fit(row, block_size, max_model_len):
    """Pure conditional capacity bookkeeping; missing evidence is never zero."""
    unknown = dict(state='UNKNOWN', Q=None, J=None, D=None, cancellation_0_or_1=None)
    keys = ('prompt_tokens', 'output_tokens', 'computed_tokens', 'held_blocks',
            'immediate_releasable_blocks', 'shared_blocks')
    if (not count(block_size) or block_size == 0 or not count(max_model_len) or max_model_len == 0
            or any(not count(row.get(key)) for key in keys)
            or row.get('request_status') != 'RUNNING'
            or row.get('release_state_error', 'NOT_RECORDED') is not None):
        return dict(unknown, reason='MISSING_OR_INVALID_PROGRESS_PHYSICAL_STATE_OR_GEOMETRY')
    prompt, output, computed, held, released, shared = (row[key] for key in keys)
    if computed > max_model_len or computed > prompt + output or released + shared > held:
        return dict(unknown, reason='INCONSISTENT_PROGRESS_OR_PHYSICAL_COUNTS')
    pages = lambda tokens: (tokens + block_size - 1) // block_size
    q = pages(min(prompt + output, max_model_len))
    full = (output > 0 and computed == prompt + output - 1
        and held == pages(computed) and released == held and shared == 0)
    state = ('FULL_PRIVATE_DECODE' if full else 'PARTIAL_PROGRESS'
        if computed < prompt + output - 1 else 'OTHER_KNOWN_STATE')
    return dict(state=state, Q=q, J=released, D=q-released,
        cancellation_0_or_1=(q-released in (0, 1)) if full else False)


def check_domain(config, store, engine, profile):
    block, limit = profile.get('block_size_tokens'), engine.get('max_model_len')
    checks = dict(prefix_off=engine.get('enable_prefix_caching') is False
            and store.get('prefix_caching') is False,
        coordinator=store.get('coordinator_type') == 'KVCacheCoordinatorNoPrefixCache',
        full_sequence_fit=engine.get('scheduler_reserve_full_isl') is True
            and config.get('reservation_policy') == 'full',
        native_calculation_unmodified=store.get('native_calc_overridden') is False,
        native_full=store.get('store_scope') == config.get('store_scope') == 'native_full',
        geometry=count(block) and block > 0 and count(limit) and limit > 0
            and profile.get('max_model_len') == config.get('max_model_len') == limit)
    if not all(checks.values()):
        raise ValueError('DOMAIN_REJECTED: ' + ', '.join(key for key, ok in checks.items() if not ok))
    return dict(checks=checks, block_size=block, max_model_len=limit,
        source_domain=SOURCE_DOMAIN)


def summarize(decisions, block_size, max_model_len, steps=()):
    """Use recorded suggestions only; never reconstruct absent old policy logs."""
    records = {name: [] for name in PROBES}
    partial_events = unknown_events = 0
    details = []
    for index, event in enumerate(decisions):
        rows = event.get('candidates', [])
        states = [reentry_fit(row, block_size, max_model_len) for row in rows]
        partial_events += any(state['state'] == 'PARTIAL_PROGRESS' for state in states)
        unknown_events += not rows or any(state['state'] == 'UNKNOWN' for state in states)
        for name in PROBES:
            probe = event.get(name)
            proposed = probe.get('proposed_request') if isinstance(probe, dict) else None
            matches = [i for i, row in enumerate(rows) if isinstance(proposed, str) and row.get('request') == proposed]
            state = (states[matches[0]] if len(matches) == 1 else
                dict(state='UNKNOWN', Q=None, J=None, D=None, cancellation_0_or_1=None,
                    reason='SUGGESTION_NOT_RECORDED_OR_CANDIDATE_NOT_UNIQUE'))
            records[name].append(state)
            if event.get('step') in steps:
                row = rows[matches[0]] if len(matches) == 1 else {}
                details.append(dict(decision_index=index, step=event.get('step'), probe=name,
                    request=proposed, action_requested=probe.get('action_requested') if isinstance(probe, dict) else None,
                    **state, observed={key: row.get(key) for key in
                        ('prompt_tokens', 'output_tokens', 'computed_tokens', 'held_blocks',
                         'shared_blocks', 'host_ready_prefix_blocks')}))
    summaries = {}
    for name, states in records.items():
        known = [state for state in states if state['D'] is not None]
        summaries[name] = dict(recorded_decisions=len(states), known=len(known), unknown=len(states)-len(known),
            state_counts=dict(Counter(state['state'] for state in states)),
            D_gt_1_known=sum(state['D'] > 1 for state in known),
            D_counts_known=dict(Counter(state['D'] for state in known)),
            full_private_decode_cancellation_verified=sum(state['cancellation_0_or_1'] is True for state in states))
    return dict(decisions=len(decisions), partial_candidate_events=partial_events,
        unknown_candidate_events=unknown_events, recorded_suggestions=summaries), details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--steps', default='', help='Comma-separated recorded steps, e.g. 673,679')
    args = parser.parse_args()
    if Path(args.cell).name != args.cell:
        parser.error('--cell must be a single cell directory name')
    try:
        steps = {int(value) for value in args.steps.split(',') if value}
        archive = args.session / args.cell / 'archive'
        read = lambda name: json.loads((archive / name).read_text())
        config, store, engine, profile = (read(name) for name in
            ('config.json', 'selective-store.json', 'engine_args.json', 'normal_capacity_profile.json'))
        domain = check_domain(config, store, engine, profile)
        events = store.get('victim_decisions')
        if not isinstance(events, list):
            raise ValueError('victim_decisions NOT_RECORDED')
        summary, details = summarize(events, domain['block_size'], domain['max_model_len'], steps)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print('REJECTED', str(error))
        return 2
    print('DOMAIN', json.dumps(domain, ensure_ascii=False))
    print('SUMMARY', json.dumps(summary, ensure_ascii=False))
    for detail in details:
        print('STEP', json.dumps(detail, ensure_ascii=False))
    print('LIMIT: Q is a necessary re-entry fit gate. J is action-time refcount-derived '
          'releasable pages, not an observed physical free for an unexecuted candidate; '
          'actual release is validated separately by raw free-counter joins for executed victims. '
          'D=Q-J is a capacity offset holding other reservations and queue conditions fixed, '
          'not actual allocation, re-entry latency, sustainable service, or a victim score; '
          'it does not establish equal actual progress windows across candidates. '
          'Host-ready KV does not reduce Q; no future other-request release is credited.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

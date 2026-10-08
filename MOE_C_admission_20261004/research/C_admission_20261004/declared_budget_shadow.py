#!/usr/bin/env python3
"""Observe a declared-token budget at v13 KV01/KV02 successful new allocations.

Usage: python3 declared_budget_shadow.py V13_ANALYSIS.json --output NEW.json
Uses ceil((known prompt tokens + declared max output tokens)/16) per live request.
This is a same-trajectory shadow, not replay or an admission intervention.
"""
import argparse
import collections
import json
from pathlib import Path

from preoutput_coverage import FIT_KEYS
from slo_failure_breakdown import read_hashed


PAGE_TOKENS = 16
USABLE_PAGES = 32768


def analyze_cell(cell):
    path = Path(cell['cell'])
    raw, raw_sha = read_hashed(path/'raw.json')
    if raw_sha != cell['raw_sha256']:
        raise ValueError('Raw differs from source service analysis')
    admission, admission_sha = read_hashed(path/'admission.json')
    config, config_sha = read_hashed(path/'config.json')
    metrics, metrics_sha = read_hashed(Path(cell['per_request_json']))
    if config['max_model_len'] != 4096 or config['target_usable_kv_blocks'] != USABLE_PAGES:
        raise ValueError('Unexpected native model length or usable page budget')
    requests = {r['request_id']: r for r in raw['requests']}
    identity = {r[k]: r['request_id'] for r in raw['requests']
        for k in ('request_id', 'internal_request_id', 'external_request_id') if r.get(k)}
    declared = {rid:r['prompt_tokens']+r['max_output_tokens'] for rid,r in requests.items()}
    if any(n > config['max_model_len'] or n <= 0 for n in declared.values()):
        raise ValueError('Declared prompt+max exceeds native model length; no implicit truncation')
    pages = {rid:(n+PAGE_TOKENS-1)//PAGE_TOKENS for rid,n in declared.items()}
    origin = raw['measurement_origin_perf_counter_s']
    offset = admission['origin_perf_s']-origin
    starts = {identity[s['request_id']]:s['first_prefill_perf_s']-origin for s in admission['starts']}
    allocations = collections.defaultdict(list)
    decisions = collections.defaultdict(list)
    for a in admission['native_allocations']:
        if a['actual'] and a['status_before'] == 'WAITING' and a['preemptions'] == 0:
            allocations[identity[a['request_id']]].append(a)
    for r in admission['decisions']:
        if r.get('native_allocation_result') is True:
            decisions[identity[r['request_id']]].append(r)
    aligned = {}
    unknown = {}
    for rid in requests:
        if len(allocations[rid]) != 1 or len(decisions[rid]) != 1 or rid not in starts:
            unknown[rid] = 'Missing unique successful new allocation/decision/first-prefill alignment'
            continue
        a, r = allocations[rid][0], decisions[rid][0]
        if (not r['t'] <= a['t'] or offset+a['t'] > starts[rid] or
                r.get('native_fit') is not True or r.get('base_allowed') is not True or r['denied'] or
                any(k not in r or r[k] != a[k] for k in FIT_KEYS)):
            unknown[rid] = 'First successful allocation cannot be exactly aligned to allowed native-fit decision'
            continue
        aligned[rid] = (a, r)
    # Count every actual first admission, including same-step allocations whose
    # first-prefill schedule return is later. Shadow rejection does not remove it.
    first_allocated = {rid:offset+rows[0]['t'] for rid,rows in allocations.items() if len(rows) == 1}
    rows = []
    for rid, (a, r) in sorted(aligned.items(), key=lambda item:item[1][1]['t']):
        now = offset+r['t']
        live = {old for old, t in first_allocated.items() if t < now and
            (requests[old]['completion_s'] is None or requests[old]['completion_s'] >= now)}
        if rid in live:
            raise ValueError('New target was already admitted')
        matches = len(live) == r['active']
        pending_start = sorted(old for old in live if starts.get(old, float('inf')) > now)
        commitment = sum(pages[old] for old in live)
        projected = commitment+pages[rid]
        valid = matches and not unknown
        rows.append(dict(request_id=rid, gate_relative_time_s=r['t'], external_time_s=now,
            first_allocation_external_time_s=offset+a['t'], first_prefill_schedule_return_s=starts[rid],
            recorded_active=r['active'], reconstructed_live_requests=len(live), active_matches=matches,
            known_prompt_tokens=requests[rid]['prompt_tokens'], declared_max_output_tokens=requests[rid]['max_output_tokens'],
            live_declared_pages=commitment, incoming_declared_pages=pages[rid], projected_declared_pages=projected,
            budget_limit_pages=USABLE_PAGES, shadow_would_deny=projected > USABLE_PAGES if valid else None,
            reconstruction_status='CHECKED' if valid else 'UNKNOWN_ACTIVE_OR_ALIGNMENT',
            earlier_allocations_with_later_first_prefill=len(pending_start),
            earlier_allocations_with_later_first_prefill_ids=pending_start,
            recovery_count=r['recovery_count'], free_blocks=r['free_blocks'],
            age_s=r['age_s'], signal_wait_limit_bypass=r['signal_wait_limit_bypass'],
            native_fit=r['native_fit'], base_allowed=r['base_allowed'],
            native_allocation_result=r['native_allocation_result'],
            reconstructed_live_request_ids=sorted(live) if not matches else None))
    targets = {r['request_id'] for r in metrics if r['max_generation_gap_s'] > 1}
    selected = {r['request_id'] for r in rows if r['shadow_would_deny'] is True}
    first = next((r for r in rows if r['shadow_would_deny'] is True), None)
    first_members = None
    if first:
        now = first['external_time_s']
        first_members = sorted(old for old,t in first_allocated.items() if t < now and
            (requests[old]['completion_s'] is None or requests[old]['completion_s'] >= now))
    return dict(cell=str(path), raw_sha256=raw_sha, admission_sha256=admission_sha, config_sha256=config_sha,
        source_per_request_json=cell['per_request_json'], source_per_request_sha256=metrics_sha,
        planned_requests=cell['planned_requests'], outcomes=cell['outcomes'],
        declaration_geometry=dict(page_tokens=PAGE_TOKENS, usable_pages=USABLE_PAGES,
            native_max_model_len=config['max_model_len'], minimum_prompt_plus_max=min(declared.values()),
            maximum_prompt_plus_max=max(declared.values()), truncation_needed=False,
            initial_recorded_free_blocks=admission['decisions'][0]['free_blocks']),
        clock_conversion=dict(gate_origin_perf_s=admission['origin_perf_s'], measurement_origin_perf_s=origin,
            gate_to_external_offset_s=offset),
        allocation_alignment=dict(aligned_requests=len(aligned), unknown_requests=unknown,
            actual_first_new_allocations=sum(len(v) for v in allocations.values())),
        active_check=dict(evaluations=len(rows), matching=sum(r['active_matches'] for r in rows),
            mismatches=sum(not r['active_matches'] for r in rows),
            maximum_reconstructed_live=max((r['reconstructed_live_requests'] for r in rows), default=None),
            decisions_with_earlier_same_step_admissions=sum(r['earlier_allocations_with_later_first_prefill'] > 0 for r in rows),
            maximum_earlier_allocations_with_later_first_prefill=max((r['earlier_allocations_with_later_first_prefill'] for r in rows),default=0)),
        shadow=dict(known_decisions=sum(r['shadow_would_deny'] is not None for r in rows),
            would_deny=len(selected), would_allow=sum(r['shadow_would_deny'] is False for r in rows),
            selected_request_ids=sorted(selected), maximum_projected_declared_pages=max((r['projected_declared_pages'] for r in rows),default=None),
            longgap_targets=len(targets), covered_longgap_targets=len(selected & targets),
            covered_longgap_target_ids=sorted(selected & targets), missed_longgap_target_ids=sorted(targets-selected),
            selected_other_requests=len(selected-targets),
            first_difference=first, first_difference_live_request_ids=first_members),
        first_allocation_decisions=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite output')
    source, sha = read_hashed(args.analysis)
    cells = {Path(c['cell']).name:c for c in source['cells']}
    names = ('probe-01-kv256', 'probe-02-kv256')
    if not all(n in cells for n in names):
        raise ValueError('Expected v13 KV01 and KV02')
    result = dict(source_analysis=str(args.analysis.resolve()), source_analysis_sha256=sha,
        cells=[analyze_cell(cells[n]) for n in names],
        semantics='At each actual first successful new native allocation, reconstruct all earlier successful '
            'status=WAITING/preemptions=0 admissions, including same-step prior allocations, then remove only '
            'requests with host completion strictly before the current gate. Verify cardinality against '
            'decision.active. The budget counts each live request once using ceil((known prompt_tokens + '
            'declared max_output_tokens)/16), plus the incoming request; the sum must be<=32768. It does not '
            'add physical used/free pages, actual output length, or predicted EOS. Completion timestamps '
            'are used only when already observed by the gate time. maxgap>1s is a post-hoc coverage label '
            'from service analysis, never a budget input. All native admissions remain in the observed '
            'trajectory even if the shadow rejects them; this is not a replay of a different budget policy '
            'and does not estimate its wait, throughput, quality, or SLO benefit. Selected-other requests '
            'are not proven harmful actions. Recovery status is context, not a budget input. Ordinary '
            'static commitment coverage does not establish recovery-specific information or novelty.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for c in result['cells']:
        print(Path(c['cell']).name, 'active', c['active_check'], 'shadowdeny', c['shadow']['would_deny'],
            'longgapcoverage', c['shadow']['covered_longgap_targets'], '/', c['shadow']['longgap_targets'])
    print(args.output.resolve())


if __name__ == '__main__':
    main()

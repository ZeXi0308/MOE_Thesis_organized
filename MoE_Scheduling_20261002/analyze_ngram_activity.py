"""Actual AR0/N4 configuration, draft bookkeeping and host chunks; no acceptance estimator."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).parent
NAMES = ('00_ar16', '01_ngram4', '02_ngram4', '03_ar16')
load = lambda p: json.loads(p.read_text())
null_itl = lambda d: d['n'] == 0 and all(d[k] is None for k in ('mean_s', 'p50_s', 'p95_s', 'max_s'))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT / 'results_ngram_r01')
    p.add_argument('--design', choices=('ngram', 'ngram_short'), default='ngram')
    p.add_argument('--output', type=Path)
    args = p.parse_args(); metrics = load(args.results / 'metrics.json'); cells = {}
    names = NAMES if args.design == 'ngram' else ('00_ar16', '01_ngram1', '02_ngram2', '03_ngram2', '04_ngram1', '05_ar16')
    if load(args.results / 'group_status.json')['status'] != 'COMPLETE':
        p.error('No report written: final group is not COMPLETE')
    for name in names:
        directory = args.results / name
        raw, config, spec, policy = [load(directory / f) for f in ('raw.json', 'config.json', 'speculation_config.json', 'policy_application.json')]
        metric = metrics['cells'][name]; n = int(name[-1]) if 'ngram' in name else 0
        if raw['status'] != 'COMPLETE' or load(directory / 'status.json')['status'] != 'COMPLETE' or not metric['all_16_complete']:
            p.error(f'No report written: {name} is not COMPLETE')
        requests = {r['request_id']: r for r in raw['requests']}; analyzed = {r['request_id']: r for r in metric['per_request']}
        algorithm = f'native_ngram_2_5_{n}' if n else 'native_without_speculation'
        expected = dict(method='ngram', num_speculative_tokens=n, prompt_lookup_min=2, prompt_lookup_max=5) if n else None
        checks = dict(configured_count=config['ngram_speculative_tokens'] == policy['speculative_tokens'] == raw['speculative_capture']['configured_tokens'] == spec['resolved_num_spec_tokens'] == n,
            requested_ngram_tuple=spec['requested'] == expected,
            observers_disabled_unqualified=all(obj[k] is False for obj in (spec, policy) for k in ('row_observer_installed', 'request_row_identity_qualified', 'readiness_observer_enabled')),
            arm_specific_warmup=spec['warmup_algorithm'] == policy['warmup_algorithm'] == algorithm and policy['warmup_algorithm_is_arm_specific'] is True,
            warmup_contract=policy['warmup_token_budget'] == 512 and policy['full_warmup_fixed_output_tokens'] == 32,
            request_inventory=set(requests) == set(analyzed), draft_bookkeeping=True, receipt_bookkeeping=True, multi_chunk_itl_null=True, max_host_gap_matches=True)
        proposals, errors, length_counts = defaultdict(Counter), [], Counter()
        for step in raw['scheduler_steps']:
            ids, counts = step['scheduled_draft_token_ids'], step['scheduled_draft_tokens']
            scheduled = {r['request_id']: r for r in step['scheduled']}
            valid = set(ids) == set(counts)
            for rid, tokens in ids.items():
                valid &= (rid in requests and rid in scheduled and counts.get(rid) == len(tokens) and len(tokens) <= n
                    and all(type(t) is int and t >= 0 for t in tokens))
                if rid in scheduled:
                    valid &= raw['internal_to_source'].get(scheduled[rid]['internal_request_id']) == rid and len(tokens) <= scheduled[rid]['scheduled_tokens']
                if tokens:
                    proposals[rid].update(request_steps=1, draft_token_rows=len(tokens)); length_counts[len(tokens)] += 1
            if not valid:
                errors.append(step['step']); checks['draft_bookkeeping'] = False
        events = defaultdict(list)
        for e in raw['output_events']:
            if e['chunk_size'] > 0:
                events[e['request_id']].append(e)
                checks['receipt_bookkeeping'] &= e['request_id'] in requests and len(e['new_token_ids']) == e['chunk_size'] and e['prefix_valid'] is True
        sizes = Counter(e['chunk_size'] for es in events.values() for e in es); multi_ids, gaps = [], {}
        for rid, request in requests.items():
            es = sorted(events[rid], key=lambda e: e['received_s']); times = [e['received_s'] for e in es]
            gaps[rid] = max((b-a for a, b in zip(times, times[1:])), default=None)
            checks['receipt_bookkeeping'] &= sum(e['chunk_size'] for e in es) == len(request['output_token_ids'])
            if any(e['chunk_size'] > 1 for e in es):
                multi_ids.append(rid); checks['multi_chunk_itl_null'] &= null_itl(analyzed[rid]['itl']) and analyzed[rid]['token_level_itl_resolved'] is False and analyzed[rid]['decode_tokens_per_s'] is None
            checks['max_host_gap_matches'] &= analyzed[rid]['maxgap_s'] == gaps[rid]
        if multi_ids:
            checks['multi_chunk_itl_null'] &= null_itl(metric['itl']) and metric['token_level_itl_resolved'] is False
        maximum = max((v for v in gaps.values() if v is not None), default=None)
        checks['max_host_gap_matches'] &= metric['request_maxgap']['max_s'] == maximum
        cells[name] = dict(checks=checks, invalid_draft_steps=errors, configured_speculative_tokens=n, warmup_algorithm=algorithm,
            proposal_request_steps=sum(v['request_steps'] for v in proposals.values()), scheduled_draft_token_rows=sum(v['draft_token_rows'] for v in proposals.values()),
            proposal_counts_by_request=dict(proposals), draft_length_distribution=dict(length_counts), positive_receipt_chunk_distribution=dict(sizes),
            output_tokens_beyond_one_per_positive_receipt=sum((size-1)*count for size, count in sizes.items()), multi_chunk_request_ids=multi_ids,
            max_host_chunk_gap_s=maximum, per_request_max_host_chunk_gap_s=gaps, exact_acceptance_rate=None)
    report = dict(all_checks_pass=all(all(c['checks'].values()) for c in cells.values()), cells=cells,
        notes=[f'All {len(names)} final cells required; true output/full-cost/quality remain in metrics.json.',
            'Draft IDs/counts are checked against the same scheduler step and raw source-request mapping; this does not qualify model-row identity or draft acceptance.',
            'sum(max(chunk_size-1,0)) is output beyond one token per positive receipt, not exact accepted drafts; EOS and unknown first-step boundaries prevent that identification.',
            'Host chunk gaps exclude TTFT and unresolved intra-chunk latency. ITL-null checks apply only to multi-chunk cells and affected requests.',
            'Resolved speculative-token count is recorded; ngram min/max are requested-config markers checked by the runner, not separately exported resolved fields.'])
    output = args.output or args.results / 'ngram_activity.json'; output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(output); print('all_checks_pass=', report['all_checks_pass'])

if __name__ == '__main__':
    main()

"""Join one-break cancellations to existing same-call async LOAD observations."""
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def read(path, hashes):
    content = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def summarize(session):
    hashes, cases = {}, []
    for output in sorted(session.glob('cell-*/output')):
        policy = read(output/'recovery-retry-defer.json', hashes)
        if policy['mode'] != 'defer_once':
            continue
        raw = read(output/'raw.json', hashes)
        capacity = read(output/'capacity-handoff.json', hashes)['events']
        order = read(output/'recovery-order.json', hashes)['events']
        selective = read(output/'selective-store.json', hashes)
        origin = raw['measurement_origin_perf_counter_s']
        mapping = raw['internal_to_source']
        returns = {}
        for event in raw['output_events']:
            returns.setdefault(event['engine_call_index'], set()).add(origin+event['received_s'])
        stamps = sorted((next(iter(values)), index) for index, values in returns.items() if len(values) == 1)
        for opportunity in policy['events']:
            if opportunity['kind'] != 'legal_retry_opportunity':
                continue
            start = opportunity['host_perf_s']
            release = opportunity.get('release_observed_perf_s')
            if release is None:
                continue
            before = max((s for s in stamps if s[0] < start), default=None)
            after = min((s for s in stamps if s[0] > start), default=None)
            bracket = bool(before and after and after[1] == before[1]+1)
            row = dict(cell=output.parent.name, target=opportunity['target'],
                target_source=mapping.get(opportunity['target']), observer_step=opportunity['step'],
                opportunity_host_perf_s=start, release_host_perf_s=release,
                release_reason=opportunity.get('release_reason'),
                actual_breaks=opportunity['executed_breaks'],
                elapsed_to_cancel_ms=(release-start)*1000,
                clock='All absolute stamps use host time.perf_counter; raw received_s adds measurement origin.',
                raw_return_before=before, raw_return_after=after,
                adjacent_raw_calls_bracket_opportunity=bracket,
                raw_engine_call_index=after[1] if bracket else None,
                direct_queue_snapshot_at_release=False, same_call_prior_async_loads=[])
            cases.append(row)
            if not bracket:
                row['status'] = 'UNVERIFIED_CALL_BOUNDARY'
                continue
            # A successful async allocation before the gate, followed by this
            # call's newly built LOAD, establishes the original async branch.
            for alloc in capacity:
                if not (alloc['kind'] == 'allocate' and alloc.get('success') is True
                        and alloc['arguments']['delay_cache_blocks'] is True
                        and before[0] < alloc['begin_host_perf_s'] <= alloc['end_host_perf_s'] < start):
                    continue
                rid = alloc['request']
                jobs = [e for e in order if e['kind'] == 'job_created' and e.get('is_store') is False
                        and e['request'] == rid and start < e['host_perf_s'] < after[0]]
                for job in jobs:
                    events = [e for e in order if e.get('job_id') == job['job_id']]
                    ack = [e['host_perf_s'] for e in events if e['kind'] == 'ack_retired']
                    evidence = dict(request=rid, source_request=mapping.get(rid), job_id=job['job_id'],
                        allocation=alloc, job_events=events,
                        allocation_begin_relative_ms=(alloc['begin_host_perf_s']-start)*1000,
                        allocation_end_relative_ms=(alloc['end_host_perf_s']-start)*1000,
                        job_created_relative_ms=(job['host_perf_s']-start)*1000,
                        ack_relative_ms=(ack[0]-start)*1000 if len(ack) == 1 else None,
                        ack_not_retired_at_cancel=len(ack) == 1 and ack[0] > release,
                        absent_from_gate_waiting_and_running=(rid not in opportunity['waiting_order']
                            and rid not in opportunity['running_cohort']),
                        native_async_branch_destination='local step_skipped_waiting; inferred from pinned branch, not directly logged')
                    row['same_call_prior_async_loads'].append(evidence)
            target_alloc = min((e for e in capacity if e['kind'] == 'allocate'
                and e['request'] == opportunity['target'] and e['begin_host_perf_s'] > release),
                key=lambda e:e['begin_host_perf_s'], default=None)
            row['first_target_allocation_after_cancel'] = target_alloc
            row['cancel_to_target_allocation_us'] = ((target_alloc['begin_host_perf_s']-release)*1e6
                                                    if target_alloc else None)
            row['victim_decisions_between_gate_and_cancel'] = [e for e in selective['victim_decisions']
                if start <= e['host_perf_counter_s'] <= release]
            row['source_backed_explanation'] = (
                'A same-call earlier async LOAD enters local step_skipped_waiting. The original loop exit '
                'prepends that local queue to self.skipped_waiting; next FCFS peek selects skipped first. '
                'The extra gate cancels before native blocked-status promotion. Native then skips the '
                'not-yet-ACKed peer and reaches the unchanged target. This explains the release predicate; '
                'the actual queue object and peeked request at release were not logged.')
            evidence = row['same_call_prior_async_loads']
            row['status'] = ('SOURCE_SUPPORTED_PENDING_SKIPPED_PEER' if len(evidence) == 1
                and evidence[0]['ack_not_retired_at_cancel']
                and evidence[0]['absent_from_gate_waiting_and_running']
                and row['release_reason'] == 'NATIVE_HEAD_OR_QUEUE_CHANGED'
                and not row['victim_decisions_between_gate_and_cancel']
                else 'UNVERIFIED_EXACT_RELEASE_CAUSE')
    scheduler = BASE.parent/'MOE_Thesis_organized/refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/liveness_pinned_sources_20260930/scheduler.py'
    source_files = [scheduler, BASE/'pkg/rotation_native.py', BASE/'recovery_retry_defer/retry_defer.py']
    return dict(cases=cases, input_sha256=hashes,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Within-run existing records only. No direct release queue snapshot, GPU completion-time inference, or counterfactual benefit.',
        source_lines=dict(local_queue_create=667, async_pop_and_local_prepend=[978,983],
                          exit_local_queue_merge=[1055,1057], fcfs_skipped_first=[1976,1978],
                          policy_cancel_predicate=[142,143]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.session/'cancel-causes.json'
    if output.exists():
        raise FileExistsError(output)
    result = summarize(args.session.resolve())
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), cases=[dict(cell=r['cell'], status=r['status'],
        prior_loads=[e['source_request'] for e in r['same_call_prior_async_loads']],
        elapsed_to_cancel_ms=r['elapsed_to_cancel_ms']) for r in result['cases']])))


if __name__ == '__main__':
    main()

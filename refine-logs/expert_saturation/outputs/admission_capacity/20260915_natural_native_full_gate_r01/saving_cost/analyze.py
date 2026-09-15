"""Native-full block reuse, using the owner's canonical lifecycle/scope analysis.

This is a diagnostic identity join, not another run or a performance comparison.
The prefix join is qualified for this single-group, no-APC, one-load-per-segment
capture; unexpected coverage, duplication or accounting fails explicitly.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(a):
    require(a['status'] == 'COMPLETE' and not a['errors'], 'Incomplete source')
    require(a['native_full_qualification'] == 'NATIVE_FULL_INCREMENTAL_SCOPE_QUALIFIED',
            'Native full scope not qualified')
    cfg = a['config']
    block_bytes = cfg['kv_block_bytes']
    jobs = a['native_full_scope']['observed_store_jobs']
    segments = a['diagnostic']['segments']
    # Both model positions and physical bytes are independently reconciled below.
    block_tokens = 16
    blocks, store_keys = {}, set()
    request_jobs = defaultdict(list)
    for job in jobs:
        require(job['accepted_dispatch'] and job['completed_perf_s'] is not None,
                'Uncompleted store')
        indices = job['logical_chunk_indices']
        keys = job['registered_key_reprs']
        require(len(indices) == len(keys) == len(job['source_gpu_blocks']),
                'Store range mismatch')
        request_jobs[job['request']].append(job)
        for index, key in zip(indices, keys):
            identity = (job['request'], index)
            require(identity not in blocks and key not in store_keys,
                    'This exact join requires once-stored unique logical keys')
            store_keys.add(key)
            blocks[identity] = job

    load_count = Counter()
    rows = []
    dispatch_ids = set()
    for s in segments:
        require(len(s['actual_load_dispatches']) == 1 and s['executed_steps'],
                'Expected exactly one load and later execution per recovery')
        load = s['actual_load_dispatches'][0]
        first = s['executed_steps'][0]
        require(load['accepted'] and not load['is_store'], 'Not an accepted load')
        require(load['job_id'] not in dispatch_ids, 'Duplicate load assignment')
        dispatch_ids.add(load['job_id'])
        prefix = first['scheduled_start_computed']
        require(prefix > 0 and prefix % block_tokens == 0, 'Non-block load prefix')
        require(first['computed_adjustment'] == 0 and first['computed_before'] == prefix,
                'Unexpected computed adjustment')
        previous = s['preempt_state_before']['computed_tokens']
        require(0 <= previous - prefix < block_tokens,
                'Recovery loses more than an incomplete tail block')
        require(s['confirmed_recompute_tokens'] == previous - prefix,
                'Recompute does not equal uncovered preemption tail')
        identities = [(s['internal_request_id'], i) for i in range(prefix // block_tokens)]
        for identity in identities:
            require(identity in blocks, 'Loaded block lacks registered store')
            require(blocks[identity]['completed_perf_s'] <= load['before_perf_s'],
                    'Store not completed before load dispatch')
        load_count.update(identities)
        rows.append(dict(request_id=s['request_id'], preempt_step=s['preempt_step'],
            load_job_id=load['job_id'], loaded_prefix_tokens=prefix,
            loaded_blocks=len(identities), load_bytes=len(identities)*block_bytes,
            recompute_tokens=s['confirmed_recompute_tokens'],
            subsequent_outputs=s['useful_outputs'], end=s['end'],
            L_s=s['L_s'], E_direct=s['E_direct'], E_fixed_most_funded=s['E_fixed_most_funded'],
            S_host_engine_call=s['S_host_engine_call'], S_load_submit=s['S'], F_s=s['F_s']))

    require(len(blocks)*block_bytes == a['completed_transfer_bytes']['store'],
            'Store bytes do not reconcile')
    require(sum(load_count.values())*block_bytes == a['completed_transfer_bytes']['load'],
            'Load bytes do not reconcile')
    require(len(dispatch_ids) == a['completed_jobs']['load'], 'Load jobs do not reconcile')
    completed_loads = {j['job_id'] for e in a['diagnostic']['offload']['completed_jobs']
                       for j in e['jobs'] if not j['is_store']}
    require(dispatch_ids == completed_loads, 'Load completion set mismatch')

    recovered_requests = {rid for rid, _ in load_count}
    by_request = []
    for rid in sorted(request_jobs):
        stored = [identity for identity in blocks if identity[0] == rid]
        used = [identity for identity in stored if identity in load_count]
        by_request.append(dict(internal_request_id=rid, store_jobs=len(request_jobs[rid]),
            stored_blocks=len(stored), loaded_unique_blocks=len(used),
            load_block_visits=sum(load_count[k] for k in used),
            blocks_without_observed_load=len(stored)-len(used)))

    transitions = []
    by_preempt = {(s['internal_request_id'], s['preempt_step']): s for s in segments}
    for s in segments:
        if s['end'] != 'repreempted':
            continue
        nxt = by_preempt[(s['internal_request_id'], s['next_preempt_step'])]
        prefix = nxt['executed_steps'][0]['scheduled_start_computed']
        old = s['discarded_computed_tokens']
        transitions.append(dict(request_id=s['request_id'],
            first_recovery_preempt_step=s['preempt_step'],
            next_preempt_step=s['next_preempt_step'], outputs_before_repreempt=s['useful_outputs'],
            gpu_computed_positions_released=old, later_loaded_positions_from_that_state=min(old, prefix),
            next_load_bytes=prefix//block_tokens*block_bytes,
            old_positions_recomputed=old-min(old, prefix)))

    unused = set(blocks)-set(load_count)
    totals = dict(stored_unique_blocks=len(blocks), loaded_unique_blocks=len(load_count),
        load_block_visits=sum(load_count.values()),
        repeat_load_block_visits=sum(load_count.values())-len(load_count),
        blocks_without_observed_load=len(unused),
        no_load_request_count=len(request_jobs)-len(recovered_requests),
        no_load_request_stored_blocks=sum(rid not in recovered_requests for rid, _ in blocks),
        recovered_request_unloaded_blocks=sum(rid in recovered_requests for rid, _ in unused),
        one_block_store_jobs=sum(len(j['logical_chunk_indices']) == 1 for j in jobs),
        finished_request_store_blocks=sum(len(j['logical_chunk_indices']) for j in jobs
                                          if j['finished_at_metadata']),
        recompute_tokens=sum(s['confirmed_recompute_tokens'] for s in segments),
        repreempt_transitions=len(transitions),
        gpu_positions_released_again=sum(r['gpu_computed_positions_released'] for r in transitions),
        those_positions_later_loaded=sum(r['later_loaded_positions_from_that_state'] for r in transitions),
        those_positions_recomputed=sum(r['old_positions_recomputed'] for r in transitions))
    return dict(status='MEASUREMENT_ONLY', block_tokens=block_tokens, block_bytes=block_bytes,
        totals=totals, load_multiplicity=dict(sorted(Counter(load_count.values()).items())),
        bytes=dict(stored=len(blocks)*block_bytes, loaded=sum(load_count.values())*block_bytes,
            stored_without_observed_load=len(unused)*block_bytes,
            repeated_load=(sum(load_count.values())-len(load_count))*block_bytes,
            loads_after_repreemption=sum(r['next_load_bytes'] for r in transitions)),
        conservation=dict(store_bytes=True, load_bytes=True, completed_load_identity=True,
            store_completion_precedes_load=True, every_recompute_is_partial_block_tail=True),
        semantics=[
            'Loaded identities are reconstructed from actual restored complete prefixes plus unique registered request/chunk/key stores; per-load source-key arrays were not separately captured.',
            'Aggregate reconstructed physical bytes exactly reconcile both actual transfer directions; jobs reconcile accepted/completed identities.',
            'No observed load is episode-bounded, hindsight-only; it is not an online skip label, zero future cache value, or proof of removable service cost.',
            'GPU release is not loss of all historical computation: later actual loaded prefix and recomputed suffix are distinct.',
            'Repeated transfer bytes are not repeated saving, pure recompute time, or exposed wall time.',
            'Capacity occupancy, cumulative transfer bytes, process RSS and shared cgroup charges must not be added.',
            'E fields are sampled necessary funding conditions, not an executable or sustainable recovery guarantee.',
            'Output times are engine returns, not client receipt; source is one diagnostic, not an independent repeat or performance estimate.'],
        requests=by_request, recoveries=rows, repreempt_transitions=transitions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.analysis.read_bytes()
    result = analyze(json.loads(source))
    result['source'] = dict(path=str(args.analysis), sha256=hashlib.sha256(source).hexdigest())
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps(result['totals'], indent=2))

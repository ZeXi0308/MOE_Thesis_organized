"""CPU-only four-arm H/R/R/H diagnostic using two existing complete summaries.

Usage: python analyze_multinews_pairs.py PROBE_GROUP REVERSE_GROUP
       [--quality QUALITY_JSON ...] [--out OUTPUT_JSON]
Missing/incomplete groups return UNRUN (exit 2) before any raw file is read.
No fitted timing correction, request-level significance test, or GPU execution.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path

import analyze as base
import analyze_first_fork as fork
from analyze_one_event import difference


def compact_difference(a, b):
    result = difference(a, b)
    first = result['first_difference']
    if first is not None:
        result['first_difference'] = dict(index=first['index'],
            left_sha256=base.fingerprint(first['left']),
            right_sha256=base.fingerprint(first['right']))
    return result


def distribution(values):
    values = list(values)
    return dict(**base.stats(values), negative=sum(x < 0 for x in values),
                zero=sum(x == 0 for x in values), positive=sum(x > 0 for x in values))


def first_joint(cell):
    raw = cell['raw']
    found = next(((i, e) for i, e in enumerate(raw['commits']) if e.get('eligible')), None)
    if found is None:
        return None, None
    index, event = found
    source = cell['lookup'][base.event_key(event)]
    row = cell['rows'][event['request_id']]
    when = event['decision_s']; times = row['token_times_s']
    n = bisect_right(times, when)
    end_step = bisect_left([s['end_s'] for s in raw['steps']], times[n]) if n < len(times) else None
    native = []
    if end_step is not None:
        for step in range(source['local_step'], end_step + 1):
            matches = [e for e in cell['trace'][step]['scheduled'] if e[0] == row['external_id']]
            native.extend(dict(step=step, count=e[1], start_computed=e[2], end_computed=e[3]) for e in matches)
    completed = [r for r in cell['rows'].values() if r['completion_s'] <= when]
    emitted = [r for r in cell['rows'].values() if r['token_times_s'] and r['token_times_s'][0] <= when]
    context = fork.context_before(cell, source)
    return dict(identity={k:cell['commits'][index][k] for k in fork.IDENTITY},
        visible_state={k:event.get(k) for k in fork.STATE},
        action=event['actual_action'], decision_s=when, allocation_s=event['allocation_s'],
        local_step=source['local_step'], next_output_step=end_step,
        next_output_index_one_based=n + 1 if end_step is not None else None,
        next_output_token_id=row['output_token_ids'][n] if end_step is not None else None,
        decision_to_next_output_s=times[n]-when if end_step is not None else None,
        allocation_to_next_output_s=times[n]-event['allocation_s'] if end_step is not None else None,
        target_native_work_through_next_output=native,
        completed_before_action=len(completed), first_output_before_action=len(emitted),
        completed_before_action_completion_latency_s=base.stats(r['completion_s']-r['arrival_s'] for r in completed),
        first_output_before_action_ttft_s=base.stats(r['token_times_s'][0]-r['arrival_s'] for r in emitted),
        target_final_output_tokens=len(times), target_completion_s=row['completion_s'],
        prior_step_wall_s=sum(s['end_s']-s['start_s'] for s in raw['steps'][:source['local_step']]),
        prior_step_driver_cpu_s=sum(s['driver_thread_cpu_s'] for s in raw['steps'][:source['local_step']])), context


def compact_tokens(a, b):
    result = base.compare_tokens(a['service'], b['service'], a['external_rows'], b['external_rows'])
    result['completion_metadata_difference_count'] = len(result.pop('completion_metadata_differences'))
    result['first_five_output_differences'] = result.pop('differences')[:5]
    return result


def matched_step_controls(a, b):
    """Descriptive controls for a matched schedule; no outlier threshold or test."""
    if a['policy'] != b['policy'] or a['trace'] != b['trace']:
        return None
    rows = []
    for i, (sa, sb, schedule) in enumerate(zip(a['raw']['steps'], b['raw']['steps'], a['trace'])):
        posts = [c['raw']['steps'][i]['end_s']-c['raw']['scheduler_steps'][i]['time_s'] for c in (a,b)]
        events = []
        for c, step in ((a, sa), (b, sb)):
            events.append({name:sum(step['start_s'] <= e[clock] <= step['end_s']
                                   for e in c['raw'][name])
                           for name,clock in [('decisions','decision_s'),('commits','allocation_s'),
                                              ('preemptions','time_s')]})
        counts = [r[1] for r in schedule['scheduled']]
        rows.append(dict(step=i, requests=len(counts), positions=sum(counts),
            all_counts_one=bool(counts) and all(n==1 for n in counts), events=events,
            wall_difference_s=(sb['end_s']-sb['start_s'])-(sa['end_s']-sa['start_s']),
            post_scheduler_difference_s=posts[1]-posts[0]))
    focus = a['first_joint']['local_step'] if a['first_joint'] else None
    result = dict(direction='right_minus_left', formal_steps=len(rows),
                  all_counts_one_steps=sum(r['all_counts_one'] for r in rows), focus_step=focus)
    for field in ('wall_difference_s','post_scheduler_difference_s'):
        ordered = sorted(rows, key=lambda r:r[field], reverse=True)
        decode = [r for r in ordered if r['all_counts_one']]
        result[field] = dict(top_five=ordered[:5], all_counts_one_top_five=decode[:5],
            focus_rank=next((i+1 for i,r in enumerate(ordered) if r['step']==focus), None),
            focus_all_counts_one_rank=next((i+1 for i,r in enumerate(decode) if r['step']==focus), None))
    result['interpretation'] = 'Matched-step descriptions, not independent repetitions or a recovery-specific causal effect. Different work shapes remain explicit.'
    return result


def pair(a, b):
    """All differences are right minus left, including descriptive request pairs."""
    left, right = a['external_rows'], b['external_rows']
    common = sorted(left.keys() & right.keys())
    firsts = [c['first_joint'] for c in (a, b)]
    before = None
    if all(firsts):
        when = [x['decision_s'] for x in firsts]
        done = [k for k in common if left[k]['completion_s'] <= when[0] and right[k]['completion_s'] <= when[1]]
        emitted = [k for k in common if left[k]['token_times_s'] and right[k]['token_times_s'] and
                   left[k]['token_times_s'][0] <= when[0] and right[k]['token_times_s'][0] <= when[1]]
        before = dict(first_joint_identity_equal=firsts[0]['identity'] == firsts[1]['identity'],
            decision_time_difference_s=when[1]-when[0], common_completed_requests=len(done),
            completed_request_set_equal={k for k in left if left[k]['completion_s'] <= when[0]} ==
                                       {k for k in right if right[k]['completion_s'] <= when[1]},
            completion_latency_difference_s=distribution((right[k]['completion_s']-right[k]['arrival_s'])-
                (left[k]['completion_s']-left[k]['arrival_s']) for k in done),
            common_first_output_requests=len(emitted),
            ttft_difference_s=distribution((right[k]['token_times_s'][0]-right[k]['arrival_s'])-
                (left[k]['token_times_s'][0]-left[k]['arrival_s']) for k in emitted),
            exact_prehistory={k:compact_difference(a['first_context'][k],b['first_context'][k])
                              for k in a['first_context']})
    # Keep the base comparison unchanged; additionally separate declared run order/path metadata.
    excluded = ('out','policy','workload','policies','telemetry_script')
    configs = [{k:v for k,v in c['service']['config'].items() if k not in excluded} for c in (a,b)]
    # Only the observer snapshot directory differs legitimately; retain all other source paths.
    source_maps = []
    for c in (a,b):
        hashes = base.read_json(Path(c['folder']).parent/'runtime_source_hashes.json', {})
        source_maps.append({('timing_observer.py' if Path(k).name=='timing_observer.py' else k):v
                            for k,v in hashes.items()})
    timing = dict(makespan_s=b['service']['makespan_s']-a['service']['makespan_s'],
        mean_completion_s=b['service']['completion_latency_s']['mean']-a['service']['completion_latency_s']['mean'],
        mean_ttft_s=b['service']['ttft_s']['mean']-a['service']['ttft_s']['mean'])
    return dict(left=a['label'],right=b['label'],direction='right_minus_left',
        token_comparison=compact_tokens(a,b),
        native_schedule=compact_difference(a['trace'],b['trace']),
        emitted_tokens_by_step=compact_difference(a['emissions'],b['emissions']),
        matched_step_controls=matched_step_controls(a,b),
        first_joint_prehistory=before, full_service_difference_s=timing,
        all_request_completion_difference_s=distribution((right[k]['completion_s']-right[k]['arrival_s'])-
            (left[k]['completion_s']-left[k]['arrival_s']) for k in common),
        declared_config_comparison=dict(equal_excluding_run_metadata=configs[0]==configs[1], excluded=list(excluded),
                                        runtime_source_hashes_equal=bool(source_maps[0]) and source_maps[0]==source_maps[1]),
        interpretation='Request pairs are descriptive, not independent experimental repetitions. Pre-action timing differences are retained; no correction is applied.')


def longest_gap_recovery_coverage(cell):
    """Locate recovery events inside each preempted request's longest output gap."""
    raw=cell['raw']; ends=[s['end_s'] for s in raw['steps']]
    preempted={e['request_id'] for e in raw['preemptions']}
    rows=[]; without_gap=[]
    for request_id in sorted(preempted):
        request=cell['rows'][request_id]; times=request['token_times_s']
        if len(times)<2:
            without_gap.append(request['external_id']);continue
        # max returns the earliest interval when two gaps have exactly equal length.
        right=max(range(1,len(times)),key=lambda i:times[i]-times[i-1])
        start,end=times[right-1],times[right]
        preempts=[e for e in raw['preemptions'] if e['request_id']==request_id and start<e['time_s']<=end]
        commits=[e for e in raw['commits'] if e['request_id']==request_id and start<e['allocation_s']<=end]
        eligible=[e for e in commits if e.get('eligible')]
        fallbacks=[e for e in commits if not e.get('eligible')]
        misses=[e for e in commits if e.get('fallback')=='host_miss_native_recompute']
        capacity_lookups=sum(e['request_id']==request_id and start<e['decision_s']<=end and
                            e.get('fallback')=='full_capacity_not_jointly_available_native'
                            for e in raw['decisions'])

        def tail(events):
            if not events:
                return None
            event=max(events,key=lambda e:e['allocation_s'])
            index=bisect_right(times,event['decision_s'])
            result=dict(event=event['event'],fallback=event.get('fallback'),
                decision_s=event['decision_s'],allocation_s=event['allocation_s'],
                decision_step=bisect_left(ends,event['decision_s']),
                allocation_step=bisect_left(ends,event['allocation_s']))
            # Find the actual first output after the decision, not an arbitrary gap endpoint.
            available=index<len(times) and start<times[index]<=end and times[index]>=event['allocation_s']
            result.update(next_output_within_gap=available,
                next_output_index_zero_based=index if available else None,
                next_output_s=times[index] if available else None,
                next_output_step=bisect_left(ends,times[index]) if available else None,
                decision_to_next_output_s=times[index]-event['decision_s'] if available else None)
            return result

        rows.append(dict(external_id=request['external_id'],gap_s=end-start,
            gap_start_s=start,gap_end_s=end,
            gap_start_step=bisect_left(ends,start),gap_end_step=bisect_left(ends,end),
            left_output_index_zero_based=right-1,right_output_index_zero_based=right,
            preemptions_in_gap=len(preempts),preemption_steps=[bisect_left(ends,e['time_s']) for e in preempts],
            commits_in_gap=len(commits),eligible_commits_in_gap=len(eligible),
            host_miss_commits_in_gap=len(misses),
            fallback_commit_counts=dict(Counter(e.get('fallback') for e in fallbacks)),
            commit_steps=[bisect_left(ends,e['allocation_s']) for e in commits],
            capacity_unavailable_lookup_count=capacity_lookups,
            last_eligible_tail=tail(eligible),last_fallback_tail=tail(fallbacks)))
    rows.sort(key=lambda r:(-r['gap_s'],r['external_id']))
    return dict(
        preempted_requests=len(preempted),requests_with_generation_gap=len(rows),
        requests_without_generation_gap=without_gap,
        longest_gaps_with_preemption=sum(r['preemptions_in_gap']>0 for r in rows),
        longest_gaps_with_commit=sum(r['commits_in_gap']>0 for r in rows),
        longest_gaps_with_joint_choice=sum(r['eligible_commits_in_gap']>0 for r in rows),
        longest_gaps_with_host_miss=sum(r['host_miss_commits_in_gap']>0 for r in rows),
        longest_gaps_with_joint_choice_and_host_miss=sum(r['eligible_commits_in_gap']>0 and
                                                       r['host_miss_commits_in_gap']>0 for r in rows),
        longest_gaps_without_commit=sum(r['commits_in_gap']==0 for r in rows),
        longest_gaps_with_capacity_unavailable_lookup=sum(r['capacity_unavailable_lookup_count']>0 for r in rows),
        requests=rows,
        definitions='Longest adjacent-token interval per preempted request, excluding TTFT; earliest tie. '
                    'Intervals are left-open/right-closed; preemptions use time_s, successful commits allocation_s, '
                    'and capacity lookups decision_s. Steps and output indices are zero-based. '
                    'Each tail selects the last successful commit of its category inside the gap, then the first '
                    'same-request output strictly after decision_s, requiring that output to remain inside the gap.',
        limitations=['Joint-choice and fallback categories can overlap; counts are descriptive event coverage.',
                     'No overlapping gaps or tails are summed into service cost or a savings upper bound.',
                     'Earlier recovery choices may affect later waiting; absence of a choice inside a gap does not establish causal irrelevance.',
                     'Observed tail durations do not establish avoidable delay, an online policy benefit, or legality of earlier intervention.'])


def arm_report(cell):
    s=cell['service']; raw=cell['raw']
    fields=('requests','completed_requests','all_requests_completed','output_tokens','makespan_s',
            'all_complete_s','service_and_drain_s','output_tokens_per_s','completion_latency_s','ttft_s',
            'admission_lag_s','token_gap_s','output_work','capacity_pressure','scheduled_native_tokens',
            'decision_count','decision_fallback_count','decision_fallbacks','committed_count','committed_actions',
            'committed_fallback_count','eligible_commits','preemptions','lookup_overhead_s','adapter_overhead_s',
            'warnings','measurement_notes')
    result={k:s[k] for k in fields}
    result.update(label=cell['label'], policy=cell['policy'], cell=s['cell'],raw_sha256=cell['raw_sha256'],
        unfinished_requests=s['requests']-s['completed_requests'],
        finish_reason_counts=dict(Counter(r.get('finish_reason') for r in raw['requests'])),
        per_request_max_generation_gap_s=base.stats(r['token_gap_s']['max'] for r in s['request_metrics']),
        native_positions=s['scheduled_position_counts']['totals'],
        transfers={kind:{k:v[k] for k in ('bytes','reported_transfer_time_s','observed_transfer_count')}
                   for kind,v in s['transfers'].items()}, first_joint_commit=cell['first_joint'],
        longest_gap_recovery_coverage=longest_gap_recovery_coverage(cell))
    return result


def quality_reports(paths, groups, cells):
    reports=[]
    for path in paths:
        if not path.is_file():
            reports.append(dict(path=str(path),status='UNRUN'));continue
        data=path.read_bytes(); q=json.loads(data); group=Path(q['group']).resolve()
        fork.require(group in groups, 'quality report belongs to neither supplied group')
        selected=[c for c in cells if c['group']==group]; arms={}
        for c in selected:
            name=c['service']['cell']
            if q.get('schema')=='E.multinews_quality.v1':
                a=q['arms'][name]
                fork.require(a['provenance']['raw_sha256']==c['raw_sha256'], 'stale long quality raw binding')
                arms[c['label']]={k:v for k,v in a['summary'].items() if k not in ('per_bundle','official_compatible_diagnostic')}
            elif q.get('schema')=='E.short_quality_reproduction.v1':
                fork.require(q['provenance']['raw_sha256'][name]==c['raw_sha256'], 'stale short quality raw binding')
                arms[c['label']]=q['summary'][name]
            else:
                raise ValueError('unsupported quality report schema')
        reports.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(data).hexdigest(),
            schema=q['schema'],status=q.get('status','SCORED'),arms=arms))
    return reports


def analyze(probe, reverse, quality=()):
    groups=[Path(probe).resolve(),Path(reverse).resolve()]
    fork.require(groups[0]!=groups[1], 'probe and reverse must be different groups')
    # This preflight must finish for BOTH groups before loading any raw, including completed probe raw.
    unavailable=[]
    for group in groups:
        status=base.read_json(group/'status.json',{})
        if status.get('status')!='COMPLETE' or not (group/'summary.json').is_file():
            unavailable.append(dict(group=str(group),status=status.get('status','MISSING'),
                                    summary_available=(group/'summary.json').is_file()))
    if unavailable:
        return dict(schema='E.multinews_pairs.v1',status='UNRUN',unavailable=unavailable,
                    interpretation='No raw read. Both groups must be complete and already base-analyzed.')
    cells=[]; summaries=[]
    for group,order in zip(groups,[['host','recompute'],['recompute','host']]):
        summary=base.read_json(group/'summary.json')
        fork.require(Path(summary['group']).resolve()==group and summary['group_status']['status']=='COMPLETE' and
            not any(summary.get(k) for k in ('invalid_cells','incomplete_cells','analysis_errors')), 'unclean base summary')
        members=sorted(summary['cells'],key=lambda s:s['cell'])
        fork.require([s['policy'] for s in members]==order,'expected exactly H/R probe and R/H reverse')
        summaries.append(summary)
        for service in members:
            c=fork.load_cell(Path(service['raw_path']).parent,summary)
            c.update(group=group,label=('H' if c['policy']=='host' else 'R')+str(len(cells)),
                     external_rows={r['external_id']:r for r in c['rows'].values()})
            c['first_joint'],c['first_context']=first_joint(c);cells.append(c)
    # The existing first-fork validator retains earlier uncommitted advice/state mismatches.
    forks=[]
    for indices,group in [((0,1),groups[0]),((3,2),groups[1])]:
        a,b=[cells[i] for i in indices]
        f=fork.analyze(a['folder'],b['folder'],group/'summary.json')
        forks.append(dict(status=f['diagnostic_status'],joint_fork_established=f['joint_fork_established'],
                          matching_metadata=f['matching_metadata']))
    forward=pair(cells[0],cells[1]); reverse_pair=pair(cells[3],cells[2])
    comparisons=[forward,reverse_pair]
    sign=lambda x: None if x is None else (x>0)-(x<0)
    deltas=[]
    for a,b in [(cells[0],cells[1]),(cells[3],cells[2])]:
        fa,fb=a['first_joint'],b['first_joint']
        matched=fa is not None and fb is not None and fa['identity']==fb['identity']
        deltas.append(fb['decision_to_next_output_s']-fa['decision_to_next_output_s']
                      if matched and fa['decision_to_next_output_s'] is not None and fb['decision_to_next_output_s'] is not None else None)
    same_policy=[pair(cells[0],cells[3]),pair(cells[1],cells[2])]
    same_identity=all(c['first_joint'] is not None for c in cells) and all(
        c['first_joint']['identity']==cells[0]['first_joint']['identity'] for c in cells)
    local_usable=all(f['joint_fork_established'] for f in forks) and all(x is not None for x in deltas) and same_identity
    return dict(schema='E.multinews_pairs.v1',status='COMPLETE_FOUR_ARM_DIAGNOSTIC',
        groups=[str(g) for g in groups],order=[c['label'] for c in cells],arms=[arm_report(c) for c in cells],
        within_group_R_minus_H=comparisons,first_fork_validation=forks,
        same_policy_across_groups=same_policy,
        repetition_diagnostic=dict(local_R_minus_H_s=deltas,local_matched_forks=local_usable,
            same_first_joint_identity_across_arms=same_identity,
            same_policy_cross_group_prehistory_equal=[all(x['equal'] for x in p['first_joint_prehistory']['exact_prehistory'].values())
                if p['first_joint_prehistory'] is not None else None for p in same_policy],
            same_local_sign=sign(deltas[0])==sign(deltas[1]) if local_usable else None,
            full_service_same_sign={k:sign(forward['full_service_difference_s'][k])==sign(reverse_pair['full_service_difference_s'][k])
                                    for k in forward['full_service_difference_s']}),
        quality_reports=quality_reports(quality,groups,cells),
        limitations=['Two serial pairs are run-level observations; requests are not independent repetitions.',
                     'Output contents and pre-action wall timing may differ. Fixed output counts, when configured, do not establish equal contents or quality equivalence.',
                     'This is a post-run diagnostic, not a preassigned one-event trial or online oracle.',
                     'No latency correction, overlapping-time addition, sunk STORE subtraction, or SLO claim.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('probe',type=Path);p.add_argument('reverse',type=Path)
    p.add_argument('--quality',type=Path,action='append',default=[]);p.add_argument('--out',type=Path)
    a=p.parse_args()
    try:
        result=analyze(a.probe,a.reverse,a.quality)
    except (KeyError,TypeError,ValueError,OSError,IndexError) as exc:
        result=dict(schema='E.multinews_pairs.v1',status='INVALID_INPUTS',error=f'{type(exc).__name__}: {exc}')
    if a.out:
        a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(dict(status=result['status'],out=str(a.out))))
    else:
        print(json.dumps(result,indent=2,ensure_ascii=False))
    return 2 if result['status']=='UNRUN' else 1 if result['status']=='INVALID_INPUTS' else 0


if __name__=='__main__':
    raise SystemExit(main())

"""CPU diagnostic of the first actual joint H/R fork in a completed probe.

Usage: python analyze_first_fork.py LEFT_CELL RIGHT_CELL SUMMARY_JSON [--out PATH]
Consumes the existing analyze.py summary; does not rerun service analysis or
invent one-event labels. Exit 0 means valid diagnostic inputs, not a fork/gain.
"""
import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
from pathlib import Path

import analyze as base
from analyze_one_event import difference
from analyze_telemetry_control import step_observations

STATE = ('host_hit_tokens', 'free_blocks', 'full_required_blocks', 'reserved_blocks',
         'watermark_blocks', 'joint_capacity', 'eligible', 'pending_load_jobs',
         'pending_transfer_jobs', 'running', 'waiting')
IDENTITY = ('external_id', 'preemptions', 'known_tokens', 'generated_tokens', 'prefix_sha256')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_cell(folder, summary):
    folder = Path(folder).resolve(); group = Path(summary['group']).resolve()
    require(group in folder.parents, 'cell is outside summary group')
    for parent in (group, folder.parent, folder):
        require(base.read_json(parent/'status.json', {}).get('status') == 'COMPLETE', f'not COMPLETE: {parent}')
    matches = [c for c in summary['cells'] if Path(c['raw_path']).resolve() == folder/'raw.json']
    require(len(matches) == 1, 'cell missing or duplicated in successful base summary')
    service = matches[0]
    data = (folder/'raw.json').read_bytes(); raw = json.loads(data)
    config = base.read_json(folder/'config.json', {})
    require(config.get('policy') in ('host', 'recompute') and config.get('target_spec') is None,
            'expected full-policy H/R probe, not one-event target')
    require(raw.get('target_selection', {}).get('selection_mode') == 'full_policy', 'not recorded full-policy episode')
    inputs = base.read_json(folder/'inputs.json')
    rows = {r['request_id']: r for r in raw['requests']}
    require(len(rows) == len(raw['requests']) == len(inputs) == service['requests'], 'request/input count mismatch')
    require(len({r['external_id'] for r in rows.values()}) == len(rows), 'duplicate external identity')
    metrics = {r['external_id']: r for r in service['request_metrics']}
    for r in rows.values():
        require(len(r['token_times_s'])==len(r['output_token_ids']) and
                all(x<=y for x,y in zip(r['token_times_s'],r['token_times_s'][1:])),
                'invalid token/time count or ordering')
        m = metrics.get(r['external_id'], {})
        require(r.get('completed') is True and m.get('output_token_sha256') == base.fingerprint(r['output_token_ids'])
                and m.get('completion_s') == r.get('completion_s'), 'raw no longer matches completed summary outputs')
    require(service['fingerprints']['prompt_token_ids'] == base.fingerprint([x['prompt_token_ids'] for x in inputs]),
            'inputs no longer match summary')
    input_by_id = dict(zip(rows, inputs))
    trace, _, drain = step_observations(raw, include_drain=True)
    starts = [s['start_s'] for s in raw['steps']]; ends = [s['end_s'] for s in raw['steps']]
    def local_step(t):
        i = bisect_right(starts, t)-1
        require(i >= 0 and starts[i] <= t < ends[i], 'decision outside formal service step')
        return i
    cache = {}
    def identity(e, time_key):
        rid=e['request_id']; r=rows[rid]; n=e['known_tokens']; g=e['generated_tokens']
        require(type(n) is int and type(g) is int and g >= 0, 'invalid known/generated count')
        require(bisect_right(r['token_times_s'], e[time_key]) == g, 'history progress differs at observation time')
        prompt=input_by_id[rid]['prompt_token_ids']
        require(len(prompt) == r['prompt_tokens'] and n == len(prompt)+g, 'known history length mismatch')
        key=(rid,n,g)
        if key not in cache:
            cache[key]=hashlib.sha256(json.dumps(prompt+r['output_token_ids'][:g]).encode()).hexdigest()
        if e.get('prefix_sha256') is not None:
            require(cache[key] == e['prefix_sha256'], 'recorded prefix hash differs from already-known history')
        return dict(external_id=r['external_id'], preemptions=e['preemptions'], known_tokens=n,
                    generated_tokens=g, prefix_sha256=cache[key])
    lookup, decisions, commits, executions = {}, [], [], []
    for records,time_key in ((raw['decisions'],'decision_s'),(raw['commits'],'allocation_s')):
        require(all(x[time_key]<=y[time_key] for x,y in zip(records,records[1:])), 'event sequence is not chronological')
    groups=base.read_json(folder/'resources.json', {}).get('group_config', [])
    block=groups[0].get('tokens_per_block') if len(groups)==1 else None
    for index,e in enumerate(raw['decisions']):
        key=base.event_key(e); require(key not in lookup, 'duplicate lookup event key')
        norm=dict(identity(e,'decision_s'), **{k:e.get(k) for k in STATE}, action=e['action'], fallback=e['fallback'])
        if e.get('eligible'):
            require(e.get('joint_capacity') is True and (e.get('host_hit_tokens') or 0)>0
                    and e.get('fallback')=='none' and e.get('request_state_preserved') is True,
                    'eligible lookup legality/state assertion missing')
            require(e['action']==config['policy'], 'eligible lookup advice differs from fixed policy')
            require(type(block) is int and block>0 and e['full_required_blocks']==(e['known_tokens']+block-1)//block,
                    'required GPU blocks mismatch')
            require(e['full_required_blocks']+e['reserved_blocks']+e['watermark_blocks']<=e['free_blocks'],
                    'joint capacity arithmetic failed')
        lookup[key]=dict(index=index, local_step=local_step(e['decision_s']), raw=e)
        decisions.append(norm)
    seen=set()
    for e in raw['commits']:
        key=base.event_key(e); require(key in lookup and key not in seen, 'orphan/duplicate commit')
        seen.add(key); source=lookup[key]; d=source['raw']
        require(all(d.get(k)==e.get(k) for k in STATE+('decision_s','action','fallback','prefix_sha256',
                   'known_tokens','generated_tokens','preemptions')), 'lookup/commit fields changed')
        require(e['decision_s']<=e['allocation_s']<=raw['steps'][source['local_step']]['end_s'], 'commit timing invalid')
        norm=identity(e,'allocation_s')
        actual='host' if e['external_tokens']>0 else 'recompute'
        require(actual==e.get('actual_action'), 'actual action/external tokens mismatch')
        if e.get('eligible'):
            require(actual==e['action']==config['policy'], 'eligible actual action differs from fixed policy')
        commits.append(dict(norm, **{k:e.get(k) for k in STATE}, action=e['action'], actual_action=actual,
                            fallback=e['fallback'], external_tokens=e['external_tokens']))
        executions.append(dict(norm, actual_action=actual, external_tokens=e['external_tokens'],
                               allocated_blocks=e.get('allocated_blocks')))
    emissions=[[] for _ in ends]
    for r in sorted(rows.values(),key=lambda r:r['external_id']):
        for token,t in zip(r['output_token_ids'],r['token_times_s']):
            i=bisect_left(ends,t); require(i<len(ends) and ends[i]==t, 'output not at a recorded step end')
            emissions[i].append((r['external_id'],token))
    return dict(folder=str(folder), raw=raw, service=service, rows=rows, trace=trace, emissions=emissions,
                lookup=lookup, decisions=decisions, commits=commits, executions=executions,
                raw_sha256=hashlib.sha256(data).hexdigest(), drain=drain, policy=config['policy'])


def context_before(c, decision):
    boundary=decision['local_step']; when=decision['raw']['decision_s']; histories=[]
    for r in sorted(c['rows'].values(),key=lambda r:r['external_id']):
        count=bisect_right(r['token_times_s'],when)
        completed=r['completion_s']<=when
        histories.append(dict(external_id=r['external_id'], generated_tokens=count,
            history_sha256=base.fingerprint(r['output_token_ids'][:count]), completed=completed,
            finish_reason=r.get('finish_reason') if completed else None,
            stop_reason=r.get('stop_reason') if completed else None))
    preempts=[dict(external_id=c['rows'][e['request_id']]['external_id'],
                  **{k:e[k] for k in ('known_tokens','generated_tokens','computed_tokens')})
              for e in c['raw'].get('preemptions',[]) if e['time_s']<when]
    prior_commits=[n for n,e in zip(c['commits'],c['raw']['commits']) if e['allocation_s']<when]
    return dict(schedule=c['trace'][:boundary], output_emissions=c['emissions'][:boundary],
                generated_histories=histories, prior_commits=prior_commits, prior_preemptions=preempts)


def boundary(c, lookup_index):
    e=c['raw']['decisions'][lookup_index]; key=base.event_key(e); d=c['lookup'][key]
    commits=[x for x in c['raw']['commits'] if base.event_key(x)==key]
    return dict(lookup_index=lookup_index, event=e['event'], local_step=d['local_step'],
                decision_s=e['decision_s'], committed=bool(commits), allocation_s=commits[0]['allocation_s'] if commits else None,
                action=e['action'], actual_action=commits[0]['actual_action'] if commits else None)


def analyze(left, right, summary_path):
    summary_path=Path(summary_path).resolve(); data=summary_path.read_bytes(); summary=json.loads(data)
    require(summary.get('group_status',{}).get('status')=='COMPLETE' and
            not any(summary.get(k) for k in ('invalid_cells','incomplete_cells','analysis_errors')), 'base summary not clean COMPLETE')
    a,b=load_cell(left,summary),load_cell(right,summary)
    require({a['policy'],b['policy']}=={'host','recompute'}, 'expected one Host and one Recompute cell')
    metadata={k:a['service']['fingerprints'][k] is not None and
              a['service']['fingerprints'][k]==b['service']['fingerprints'][k] for k in a['service']['fingerprints']}
    look=difference(a['decisions'],b['decisions']); commit=difference(a['commits'],b['commits'])
    execution=difference(a['executions'],b['executions'])
    result=dict(schema='E.first_fork_analysis.v1', diagnostic_status='NO_ACTUAL_COMMIT_DIVERGENCE',
        inputs_valid=True, joint_fork_established=False, left_cell=str(Path(left).resolve()), right_cell=str(Path(right).resolve()),
        summary_sha256=hashlib.sha256(data).hexdigest(), raw_sha256=[a['raw_sha256'],b['raw_sha256']],
        matching_metadata=metadata, eligible_commits=[sum(x['eligible'] for x in c['commits']) for c in (a,b)],
        lookup_sequence=look, commit_sequence=commit, actual_commit_execution_sequence=execution,
        normalized_native_trace=difference(a['trace'],b['trace']), step_output_emissions=difference(a['emissions'],b['emissions']),
        first_lookup_boundaries=None, candidate=None,
        interpretation='Post-run diagnosis of the first execution difference, not a preassigned one-event trial or an online oracle. Event numbers only link records within one arm. Conservative prehistory stops before the first differing lookup step; unrecorded system state and physical KV equivalence are not established. Service costs remain in the supplied base summary; no latency correction.')
    if not look['equal']:
        i=look['first_difference']['index']
        result['first_lookup_boundaries']=[boundary(c,i) if i<len(c['decisions']) else None for c in (a,b)]
    if not any(result['eligible_commits']):
        result['diagnostic_status']='NO_JOINT_ELIGIBLE_COMMITS'
        return result
    if execution['equal']:
        return result
    i=execution['first_difference']['index']
    if i>=min(len(a['commits']),len(b['commits'])):
        result['diagnostic_status']='UNMATCHED_FIRST_COMMIT_EXECUTION'
        return result
    ca,cb=a['commits'][i],b['commits'][i]
    if not (all(ca[k]==cb[k] for k in IDENTITY) and ca['eligible'] and cb['eligible']
            and {ca['actual_action'],cb['actual_action']}=={'host','recompute'}):
        result['diagnostic_status']='UNMATCHED_FIRST_COMMIT_EXECUTION'
        return result
    sources=[c['lookup'][base.event_key(c['raw']['commits'][i])] for c in (a,b)]
    # Earlier differing suggestions remain visible even if native allocation never committed them.
    prior_lookup=difference(a['decisions'][:sources[0]['index']],b['decisions'][:sources[1]['index']])
    cuts=[]
    for c,source in zip((a,b),sources):
        first_index=look['first_difference']['index'] if not look['equal'] else source['index']
        cuts.append(c['lookup'][base.event_key(c['raw']['decisions'][min(first_index,source['index'])])])
    contexts=[context_before(c,cut) for c,cut in zip((a,b),cuts)]
    checks={k:difference(contexts[0][k],contexts[1][k]) for k in contexts[0]}
    state_equal=all(ca[k]==cb[k] for k in STATE)
    uncommitted=[]
    for side,c,source,other,other_source in ((0,a,sources[0],b,sources[1]),(1,b,sources[1],a,sources[0])):
        committed_keys={base.event_key(e) for e in c['raw']['commits']}
        for j,norm in enumerate(c['decisions'][:source['index']]):
            if base.event_key(c['raw']['decisions'][j]) in committed_keys or norm['action'] not in ('host','recompute'):
                continue
            others=[x['action'] for x in other['decisions'][:other_source['index']+1]
                    if all(norm[k]==x[k] for k in IDENTITY)]
            if any(action!=norm['action'] for action in others):
                uncommitted.append(dict(side=side, identity={k:norm[k] for k in IDENTITY},
                    boundary=boundary(c,j), other_observed_advice=sorted(set(others))))
    result['candidate']=dict(identity={k:ca[k] for k in IDENTITY}, commit_index=i,
        commits=[ca,cb], commit_boundaries=[boundary(c,s['index']) for c,s in zip((a,b),sources)],
        conservative_boundaries=[boundary(c,d['index']) for c,d in zip((a,b),cuts)],
        visible_state_equal=state_equal, prior_lookup_comparison=prior_lookup, prehistory_comparisons=checks,
        prior_uncommitted_different_lookup=bool(uncommitted), earlier_uncommitted_advice_differences=uncommitted)
    result['joint_fork_established']=all(metadata.values()) and state_equal and prior_lookup['equal'] and all(x['equal'] for x in checks.values())
    result['diagnostic_status']=('MATCHED_FIRST_JOINT_FORK' if result['joint_fork_established'] else
        'PRIOR_LOOKUP_DIVERGENCE' if not prior_lookup['equal'] else 'JOINT_FORK_WITH_STATE_OR_PREHISTORY_MISMATCH')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('left_cell',type=Path);p.add_argument('right_cell',type=Path);p.add_argument('summary',type=Path)
    p.add_argument('--out',type=Path);args=p.parse_args()
    try:
        result=analyze(args.left_cell,args.right_cell,args.summary)
    except (KeyError,TypeError,ValueError,OSError,IndexError) as exc:
        result=dict(diagnostic_status='INVALID_INPUTS',inputs_valid=False,joint_fork_established=False,error=f'{type(exc).__name__}:{exc}')
    out=args.out or args.summary.parent/'first_fork_summary.json'
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ('diagnostic_status','inputs_valid','joint_fork_established')}))
    return 0 if result['inputs_valid'] else 1


if __name__=='__main__':
    raise SystemExit(main())

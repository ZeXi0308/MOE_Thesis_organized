#!/usr/bin/env python3
"""Source-ID action/tail diagnostics after the entire primary campaign qualifies."""
import argparse
from collections import Counter
from itertools import zip_longest
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gate(bundle, primary_path, run_dir=None):
    run_dir = run_dir or bundle/'execution'
    if not primary_path.exists():
        return None, dict(status='UNRUN_OR_UNQUALIFIED', reason='qualified primary analysis is absent', comparisons=[])
    primary = read(primary_path)
    manifest = read(bundle/'preparation/source/campaign.json')
    execution_path = run_dir/'execution.json'
    execution = read(execution_path) if execution_path.exists() else {}
    labels = [c['label'] for c in manifest['cells']]
    rows = primary.get('cells', [])
    qualified = (primary.get('status') == 'MEASUREMENT_ONLY' and primary.get('comparisons_eligible')
        and len(rows) == len(labels) and {r['label'] for r in rows} == set(labels)
        and all(r['status'] == 'COMPLETE' and r['full_episode_comparison_eligible'] for r in rows)
        and execution.get('status') == 'COMPLETE'
        and [c['label'] for c in execution.get('cells', [])] == labels
        and all(c['status'] == 'READ_BACK' for c in execution.get('cells', [])))
    if not qualified:
        return None, dict(status='UNRUN_OR_UNQUALIFIED', reason='the full manifest campaign is not qualified/read back', comparisons=[])
    assert manifest == primary['campaign_manifest'] and digest(bundle/'preparation/source/campaign.json') == primary['campaign_manifest_sha256']
    accounts = {r['label']: r for r in primary['recovery_accounting']}
    assert set(accounts) == set(labels)
    for label, row in accounts.items():
        for artifact in row['inputs']:
            path = run_dir/'gpu_results'/label/Path(artifact['path']).name
            assert digest(path) == artifact['sha256'], f'raw changed since primary qualification: {path}'
    return (manifest, primary, accounts), None


def actions(raw, decisions, role):
    """Candidate check adapted from V extract_paths; C uses each decision's mode."""
    ids = raw['internal_to_source']; traces = {t['attempted_step']: t for t in raw['memory_trace']}
    absent, resident, count, selected = {}, {}, Counter(), []
    for d in decisions:
        step = d['step']; before = traces[step]['before']; states = before['requests']; proposal = d['proposal']
        effective = d.get('effective_victim_order', role)
        assert effective in ('least_progress', 'most_output'), 'C requires recorded effective_victim_order'
        if proposal and proposal['action'] == 'rotate':
            candidates = []
            for rid in before['running_ids']:
                state = states[rid]; progress = max(0, state['computed_tokens']-state['prompt_tokens'])/1024
                if progress < .9 and count[rid] < 8 and step-resident.get(rid, -10**9) >= 30:
                    candidates.append(dict(internal_id=rid, request_id=ids[rid], output_tokens=state['output_tokens'],
                        computed_tokens=state['computed_tokens'], computed_progress=progress, blocks=sum(state['block_counts'])))
            key = (lambda r: (-r['output_tokens'], r['internal_id'])) if effective == 'most_output' else (lambda r: (r['computed_progress'], r['internal_id']))
            candidates.sort(key=key)
            victim, target = proposal['victim_id'], proposal['resume_id']
            assert candidates and candidates[0]['internal_id'] == victim, 'actual proposal violates effective victim order'
            if d['forced_preempted']:
                assert d['forced_preempted'] == [victim]
                normalized = dict(running_ids=sorted(ids[r] for r in before['running_ids']), free_blocks=before['pool']['free_blocks'],
                    requests={ids[r]: s for r, s in states.items()})
                selected.append(dict(step=step, effective_victim_order=effective, victim=ids[victim], resume=ids[target],
                    victim_output_tokens=states[victim]['output_tokens'], victim_computed_tokens=states[victim]['computed_tokens'],
                    eligible=[{k:v for k,v in r.items() if k!='internal_id'} for r in candidates],
                    prestate=normalized))
        for rid in d['preempted']:
            absent.setdefault(rid, step); count[rid] += 1; resident.pop(rid, None)
        for rid in d['resumed']:
            absent.pop(rid, None); resident[rid] = step
    assert selected, 'qualified rotation must have a successful forced swap'
    return selected


def tails(raw):
    """V tail ledger with widths and labels taken from actual execution."""
    calls = {c['scheduler_step_start']: c for c in raw['engine_steps']}
    bytime = {c['returned_s']: c for c in raw['engine_steps']}
    ranked = sorted(raw['requests'], key=lambda r:(r['completion_s'], r['request_id']))
    ranks = {r['request_id']: i+1 for i,r in enumerate(ranked)}
    requests = {r['request_id']: dict(completion_s=r['completion_s'], completion_rank=ranks[r['request_id']],
        completion_step=bytime[r['completion_s']]['scheduler_step_start']) for r in raw['requests']}
    pure = [s for s in raw['scheduler_steps'] if s['recompute_tokens']==0 and not any(r['prefill_tokens'] for r in s['scheduled'])]
    widths = {}
    for width in sorted({len(s['scheduled']) for s in pure} | {1,2,4}):
        rows = [s for s in pure if len(s['scheduled']) == width]
        widths[str(width)] = dict(calls=len(rows), engine_duration_s=sum(calls[s['step']]['returned_s']-calls[s['step']]['start_s'] for s in rows),
            first_step=rows[0]['step'] if rows else None, last_step=rows[-1]['step'] if rows else None,
            first_output_counts={r['request_id']:r['output_tokens_before'] for r in rows[0]['scheduled']} if rows else {},
            request_sets=dict(Counter('|'.join(sorted(r['request_id'] for r in s['scheduled'])) for s in rows)))
    return dict(requests=requests, completion_order=[r['request_id'] for r in ranked], pure_decode_widths=widths)


def diagnose(bundle, primary_path, run_dir=None):
    run_dir = run_dir or bundle/'execution'
    qualified, blocked = gate(bundle, primary_path, run_dir)
    if blocked:
        return blocked
    manifest, primary, accounts = qualified
    cells = {}
    for spec in manifest['cells']:
        directory = run_dir/'gpu_results'/spec['label']
        raw = read(next(p for p in (directory/'raw.json', directory/'raw.json.gz') if p.exists()))
        decisions = read(directory/'headroom-decisions.json')
        cells[spec['label']] = dict(spec=spec, forced_actions=actions(raw, decisions, spec['role']),
            **tails(raw), recovery_accounting=accounts[spec['label']])
    pairs = []
    for p in primary['comparisons']:
        assert p['status'] == 'DESCRIPTIVE_MATCHED_PAIR'
        a,b = cells[p['baseline']],cells[p['action']]
        assert a['spec']['cohort_id'] == b['spec']['cohort_id'] and a['spec']['block'] == b['spec']['block']
        differences = [dict(request_id=rid, baseline=a['requests'][rid], action=b['requests'][rid],
            completion_change_s=b['requests'][rid]['completion_s']-r['completion_s']) for rid,r in a['requests'].items()]
        first = next((dict(index=i, baseline=x, action=y) for i,(x,y) in enumerate(zip_longest(a['forced_actions'],b['forced_actions']))
            if x is None or y is None or (x['victim'],x['resume']) != (y['victim'],y['resume'])), None)
        widths = {w:dict(calls_change=b['pure_decode_widths'].get(w,{}).get('calls',0)-a['pure_decode_widths'].get(w,{}).get('calls',0),
            engine_duration_change_s=b['pure_decode_widths'].get(w,{}).get('engine_duration_s',0)-a['pure_decode_widths'].get(w,{}).get('engine_duration_s',0))
            for w in set(a['pure_decode_widths']) | set(b['pure_decode_widths'])}
        anchor_label = next(s['label'] for s in manifest['cells'] if s['cohort_id']==a['spec']['cohort_id'] and s['block']==a['spec']['block'] and s['role']=='least_progress')
        anchors = cells[anchor_label]['completion_order']
        pairs.append(dict(baseline=p['baseline'], action=p['action'], first_choice_difference=first,
            first_action_prestate_equal=a['forced_actions'][0]['prestate']==b['forced_actions'][0]['prestate'],
            per_request_changes=differences, width_changes=widths, least_progress_early_four_ids=anchors[:4],
            least_progress_tail_two_ids=anchors[-2:], both_keep_A_tail_two=set(a['completion_order'][-2:])==set(b['completion_order'][-2:])==set(anchors[-2:]),
            early_four_changes=[r for r in differences if r['request_id'] in anchors[:4]], tail_two_changes=[r for r in differences if r['request_id'] in anchors[-2:]]))
    return dict(status='MEASUREMENT_ONLY', experiment_id=manifest['experiment_id'], cells=cells, comparisons=pairs,
        primary_input=dict(path=str(primary_path),sha256=digest(primary_path)),
        scope='Manifest roles/blocks; source-ID joins; C uses recorded effective order. Main analysis qualifies successful-swap counts, costs and recovery chains. Output counts exclude recompute. Width bucket time changes and displaced completion are actual paths, not additive causal savings or hard bounds. No new C GPU evidence exists until the full primary campaign qualifies.')


def readable(result):
    if result['status'] != 'MEASUREMENT_ONLY':
        return result['status']+'\n\n'+result['reason']+'\n'
    lines = ['# Actual first-swap and tail paths', '', '| Baseline → action | width2 calls Δ | width2 host ms Δ | A early four completion ms Δ | A tail two completion ms Δ | Same A tail two |', '|---|---:|---:|---|---|---|']
    for p in result['comparisons']:
        w=p['width_changes']['2']; early=', '.join(f"{x['request_id'].rsplit('-',1)[-1]}: {x['completion_change_s']*1000:+.3f}" for x in p['early_four_changes']); tail=', '.join(f"{x['request_id'].rsplit('-',1)[-1]}: {x['completion_change_s']*1000:+.3f}" for x in p['tail_two_changes'])
        lines.append(f"| {p['baseline']} → {p['action']} | {w['calls_change']:+} | {w['engine_duration_change_s']*1000:+.3f} | {early} | {tail} | {p['both_keep_A_tail_two']} |")
    return '\n'.join(lines+['',result['scope'],''])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--primary',type=Path)
    parser.add_argument('--run-dir',type=Path,help='execution directory; defaults to bundle/execution')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output_dir.exists(), 'refusing overwrite'
    result=diagnose(args.bundle,args.primary or args.bundle/'analysis/analysis.json',args.run_dir)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'report.md').write_text(readable(result))
    print(json.dumps(dict(status=result['status'],pairs=len(result['comparisons']),output=str(args.output_dir))))


if __name__=='__main__':
    main()

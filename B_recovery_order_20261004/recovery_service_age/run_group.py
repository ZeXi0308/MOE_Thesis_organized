#!/usr/bin/env python3
"""One age8/stall8 comparison using the frozen shared-resource controller."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT = BASE/'recovery_repeat_unique/run_group.py'
PARENT_SHA = '47280db3948cc71a194e0d5283f63b8816df6fdc9949eaf2a8d8b42e3d485b86'


def load_parent():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen common resource controller changed')
    spec = importlib.util.spec_from_file_location('service_age_group_parent', PARENT)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent


def action_evidence(data, mode):
    """Count actual changes; a different shadow suggestion alone does not count."""
    if (data.get('mode') != mode or data.get('status') != 'UNINSTALLED'
            or data.get('action_limit') != 8 or not isinstance(data.get('events'), list)):
        raise RuntimeError('Missing service-age action evidence')
    episodes, ids = [], set()
    different_moves = suppressions = suggestions = fallbacks = 0
    for event in data['events']:
        before, after = event.get('waiting_before'), event.get('waiting_after')
        if (not isinstance(before, list) or not before or not isinstance(after, list)
                or len(set(before)) != len(before) or sorted(before) != sorted(after)
                or event.get('action_count_before') != len(episodes)):
            raise RuntimeError('Service-age queue or count evidence differs')
        eligible = {row['request']:row for row in event['candidates'] if row['eligible']}
        age, stall = event['age8_suggestion'], event['stall8_suggestion']
        for name, choice in [('age8',age), ('stall8',stall)]:
            if choice not in eligible:
                raise RuntimeError('Service-age suggestion is not legal')
            episode = (choice, eligible[choice]['num_preemptions'])
            executable = len(episodes) < 8 and episode not in episodes
            if (event[name+'_suggestion_num_preemptions'] != episode[1]
                    or event[name+'_would_execute'] is not executable):
                raise RuntimeError('Suggestion guard does not match actual prior actions')
        suggestions += age != stall
        fallbacks += event.get('stall8_fallback_reason') is not None
        chosen = age if mode == 'age8' else stall
        episode = (chosen, eligible[chosen]['num_preemptions'])
        reasons = []
        if episode in episodes: reasons.append('EPISODE_ALREADY_BYPASSED')
        if len(episodes) >= 8: reasons.append('BUDGET_EXHAUSTED')
        if event.get('candidate_head') != chosen or event.get('action_skip_reasons') != reasons:
            raise RuntimeError('Effective choice or guard differs')
        if event['action'] == 'REORDERED':
            if (reasons or chosen == before[0]
                    or after != [chosen]+[rid for rid in before if rid != chosen]
                    or event.get('queue_changed') is not True
                    or event.get('action_count_after') != len(episodes)+1):
                raise RuntimeError('Invalid service-age mutation')
            episodes.append(episode); ids.add(chosen)
            different_moves += chosen != age
        elif event['action'] == 'SHADOW_ONLY':
            if (not reasons or before != after or event.get('queue_changed') is not False
                    or event.get('action_count_after') != len(episodes)):
                raise RuntimeError('Invalid service-age shadow')
            suppressions += mode == 'stall8' and event['age8_would_execute'] and not event['stall8_would_execute']
        else:
            raise RuntimeError('Incomplete service-age action')
        if event.get('final_head') != after[0]:
            raise RuntimeError('Final service-age head differs')
    expected = [dict(request=rid,num_preemptions=number) for rid,number in episodes]
    if data.get('action_count') != len(episodes) or data.get('bypassed_episodes') != expected:
        raise RuntimeError('Final service-age count differs')
    return dict(reorder_count=len(episodes), unique_ids=sorted(ids), unique_id_count=len(ids),
        suggestion_difference_count=suggestions, different_queue_mutations=different_moves,
        executable_age_suppressions=suppressions, difference_decisions=different_moves+suppressions,
        fallback_decisions=fallbacks,
        evidence_semantics='Only successful queue mutations or suppression of an executable age suggestion count; actual observed prefixes, not hypothetical alternate trajectories.')


def adapted_source():
    parent = load_parent()
    text, sources = parent.adapted_source()
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('Service-age group boundary changed: '+old[:100])
        text = text.replace(old,new)
    replace(inspect.getsource(parent.action_evidence), inspect.getsource(action_evidence))
    replace("['repeat8', 'unique8', 'unique8', 'repeat8']", "['age8', 'stall8', 'stall8', 'age8']")
    replace('Requires fixed cap256 repeat8/unique8/unique8/repeat8,450s per child,current GPU/110GiB host',
            'Requires fixed cap256 age8/stall8/stall8/age8,450s per child,current GPU/110GiB host')
    replace('B_RECOVERY_REPEAT_UNIQUE=mode,', 'B_RECOVERY_SERVICE_AGE=mode,')
    replace("receipt['cells'][-1]['recovery_repeat_unique_mode']", "receipt['cells'][-1]['recovery_service_age_mode']")
    replace("cell/'output/recovery-repeat-unique.json'", "cell/'output/recovery-service-age.json'")
    replace("receipt['cells'][-1]['recovery_repeat_unique_actions']", "receipt['cells'][-1]['recovery_service_age_actions']")
    replace('First unique8 had no different executed queue mutation or executable repeat8 suppression; reverse cells not started; suggestions alone do not count',
            'First stall8 had no different executed queue mutation or executable age8 suppression; reverse cells not started; suggestions alone do not count')
    return text, [*sources,PARENT,Path(__file__)]


def self_check():
    # Resource/whole-group lifecycle is unchanged and was already exercised by
    # the parent. Check only the changed action accounting and assembled code.
    text, _ = adapted_source(); compile(text,'<service-age-group>','exec')
    assert 'B_RECOVERY_REPEAT_UNIQUE' not in text
    assert "B_RECOVERY_SERVICE_AGE=mode" in text
    assert "str(ROOT/'run_cell.py')" in text
    data = dict(mode='stall8',status='UNINSTALLED',action_limit=8,action_count=1,
                bypassed_episodes=[dict(request='s',num_preemptions=1)],events=[])
    common = dict(waiting_before=['h','a','s'],age8_suggestion='a',stall8_suggestion='s',
        age8_suggestion_num_preemptions=1,stall8_suggestion_num_preemptions=1,
        age8_would_execute=True,candidate_head='s',
        candidates=[dict(request=r,eligible=True,num_preemptions=1) for r in ('a','s')])
    data['events'].append(dict(common,stall8_would_execute=True,action_count_before=0,
        action_count_after=1,action='REORDERED',action_skip_reasons=[],
        waiting_after=['s','h','a'],final_head='s',queue_changed=True))
    data['events'].append(dict(common,stall8_would_execute=False,action_count_before=1,
        action_count_after=1,action='SHADOW_ONLY',action_skip_reasons=['EPISODE_ALREADY_BYPASSED'],
        waiting_after=['h','a','s'],final_head='h',queue_changed=False))
    e = action_evidence(data,'stall8')
    assert e['different_queue_mutations']==1 and e['executable_age_suppressions']==1
    assert e['difference_decisions']==2 and e['reorder_count']==1
    data['events'][1]['age8_would_execute']=False
    try: action_evidence(data,'stall8')
    except RuntimeError: pass
    else: raise AssertionError('Invalid guard accepted')
    print('PASS: assembled common controller, actual different reorder, executable age suppression, malformed guard rejection; resource lifecycle unchanged. CPU only.')


def main():
    if sys.argv[1:] == ['--self-check']:
        self_check(); return 0
    parent = load_parent()
    parent.adapted_source = adapted_source
    parent.__file__ = str(__file__)
    return parent.main()


if __name__ == '__main__': raise SystemExit(main())

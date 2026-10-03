"""Current-call frequency protection feasibility only; no future trace replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import wisp_expert_groups as planner


def read(p): return json.loads(p.read_text())


def inspect(record, cap):
    rows, context, entry = record['row_topk_experts'], record['context'], set(record['entry_resident_experts'])
    active = set(e for row in rows for e in row)
    assert active == set(record['active_experts']) and record['status'] == 'complete'
    plan, info = planner.retention_plan(rows, context, entry, cap, 'frequency', order='late', salt='unused-frequency')
    protected = set(info['chosen_protected_experts']); a = len(protected & entry)
    ordinary = planner.partition_experts(active, entry, cap)
    gp, g0 = len(plan), len(ordinary)
    assert gp == (max(1, (len(active)-a+cap-a-1)//(cap-a)) if active else 0)
    assert g0 == (len(active)+cap-1)//cap and gp >= g0
    assert Counter(e for g in plan for e in g['execute_experts']) == Counter(active)
    for reverse in (False, True):
        current, loaded, held = set(entry), Counter(), protected & entry
        for group in plan:
            execute, ensure = set(group['execute_experts']), set(group['ensure_experts'])
            held |= protected & execute
            assert ensure == execute | held and set(group['protected_after']) == held and len(ensure) <= cap
            loaded.update(ensure-current)
            victims = sorted(current-ensure, reverse=reverse)[:max(0, len(current | ensure)-cap)]
            current = (current-set(victims)) | ensure
            assert len(current) <= cap and held <= current
        assert loaded == Counter(active-entry) and protected <= current
    different = [(set(g['execute_experts']),set(g['ensure_experts'])) for g in plan] != [(set(g),set(g)) for g in ordinary]
    assert info['applied'] == (bool(protected) and different)
    if record['retention']['mode'] == 'frequency':
        assert info['chosen_protected_experts'] == record['retention']['chosen_protected_experts']
        assert plan == [dict(execute_experts=g['required_experts'],ensure_experts=g['ensure_experts'],protected_after=g['protected_after']) for g in record['groups']]
    return info, protected, protected & entry, gp, g0, different


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    execution=read(args.input_dir/'results/execution.json'); protocol=read(args.input_dir/'protocol.json')
    source_hash=hashlib.sha256(Path(planner.__file__).read_bytes()).hexdigest()
    assert source_hash == protocol['source_sha256']['wisp_expert_groups.py']
    assert execution['status']=='COMPLETE' and len(execution['cells'])==2
    engines={}; by_mode={}; total=Counter()
    for cell in execution['cells']:
        root=args.input_dir/'results'/cell['label']; phases={}; engine_counts=Counter(); cap=read(root/'pager_summary.json')['cap']; assert cap==24
        with (root/'pager/calls.jsonl').open() as stream:
            for line in stream:
                r=json.loads(line)
                if not r['measurement']: continue
                phase=r['context']['phase']; mode=r['retention']['mode']+'_'+r['retention']['order']
                row=phases.setdefault(phase,dict(actual_mode=mode,counts=Counter(),distributions={},zero_extra_call_ids=[]))
                assert row['actual_mode']==mode
                info,P,H,gp,g0,different=inspect(r,cap)
                keep=info['applied'] and gp==g0 and different; reject=info['applied'] and gp>g0
                counts=dict(layer_calls=1,eligible=int(info['eligible']),candidate_applied=int(info['applied']),
                    zero_extra_group_different_calls=int(keep),guard_excluded_calls=int(reject),
                    candidate_groups=gp,ordinary_groups=g0,guard_excluded_extra_groups=(gp-g0) if reject else 0)
                for target in (row['counts'],engine_counts,total,by_mode.setdefault(mode,Counter())): target.update(counts)
                if keep: row['zero_extra_call_ids'].append(r['call_id'])
                for scope,selected in (('eligible',info['eligible']),('zero_extra',keep),('guard_excluded',reject)):
                    if not selected: continue
                    hist=row['distributions'].setdefault(scope,{k:Counter() for k in ('protected_size','entry_intersection_size','size_pair','protected_experts','entry_protected_experts')})
                    hist['protected_size'][str(len(P))]+=1; hist['entry_intersection_size'][str(len(H))]+=1
                    hist['size_pair'][f'{len(P)},{len(H)}']+=1; hist['protected_experts'].update(map(str,sorted(P))); hist['entry_protected_experts'].update(map(str,sorted(H)))
        assert set(phases)=={e['phase'] for e in read(root/'episodes.json')} and len(phases)==8
        assert engine_counts['layer_calls']==read(root/'pager_summary.json')['measurement_calls']
        engines[cell['label']]=dict(counts=engine_counts,phases=phases)
    output=dict(status='POST_HOC_STRUCTURAL_OPPORTUNITY_ONLY',input_dir=str(args.input_dir.resolve()),planner_sha256=source_hash,
        checks='All measurement calls: exact bounds, disjoint active coverage, cap, final P and A\\entry loaded once under two legal victim orders.',
        formulas=dict(gP='0 if A empty else max(1,ceil((|A|-|P intersect entry|)/(24-|P intersect entry|)))',g0='ceil(|A|/24)',
            guard='candidate applied AND gP==g0 AND ordered execute/ensure group sets differ from ordinary; within-group list order ignored'),
        counts=total,by_actual_mode=by_mode,engines=engines,
        scope=['P is the unchanged current-row frequency selection; both modes use their own actual entry state and routes.',
            'Distributions count layer-call occurrences: sizes/joint sizes, and per-expert inclusion counts within each scope.',
            'The guard was not executed. No next-step masking, counterfactual traffic/latency or deployment benefit is estimated.',
            'Repeated episodes reuse the same three documents; this only tests whether a real guarded comparison has nonempty action space.'])
    with args.out.open('x') as f: json.dump(output,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__': main()

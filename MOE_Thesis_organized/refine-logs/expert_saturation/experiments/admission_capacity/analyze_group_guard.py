"""Three-arm executed comparison; group guard uses only each call's own state."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from analyze_retention_lifecycle import FIELDS, check, digest, engine_result, read

ARMS=('none_early','frequency_late','frequency_late_no_extra_groups')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.input_dir;protocol=read(root/'protocol.json');execution=read(root/'results/execution.json');issues=[];engines=[];pairs=[]
    check(execution['status']=='COMPLETE' and len(execution['cells'])==len({c['label'] for c in execution['cells']})==2,'campaign incomplete',issues)
    for cell in execution['cells']:
        try:
            e=engine_result(root,cell,protocol);engines.append(e);issues.extend(cell['label']+': '+s for s in e['issues'])
            rows=e['rows'];check([r.get('arm') for r in rows]==protocol['engine_sequences'][cell['label']],cell['label']+': arm sequence',issues)
            check(Counter(r.get('arm') for r in rows)==Counter({arm:2 for arm in ARMS}),cell['label']+': arm counts',issues)
            meta=defaultdict(Counter)
            with (root/'results'/cell['label']/'pager/calls.jsonl').open() as f:
                for line in f:
                    r=json.loads(line)
                    if not r['measurement']:continue
                    t=r['retention'];A=set(r['active_experts']);C=set(r['entry_resident_experts']);P=set(t['candidate_protected_experts']);chosen=set(t['chosen_protected_experts'])
                    cap=24;h=len(P&C);g0=(len(A)+cap-1)//cap;gp=max(1,(len(A)-h+cap-h-1)//(cap-h)) if A else 0
                    check(t['ordinary_group_count']==g0 and t['candidate_group_lower_bound']==gp and t['executed_group_count']==len(r['groups']),'guard count metadata mismatch',issues)
                    guarded=t['guard']=='no_extra_groups';reject=guarded and gp>g0
                    check(t['guard_rejected']==reject and chosen==(set() if reject else P),'guard decision mismatch',issues)
                    if guarded:check(len(r['groups'])==g0,'executed guard increased own-state groups',issues)
                    meta[r['context']['phase']].update(layer_calls=1,candidate_eligible=int(t['candidate_eligible']),candidate_selected=int(bool(P)),
                        guard_rejected=int(reject),applied=int(t['applied']),groups=len(r['groups']),ordinary_groups=g0,
                        candidate_min_groups=gp,guarded_calls=int(guarded),protected_occurrences=len(chosen))
            for r in rows:r['guard_counts']=dict(meta[r['phase']])
            for start in (0,3):
                block={r['arm']:r for r in rows[start:start+3]};check(set(block)==set(ARMS),'block missing arm',issues)
                for baseline,treatment in ((ARMS[0],ARMS[2]),(ARMS[1],ARMS[2]),(ARMS[0],ARMS[1])):
                    x,y=block[baseline],block[treatment]
                    pairs.append(dict(engine=cell['label'],block=start//3,positions=[x['repeat'],y['repeat']],baseline=baseline,treatment=treatment,
                        delta={k:y[k]-x[k] for k in FIELDS},delta_pct={k:100*(y[k]/x[k]-1) if x[k] else None for k in FIELDS},
                        request_deltas={rid:{k:y['requests'][rid][k]-x['requests'][rid][k] for k in ('ttft_s','tpot_s','max_itl_s','completion_latency_s')} for rid in x['requests']}))
        except (OSError,KeyError,ValueError,TypeError,IndexError,StopIteration) as exc:issues.append(cell['label']+': '+repr(exc))
    rows=[r for e in engines for r in e['rows'] if 'trace_sha256' in r]
    check(len(rows)==12 and len(pairs)==12,'incomplete usable rows/pairs',issues)
    check(len({digest(r['fixed_resources']) for r in rows})==1,'fixed resources differ',issues)
    equality={arm:{field:len({r[field] for r in rows if r['arm']==arm})==1 for field in ('trace_sha256','output_sha256')} for arm in ARMS}
    result=dict(status='ISSUES' if issues else 'DESCRIPTIVE_GROUP_GUARD_COMPARISON',issues=issues,engines=engines,pairs=pairs,within_mode_equal=equality,
        all_outputs_equal=len({r['output_sha256'] for r in rows})==1,
        scope=['Two engines, four dependent episodes per arm, same three documents; no significance, population noise bound or production SLO.',
            'Group bound is checked on each executed call own A/C/P. Total arm group counts may differ because future routes and cache evolve independently.',
            'Capture, per-repeat cycle and whole engine/process cost retained; repeat marker/shared tail covered by engine totals, not assigned to arms.',
            'GC covers observed callback intervals only and closes before event export/shutdown. CPU/CUDA/GC spans overlap; no subtraction as counterfactual benefit.'])
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    if issues:raise SystemExit('issues retained in output')


if __name__=='__main__':main()

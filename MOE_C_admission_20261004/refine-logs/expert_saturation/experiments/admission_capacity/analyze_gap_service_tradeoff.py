#!/usr/bin/env python3
"""All empirical gap thresholds, without choosing a business SLO from results."""
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path


def request_metrics(raw):
    assert raw['status'] == 'COMPLETE' and raw['error'] is None
    rows = []
    for r in raw['requests']:
        times = r['token_times_s']
        assert r['status'] == 'completed' and len(times) == len(r['output_token_ids']) > 1
        assert times == sorted(times) and r['completion_s'] == times[-1]
        rows.append(dict(request_id=r['request_id'],
                         ttft_s=times[0]-r['arrival_s'],
                         completion_s=times[-1]-r['arrival_s'],
                         max_gap_s=max(b-a for a,b in zip(times,times[1:]))))
    return rows


def compare(a, b):
    """Piecewise-constant rate differences on every union breakpoint, [lo, hi)."""
    ag, bg = sorted(r['max_gap_s'] for r in a['requests']), sorted(r['max_gap_s'] for r in b['requests'])
    points = sorted({0.0, *ag, *bg})
    intervals = []
    for index, lo in enumerate(points):
        na, nb = bisect_right(ag,lo), bisect_right(bg,lo)
        delta = nb/b['wall_s'] - na/a['wall_s']
        relation = 'higher' if delta > 1e-12 else ('lower' if delta < -1e-12 else 'equal')
        intervals.append(dict(lower_inclusive_s=lo,
            upper_exclusive_s=points[index+1] if index+1<len(points) else None,
            baseline_qualified=na, action_qualified=nb,
            baseline_rate=na/a['wall_s'], action_rate=nb/b['wall_s'], relation=relation))
    runs = []
    for x in intervals:
        if runs and runs[-1]['relation']==x['relation']:
            runs[-1]['upper_exclusive_s']=x['upper_exclusive_s']
        else:
            runs.append({k:x[k] for k in ('lower_inclusive_s','upper_exclusive_s','relation')})
    qa={r['request_id']:r for r in a['requests']};qb={r['request_id']:r for r in b['requests']}
    assert qa.keys()==qb.keys()
    return dict(baseline=a['label'],action=b['label'],intervals=intervals,relation_ranges=runs,
                gap_distribution_action_first_order_dominates=all(y<=x for x,y in zip(ag,bg)),
                gap_distribution_baseline_first_order_dominates=all(x<=y for x,y in zip(ag,bg)),
                action_completion_slower_requests=sum(qb[k]['completion_s']>qa[k]['completion_s'] for k in qa))


def baseline_envelope(baselines, action):
    """Descriptive envelope of separately executed complete static policies."""
    cells=[*baselines,action]
    assert baselines and all({r['request_id'] for r in c['requests']} ==
                            {r['request_id'] for r in action['requests']} for c in baselines)
    gaps={c['label']:sorted(r['max_gap_s'] for r in c['requests']) for c in cells}
    points=sorted({0.0,*(g for values in gaps.values() for g in values)})
    rows=[]
    for i, lo in enumerate(points):
        rates={c['label']:bisect_right(gaps[c['label']],lo)/c['wall_s'] for c in cells}
        best=max(rates[c['label']] for c in baselines)
        delta=rates[action['label']]-best
        rows.append(dict(lower_inclusive_s=lo,upper_exclusive_s=points[i+1] if i+1<len(points) else None,
            rates=rates,best_baselines=[c['label'] for c in baselines if rates[c['label']]==best],
            action_minus_baseline_envelope=delta,
            relation='higher' if delta>1e-12 else ('lower' if delta < -1e-12 else 'equal')))
    return dict(action=action['label'],baselines=[c['label'] for c in baselines],intervals=rows,
        action_above_envelope_anywhere=any(r['relation']=='higher' for r in rows),
        boundary='Measured complete-policy envelope for a hypothetical fixed gap requirement. No within-episode switching, future action Oracle, joint TTFT constraint, or selected acceptance threshold.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--campaign',default='20260914_restore_token_reservation_r01')
    p.add_argument('--roles',nargs='+',default=['fit_scan','guard_all','guard_residual'])
    p.add_argument('--label-prefix',default='')
    p.add_argument('--plot-only', action='store_true', help='Render retained analysis without rewriting it')
    args=p.parse_args()
    if args.plot_only:
        plot(json.loads((args.output_dir/'analysis.json').read_text()), args.output_dir)
        return
    assert not args.output_dir.exists(), 'retain prior analysis'
    folder=args.workspace/'refine-logs/expert_saturation/outputs/admission_capacity'/args.campaign
    qualified=json.loads((folder/'analysis/analysis.json').read_text())
    assert qualified['status']=='MEASUREMENT_ONLY' and len(qualified['cells'])==2*len(args.roles)
    assert args.roles[-1]=='guard_residual' and len(set(args.roles))==len(args.roles)
    cells=[]
    for c in qualified['cells']:
        raw_path=args.workspace/c['raw_path']
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==c['raw_sha256']
        raw=json.loads(raw_path.read_text());rows=request_metrics(raw)
        assert len(rows)==32 and c['eligible'] and c['status']=='COMPLETE'
        assert raw['observation_end_s']==c['wall_s']
        cells.append(dict(label=c['label'],wall_s=c['wall_s'],requests=rows,
                          raw_path=str(raw_path),raw_sha256=c['raw_sha256'],
                          all_completed_rate=len(rows)/c['wall_s'],
                          worst_ttft_s=max(r['ttft_s'] for r in rows)))
    by_label={c['label']:c for c in cells}
    comparisons=[compare(by_label[f'{args.label_prefix}block{block}-{baseline}'],by_label[f'{args.label_prefix}block{block}-guard_residual'])
                 for block in (0,1) for baseline in args.roles[:-1]]
    result=dict(status='DESCRIPTIVE_FULL_THRESHOLD_CURVES',cells=cells,comparisons=comparisons,
        roles=args.roles,label_prefix=args.label_prefix,
        definition='Q(g) = completed requests whose own max engine-return gap <= g / full episode wall seconds',
        boundary='No TTFT or TPOT constraint is imposed. This is not joint SLO-goodput or client QoE. Thresholds are descriptive breakpoints, never selected acceptance limits.',
        repetitions='Two execution-order blocks on the same fixed-length document cohort. No independent-request statistical claim.')
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if {'native','most_output'}.issubset(args.roles):
        envelopes=[baseline_envelope([by_label[f'{args.label_prefix}block{b}-{v}']
                    for v in ('native','most_output')],
                    by_label[f'{args.label_prefix}block{b}-guard_residual']) for b in (0,1)]
        sidecar=dict(source_analysis_sha256=hashlib.sha256((args.output_dir/'analysis.json').read_bytes()).hexdigest(),
                     envelopes=envelopes)
        (args.output_dir/'baseline_envelope.json').write_text(json.dumps(sidecar,indent=2,allow_nan=False)+'\n')
    plot(result, args.output_dir)
    print(json.dumps([dict(baseline=c['baseline'],action=c['action'],ranges=c['relation_ranges'],
                          dominates=c['gap_distribution_action_first_order_dominates']) for c in comparisons],indent=2))


def plot(result, output_dir):
    assert not (output_dir/'gap_service_curves.png').exists(), 'retain prior plot'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'native':'#8B59A3','most_output':'#D18F10','fit_scan':'#2878B5','guard_all':'#D95319','guard_residual':'#27864A'}
    by_label={c['label']:c for c in result['cells']}
    xmax=max(r['max_gap_s'] for c in result['cells'] for r in c['requests'])*1.04
    fig, axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey='row')
    for block in (0,1):
        for role in result.get('roles',['fit_scan','guard_all','guard_residual']):
            color=colors[role]
            c=by_label[f'{result.get("label_prefix", "")}block{block}-{role}'];g=sorted(r['max_gap_s'] for r in c['requests'])
            x=[0.,*sorted(set(g)),xmax]
            counts=[bisect_right(g,t) for t in x]
            axes[0,block].step(x,counts,where='post',label=role,color=color)
            axes[1,block].step(x,[n/c['wall_s'] for n in counts],where='post',color=color)
        axes[0,block].set_title(f'Execution block {block}')
        axes[1,block].set_xlabel('Allowed maximum engine-return gap (s)')
        for ax in axes[:,block]:
            ax.grid(alpha=.25);ax.set_xlim(0,xmax)
    axes[0,0].set_ylabel('Qualified completed requests / 32')
    axes[1,0].set_ylabel('Gap-qualified completions / s')
    axes[0,0].legend(loc='lower right',fontsize=9)
    fig.suptitle('All observed gap thresholds; no chosen business SLO',fontsize=13)
    fig.text(.5,.015,'Same 32-document cohort; TTFT/TPOT unconstrained; all episode costs retained.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95))
    fig.savefig(output_dir/'gap_service_curves.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()

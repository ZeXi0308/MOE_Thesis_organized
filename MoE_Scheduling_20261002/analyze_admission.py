"""Actual six-arm admission runs; best static is a hindsight reference only."""
import argparse
import collections
import json
from pathlib import Path
from analyze_optimization import analyze_cell, equivalent, load

def controller_summary(raw):
    decisions, observations = raw.get('admission_decisions', []), raw.get('admission_observations', [])
    holds = [d for d in decisions if d.get('reason') in ('age_gate_hold', 'myopic_cost_hold')]
    used = [o for o in observations if o.get('used') is True]
    changed = [o for o in used if o.get('prior_s') != o.get('updated_s')]
    return dict(decisions=len(decisions), open_decisions=sum(d.get('admit_one') is True for d in decisions),
        hold_decisions=len(holds), fallback_decisions=sum(bool(d.get('fallback')) for d in decisions),
        reason_counts=dict(collections.Counter(d.get('reason', 'MISSING') for d in decisions)),
        fallback_counts=dict(collections.Counter(d['fallback'] for d in decisions if d.get('fallback'))),
        holds_with_running_prefill=sum(d.get('running_prefill_requests', 0) > 0 for d in holds),
        observations=len(observations), ewma_used_observations=len(used), ewma_changed_observations=len(changed),
        ewma_really_updated=bool(changed), ewma_updates_by_kind=dict(collections.Counter(o.get('kind') for o in used)),
        decisions_with_completed_signal=sum(d.get('signal_available_s') is not None for d in decisions),
        first_ewma_update=used[0] if used else None, last_ewma_update=used[-1] if used else None,
        frozen_training=raw.get('admission_training'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).parent / 'results_admission_r01')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    names = ('00_native16', '01_static2', '02_static4', '03_static8', '04_age_gate', '05_model')
    cells, outputs = {}, {}
    for name in names:
        directory = args.results / name
        cells[name], outputs[name] = analyze_cell(directory)
        raw = load(directory / 'raw.json')
        cells[name].update(preemption=raw.get('preemption_summary'), controller=controller_summary(raw))

    def ratio(candidate, baseline):
        a, b = cells[candidate], cells[baseline]
        if not all(c['complete_16_requests_512_tokens'] for c in (a, b)):
            return None
        return dict(candidate=candidate, baseline=baseline, wall_ratio=a['episode_wall_s'] / b['episode_wall_s'],
            mean_flow_ratio=a['flow']['mean_s'] / b['flow']['mean_s'],
            ttft_mean_ratio=a['ttft']['mean_s'] / b['ttft']['mean_s'],
            itl_mean_ratio=a['itl']['mean_s'] / b['itl']['mean_s'])

    statics = names[1:4]
    best_static = None
    if all(cells[n]['complete_16_requests_512_tokens'] for n in statics):
        wall = min(statics, key=lambda n: cells[n]['episode_wall_s'])
        flow = min(statics, key=lambda n: cells[n]['flow']['mean_s'])
        best_static = dict(scope='HINDSIGHT_STRONG_REFERENCE_NOT_AN_ONLINE_POLICY',
            candidates=list(statics), wall_selected_arm=wall, mean_flow_selected_arm=flow,
            model_vs_wall_selected=ratio(names[-1], wall), model_vs_mean_flow_selected=ratio(names[-1], flow))
    report = dict(evidence_type='INDEPENDENT_ACTUAL_EXECUTIONS_ONLY', order=list(names),
        group_status=load(args.results / 'group_status.json').get('status', 'MISSING'),
        all_six_complete=all(c['complete_16_requests_512_tokens'] for c in cells.values()),
        cells=cells, vs_native={n: ratio(n, names[0]) for n in names[1:]},
        model_vs_age_gate=ratio(names[-1], names[-2]), hindsight_best_static=best_static,
        output_equivalence_vs_native=[equivalent(n, names[0], outputs) for n in names[1:]],
        notes=['Every cell is an independently executed policy; no fixed-route replay or synthetic policy outcome.',
               'Best static is selected after observing all three actual static arms, separately by wall and mean flow; it is a hindsight strong reference, not an online baseline.',
               'Raw episode wall includes measurement-loop overhead and excludes engine initialization/warmup; no subtractions.',
               'Flow/TTFT start at workload arrival; ITL uses adjacent host-observed token times; quantiles use linear interpolation.',
               'Open/hold/fallback count controller decisions, not necessarily realized admissions; holds mean explicit age_gate_hold/myopic_cost_hold.',
               'EWMA used counts logged updates; changed counts require prior_s != updated_s, and both are reported.',
               'Copy bytes/groups and CUDA peak use the measurement scope; preemption is the native scheduler summary.',
               'The workload is 16 requests with 32 forced output tokens each; no natural-completion, quality or repeated-run confidence claim.',
               'Ratios below one favor the named candidate.'])
    destination = args.output or args.results / 'metrics.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(destination)
    for name, c in cells.items():
        print(name, c['status'], 'completed=', c['completed_requests'], 'tokens=', c['output_tokens'],
              'wall=', c['episode_wall_s'], 'mean_flow=', c['flow']['mean_s'])
    print(json.dumps(dict(model_vs_age_gate=report['model_vs_age_gate'], hindsight_best_static=best_static), indent=2))

if __name__ == '__main__':
    main()

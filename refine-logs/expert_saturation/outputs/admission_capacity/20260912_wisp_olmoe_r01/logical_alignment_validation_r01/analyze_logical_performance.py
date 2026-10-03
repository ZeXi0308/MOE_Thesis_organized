"""Six same-budget F/X/Y engines; all outcomes and costs, no inferential noise floor."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'instrumentation'))
import analyze_covered
import finite_metrics
from logical_execution import call_identity_bytes
from compile_domain import EXPECTED as INSTALLED


def read(path):
    return json.loads(path.read_text())


def check_logical(root, plan, entry, row):
    directory = root / 'results' / plan['label']
    report = read(directory / 'logical_execution.json')
    issues = row['issues']
    def check(ok, why):
        if not ok: issues.append(why)
    enabled = plan['logical_alignment']
    check(report['schema'] == 'logical_alignment_execution_v1' and report['status'] == 'COMPLETE'
          and report['logical_alignment'] is enabled and report['hooks_restored'] is True, 'logical execution terminal/flag/restoration')
    check(('--logical-alignment' in entry['command']) is enabled, 'logical flag command mismatch')
    if not enabled:
        check(not report['phases'], 'disabled arm executed logical rewrite')
        return
    check(report['installed_sources'] == INSTALLED, 'logical helper installed sources')
    trace = [json.loads(line) for line in (directory/'pager/calls.jsonl').read_text().splitlines()]
    phases = {record['context']['phase'] for record in trace}
    check(set(report['phases']) == phases, 'logical actual phase coverage')
    expected_parameters = dict(global_num_experts=64,map_shape=[64],top_k=8,
        incoming_ignore_invalid_experts=True,effective_ignore_invalid_experts=False)
    for phase in phases:
        records = [record for record in trace if record['context']['phase'] == phase]
        stats = report['phases'][phase]
        total_rows = sum(record['rows'] for record in records)
        check(stats['calls'] == stats['completed_calls'] == stats['helper_calls'] == len(records)
              and stats['failed_calls'] == 0 and stats['assignment_restored'] is True,
              'logical exactly one kernel/helper per original call: ' + phase)
        check(stats['token_rows'] == stats['helper_token_rows'] == total_rows
              and stats['rows_histogram'] == dict(Counter(str(record['rows']) for record in records)),
              'logical rows vs actual trace: ' + phase)
        check(stats['helper_parameters'] == expected_parameters, 'logical actual helper interface: ' + phase)
        rebuilt = hashlib.sha256(b''.join(call_identity_bytes(record) for record in records)).hexdigest()
        check(rebuilt == stats['call_identity_sha256'], 'logical ordered call identity: ' + phase)
    row['logical_execution'] = report


def analyze(root):
    plans, protocol = read(root/'run_cells.json'), read(root/'protocol.json')
    path = root/'results/execution.json'
    execution = read(path) if path.exists() else dict(status='UNRUN',cells=[])
    entries = {entry['label']:entry for entry in execution['cells']}
    rows, missing, issues = [], [], []
    expected = [(f'{i}_{arm}',arm,'fullstage' if arm=='F' else 'oneshot',arm=='Y')
                for i,arm in enumerate(('Y','X','F','F','X','Y'))]
    if [(p['label'],p['arm'],p['mode'],p['logical_alignment']) for p in plans] != expected:
        issues.append('plan differs from frozen Y/X/F/F/X/Y')
    if len(entries) != len(execution['cells']) or set(entries) - {p['label'] for p in plans}:
        issues.append('duplicate or undeclared result cells')
    work = read(root/'prepared/workload.json')
    if (len(work['source_requests']) != 16 or len(work['actual_prompt_token_ids']) != 16
        or len({r['request_id'] for r in work['source_requests']}) != 16
        or any(len(tokens) != 128 for tokens in work['actual_prompt_token_ids'])):
        issues.append('prepared input schema/count')
    for plan in plans:
        if entries.get(plan['label'],{}).get('status') != 'COMPLETE':
            missing.append(plan['label']); continue
        try:
            entry = entries[plan['label']]
            row = analyze_covered.analyze_cell(root,plan,entry,root/'prepared/workload.json',work['source_requests'][0]['document_id'])
            row.update(arm=plan['arm'],block=plan['block'],logical_alignment=plan['logical_alignment'])
            raw = read(root/'results'/plan['label']/'raw.json')
            if raw.get('cpu_diagnostics') is not False or raw.get('runtime_observer_enabled') is not False:
                row['issues'].append('unexpected CPU/runtime observer')
            if sum(str(x).endswith('/instrumentation/run_covered.py') for x in entry['command']) != 1:
                row['issues'].append('wrong execution wrapper')
            check_logical(root,plan,entry,row)
            rows.append(row); issues.extend(plan['label']+': '+item for item in row['issues'])
        except Exception as exc:
            issues.append(plan['label']+': '+repr(exc))
    pairs, repeats, mean_comparisons, means = [], [], [], {}
    by_label = {row['label']:row for row in rows}
    for block, labels in [('forward',('2_F','1_X','0_Y')),('reverse',('3_F','4_X','5_Y'))]:
        for a,b in ((labels[0],labels[2]),(labels[1],labels[2]),(labels[0],labels[1])):
            if a in by_label and b in by_label:
                item = finite_metrics.comparison(by_label[a],by_label[b]); item['block'] = block; pairs.append(item)
    for a,b in (('2_F','3_F'),('1_X','4_X'),('0_Y','5_Y')):
        if a in by_label and b in by_label: repeats.append(finite_metrics.comparison(by_label[a],by_label[b]))
    for arm in ('F','X','Y'):
        selected = [r for r in rows if r['arm']==arm]
        if len(selected)==2:
            means[arm] = {key:statistics.mean(r['metrics'][key] for r in selected) for key in selected[0]['metrics']}
    for a,b in (('F','Y'),('X','Y'),('F','X')):
        if a in means and b in means:
            mean_comparisons.append(dict(baseline=a,target=b,delta_pct={k:100*(means[b][k]/v-1) if v else None for k,v in means[a].items()}))
    done = execution['status']=='COMPLETE' and len(rows)==6 and len(pairs)==6 and not missing and not issues
    return dict(status='DESCRIPTIVE_LOGICAL_ALIGNMENT_FXY' if done else 'INVALID_ANALYSIS_INPUT' if issues else 'UNRUN' if not entries else 'INCOMPLETE',
        issues=issues,missing_cells=missing,cells=rows,paired_comparisons=pairs,same_arm_repeats=repeats,engine_metric_means=means,
        mean_comparisons=mean_comparisons,research_question=protocol['question'],primary='capture_wall_s',
        claim_ceiling='One new B-line16-document cohort, two engines per arm in mirrored order. All costs and adverse metrics retained; no statistical noise floor, significance, quality, SLO, stable benefit or method GO.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT); parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(); result = analyze(args.input_dir.resolve())
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','issues','missing_cells')}))

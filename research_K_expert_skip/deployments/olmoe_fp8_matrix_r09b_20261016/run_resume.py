#!/usr/bin/env python3
"""Resume r09's full FP8 matrix without rerunning successful scientific stages.

All execution belongs to run_locked. Plan-only reads local inputs and emits JSON.
The original padded failures remain failures; only exact seven-B graph checks
can release the new performance stage. The original r09 executor owns timeouts.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[1]
BATCHES = [1, 2, 4, 8, 16, 32, 64]


def native_padded_diagnostic(root, output, config, base):
    raw = output.with_name('native_padded_strict.json')
    command = [sys.executable, str(HERE/'smoke_graph.py'), '--model', config['model'],
        '--dtype', 'bfloat16', '--quantization', 'fp8', '--gpu-memory-utilization', '.35',
        '--spec-json', 'null', '--no-count-routes', '--output', str(raw)]
    if output.exists() or raw.exists():
        raise FileExistsError(output)
    result = subprocess.run(command, cwd=root, check=False)
    report = {'status': 'FAILED', 'command': command, 'strict_exit_code': result.returncode,
              'scope': 'Native padded B3->4/B5->8 observation; not a successful parity assertion'}
    try:
        strict = json.loads(raw.read_text())
        report.update(strict_status=strict.get('status'), strict_error=strict.get('error'),
                      strict_artifact=str(raw), strict_sha256=base.sha(raw))
        children = {kind: json.loads(Path(path).read_text()) for kind, path in strict['children'].items()}
        if set(children) != {'eager', 'graph'} or any(r.get('status') != 'PASS' for r in children.values()):
            raise RuntimeError('Native padded diagnostic child failed; not a measured parity difference')
        erows, grows = children['eager']['rows'], children['graph']['rows']
        if [r['batch'] for r in erows] != [3, 5] or [r['batch'] for r in grows] != [3, 5]:
            raise RuntimeError('Incomplete native padded batch coverage')
        equal = all(e['outputs'] == g['outputs'] for e, g in zip(erows, grows, strict=True))
        accepted_identity = equal and result.returncode == 0 and strict['status'] == 'PASS'
        accepted_difference = (not equal and result.returncode != 0 and strict['status'] == 'FAILED'
            and 'eager/graph deterministic outputs differ' in strict.get('error', ''))
        if not (accepted_identity or accepted_difference):
            raise RuntimeError('Diagnostic failure is not the expected native output comparison')
        report.update(status='DIAGNOSTIC_COMPLETE', observation='IDENTICAL' if equal else 'DIFFERENT',
            native_padded_parity_passed=accepted_identity,
            children={kind: {'path': path, 'sha256': base.sha(Path(path)), 'status': children[kind]['status']}
                      for kind, path in strict['children'].items()})
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        report['error'] = f'{type(error).__name__}: {error}'
    base.save(output, report)
    return 0 if report['status'] == 'DIAGNOSTIC_COMPLETE' else 1


def source_evidence(source, config, strict, base):
    records, warnings = {}, []
    manifest_path = source/'SHA256.json'
    original_hashes = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if strict and not original_hashes:
        raise ValueError('r09 final SHA256.json must exist before recovery execution')

    def read(relative, required=True):
        path = source/relative
        if not path.is_file():
            if strict and required:
                raise FileNotFoundError(path)
            warnings.append(f'Planning snapshot does not contain {relative}; execution rechecks final source')
            return {}
        digest = base.sha(path)
        if (strict or original_hashes) and original_hashes.get(relative) != digest:
            raise ValueError(f'r09 source SHA mismatch or missing original hash: {relative}')
        row = json.loads(path.read_text())
        records[relative] = {'path': str(path), 'sha256': digest, 'status': row.get('status', row.get('state'))}
        return row

    status, stages = read('status.json'), read('stages.json')
    if strict:
        if status.get('state') not in ('COMPLETE', 'FAILED') or stages.get('status') not in ('COMPLETE', 'INCOMPLETE', 'INTERRUPTED'):
            raise ValueError('r09 must be terminal; never read a running predecessor as reusable')
        for key in ('controller_pid', 'worker_pid'):
            pid = status.get(key)
            if not isinstance(pid, int):
                raise ValueError(f'Missing r09 {key}')
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f'r09 {key} is still alive: {pid}')
    stage_status = {r['name']: r['status'] for r in stages.get('stages', [])}
    trace = read('native_traces/status.json')
    if trace.get('status') != 'COMPLETE' or trace.get('batches') != BATCHES:
        raise ValueError('All seven r09 native traces must already be complete')
    for batch in BATCHES:
        read(f'native_traces/trace_native_b{batch}/snapshot.json')
    tune = read('native_tuning/summary.json')
    base.check_artifact({'name':'native_tuning', 'artifact':str(source/'native_tuning/summary.json'),
        'expected_artifact_status':'PASS'}, {'config':config})
    for relative in ('plan.json', 'weights.json', 'fp8_conversion.json', 'tuned_configs/'+tune['config_filename']):
        read('native_tuning/'+relative)
    if strict and any(stage_status.get(name) != 'COMPLETE' for name in ('native_traces', 'native_tuning')):
        raise ValueError('r09 controller did not accept trace/tuning success')
    failed = {}
    for name in ('graph_dense', 'graph_marginal'):
        row = read(name+'.json')
        failed[name] = {'status':row.get('status'), 'error':row.get('error'), 'artifact':str(source/(name+'.json'))}
    reused, needs_quality = [], []
    for batch in config['quality']['batches']:
        name = f'quality_b{batch}'; row = read(name+'.json', required=False)
        accepted = row.get('status') == 'PASS' and (not stages or stage_status.get(name) == 'COMPLETE')
        if accepted:
            if row.get('selected_documents') != 32 or len(row.get('quality_by_arm', {})) != 10 or any(q['tokens'] != 1024 for q in row['quality_by_arm'].values()):
                raise ValueError(f'{name} does not contain the complete expected quality population')
            reused.append(name)
        else:
            needs_quality.append(name)
    return {'source_run':str(source), 'state':status.get('state'), 'stages_status':stages.get('status'),
        'source_manifest':{'path':str(manifest_path), 'sha256':base.sha(manifest_path) if manifest_path.is_file() else None},
        'files':records, 'reused_stages':['native_traces','native_tuning',*reused],
        'prior_padded_failures_preserved':failed, 'quality_to_run':needs_quality,
        'warnings':warnings}, stages


def make_plan(root, output, config, strict, base):
    source = base.absolute(root, config['source_run'])
    evidence, old = source_evidence(source, config, strict, base)
    plan = base.build_plan(root, output, config, 7170)
    if old:
        for name, value in plan['input_files'].items():
            if old.get('input_files', {}).get(name, {}).get('sha256') != value['sha256']:
                raise ValueError(f'Recovery input changed relative to r09: {name}')
    stages = [s for s in plan['stages'] if s['name'] not in evidence['reused_stages']]
    tuned = source/'native_tuning/tuned_configs'
    diagnostic = {'name':'native_padded_diagnostic',
        'command':[sys.executable,str(Path(__file__).resolve()),'--native-padded-diagnostic',
                   '--root',str(root),'--output',str(output/'native_padded_diagnostic.json')],
        'dependencies':[], 'timeout_cap_seconds':config['stage_timeout_seconds']['native_padded_diagnostic'],
        'tuned_config_folder':str(tuned),'artifact':str(output/'native_padded_diagnostic.json'),
        'expected_artifact_status':'DIAGNOSTIC_COMPLETE','log':str(output/'stage_logs/native_padded_diagnostic.log'),
        'status':'NOT_STARTED','started_unix':None,'ended_unix':None,'exit_code':None}
    for stage in stages:
        stage['tuned_config_folder'] = str(tuned)
        stage['dependencies'] = [d for d in stage['dependencies'] if d not in evidence['reused_stages']]
        if stage['name'] in ('graph_dense','graph_marginal'):
            stage['command'][1] = str(HERE/'smoke_graph.py')
            stage['command'] += ['--batches', *map(str,BATCHES)]
            stage['dependencies'] = ['native_padded_diagnostic']
    rolling = next(s for s in stages if s['name']=='rolling_path')
    tail = copy.deepcopy(rolling); tail['name']='rolling_tail'
    tail['artifact']=str(output/'quality_rolling_tail.json')
    tail['log']=str(output/'stage_logs/rolling_tail.log')
    tail['timeout_cap_seconds']=config['stage_timeout_seconds']['rolling_tail']
    cmd=tail['command']; cmd[cmd.index('--tokenized-windows')+1]=str(HERE/'rolling_tail/manifest.json')
    index=cmd.index('--limit-windows'); del cmd[index:index+2]
    cmd[cmd.index('--output')+1]=tail['artifact']
    stages.insert(stages.index(rolling)+1,tail)
    plan.update(stages=[diagnostic,*stages],reuse=evidence,trace_files=[],
        scope='Complete fixed seven-B online-FP8 matrix only after exact no-padding parity. Prior padded failures remain unresolved and are not overwritten.',
        diagnostic_policy='Native padded output difference is measured and kept as strict FAILED; DIAGNOSTIC_COMPLETE means both executions succeeded, not parity passed. Other diagnostic failures block graph verification.',
        quality_policy='Every accepted r09 quality B is reused by original SHA/status. Only unsuccessful or absent B is rescheduled, never a successful one.')
    plan['input_files']['tail_manifest']={'path':str(HERE/'rolling_tail/manifest.json'),'sha256':base.sha(HERE/'rolling_tail/manifest.json')}
    plan['input_files']['smoke_graph']={'path':str(HERE/'smoke_graph.py'),'sha256':base.sha(HERE/'smoke_graph.py')}
    return plan


def check_extra(stage, plan, ordinary_check):
    ordinary_check(stage,plan)
    row=json.loads(Path(stage['artifact']).read_text())
    if stage['name'] in ('graph_dense','graph_marginal'):
        if row.get('args',{}).get('batches') != BATCHES:
            raise RuntimeError('Graph verification did not cover all seven exact batches')
        if stage['name']=='graph_dense' and not all(row.get('native_dense_'+mode+'_identity_checked') for mode in ('eager','graph')):
            raise RuntimeError('Native/dense identity was not checked in both execution modes')
        for kind,path in row['children'].items():
            child=json.loads(Path(path).read_text())
            if child.get('status')!='PASS' or [r['batch'] for r in child['rows']]!=BATCHES:
                raise RuntimeError('Incomplete no-padding graph child')
            if kind != 'eager' and any(r['batch']!=r['target_bucket'] for r in child['rows']):
                raise RuntimeError('Unexpected graph padding in the exact-batch verification')
    if stage['name'] in ('rolling_path','rolling_tail'):
        expected=1152 if stage['name']=='rolling_path' else 65
        population=9 if stage['name']=='rolling_path' else 1
        if row.get('selected_documents')!=population or len(row.get('quality_by_arm',{}))!=10 or any(q['tokens']!=expected for q in row['quality_by_arm'].values()):
            raise RuntimeError('Rolling path did not score the expected real input population')
        if not row.get('rolling_coverage_verified',{}).get('all_scored_predictions_decode'):
            raise RuntimeError('Rolling predictions were not verified as real decode')


def main():
    started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--output',type=Path,default=Path('runs/olmoe_fp8_matrix_r09b'))
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--native-padded-diagnostic',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();root=args.root.expanduser().resolve()
    sys.path.insert(0,str(root/'code'))
    import run_olmoe_fp8_r09 as base
    config=json.loads((HERE/'r09b.json').read_text())
    output=base.absolute(root,args.output)
    if args.native_padded_diagnostic:
        return native_padded_diagnostic(root,output,config,base)
    plan=make_plan(root,output,config,not args.plan_only,base)
    plan['config_source']={'path':str(HERE/'r09b.json'),'sha256':base.sha(HERE/'r09b.json')}
    if args.plan_only:
        print(json.dumps(plan,indent=2));return 0
    if (output/'stages.json').exists():
        raise FileExistsError('Choose a new run directory; r09 and earlier recovery evidence are preserved')
    plan['outer_run']=base.verify_outer(SimpleNamespace(outer_run_dir=output,budget_seconds=7170))
    ordinary_check=base.check_artifact
    base.check_artifact=lambda stage,current:check_extra(stage,current,ordinary_check)
    return base.execute(root,output,plan,started)


if __name__=='__main__':
    raise SystemExit(main())

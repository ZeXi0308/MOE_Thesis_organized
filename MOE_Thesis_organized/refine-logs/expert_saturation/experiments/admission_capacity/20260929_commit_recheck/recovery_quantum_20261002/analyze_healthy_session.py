#!/usr/bin/env python3
"""One offline command for healthy-session quality, endings, and full service.

Reads cell-NN-ARM/archive (or output before archival), fixed reference answers,
and a local pinned tokenizer. It never edits raw files or replaces a report.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'healthy_workload'))
from evaluate_quality import evaluate, compare, distribution, digest
from tokenizer_backend import OfflineTokenizer
from analyze_service_session import read_cell
from evaluate_goodput import pair as service_pair


def read(path):
    return json.loads(path.read_text())


def load_inputs(folder):
    config, workload = read(folder/'config.json'), read(folder/'workload.json')
    if hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest() != config['workload_sha256']:
        raise ValueError('Prepared workload identity changed')
    healthy = ROOT/'healthy_workload'
    if (digest(healthy/'reference_answers.json') != config['reference_answers_sha256']
            or digest(healthy/'tasks_text.json') != config['tasks_text_sha256']):
        raise ValueError('Question/reference identity changed')
    if workload['sampling'] != dict(ignore_eos=False,min_tokens=0,max_tokens=config['output_tokens'],
                                    temperature=0.0,stop=[],stop_token_ids=[]):
        raise ValueError('Expected natural EOS with no extra stop rule')
    return config, workload


def analyze(args):
    directories = sorted(args.session.glob('cell-[0-9][0-9]-*'))
    if not directories:
        raise ValueError('No cell-NN-ARM directories found')
    inputs = args.inputs or directories[0]/'selected-inputs/inputs'
    config, workload = load_inputs(inputs)
    if len(workload['source_requests']) != args.expected_requests:
        raise ValueError('Prepared task count differs from expected cohort')
    hashes = {name:digest(args.tokenizer_dir/name) for name in config['tokenizer_files_sha256']}
    if hashes != config['tokenizer_files_sha256']:
        raise ValueError('Tokenizer does not match prepared model revision')
    tokenizer = OfflineTokenizer(args.tokenizer_dir)
    references = {r['request_id']:r for r in read(ROOT/'healthy_workload/reference_answers.json')['references']}
    cells = {}
    for directory in directories:
        arm = directory.name[8:]
        if arm in cells:
            raise ValueError('Duplicate arm name: '+arm)
        archive = directory/'archive' if (directory/'archive').exists() else directory/'output'
        cell = dict(directory=str(directory),archive=str(archive),status='NO_MEASUREMENT')
        cells[arm] = cell
        if not (archive/'raw.json').is_file():
            continue
        try:
            # Parse full documents before computing metrics. A copied partial
            # JSON/EOF is visible as UNREADABLE_JSON, never a shortened cohort.
            readable = {}
            for filename in ('raw.json','config.json','status.json','selective-store.json',
                             'timing.json','resolved-eos.json'):
                path=archive/filename
                if path.exists():
                    value=read(path)
                    readable[filename]=dict(bytes=path.stat().st_size,sha256=digest(path))
                    if filename=='resolved-eos.json':cell['resolved_eos']=value
            cell['fully_readable_json_files']=readable
            quality=evaluate(archive,workload,config,references,tokenizer)
            service,_=read_cell(directory,args.expected_requests,args.timeout_s)
            rows=quality['per_request']
            quality['cap_correct_count']=sum(r['strict_correct'] and r['cap_truncated'] for r in rows)
            quality['strict_answer_parsed_count']=sum(r['answer'] is not None for r in rows)
            quality['output_token_count_distribution']=distribution([r['output_tokens'] for r in rows])
            quality['native_stop_identifier_counts']=dict(Counter(
                json.dumps(r['stop_identifier'],ensure_ascii=False) if r['stop_identifier_available']
                else 'NOT_RECORDED' for r in rows))
            quality['native_stop_identifier_field_count']=sum(r['stop_identifier_available'] for r in rows)
            cell.update(status='COMPLETE' if service['status']=='COMPLETE' and quality['all_complete'] else 'INCOMPLETE',
                        quality=quality,service=service)
        except json.JSONDecodeError as error:
            cell.update(status='UNREADABLE_JSON',error=str(error),error_line=error.lineno,error_column=error.colno)
        except (OSError,ValueError,KeyError,TypeError) as error:
            cell.update(status='ANALYSIS_ERROR',error=f'{type(error).__name__}: {error}')
    comparisons={}
    reference=cells.get(args.reference)
    if reference and 'quality' in reference:
        for arm,cell in cells.items():
            if arm==args.reference or 'quality' not in cell:
                continue
            entry=dict(candidate=arm,reference=args.reference,
                       quality=compare(cell['quality'],reference['quality']))
            entry['quality']['direction']='left=candidate, right=reference; accuracy_delta=candidate-reference'
            if 'metrics' in cell['service'] and 'metrics' in reference['service']:
                entry['service']=service_pair(reference['service']['metrics'],cell['service']['metrics'])
            comparisons[arm+'_vs_'+args.reference]=entry
    missing=sorted(set(args.expected_arms)-set(cells))
    complete=not missing and all(c['status']=='COMPLETE' for c in cells.values())
    return dict(schema='a-healthy-session-quality-service-v1',
                status='ALL_CELLS_COMPLETE' if complete else 'PARTIAL_OR_INVALID',
                session=str(args.session),inputs=str(inputs),expected_requests=args.expected_requests,
                expected_arms=args.expected_arms,missing_arms=missing,reference=args.reference,
                source_receipt_status=read(args.session/'receipt.json').get('status') if (args.session/'receipt.json').exists() else None,
                workload_sha256=config['workload_sha256'],model=config['model'],tokenizer_files_sha256=hashes,
                cells=cells,comparisons=comparisons,
                notes=['All planned requests remain in quality denominators; incomplete/missing score zero.',
                       'Primary score requires an explicit numeric answer marker. Historical last-number string match is separate.',
                       'cap_correct_count counts correct truncated responses; correct_and_natural_stop excludes them.',
                       'Natural stop is EOS-compatible under the empty extra-stop contract; actual native_stop_reason, including explicit null, is preserved per request.',
                       'Output counts/rates use actual returned tokens and full episode wall; flow begins at planned arrival.',
                       'Service goodput thresholds remain the established development grid, not production SLOs.',
                       'Host output-gap distributions exclude TTFT and do not interpolate within chunks.',
                       'Independent scheduler trajectories may alter output length/content. No equal-work speedup, quality-equivalence, action-causality, or novelty inference.',
                       'Lease action qualification is separate from this quality/service report.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--inputs',type=Path,help='Default: first cell selected-inputs/inputs')
    p.add_argument('--tokenizer-dir',type=Path,required=True)
    p.add_argument('--expected-requests',type=int,default=16)
    p.add_argument('--expected-arms',nargs='+',default=['q1','fixed4','adaptive'])
    p.add_argument('--reference',default='q1')
    p.add_argument('--timeout-s',type=float,default=180.)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Refusing to replace an existing report')
    result=analyze(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:json.dump(result,out,indent=2,ensure_ascii=False,allow_nan=False);out.write('\n')
    print(args.output)
    for arm,cell in result['cells'].items():
        quality=cell.get('quality',{})
        print(arm,cell['status'],'accuracy=',quality.get('strict_accuracy'),
              'natural_stop=',quality.get('natural_stop_count'),'cap=',quality.get('cap_truncated_count'))


if __name__=='__main__':
    main()

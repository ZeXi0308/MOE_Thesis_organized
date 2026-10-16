#!/usr/bin/env python3
"""Full-population OLMoE BF16 MC identity controls via the existing task runner.

Select one family per locked run; default all ten existing quality_pareto arms.
An explicit native-first subset changes arms only, never the task population.
Standard MC loglikelihood executes prefill only: this is not decode task quality.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POPULATION = {'mmlu': 14042, 'hellaswag': 10042, 'arc_challenge': 1172}
SHOTS = {'mmlu': 5, 'hellaswag': 0, 'arc_challenge': 0}
ARM_NAMES = ['native', 'dense', 'global_j1', 'global_j2', 'global_j4',
             'gate_low', 'gate_high', 'marginal_low', 'marginal_mid', 'marginal_high']


def verify_population(summary, folder, names, family):
    """Use existing harness results; ensure complete effective/original counts."""
    if summary.get('status') != 'COMPLETE' or len(summary['evaluations']) != len(names):
        raise ValueError('Family/arm sequence did not complete')
    if [row['arm'] for row in summary['evaluations']] != names:
        raise ValueError('Evaluated arm sequence differs from selected set')
    for row in summary['evaluations']:
        if row['task'] != family or not row['control_only']:
            raise ValueError('Expected prefill-only MC identity control')
        metrics = row['sample_weighted_metrics']
        for metric in (['acc,none'] if family == 'mmlu' else ['acc,none', 'acc_norm,none']):
            if metrics[metric]['sample_rows'] != POPULATION[family]:
                raise ValueError('A reported metric lacks the complete task population')
        result = json.loads((folder/row['file']).read_text())
        counts = result['n-samples']
        if (result['config']['limit'] is not None or
                any(n['effective'] != n['original'] for n in counts.values()) or
                sum(n['effective'] for n in counts.values()) != POPULATION[family] or
                len(counts) != (57 if family == 'mmlu' else 1)):
            raise ValueError('Task leaves were omitted or truncated')
        if any(n != SHOTS[family] for n in result['n-shot'].values()):
            raise ValueError('Official shot count changed')
        if any(len(result['samples'][leaf]) != n['effective'] or
               len({row['doc_id'] for row in result['samples'][leaf]}) != n['effective']
               for leaf, n in counts.items()):
            raise ValueError('Raw per-document result population is incomplete or duplicated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--model', type=Path, default=Path('/root/autodl-tmp/moe-research-20261002/model'))
    parser.add_argument('--family', choices=POPULATION, required=True)
    parser.add_argument('--arms', nargs='+', choices=ARM_NAMES, default=ARM_NAMES)
    parser.add_argument('--budget-seconds', type=int, required=True, help='Explicit internal cap, at most14370; outer timeout at least30s longer')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--outer-run-dir', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if (args.arms[0] != 'native' or len(set(args.arms)) != len(args.arms)
            or not 30 < args.budget_seconds <= 14370):
        parser.error('Unique native-first arms and an internal budget31..14370 are required')
    args.root = args.root.expanduser().resolve()
    sys.path.insert(0, str(args.root/'code'))
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    import run_task_quality as runner
    for key in ('model', 'output', 'outer_run_dir'):
        if getattr(args, key) is not None:
            setattr(args, key, runner.resolved(getattr(args, key), args.root))
    args.data_dir = args.root/'datasets/task_eval_r02'
    args.prompts = args.root/'datasets/free_generation_validation_r01/prompts.jsonl'
    args.arms_file = args.root/'configs/quality_pareto.json'
    args.formats = ['bf16']; args.tasks = [args.family]; args.limit = None
    args.free_limit = 200; args.prompt_tokens = 128; args.batch = 8
    args.max_tokens = 256; args.max_model_len = 4096; args.seed = 1701
    args.gpu_memory_utilization = .35; args.job_timeout = args.budget_seconds-10
    args.per_arm_isolation = False
    lengths_path = args.data_dir/'context_lengths_olmoe_full.json'
    lengths = json.loads(lengths_path.read_text())
    protocol, checked = lengths['protocol'], lengths['families'][args.family]
    if (lengths['status'] != 'FULL_INPUT_LENGTHS_CHECKED' or protocol['limit_per_task_leaf'] is not None
            or protocol['seed'] != 1701 or protocol['shots'][args.family] != SHOTS[args.family]
            or protocol['max_model_len'] != 4096 or protocol['apply_chat_template']
            or checked['documents'] != POPULATION[args.family] or checked['over_requests'] != 0):
        raise ValueError('Full-context preparation does not match this fixed protocol')
    full_arms, provenance, warnings = runner.inputs(args)
    if [entry['name'] for entry in full_arms['arms']] != ARM_NAMES:
        raise ValueError('Existing ten-arm quality_pareto set changed')
    if provenance['dataset_source']['sha256'] != lengths['dataset_source']['sha256']:
        raise ValueError('Task source changed since full request preparation')
    if 'model_config' in provenance and provenance['model_config']['sha256'] != lengths['tokenizer']['files']['config.json']['sha256']:
        raise ValueError('This entry requires the checked OLMoE checkpoint')
    lookup = {entry['name']: entry for entry in full_arms['arms']}
    selected = {'purpose': 'Full-population prefill MC identity control; explicit selected arm set',
                'arms': [lookup[name] for name in args.arms]}
    provenance['full_context_lengths'] = {'path': str(lengths_path), 'sha256': runner.sha(lengths_path)}
    provenance['deployment_entry'] = {'path': str(Path(__file__)), 'sha256': runner.sha(Path(__file__))}
    plan = runner.build_plan(args, selected, provenance, warnings)
    plan['jobs'] = [job for job in plan['jobs'] if job['unit'] == args.family]
    assert len(plan['jobs']) == 1 and '--limit' not in plan['jobs'][0]['command']
    plan.update(expected_documents_per_arm=POPULATION[args.family], expected_requests_per_arm=checked['requests'],
                exact_selected_arms=args.arms, full_ten_arm_matrix=set(args.arms) == set(ARM_NAMES),
                total_document_arm_evaluations=POPULATION[args.family]*len(args.arms))
    plan['semantics'].update(
        generation='No generation task/free job is run. max_tokens256 is an unused common CLI setting for MC.',
        mc='Official MMLU5-shot/HellaSwag0-shot/ARC-c0-shot prompt loglikelihood. All arms have zero decode and are prefill identity controls, not evidence of decode quality preservation.',
        limit='No --limit; every selected arm evaluates the full chosen family. The arm subset is recorded separately and is not called a full ten-arm matrix.',
        statistics='Same seed1701 and official samples/prompts. Existing runner checks native/candidate response identity. Sample stderr is not independent system-run uncertainty.',
        failure='Keep all raw files and completed arms. One family per fresh output permits independent family reruns without repeating successful families; no automatic retry or partial-arm reuse.')
    if not args.plan_only:
        plan['outer_run'] = runner.verify_outer(args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'arms').mkdir()
    runner.save(args.output/'arms/all_arms.json', plan['arm_files']['all_arms'])
    runner.save(args.output/'suite.json', plan)
    print(json.dumps({'status': 'PLANNED', 'family': args.family, 'arms': args.arms,
        'documents_per_arm': POPULATION[args.family], 'warnings': warnings}), flush=True)
    if args.plan_only:
        return 0
    code = runner.execute(args, plan)
    if code == 0:
        job = plan['jobs'][0]; folder = Path(job['output'])
        try:
            verify_population(json.loads((folder/'summary.json').read_text()), folder, args.arms, args.family)
            plan['full_population_verified'] = True
        except (OSError, ValueError, KeyError) as error:
            plan['status'] = 'INCOMPLETE'; code = 1
            job.update(status='FAILED', error=f'Full-population validation: {error}')
        runner.save(args.output/'suite.json', plan)
    print(json.dumps({'status': plan['status'], 'manifest': str(args.output/'suite.json')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())

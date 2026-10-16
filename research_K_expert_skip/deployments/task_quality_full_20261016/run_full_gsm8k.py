#!/usr/bin/env python3
"""Full OLMoE BF16 GSM8K deployment; reuse the existing bounded suite runner.

This local deployment selects only the GSM8K job from run_task_quality's plan.
It never edits live code, lowers the sample population, or runs free generation.
Execution must be a direct run_locked child. --plan-only is CPU/stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--model', type=Path, default=Path('/root/autodl-tmp/moe-research-20261002/model'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--outer-run-dir', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    sys.path.insert(0, str(args.root/'code'))
    # Set offline mode before any possible downstream HF/dataset imports.
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    import run_task_quality as runner

    args.model = runner.resolved(args.model, args.root)
    args.output = runner.resolved(args.output, args.root)
    if args.outer_run_dir:
        args.outer_run_dir = runner.resolved(args.outer_run_dir, args.root)
    args.data_dir = args.root/'datasets/task_eval_r02'
    # Required by the common input validator; this deployment drops its free job.
    args.prompts = args.root/'datasets/free_generation_validation_r01/prompts.jsonl'
    args.arms_file = HERE/'arms.json'
    args.formats = ['bf16']; args.tasks = ['gsm8k']; args.limit = None
    args.free_limit = 200; args.prompt_tokens = 128
    args.batch = 8; args.max_tokens = 256; args.max_model_len = 4096
    args.gpu_memory_utilization = .35; args.seed = 1701
    args.budget_seconds = 7080; args.job_timeout = 7070
    args.per_arm_isolation = False

    lengths_path = args.data_dir/'context_lengths_olmoe_full.json'
    lengths = json.loads(lengths_path.read_text())
    gsm = lengths['families']['gsm8k']; protocol = lengths['protocol']
    if (lengths['status'] != 'FULL_INPUT_LENGTHS_CHECKED'
            or protocol['limit_per_task_leaf'] is not None or protocol['seed'] != 1701
            or protocol['shots']['gsm8k'] != 5 or protocol['generation_max_tokens'] != 256
            or protocol['max_model_len'] != 4096 or gsm['documents'] != 1319
            or gsm['requests'] != 1319 or gsm['over_requests'] != 0):
        raise ValueError('This deployment requires the complete checked 1319-question protocol')
    arms, provenance, warnings = runner.inputs(args)
    if provenance['dataset_source']['sha256'] != lengths['dataset_source']['sha256']:
        raise ValueError('Task source changed since the complete CPU request check')
    if 'model_config' in provenance and provenance['model_config']['sha256'] != lengths['tokenizer']['files']['config.json']['sha256']:
        raise ValueError('This deployment is restricted to the checked OLMoE checkpoint')
    provenance['full_context_lengths'] = {'path': str(lengths_path), 'sha256': runner.sha(lengths_path)}
    provenance['deployment_entry'] = {'path': str(Path(__file__).resolve()), 'sha256': runner.sha(Path(__file__))}
    plan = runner.build_plan(args, arms, provenance, warnings)
    plan['jobs'] = [job for job in plan['jobs'] if job['unit'] == 'gsm8k']
    assert len(plan['jobs']) == 1
    assert '--limit' not in plan['jobs'][0]['command']
    plan['expected_documents_per_arm'] = 1319
    plan['total_planned_completions'] = 1319*len(arms['arms'])
    plan['semantics'].update(
        generation='Only full official GSM8K 5-shot, 1319 test questions per arm, seed1701, no chat template, greedy, natural EOS/task stop strings, output cap256; first prediction is prefill.',
        limit='No task --limit is passed. All arms use the complete same 1319-question population; no filtering or subsampling for budget.',
        native_reuse='One model load and one full native reference shared by all 11 arms. Dense checks identical generated responses; every candidate checks identical document/prompt/target signatures.',
        failure='7070s job timeout within 7080s suite budget; timed-out raw outputs and accepted complete arms remain. The incomplete arm is not assigned an accuracy. A new run is required for continuation; no automatic rerun or sample reduction.',
        statistics='Single run, paired document-level task accuracy; strict-match and flexible-extract retained separately, sample stderr is not a system-run CI.')
    if not args.plan_only:
        plan['outer_run'] = runner.verify_outer(args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'arms').mkdir()
    for tag, value in plan['arm_files'].items():
        runner.save(args.output/'arms'/f'{tag}.json', value)
    runner.save(args.output/'suite.json', plan)
    print(json.dumps({'status': 'PLANNED', 'jobs': len(plan['jobs']),
        'arms': len(arms['arms']), 'questions_per_arm': 1319,
        'manifest': str(args.output/'suite.json'), 'warnings': warnings}), flush=True)
    if args.plan_only:
        return 0
    code = runner.execute(args, plan)
    if code == 0:
        job = plan['jobs'][0]
        summary = json.loads((Path(job['output'])/'summary.json').read_text())
        entries = summary.get('evaluations', [])
        accepted = {row['arm'] for row in entries}
        valid = accepted == set(job['arms']) and all(
            row.get('sample_weighted_metrics', {}).get('exact_match,'+filter_name, {}).get('sample_rows') == 1319
            for row in entries for filter_name in ('strict-match', 'flexible-extract'))
        if not valid:
            plan['status'] = 'INCOMPLETE'
            job.update(status='FAILED', error='Full 1319-question metrics missing for one or more arms/filters')
            code = 1
        else:
            plan['full_population_verified'] = True
        runner.save(args.output/'suite.json', plan)
    print(json.dumps({'status': plan['status'], 'manifest': str(args.output/'suite.json')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())

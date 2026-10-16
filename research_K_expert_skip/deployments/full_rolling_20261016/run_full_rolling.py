#!/usr/bin/env python3
"""Complete OLMoE test rolling quality, B8/seven arms, three resumable chunks.

This deployment only invokes the unmodified quality_decode CLI. A checkpoint is
one entire chunk/all arms. Failed attempts remain; explicit --resume restarts
the whole incomplete chunk. Chunks preserve the original B8 cohort boundaries
and are parts of one population, not independent experimental repetitions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT = 'datasets/wikitext2_rolling_olmoe_r01/test/manifest.json'
NAMES = ['native', 'dense', 'global_j2', 'gate_mid', 'gate_high', 'marginal_mid', 'marginal_high']
TUNING = {'bf16': 'runs/characterize_r06/native_tuning',
          'fp8': 'runs/olmoe_fp8_matrix_r09/native_tuning'}
MODEL_SHA = '42ea6f5fe31e9243290ba3c348a4ad615b88a6fbc6e10544e21a87b6dff5da6c'


def initialize(root):
    global rolling, quality, common
    sys.path.insert(0, str(root / 'code'))
    import prepare_wikitext_rolling as rolling
    import quality_decode as quality
    import run_task_quality as common


def prepare_chunks(root):
    """Build once, then verify exact source IDs and original cohort membership."""
    source = root / INPUT
    docs, selection, manifest = rolling.load_rolling_documents(source)
    if (manifest['split'] != 'test' or len(docs) != 2255
            or selection['selected_scored_tokens'] != 288602
            or manifest['context_prefix_tokens'] != 128 or manifest['max_scored_tokens'] != 128):
        raise ValueError('Expected the prepared complete OLMoE P128/S128 test population')
    groups = list(quality.document_groups(docs, 8))
    if len(groups) != 283:
        raise ValueError('Original B8 cohort population changed')
    raw = [json.loads(line) for line in (source.parent / manifest['windows_file']['path']).read_text().splitlines()]
    plans, window_start = [], 0
    for chunk, (first, stop) in enumerate(((0, 96), (96, 192), (192, 283))):
        count = sum(map(len, groups[first:stop]))
        originals = raw[window_start:window_start+count]
        stream = originals[0]['token_ids'][:128] + [t for row in originals for t in row['token_ids'][128:]]
        rows = list(rolling.rolling_windows(stream, 128, 128, 'test'))
        for row, original in zip(rows, originals):
            if row['token_ids'] != original['token_ids']:
                raise ValueError('Chunk changed an original token ID')
            row['id'] = original['id']
            for field in ('doc_key', 'window_index', 'context_start_token', 'score_start_token', 'score_stop_token'):
                row['source_' + field] = original[field]
        provenance = {'manifest_path_from_root': INPUT, 'manifest_sha256': rolling.file_hash(source),
                      'windows_sha256': manifest['windows_file']['sha256'],
                      'token_stream_sha256': manifest['token_stream']['sha256'],
                      'global_scored_span': [originals[0]['score_start_token'], originals[-1]['score_stop_token']],
                      'window_range': [window_start, window_start+count], 'cohort_range': [first, stop],
                      'original_cohort_batch': 8, 'chunk_index': chunk, 'chunks': 3,
                      'identity_note': 'Runtime keys/local offsets rebased to chunk stream; source_* fields preserve original keys/global offsets. Exact IDs; no decode/encode.'}
        folder = HERE / 'chunks' / f'chunk{chunk}'
        path = folder / 'manifest.json'
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=False)
            windows = folder / 'windows.jsonl'
            windows.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
            derived = dict(manifest)
            derived.update(created_unix=time.time(), derived_chunk=provenance,
                text_construction={'method': 'Exact consecutive token-ID windows copied from source manifest; no retokenization'},
                tokenization={'calls': 0, 'method': 'copy source token IDs'},
                token_stream={'tokens': len(stream), 'sha256': rolling.json_hash(stream), 'hash_encoding': 'canonical JSON integer array'},
                windows_file={'path': 'windows.jsonl', 'rows': count, 'sha256': rolling.file_hash(windows)},
                coverage={'excluded_initial_span': [0, 128], 'scored_span': [128, len(stream)],
                          'scored_tokens': len(stream)-128, 'scored_spans_overlap': False,
                          'scored_spans_have_gaps': False, 'last_window_scored_tokens': rows[-1]['scored_tokens'],
                          'tail_policy': 'Retain short tail and incomplete cohort; complete chunk is only part of full test'},
                scope='Contiguous chunk of complete test; rolling teacher-forced decode P128/S<=128, context resets per window; union of all three chunks is required for full test',
                statistical_unit='Consecutive windows of one ordered test stream, not independent articles/runs; chunks are restart boundaries only')
            common.save(path, derived)
        chunk_docs, chunk_selection, found = rolling.load_rolling_documents(path)
        actual = [json.loads(line) for line in (folder/'windows.jsonl').read_text().splitlines()]
        if actual != rows or found['derived_chunk'] != provenance or found['tokenizer'] != manifest['tokenizer']:
            raise ValueError('Existing chunk differs from exact source partition')
        if [len(g) for g in quality.document_groups(chunk_docs, 8)] != [len(g) for g in groups[first:stop]]:
            raise ValueError('Chunk changed original cohort boundaries')
        plans.append({'id': f'chunk{chunk}', 'manifest': str(path), 'manifest_sha256': rolling.file_hash(path),
                      'windows_sha256': found['windows_file']['sha256'], 'windows': count,
                      'scored_tokens': chunk_selection['selected_scored_tokens'], 'cohorts': stop-first,
                      'global_scored_span': provenance['global_scored_span'],
                      'original_window_range': provenance['window_range'],
                      'actual_batch_distribution': dict(Counter(map(len, quality.document_groups(chunk_docs, 8))))})
        window_start += count
    if window_start != len(docs) or sum(p['scored_tokens'] for p in plans) != 288602:
        raise ValueError('Incomplete source partition')
    return plans


def provenance(args, chunks):
    code_names = ['quality_decode.py', 'prepare_wikitext_rolling.py', 'teacher_force.py',
                  'skip_worker.py', 'expert_skip.py', 'run_task_quality.py']
    sources = {name: rolling.file_hash(args.root/'code'/name) for name in code_names}
    sources['deployment_entry'] = rolling.file_hash(Path(__file__))
    tunes = {}
    for fmt in args.formats:
        folder = args.root / TUNING[fmt]
        summary = json.loads((folder/'summary.json').read_text())
        if summary['status'] != 'PASS':
            raise ValueError(f'{fmt} native-only tuning did not pass')
        config = folder/'tuned_configs'/summary['config_filename']
        if (fmt == 'fp8') != ('dtype=fp8_w8a8' in config.name):
            raise ValueError('Weight format/tuning config mismatch')
        tunes[fmt] = {'source_from_root': TUNING[fmt], 'summary_sha256': rolling.file_hash(folder/'summary.json'),
                      'config': config.name, 'config_sha256': rolling.file_hash(config)}
    config = args.model/'config.json'
    if config.exists():
        if rolling.file_hash(config) != MODEL_SHA:
            raise ValueError('Different checkpoint config: this entry is only OLMoE')
    elif not args.plan_only:
        raise FileNotFoundError(config)
    return {'code_sha256': sources, 'arms_sha256': rolling.file_hash(HERE/'arms.json'),
            'model_config_sha256': MODEL_SHA, 'model_path': str(args.model),
            'original_manifest_sha256': rolling.file_hash(args.root/INPUT),
            'chunks': [{k: v for k, v in c.items() if k != 'manifest'} for c in chunks],
            'native_tuning': tunes, 'formats': args.formats,
            'batch': 8, 'seed': 1701, 'gpu_memory_utilization': .35, 'reference_max_mib': 2048}


def command(args, fmt, chunk, output):
    cmd = [sys.executable, str(args.root/'code/quality_decode.py'), '--model', str(args.model),
           '--tokenized-windows', chunk['manifest'], '--output', str(output), '--batch', '8',
           '--arms-file', str(HERE/'arms.json'), '--dtype', 'bfloat16',
           '--gpu-memory-utilization', '.35', '--reference-max-mib', '2048', '--seed', '1701']
    if fmt == 'fp8':
        cmd += ['--quantization', 'fp8']
    return cmd


def validate_result(path, chunk, fmt):
    """Check every cohort/window and supplied SHA, without replacing GPU checks."""
    value = json.loads(path.read_text())
    docs, _, _ = rolling.load_rolling_documents(chunk['manifest'])
    artifact = path.with_name(path.stem+'_artifacts')
    if json.loads((artifact/'documents.json').read_text()) != docs:
        raise ValueError('Returned document population changed')
    expected_arms = [{'name': n, 'spec': s} for n, s in quality.resolve_arm_specs(quality.load_arms_file(HERE/'arms.json')[0])]
    selection = value['selection']
    if (value['status'] != 'PASS' or value['arms'] != expected_arms
            or value['arm_source']['sha256'] != rolling.file_hash(HERE/'arms.json')
            or selection['source_manifest_sha256'] != chunk['manifest_sha256']
            or selection['selected_windows'] != chunk['windows']
            or selection['selected_scored_tokens'] != chunk['scored_tokens']
            or selection['unselected_suffix_tokens'] != 0
            or not value['rolling_coverage_verified']['all_scored_predictions_decode']):
        raise ValueError('Incomplete chunk population or changed input/arms')
    a = value['args']
    if (a['batch'] != 8 or a['seed'] != 1701 or a['limit_windows'] is not None
            or a['quantization'] != ('fp8' if fmt == 'fp8' else None)):
        raise ValueError('Changed execution protocol')
    groups = list(quality.document_groups(docs, 8))
    expected = [(f'batch{i:04d}', arm, group) for i, group in enumerate(groups) for arm in NAMES]
    if len(value['batches']) != len(expected):
        raise ValueError('Missing or duplicate cohort/arm')
    for row, (group_name, arm, population) in zip(value['batches'], expected):
        sizes = {d['doc_key']: len(d['forced_tokens'])-1 for d in population}
        if (not row['valid'] or row['group'] != group_name or row['arm'] != arm
                or row['actual_batch'] != len(population) or set(row['per_document']) != set(sizes)
                or any(row['per_document'][key]['tokens'] != size for key, size in sizes.items())
                or row['quality']['tokens'] != sum(sizes.values())):
            raise ValueError('A cohort has incomplete or duplicate score population')
        for field in ('policy_counter_file', 'token_file'):
            if not (artifact/Path(row[field]).name).is_file():
                raise FileNotFoundError(row[field])
    for arm in NAMES:
        if (value['quality_by_arm'][arm]['tokens'] != chunk['scored_tokens']
                or value['quality_by_arm'][arm]['documents'] != chunk['windows']):
            raise ValueError('Wrong aggregate population')
    return value


def accept_attempt(folder, chunk, fmt, contract_sha):
    receipt_path = folder/'checkpoint.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['contract_sha256'] != contract_sha or receipt['format'] != fmt or receipt['chunk'] != chunk['id']:
        raise ValueError('Completed checkpoint source/protocol changed; choose a new output')
    for relative, expected in receipt['files_sha256'].items():
        if rolling.file_hash(folder/relative) != expected:
            raise ValueError(f'Completed checkpoint changed: {folder/relative}')
    return validate_result(folder/'quality.json', chunk, fmt)


def merge_results(args, chunks, accepted):
    formats = {}
    for fmt in args.formats:
        entries = [accepted[(fmt, c['id'])] for c in chunks if (fmt, c['id']) in accepted]
        complete = len(entries) == len(chunks)
        pooled = {arm: quality.merge_quality_summaries([value['quality_by_arm'][arm] for value in entries]) for arm in NAMES} if entries else {}
        if complete and any(q['tokens'] != 288602 or q['documents'] != 2255 for q in pooled.values()):
            raise ValueError('Full split token/window population mismatch')
        if pooled:
            for stats in pooled.values():
                stats['delta_ppl_vs_native'] = stats['ppl'] - pooled['native']['ppl']
        formats[fmt] = {'status': 'COMPLETE' if complete else 'INCOMPLETE',
            'complete_test_split': complete, 'accepted_chunks': [c['id'] for c in chunks if (fmt, c['id']) in accepted],
            'scored_global_spans': [c['global_scored_span'] for c in chunks if (fmt, c['id']) in accepted],
            'quality_by_arm': pooled, 'windows_are_not_independent_articles_or_repeats': True}
    return {'status': 'COMPLETE' if all(v['complete_test_split'] for v in formats.values()) else 'INCOMPLETE',
            'scope': 'Seven-arm subset, not the ten-arm quality matrix; complete ordered test [128,288730) only when all three chunks pass. B8 except original B6 cohort and B1 short tail.',
            'aggregation': 'Token-weighted NLL/KL; PPL=exp(pooled NLL). Each format has its own native. Three chunks are restart boundaries, not repetitions.',
            'formats': formats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--model', type=Path, default=Path('/root/autodl-tmp/moe-research-20261002/model'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--outer-run-dir', type=Path)
    parser.add_argument('--formats', nargs='+', choices=['bf16', 'fp8'], default=['bf16', 'fp8'])
    parser.add_argument('--budget-seconds', type=int, default=14370)
    parser.add_argument('--chunk-timeout', type=int, default=2700)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    args.root = args.root.resolve()
    initialize(args.root)
    for name in ('model', 'output', 'outer_run_dir'):
        if getattr(args, name) is not None:
            setattr(args, name, common.resolved(getattr(args, name), args.root))
    if len(set(args.formats)) != len(args.formats) or not 0 < args.budget_seconds <= 14370 or args.chunk_timeout <= 0:
        parser.error('Unique formats and bounded positive budgets required')
    chunks = prepare_chunks(args.root)
    contract = provenance(args, chunks)
    contract_sha = rolling.json_hash(contract)
    plan = {'status': 'PLANNED', 'contract': contract, 'contract_sha256': contract_sha,
            'budget_seconds': args.budget_seconds, 'chunk_timeout_seconds': args.chunk_timeout,
            'chunks': chunks, 'jobs': []}
    for fmt in args.formats:
        for chunk in chunks:
            plan['jobs'].append({'format': fmt, 'chunk': chunk['id'], 'status': 'NOT_STARTED',
                'command': command(args, fmt, chunk, args.output/f'{fmt}.{chunk["id"]}'/'attempt001/quality.json'),
                'tuned_config_folder': str(args.root/TUNING[fmt]/'tuned_configs')})
    if args.output.exists():
        if not args.resume:
            raise FileExistsError('Use a new output, or explicit --resume for identical inputs/code')
        previous = json.loads((args.output/'suite.json').read_text())
        if previous['contract_sha256'] != contract_sha:
            raise ValueError('Resume source/format/protocol SHA differs')
    else:
        args.output.mkdir(parents=True)
    if not args.plan_only:
        plan['outer_run'] = common.verify_outer(args)
    common.save(args.output/'suite.json', plan)
    if args.plan_only:
        print(json.dumps({'status': 'PLANNED', 'jobs': len(plan['jobs']), 'windows_per_format': 2255,
                          'scored_tokens_per_arm': 288602, 'contract_sha256': contract_sha}))
        return 0
    started, accepted, child = time.monotonic(), {}, None
    def interrupted(signum, frame):
        raise InterruptedError(f'Received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    plan.update(status='RUNNING', started_unix=time.time())
    try:
        for job in plan['jobs']:
            fmt = job['format']; chunk = next(c for c in chunks if c['id'] == job['chunk'])
            base = args.output/f'{fmt}.{chunk["id"]}'
            attempts = sorted(base.glob('attempt*'))
            done = [p for p in attempts if (p/'checkpoint.json').is_file()]
            if len(done) > 1:
                raise ValueError('Multiple accepted attempts for one chunk')
            if done:
                accepted[(fmt, chunk['id'])] = accept_attempt(done[0], chunk, fmt, contract_sha)
                job.update(status='REUSED_COMPLETE', attempt=str(done[0]))
                continue
            remaining = args.budget_seconds-(time.monotonic()-started)
            if remaining < 120:
                job['status'] = 'NOT_RUN_BUDGET'
                break
            folder = base/f'attempt{len(attempts)+1:03d}'
            folder.mkdir(parents=True)
            job.update(status='RUNNING', attempt=str(folder), started_unix=time.time(),
                       timeout_seconds=min(args.chunk_timeout, remaining-15))
            job['command'] = command(args, fmt, chunk, folder/'quality.json')
            env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       VLLM_ENABLE_V1_MULTIPROCESSING='0', TOKENIZERS_PARALLELISM='false',
                       VLLM_TUNED_CONFIG_FOLDER=job['tuned_config_folder'])
            common.save(args.output/'suite.json', plan)
            with (folder/'stdout.log').open('x') as log:
                child = subprocess.Popen(job['command'], cwd=args.root, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    rc = child.wait(timeout=job['timeout_seconds'])
                    job.update(exit_code=rc, status='RETURNED' if rc == 0 else 'FAILED')
                except subprocess.TimeoutExpired:
                    job['status'] = 'TIMEOUT'
                finally:
                    common.stop_child(child)
                    child = None
            job['ended_unix'] = time.time()
            if job['status'] != 'RETURNED':
                break
            value = validate_result(folder/'quality.json', chunk, fmt)
            files = {str(p.relative_to(folder)): rolling.file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()}
            common.save(folder/'checkpoint.json', {'status': 'COMPLETE', 'contract_sha256': contract_sha,
                        'format': fmt, 'chunk': chunk['id'], 'files_sha256': files,
                        'scored_tokens_per_arm': chunk['scored_tokens'], 'windows': chunk['windows']})
            accepted[(fmt, chunk['id'])] = value
            job['status'] = 'COMPLETE'
            common.save(args.output/'summary.json', merge_results(args, chunks, accepted))
            common.save(args.output/'suite.json', plan)
    except BaseException as error:
        plan['error'] = f'{type(error).__name__}: {error}'
        for job in plan['jobs']:
            if job['status'] in ('RUNNING', 'RETURNED'):
                job['status'] = 'FAILED_OR_INTERRUPTED'
    finally:
        if child is not None:
            common.stop_child(child)
        result = merge_results(args, chunks, accepted)
        plan.update(status=result['status'], ended_unix=time.time(), elapsed_seconds=time.monotonic()-started)
        common.save(args.output/'summary.json', result)
        common.save(args.output/'suite.json', plan)
    print(json.dumps({'status': plan['status'], 'output': str(args.output)}), flush=True)
    return 0 if plan['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())

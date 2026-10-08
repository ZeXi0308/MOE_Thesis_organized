#!/usr/bin/env python3
"""One independent document-QA qualification; no automatic capacity experiment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
import qwen7b_launch_v2 as qualified
from health_analyze_v2 import analyze as analyze_q0

BASE = Path(__file__).resolve().parent
FREEZE = BASE / 'docqa_freeze_v1.json'
OUT = BASE / 'qwen7b-docqa16-v1'
INPUTS = BASE / 'docqa_inputs16_v1.json'
PREFLIGHT = BASE / 'docqa_context_preflight_v1.json'
resource = qualified.resource
require, sha, save = resource.require, resource.sha, resource.save


def read(path):
    return json.loads(path.read_text())


def prerequisites():
    frozen = read(FREEZE)['files_sha256']
    required = {'docqa_launch_v1.py', 'docqa_native_v1.py', 'docqa_integrity_analyze_v1.py',
        'docqa_input_build_v1.py', 'docqa_inputs16_v1.json', 'docqa_inputs128_v1.json',
        'docqa_inputs128_candidate_v1.json', 'docqa_answers128_v1.json',
        'docqa_questions_part_a_v1.json', 'docqa_questions_part_b_v1.json',
        'docqa_context_preflight_v1.json', 'DOCUMENT_QA_NEXT_PROBE_PLAN.md',
        'summary128_inputs_v1.json', 'summary128_reference_v1.json',
        'qwen7b_launch_v2.py', 'qwen7b_freeze_v2.json', 'qwen7b_q1_quality_result_v2.json'}
    require(required <= set(frozen), 'document QA freeze omits required dependency')
    for name, expected in frozen.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and sha(BASE / path) == expected,
                'document QA frozen file differs: ' + name)
    require(set(read(qualified.FREEZE)['files_sha256']) <= set(frozen), 'inherited files not frozen')
    manifest = qualified.frozen_manifest()
    q0_root = BASE / 'qwen7b-q0-v2'
    prior = read(q0_root / 'launcher-receipt.json')
    require(prior['status'] == 'COMPLETE' and prior['action'] == 'q0' and
        prior['model_id'] == qualified.MODEL and prior['revision'] == qualified.REVISION and
        prior['manifest_sha256'] == sha(qualified.MANIFEST) and
        prior['freeze_sha256'] == sha(qualified.FREEZE) and
        prior['runtime_environment']['VLLM_USE_FLASHINFER_SAMPLER'] == '0', 'Q0 model/lifecycle differs')
    q0 = analyze_q0(q0_root / 'native', BASE / 'workload.json')
    require(q0['validity'] == 'COMPLETE' and q0['screening'] == 'PASS_DEVELOPMENT_SCREEN',
            'Q0 prerequisite failed')
    preflight, inputs = read(PREFLIGHT), read(INPUTS)
    all_inputs = read(BASE / 'docqa_inputs128_v1.json')
    require(preflight['status'] == 'PASS_INPUT_GEOMETRY' and
        preflight['candidate_sha256'] == sha(BASE / 'docqa_inputs128_candidate_v1.json') and
        preflight['model_id'] == qualified.MODEL and preflight['revision'] == qualified.REVISION,
        'input geometry/model mismatch')
    context = preflight['max_model_len']
    selected = next((n for n in (4096, 8192) if preflight['prompt_tokens_max'] + 512 <= n), None)
    require(context == selected == inputs['max_model_len'] == all_inputs['max_model_len'],
            'context was not selected by frozen legal-input rule')
    require(len(all_inputs['requests']) == 128 and inputs['requests'] == all_inputs['requests'][:16]
        and inputs['arrival_traces_s'] == [0.0] * 16, 'fixed first16 differ')
    require(inputs['questions_answer_key_sha256'] == sha(BASE / 'docqa_answers128_v1.json'),
            'questions/answer-key identity differs')
    old_summary = read(BASE / 'qwen7b_q1_quality_result_v2.json')
    require(old_summary['qualification'] == 'FAIL_DEVELOPMENT_SCREEN',
            'independent-task provenance changed')
    return manifest, q0, preflight, context


def main():
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    manifest, q0, preflight, context = prerequisites()
    require(not OUT.exists() and not OUT.is_symlink(), 'immutable output already exists')
    fd = os.open(resource.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        qualified.lock_identity(fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status='GPU_DEFERRED', reason='existing shared lock busy')), flush=True)
            return 75
        OUT.mkdir(); resource.OUT = OUT
        record = dict(status='RUNNING', action='document-qa16', start_unix_s=time.time(),
            model_id=qualified.MODEL, revision=qualified.REVISION,
            manifest_sha256=sha(qualified.MANIFEST), freeze_sha256=sha(FREEZE),
            inputs_sha256=sha(INPUTS), answer_key_sha256=sha(BASE / 'docqa_answers128_v1.json'),
            context_preflight_sha256=sha(PREFLIGHT), max_model_len=context,
            launcher_pid=os.getpid(), lock_inode=resource.LOCK_ID,
            runtime_environment={'VLLM_USE_FLASHINFER_SAMPLER': '0'})
        try:
            record['gpu_before'] = resource.gpu_idle()
            save(OUT / 'launcher-receipt.json', record)
            save(OUT / 'q0-prerequisite.json', q0)
            save(OUT / 'context-preflight.json', preflight)
            require(shutil.disk_usage(BASE).free >= qualified.GIB, 'root disk reserve below 1 GiB')
            identity = qualified.stage_identity(manifest)
            files = qualified.inventory(manifest, complete=True)
            save(OUT / 'model-reverified.json', dict(status='VERIFIED', files=files, **identity))
            require(qualified.memory_headroom() >= 24 * qualified.GIB, 'RAM reserve below 24 GiB')
            resource.gpu_idle()
            resource.bounded([sys.executable, str(BASE / 'docqa_native_v1.py'),
                '--model-dir', str(qualified.STAGE), '--inputs', str(INPUTS),
                '--output', str(OUT / 'native'), '--expected-gpu-uuid', resource.GPU,
                '--kv-bytes', '8589934592', '--max-seqs', '32', '--batch-tokens', '1024',
                '--max-model-len', str(context)], 'native.log', 1200, fd)
            require(read(OUT / 'native/status.json')['status'] == 'COMPLETE' and
                    read(OUT / 'native/native-drain.json')['status'] == 'QUALIFIED',
                    'document QA native unit incomplete')
            record.update(status='COMPLETE', gpu_after=resource.gpu_idle())
        except BaseException as exc:
            record.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            try:
                record['gpu_after'] = resource.gpu_idle()
            except Exception as exc:
                record.update(status='INCOMPLETE', gpu_after_error=f'{type(exc).__name__}: {exc}')
            record.update(end_unix_s=time.time(), retained_stage=str(qualified.STAGE),
                scope='Independent document-QA quality unit; old summary failure retained. No policy comparison.')
            save(OUT / 'launcher-receipt.json', record)
        return 0 if record['status'] == 'COMPLETE' else 70
    finally:
        os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    qualified.install_signals()
    sys.exit(main())

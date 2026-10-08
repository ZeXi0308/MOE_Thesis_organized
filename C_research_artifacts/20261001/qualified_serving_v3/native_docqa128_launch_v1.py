#!/usr/bin/env python3
"""One frozen native document-QA128 capacity existence unit after QA16 qualification."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
import docqa_launch_v1 as prior
from docqa_integrity_analyze_v1 import analyze as analyze_qa16

qualified, resource = prior.qualified, prior.resource
require, sha, save, read = prior.require, prior.sha, prior.save, prior.read
BASE = Path(__file__).resolve().parent
FREEZE = BASE / 'native_docqa128_freeze_v1.json'
PROTOCOL = BASE / 'native_docqa128_protocol_v1.json'
INPUTS = BASE / 'docqa_inputs128_v1.json'
QUALITY = BASE / 'docqa16_quality_v1.json'
OUT = BASE / 'qwen7b-native-docqa128-v1'


def prerequisites():
    frozen = read(FREEZE)['files_sha256']
    required = {'native_docqa128_launch_v1.py', 'native_docqa128_cell_v1.py',
        'docqa128_integrity_analyze_v1.py', 'native_pressure_observer_v1.py',
        'native_pressure_analyze_v1.py', PROTOCOL.name, QUALITY.name, prior.FREEZE.name,
        'qwen7b_native_default_source_v1.json'}
    require(required <= set(frozen), 'QA128 freeze omits dependencies')
    require(set(read(prior.FREEZE)['files_sha256']) <= set(frozen), 'QA16 dependencies omitted')
    for name, expected in frozen.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and sha(BASE / path) == expected,
                'QA128 frozen file differs: ' + name)
    manifest, q0, geometry, context = prior.prerequisites()
    require(context == 8192 and read(INPUTS)['arrival_traces_s'] == [0.0] * 128,
            'frozen context or arrival trace differs')
    source = prior.OUT
    parent = read(source / 'launcher-receipt.json')
    require(parent['status'] == 'COMPLETE' and parent['action'] == 'document-qa16' and
        parent['model_id'] == qualified.MODEL and parent['revision'] == qualified.REVISION and
        parent['manifest_sha256'] == sha(qualified.MANIFEST) and
        parent['freeze_sha256'] == sha(prior.FREEZE) and
        parent['runtime_environment']['VLLM_USE_FLASHINFER_SAMPLER'] == '0',
        'QA16 model/lifecycle binding differs')
    integrity = analyze_qa16(source, prior.INPUTS)
    quality = read(QUALITY)
    require(integrity['validity'] == 'COMPLETE' and integrity['structural_gate'] is True,
            'QA16 structural gate failed')
    require(quality['validity'] == 'COMPLETE' and
        quality['qualification'] == 'PASS_DEVELOPMENT_SCREEN' and
        12 <= quality['sensitivity']['conservative_successful_requests'] <= quality['successful_requests'] <= 16 and
        quality['required_successful_requests'] == 12 and quality['total_requests'] == 16,
        'QA16 conservative semantic gate failed')
    require(quality['raw_measured_outputs_sha256'] == sha(source / 'native/measured-outputs.json') and
        quality['source_sha256'] == sha(prior.INPUTS) and
        quality['answers_sha256'] == sha(BASE / 'docqa_answers128_v1.json') and
        quality['freeze_sha256'] == sha(prior.FREEZE), 'QA16 semantic identity differs')
    protocol = read(PROTOCOL)
    require(protocol['inputs_sha256'] == sha(INPUTS) and protocol['max_model_len'] == 8192 and
        protocol['max_num_seqs'] == 128 and protocol['max_num_batched_tokens'] == 2048 and
        protocol['kv_cache_memory_bytes'] == 8589934592 and protocol['max_tokens'] == 512,
        'QA128 protocol binding differs')
    return manifest, q0, geometry, integrity, quality


def main():
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    manifest, q0, geometry, integrity, quality = prerequisites()
    require(not OUT.exists() and not OUT.is_symlink(), 'immutable output already exists')
    fd = os.open(resource.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        qualified.lock_identity(fd)
        wait_deadline = time.monotonic() + 1200
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= wait_deadline:
                    print(json.dumps(dict(status='GPU_DEFERRED', reason='shared lock wait expired',
                        launcher_pid=os.getpid())), flush=True)
                    return 75
                print(json.dumps(dict(status='WAITING_SHARED_LOCK', launcher_pid=os.getpid(),
                    observed_unix_s=time.time())), flush=True)
                time.sleep(15)
        OUT.mkdir(); resource.OUT = OUT
        record = dict(status='RUNNING', action='native-docqa128', start_unix_s=time.time(),
            model_id=qualified.MODEL, revision=qualified.REVISION,
            manifest_sha256=sha(qualified.MANIFEST), freeze_sha256=sha(FREEZE),
            inputs_sha256=sha(INPUTS), answer_key_sha256=sha(BASE / 'docqa_answers128_v1.json'),
            protocol_sha256=sha(PROTOCOL), qa16_quality_sha256=sha(QUALITY), max_model_len=8192,
            launcher_pid=os.getpid(), lock_inode=resource.LOCK_ID,
            generation_timeout_s=900, child_timeout_s=1200,
            runtime_environment={'VLLM_USE_FLASHINFER_SAMPLER': '0'})
        try:
            record['gpu_before'] = resource.gpu_idle()
            save(OUT / 'launcher-receipt.json', record)
            save(OUT / 'q0-prerequisite.json', q0)
            save(OUT / 'context-preflight.json', geometry)
            save(OUT / 'qa16-structural-gate.json', integrity)
            save(OUT / 'qa16-quality-gate.json', quality)
            save(OUT / 'protocol.json', read(PROTOCOL))
            require(shutil.disk_usage(BASE).free >= qualified.GIB, 'root disk reserve below 1 GiB')
            identity = qualified.stage_identity(manifest)
            files = qualified.inventory(manifest, complete=True)
            save(OUT / 'model-reverified.json', dict(status='VERIFIED', files=files, **identity))
            require(qualified.memory_headroom() >= 24 * qualified.GIB, 'RAM reserve below 24 GiB')
            resource.gpu_idle()
            resource.bounded([sys.executable, str(BASE / 'native_docqa128_cell_v1.py'),
                '--model-dir', str(qualified.STAGE), '--inputs', str(INPUTS),
                '--output', str(OUT / 'native'), '--expected-gpu-uuid', resource.GPU,
                '--kv-bytes', '8589934592', '--max-seqs', '128', '--batch-tokens', '2048',
                '--max-model-len', '8192'], 'native.log', 1200, fd)
            require(read(OUT / 'native/status.json')['status'] == 'COMPLETE' and
                read(OUT / 'native/native-drain.json')['status'] == 'QUALIFIED' and
                read(OUT / 'native/measured-pressure.json')['status'] == 'COMPLETE',
                'QA128 native unit or pressure observer incomplete')
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
                scope='One native document-QA128 capacity existence unit; no policy comparison.')
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

#!/usr/bin/env python3
"""One conditional native128 summary unit; preparation alone does not authorize a run.

Requires a separate freeze and completed Q0/Q1 v2 structural and semantic gates.
No model download, retry chain, controller or performance verdict is implemented.
"""
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
from summary_integrity_analyze_v2 import analyze as analyze_q1

BASE = Path(__file__).resolve().parent
FREEZE = BASE / 'native_summary128_freeze_v1.json'
OUT = BASE / 'qwen7b-native-summary128-v1'
INPUTS = BASE / 'summary128_inputs_v1.json'
REFERENCES = BASE / 'summary128_reference_v1.json'
Q0, Q1 = BASE / 'qwen7b-q0-v2', BASE / 'qwen7b-q1-v2'
QUALITY = BASE / 'qwen7b_q1_quality_result_v2.json'
resource = qualified.resource
require, sha, save = resource.require, resource.sha, resource.save


def read(path):
    return json.loads(path.read_text())


def frozen_manifest():
    frozen = read(FREEZE)['files_sha256']
    required = {
        'native_summary128_launch_v1.py', 'native_summary128_cell_v1.py',
        'native_pressure_observer_v1.py', 'native_pressure_analyze_v1.py',
        'qwen7b_launch_v2.py', 'qwen7b_freeze_v2.json', 'qwen7b_model_manifest.json',
        'health_launch.py', 'range_download_v2.py', 'health_native.py',
        'health_native_summary.py', 'health_analyze_v2.py',
        'summary_integrity_analyze_v2.py', 'workload.json',
        'summary_inputs_v1.json', 'summary_reference_v1.json',
        INPUTS.name, REFERENCES.name, QUALITY.name,
    }
    require(required <= set(frozen), 'native128 freeze omits a required dependency or quality gate')
    for name, digest in frozen.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and sha(BASE / path) == digest,
                'native128 frozen file differs: ' + name)
    inherited = read(qualified.FREEZE)['files_sha256']
    require(set(inherited) <= set(frozen), 'native128 freeze omits inherited v2 dependencies')
    return qualified.frozen_manifest()


def qualification_gates(manifest):
    """CPU-only gates execute before acquiring a GPU unit or creating its output."""
    manifest_sha = sha(qualified.MANIFEST)
    for root, action, cell in ((Q0, 'q0', 'health_native.py'),
                               (Q1, 'q1', 'health_native_summary.py')):
        parent = read(root / 'launcher-receipt.json')
        model = read(root / 'model-reverified.json')
        require(parent['status'] == 'COMPLETE' and parent['action'] == action and
                parent['model_id'] == qualified.MODEL and parent['revision'] == qualified.REVISION and
                parent['manifest_sha256'] == manifest_sha and
                parent['freeze_sha256'] == sha(qualified.FREEZE),
                action + ' v2 lifecycle/model/freeze mismatch')
        require(parent['runtime_environment']['VLLM_USE_FLASHINFER_SAMPLER'] == '0',
                action + ' v2 sampler environment differs')
        require(model['status'] == 'VERIFIED' and model['model_id'] == qualified.MODEL and
                model['revision'] == qualified.REVISION and model['manifest_sha256'] == manifest_sha and
                model['files'] == manifest['files'], action + ' verified model inventory differs')
        require(read(root / 'native/status.json')['status'] == 'COMPLETE', action + ' native incomplete')
        provenance = read(root / 'native/provenance.json')
        require(provenance['cell_sha256'] == sha(BASE / cell) and
                sha(root / 'native/cell-source.py') == sha(BASE / cell),
                action + ' original qualification cell differs')
    q0 = analyze_q0(Q0 / 'native', BASE / 'workload.json')
    require(q0['validity'] == 'COMPLETE' and q0['screening'] == 'PASS_DEVELOPMENT_SCREEN',
            'Q0 v2 original health gate failed')
    q1 = analyze_q1(Q1, BASE / 'summary_inputs_v1.json', expected_max_seqs=32)
    require(q1['validity'] == 'COMPLETE' and q1['structural_gate'] is True,
            'Q1 v2 original structural gate failed')
    quality = read(QUALITY)
    require(quality['validity'] == 'COMPLETE' and
            quality['qualification'] == 'PASS_DEVELOPMENT_SCREEN' and
            type(quality['useful_strict']) is int and 12 <= quality['useful_strict'] <= 16,
            'Q1 v2 semantic quality gate failed')
    require(quality['references_sha256'] == sha(BASE / 'summary_reference_v1.json'),
            'Q1 semantic reference identity differs')
    require(quality['raw_measured_outputs_sha256'] == sha(Q1 / 'native/measured-outputs.json'),
            'Q1 semantic judgment refers to different raw outputs')
    return q0, q1, quality


def main():
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    manifest = frozen_manifest()
    require(not OUT.exists() and not OUT.is_symlink(), 'existing immutable output; no repeat')
    q0, q1, quality = qualification_gates(manifest)
    fd = os.open(resource.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        qualified.lock_identity(fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status='GPU_DEFERRED', reason='shared lock busy')), flush=True)
            return 75
        OUT.mkdir()
        resource.OUT = OUT
        record = dict(status='RUNNING', action='native-summary128', start_unix_s=time.time(),
            model_id=qualified.MODEL, revision=qualified.REVISION,
            manifest_sha256=sha(qualified.MANIFEST), freeze_sha256=sha(FREEZE),
            inputs_sha256=sha(INPUTS), references_sha256=sha(REFERENCES),
            q1_quality_sha256=sha(QUALITY), launcher_pid=os.getpid(), lock_inode=resource.LOCK_ID,
            generation_timeout_s=900, child_timeout_s=1200,
            runtime_environment={'VLLM_USE_FLASHINFER_SAMPLER': '0'})
        try:
            record['gpu_before'] = resource.gpu_idle()
            save(OUT / 'launcher-receipt.json', record)
            save(OUT / 'q0-health-gate.json', q0)
            save(OUT / 'q1-structural-gate.json', q1)
            save(OUT / 'q1-quality-gate.json', quality)
            require(shutil.disk_usage(BASE).free >= qualified.GIB, 'root disk reserve below 1 GiB')
            identity = qualified.stage_identity(manifest)
            verified = qualified.inventory(manifest, complete=True)
            save(OUT / 'model-reverified.json', dict(status='VERIFIED', files=verified, **identity))
            require(qualified.memory_headroom() >= 24 * qualified.GIB,
                    'native RAM reserve below 24 GiB')
            resource.gpu_idle()
            resource.bounded([sys.executable, str(BASE / 'native_summary128_cell_v1.py'),
                '--model-dir', str(qualified.STAGE), '--inputs', str(INPUTS),
                '--output', str(OUT / 'native'), '--expected-gpu-uuid', resource.GPU,
                '--kv-bytes', '8589934592', '--max-seqs', '128', '--batch-tokens', '2048',
                '--max-model-len', '4096'], 'native.log', 1200, fd)
            require(read(OUT / 'native/status.json')['status'] == 'COMPLETE' and
                    read(OUT / 'native/native-drain.json')['status'] == 'QUALIFIED' and
                    read(OUT / 'native/measured-pressure.json')['status'] == 'COMPLETE',
                    'native128 cell/observation/drain incomplete')
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
                scope='One conditional native128 capacity observation; no controller or policy comparison.')
            save(OUT / 'launcher-receipt.json', record)
        return 0 if record['status'] == 'COMPLETE' else 70
    finally:
        os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True,
                        help='run the single unit only after all frozen qualification gates pass')
    parser.parse_args()
    qualified.install_signals()
    sys.exit(main())

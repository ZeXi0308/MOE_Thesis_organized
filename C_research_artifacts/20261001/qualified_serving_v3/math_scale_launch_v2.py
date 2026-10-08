#!/usr/bin/env python3
"""Two same-task load-scaling arms; preserve KV, APC and native decisions."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
import qwen7b_launch_v2 as model

BASE = Path(__file__).resolve().parent
OUTROOT = Path('/dev/shm/c-math-scale-runs-20261002-v2')
INPUTS = BASE / 'math_inputs1024_v1.json'
PROTOCOL = BASE / 'math_scale_protocol_v2.json'
CACHE = BASE / 'math_scale_runtime_cache_v2'
resource = model.resource
require, sha, save = resource.require, resource.sha, resource.save


def retire_unused_small_model():
    """Release only the pinned public 1.5B cache from the retired branch."""
    stage = Path('/dev/shm/c-qwen-qualified-20261001-v1')
    manifest = json.loads((BASE / 'model_manifest.json').read_text())
    require(manifest['model_id'] == 'Qwen/Qwen2.5-1.5B-Instruct' and
            manifest['revision'] == '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
            'retired model identity differs')
    require(stage.is_dir() and not stage.is_symlink() and stage.resolve() == stage,
            'retired stage missing or symlinked')
    require(set(x.name for x in stage.iterdir()) ==
            set(x['filename'] for x in manifest['files']), 'retired stage inventory differs')
    for item in manifest['files']:
        path = stage / item['filename']
        require(path.is_file() and not path.is_symlink() and
                path.stat().st_size == item['size'] and sha(path) == item['sha256'],
                'retired public model cache differs: ' + item['filename'])
    record = dict(status='VERIFIED_PUBLIC_CACHE', model_id=manifest['model_id'],
        revision=manifest['revision'], manifest_sha256=sha(BASE / 'model_manifest.json'),
        stage=str(stage), bytes=sum(x['size'] for x in manifest['files']),
        reason='Retired 1.5B branch; all raw experiments retained; public pinned weights can be fetched again.',
        memory_headroom_before=model.memory_headroom())
    save(OUTROOT / 'retired-small-model-cache.json', record)
    shutil.rmtree(stage)
    record.update(status='RELEASED_VERIFIED_PUBLIC_CACHE',
                  memory_headroom_after=model.memory_headroom())
    save(OUTROOT / 'retired-small-model-cache.json', record)


def unit(seqs):
    out = OUTROOT / ('math1024-seqs' + str(seqs))
    require(not out.exists() and not out.is_symlink(), 'immutable output exists')
    fd = os.open(resource.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        model.lock_identity(fd)
        deadline = time.monotonic() + 1200
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                require(time.monotonic() < deadline, 'shared lock wait expired')
                print(json.dumps(dict(status='WAITING_SHARED_LOCK', seqs=seqs,
                    launcher_pid=os.getpid(), time=time.time())), flush=True)
                time.sleep(15)
        out.mkdir(); resource.OUT = out
        record = dict(status='RUNNING', start_unix_s=time.time(), launcher_pid=os.getpid(),
            lock_inode=resource.LOCK_ID, model_id=model.MODEL, revision=model.REVISION,
            model_manifest_sha256=sha(model.MANIFEST), input_sha256=sha(INPUTS),
            protocol_sha256=sha(PROTOCOL), launcher_sha256=sha(__file__),
            cell_sha256=sha(BASE / 'math_scale_cell_v1.py'), max_num_seqs=seqs,
            requests_planned=1024, output_root=str(OUTROOT),
            scope='Exploratory same-task load scaling; not held-out confirmation or new policy.')
        try:
            record['gpu_before'] = resource.gpu_idle()
            save(out / 'launcher-receipt.json', record)
            require(shutil.disk_usage(OUTROOT).free >= 2 * model.GIB, 'raw log tmpfs reserve below 2 GiB')
            if seqs == 512:
                retire_unused_small_model()
            # Large measured logs live on tmpfs. Historical compile entries
            # used 28-35 MB per configuration; 768 MiB retains ample root margin.
            require(shutil.disk_usage(BASE).free >= 768 * 1024**2, 'root disk reserve below 768 MiB')
            record['resource_profile'] = dict(raw_outputs='private tmpfs',
                executable_cache_root=str(CACHE), root_reserve_bytes=768 * 1024**2,
                shm_reserve_bytes=2 * model.GIB, ram_headroom_bytes=26 * model.GIB,
                environment={k: os.environ[k] for k in ('VLLM_CACHE_ROOT',
                    'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR',
                    'FLASHINFER_WORKSPACE_BASE', 'XDG_CACHE_HOME', 'TMPDIR', 'CUDA_CACHE_PATH')})
            save(out / 'launcher-receipt.json', record)
            manifest = model.frozen_manifest()
            identity = model.stage_identity(manifest)
            files = model.inventory(manifest, complete=True)
            save(out / 'model-reverified.json', dict(status='VERIFIED', files=files, **identity))
            require(model.memory_headroom() >= 26 * model.GIB, 'RAM headroom below 26 GiB')
            resource.gpu_idle()
            resource.bounded([sys.executable, str(BASE / 'math_scale_cell_v1.py'),
                '--model-dir', str(model.STAGE), '--inputs', str(INPUTS),
                '--output', str(out / 'native'), '--expected-gpu-uuid', resource.GPU,
                '--kv-bytes', '8589934592', '--max-seqs', str(seqs),
                '--batch-tokens', '2048', '--max-model-len', '4096'], 'native.log', 1200, fd)
            for f, expected in [('status.json', 'COMPLETE'), ('native-drain.json', 'QUALIFIED'),
                                ('measured-pressure.json', 'COMPLETE')]:
                require(json.loads((out / 'native' / f).read_text())['status'] == expected,
                        'incomplete scale cell: ' + f)
            record.update(status='COMPLETE', gpu_after=resource.gpu_idle())
        except BaseException as exc:
            record.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record['end_unix_s'] = time.time()
            save(out / 'launcher-receipt.json', record)
    finally:
        os.close(fd)


def main():
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    model.install_signals()
    protocol = json.loads(PROTOCOL.read_text())
    for name, digest in protocol['execution_sha256'].items():
        require(sha(BASE / name) == digest, 'execution file differs: ' + name)
    OUTROOT.mkdir(mode=0o700, exist_ok=False)
    CACHE.mkdir(mode=0o700, exist_ok=False)
    env = dict(VLLM_CACHE_ROOT=str(CACHE / 'vllm'),
        TORCHINDUCTOR_CACHE_DIR=str(CACHE / 'inductor'), TRITON_CACHE_DIR=str(CACHE / 'triton'),
        TORCH_EXTENSIONS_DIR=str(CACHE / 'extensions'),
        FLASHINFER_WORKSPACE_BASE=str(CACHE / 'flashinfer'), XDG_CACHE_HOME=str(CACHE / 'xdg'),
        TMPDIR=str(CACHE / 'tmp'), TEMP=str(CACHE / 'tmp'), TMP=str(CACHE / 'tmp'),
        CUDA_CACHE_PATH=str(OUTROOT / 'cuda-driver-cache'))
    for value in set(env.values()):
        Path(value).mkdir(parents=True, exist_ok=False)
    os.environ.update(env, CUDA_CACHE_MAXSIZE='268435456', PYTHONDONTWRITEBYTECODE='1')
    save(OUTROOT / 'owner.json', dict(purpose='C research math development scale runs',
        launcher_pid=os.getpid(), protocol_sha256=sha(PROTOCOL), created_unix_s=time.time()))
    for seqs in (512, 1024):
        unit(seqs)


if __name__ == '__main__':
    main()

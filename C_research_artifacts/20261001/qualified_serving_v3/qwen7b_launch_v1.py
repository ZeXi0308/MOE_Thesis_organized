#!/usr/bin/env python3
"""Independent one-shard or Q0/Q1 units; immutable outputs, persistent owned stage."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import sys
import time

sys.dont_write_bytecode = True
import health_launch as resource
from range_download_v2 import download_ranged, memory_headroom

BASE = Path(__file__).resolve().parent
STAGE = Path('/dev/shm/c-qwen7b-qualified-20261001-v1')
MARKER = 'MODEL_IDENTITY.json'
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
MANIFEST = BASE / 'qwen7b_model_manifest.json'
FREEZE = BASE / 'qwen7b_freeze_v1.json'
SHARDS = tuple(f'model-{i:05d}-of-00004.safetensors' for i in range(1, 5))
Q0 = BASE / 'qwen7b-q0-v1'
SUMMARY_INPUTS = BASE / 'summary_inputs_v1.json'
GIB = 1024 ** 3
require, sha, save = resource.require, resource.sha, resource.save


def frozen_manifest():
    frozen = json.loads(FREEZE.read_text())['files_sha256']
    required = {'qwen7b_launch_v1.py', 'health_launch.py', 'range_download_v2.py',
        'qwen7b_model_manifest.json', 'health_native.py', 'health_native_summary.py',
        'health_analyze_v2.py', 'workload.json', 'summary_inputs_v1.json'}
    require(required <= set(frozen), 'freeze omits a required execution dependency')
    for name, digest in frozen.items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts and sha(BASE / p) == digest,
                'frozen file differs: ' + name)
    m = json.loads(MANIFEST.read_text())
    require(m['model_id'] == MODEL and m['revision'] == REVISION, 'model identity differs')
    names = [i['filename'] for i in m['files']]
    require(len(names) == len(set(names)) and
            set(n for n in names if n.endswith('.safetensors')) == set(SHARDS), 'weight inventory differs')
    for item in m['files']:
        n = item['filename']
        require(Path(n).name == n and n not in ('.', '..', MARKER) and
                type(item['size']) is int and item['size'] > 0 and
                len(item['sha256']) == 64, 'invalid manifest entry')
        if n not in SHARDS:
            require('qwen7b_metadata/' + n in frozen, 'metadata absent from freeze: ' + n)
    return m


def verify_file(path, item):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == item['size']
            and sha(path) == item['sha256'], 'size/SHA differs: ' + str(path))


def stage_identity(m, create=False):
    fresh = not STAGE.exists() and not STAGE.is_symlink()
    if fresh:
        require(create, 'model stage missing')
        STAGE.mkdir(mode=0o700)
    require(STAGE.is_dir() and not STAGE.is_symlink() and
            stat.S_IMODE(STAGE.stat().st_mode) == 0o700, 'stage ownership/path differs')
    info = STAGE.stat()
    expected = dict(schema='qwen7b-owned-stage-v1', model_id=MODEL, revision=REVISION,
                    manifest_sha256=sha(MANIFEST), stage_inode=[info.st_dev, info.st_ino])
    marker = STAGE / MARKER
    if fresh:
        with marker.open('x') as f:
            json.dump(expected, f, indent=2)
    require(marker.is_file() and not marker.is_symlink() and
            json.loads(marker.read_text()) == expected, 'model marker differs; never adopt this stage')
    allowed = {i['filename'] for i in m['files']}
    require(all(p.name in allowed | {n + '.part' for n in allowed} | {MARKER}
                for p in STAGE.iterdir()), 'unknown file in model stage')
    return expected


def inventory(m, complete=False):
    verified = []
    for item in m['files']:
        p = STAGE / item['filename']
        if p.exists() or p.is_symlink():
            verify_file(p, item)
            verified.append(item)
        else:
            require(not complete, 'incomplete model cache: ' + item['filename'])
    return verified


def prep_out(shard):
    return BASE / f'qwen7b-prepare-shard{SHARDS.index(shard) + 1}-v1'


def lock_identity(fd):
    s = os.fstat(fd)
    require(stat.S_ISREG(s.st_mode) and (s.st_dev, s.st_ino) == resource.LOCK_ID,
            'existing shared lock identity differs')


def install_signals():
    def interrupted(signum, _frame):
        if resource.SPAWNING:
            resource.PENDING_SIGNAL = signum
            return
        raise InterruptedError('qwen7b launcher received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)


def prepare_worker(shard, lock_fd):
    out = prep_out(shard)
    record = dict(status='PREPARING', shard=shard, start_unix_s=time.time(),
                  network_deadline_s=1200, partial_policy='Retain failures; .part is never a verified shard.')
    try:
        lock_identity(lock_fd)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m = frozen_manifest(); deadline = time.monotonic() + 1200
        record['stage_identity'] = stage_identity(m, create=True)
        existing = inventory(m)
        present = {i['filename'] for i in existing}
        selected = next(i for i in m['files'] if i['filename'] == shard)
        missing = [i for i in m['files'] if i['filename'] not in present and
                   (i['filename'] not in SHARDS or i['filename'] == shard)]
        pending_bytes = sum(i['size'] for i in missing)
        record.update(verified_files_before=existing, pending_bytes=pending_bytes,
                      memory_headroom_before=memory_headroom())
        require(shutil.disk_usage(STAGE).free >= pending_bytes + GIB and
                memory_headroom() >= pending_bytes + 24 * GIB, 'insufficient incremental RAM headroom')
        save(out / 'shard-prepare.json', record)
        for item in missing:
            name = item['filename']; partial = STAGE / (name + '.part')
            require(not partial.exists() and not partial.is_symlink(), 'retained partial exists; no automatic retry')
            if name in SHARDS:
                def progress(count):
                    record['received_bytes'] = record.get('received_bytes', 0) + count
                    require(memory_headroom() >= 24 * GIB, 'RAM reserve fell below 24 GiB')
                    save(out / 'shard-prepare.json', record)
                url = f'https://hf-mirror.com/{MODEL}/resolve/{REVISION}/{name}'
                record['download'] = download_ranged(url, partial, item['size'], item['sha256'],
                                                    deadline, progress_callback=progress)
            else:
                source = BASE / 'qwen7b_metadata' / name
                verify_file(source, item)
                with source.open('rb') as src, partial.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
            verify_file(partial, item)
            os.link(partial, STAGE / name, follow_symlinks=False)
            partial.unlink()
        verify_file(STAGE / shard, selected)
        record.update(status='ALREADY_VERIFIED' if shard in present else 'VERIFIED',
                      sha256=selected['sha256'], size=selected['size'])
    except BaseException as exc:
        record.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(end_unix_s=time.time(), retained_stage=str(STAGE))
        save(out / 'shard-prepare.json', record)


def main(args):
    m = frozen_manifest()
    action = 'prepare' if args.prepare_shard else 'q0' if args.run_q0 else 'q1'
    out = prep_out(args.prepare_shard) if action == 'prepare' else BASE / f'qwen7b-{action}-v1'
    require(not out.exists() and not out.is_symlink(), 'existing immutable output; no repeat')
    fd = os.open(resource.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        lock_identity(fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status='GPU_DEFERRED', reason='shared lock busy')), flush=True)
            return 75
        out.mkdir(); resource.OUT = out
        record = dict(status='RUNNING', action=action, start_unix_s=time.time(),
            model_id=MODEL, revision=REVISION, manifest_sha256=sha(MANIFEST),
            freeze_sha256=sha(FREEZE), launcher_pid=os.getpid(), lock_inode=resource.LOCK_ID)
        try:
            record['gpu_before'] = resource.gpu_idle()
            save(out / 'launcher-receipt.json', record)
            require(shutil.disk_usage(BASE).free >= GIB, 'root disk reserve below 1 GiB')
            if action == 'prepare':
                code = ('import qwen7b_launch_v1 as m\nm.install_signals()\n'
                        f'm.prepare_worker({args.prepare_shard!r}, {fd})\n')
                resource.bounded([sys.executable, '-c', code], 'prepare.log', 1260, fd)
                result = json.loads((out / 'shard-prepare.json').read_text())
                require(result['status'] in ('VERIFIED', 'ALREADY_VERIFIED'), 'shard preparation incomplete')
            else:
                if action == 'q1':
                    prior = json.loads((Q0 / 'launcher-receipt.json').read_text())
                    require(prior['status'] == 'COMPLETE' and prior['action'] == 'q0' and
                            prior['manifest_sha256'] == sha(MANIFEST), 'new 7B Q0 lifecycle/model mismatch')
                    from health_analyze_v2 import analyze
                    gate = analyze(Q0 / 'native', BASE / 'workload.json')
                    save(out / 'q0-health-gate.json', gate)
                    require(gate['validity'] == 'COMPLETE' and gate['screening'] == 'PASS_DEVELOPMENT_SCREEN',
                            'new 7B Q0 health gate failed')
                identity = stage_identity(m)
                verified = inventory(m, complete=True)
                save(out / 'model-reverified.json', dict(status='VERIFIED', files=verified, **identity))
                require(memory_headroom() >= 24 * GIB, 'native RAM reserve below 24 GiB')
                resource.gpu_idle()
                cell = 'health_native.py' if action == 'q0' else 'health_native_summary.py'
                inputs = BASE / 'workload.json' if action == 'q0' else SUMMARY_INPUTS
                resource.bounded([sys.executable, str(BASE / cell), '--model-dir', str(STAGE),
                    '--inputs', str(inputs), '--output', str(out / 'native'), '--expected-gpu-uuid', resource.GPU,
                    '--kv-bytes', '8589934592', '--max-seqs', '32', '--batch-tokens', '1024',
                    '--max-model-len', '4096'], 'native.log', 1200, fd)
                require(json.loads((out / 'native/status.json').read_text())['status'] == 'COMPLETE',
                        'native cell incomplete')
            record.update(status='COMPLETE', gpu_after=resource.gpu_idle())
        except BaseException as exc:
            record.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            try:
                record['gpu_after'] = resource.gpu_idle()
            except Exception as exc:
                record.update(status='INCOMPLETE', gpu_after_error=f'{type(exc).__name__}: {exc}')
            record.update(end_unix_s=time.time(), retained_stage=str(STAGE),
                          scope='One independent shard or model-health unit; no policy comparison.')
            save(out / 'launcher-receipt.json', record)
        return 0 if record['status'] == 'COMPLETE' else 70
    finally:
        os.close(fd)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare-shard', choices=SHARDS)
    actions.add_argument('--run-q0', action='store_true')
    actions.add_argument('--run-q1', action='store_true')
    install_signals()
    sys.exit(main(p.parse_args()))

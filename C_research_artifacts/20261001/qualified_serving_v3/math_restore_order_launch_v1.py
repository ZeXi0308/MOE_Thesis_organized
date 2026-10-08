#!/usr/bin/env python3
"""Restoration-object ordering at the strongest1024/B4096 fixed48 configuration."""
import fcntl
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import stat
import sys
import time

sys.dont_write_bytecode = True
import qwen7b_launch_v2 as model

BASE = Path(__file__).resolve().parent
OUTROOT = Path('/dev/shm/c-math-restore-order-runs-20261002-v1')
INPUTS = BASE / 'math_inputs1024_v1.json'
PROTOCOL = BASE / 'math_restore_order_protocol_v1.json'
CACHE = BASE / 'math_scale_runtime_cache_v2'
resource = model.resource
require, sha, save = resource.require, resource.sha, resource.save
COMPRESS_FILES = ('measured-pressure.json', 'measured-host-returns.jsonl', 'measured-steps.json')
ROOT_RESERVE, SHM_RESERVE, RAM_RESERVE = 768 * 1024**2, 2 * model.GIB, 26 * model.GIB


def time_left(deadline):
    return deadline - time.monotonic()


def resource_snapshot():
    snap = dict(time_unix_s=time.time(), root_free_bytes=shutil.disk_usage(BASE).free,
        shm_free_bytes=shutil.disk_usage(OUTROOT).free,
        ram_headroom_bytes=model.memory_headroom(), reasons=[])
    for field, minimum, reason in [('root_free_bytes', ROOT_RESERVE, 'root_below_768MiB'),
            ('shm_free_bytes', SHM_RESERVE, 'shm_below_2GiB'),
            ('ram_headroom_bytes', RAM_RESERVE, 'RAM_below_26GiB')]:
        if snap[field] < minimum:
            snap['reasons'].append(reason)
    try:
        snap['gpu'] = resource.gpu_idle()
    except RuntimeError as exc:
        if str(exc) != 'GPU unavailable; never terminate another task':
            raise
        snap['reasons'].append('gpu_unavailable')
        snap['gpu_error'] = str(exc)
    return snap


def stream_digest(stream, deadline):
    digest, size = hashlib.sha256(), 0
    while True:
        require(time_left(deadline) > 0, 'unit deadline reached during compression verification')
        chunk = stream.read(1024 * 1024)
        if not chunk:
            return digest.hexdigest(), size
        digest.update(chunk); size += len(chunk)


def compress_completed(native, receipt_path, deadline):
    """Only these completed-cell logs; verify decoded bytes before removing source."""
    require(native.is_dir() and not native.is_symlink(), 'native output path invalid')
    require(json.loads((native / 'status.json').read_text())['status'] == 'COMPLETE',
            'never compress a running/incomplete cell')
    record = dict(schema='c-postrun-lossless-xz-v1', status='RUNNING',
        start_unix_s=time.time(), native_root=str(native), files=[],
        scope='Post-child compression only; decoded SHA256 and byte count verified before each original removal.')
    require(not receipt_path.exists() and not receipt_path.is_symlink(), 'compression receipt exists')
    save(receipt_path, record)
    try:
        for name in COMPRESS_FILES:
            source = native / name
            packed, partial = native / (name + '.xz'), native / (name + '.xz.part')
            require(source.is_file() and not source.is_symlink(), 'raw log missing/symlinked: ' + name)
            require(not packed.exists() and not packed.is_symlink() and
                    not partial.exists() and not partial.is_symlink(), 'archive path already exists')
            item = dict(source=name, archive=packed.name, status='COMPRESSING', raw_removed=False)
            record['files'].append(item); save(receipt_path, record)
            digest, size = hashlib.sha256(), 0
            fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as src:
                before = os.fstat(src.fileno())
                require(stat.S_ISREG(before.st_mode), 'raw log not regular')
                with lzma.open(partial, 'xb', format=lzma.FORMAT_XZ,
                               check=lzma.CHECK_CRC64, preset=3) as dst:
                    while True:
                        require(time_left(deadline) > 0, 'unit deadline reached during compression')
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk); size += len(chunk); dst.write(chunk)
                after = os.fstat(src.fileno())
            source_identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
            require(source_identity(before) == source_identity(after) ==
                    source_identity(source.lstat()) and size == before.st_size,
                    'raw log changed during compression')
            with lzma.open(partial, 'rb') as decoded:
                restored_digest, restored_size = stream_digest(decoded, deadline)
            require(restored_digest == digest.hexdigest() and restored_size == size,
                    'decoded compression checksum/size differs; raw retained')
            # Link refuses replacement of a pre-existing final archive.
            os.link(partial, packed, follow_symlinks=False); partial.unlink()
            item.update(status='VERIFIED_ARCHIVE_RAW_RETAINED', source_sha256=digest.hexdigest(),
                source_bytes=size, decoded_sha256=restored_digest, decoded_bytes=restored_size,
                archive_sha256=sha(packed), archive_bytes=packed.stat().st_size,
                source_identity=list(source_identity(before)))
            save(receipt_path, record)
            require(not source.is_symlink() and source_identity(source.lstat()) ==
                    source_identity(before), 'raw log identity changed before replacement')
            source.unlink()
            item.update(status='COMPRESSED_VERIFIED', raw_removed=True)
            save(receipt_path, record)
        record['status'] = 'COMPLETE'
    except BaseException as exc:
        record.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['end_unix_s'] = time.time(); save(receipt_path, record)
    return record


def unit(name, seqs, policy, batch_tokens, margin_blocks):
    require(Path(name).name == name and name not in ('.', '..'), 'invalid unit name')
    require(seqs == 1024 and policy in ('restore-fixed-margin','restore-first-fit','restore-min-recompute') and batch_tokens == 4096 and margin_blocks == 48,
            'only the specified fixed48 restoration-order arms are authorized')
    out = OUTROOT / name
    require(not out.exists() and not out.is_symlink(), 'immutable output exists')
    out.mkdir(mode=0o700); resource.OUT = out
    deadline = time.monotonic() + 1200
    record = dict(status='WAITING_RESOURCES', start_unix_s=time.time(),
        launcher_pid=os.getpid(), lock_inode=resource.LOCK_ID, model_id=model.MODEL,
        revision=model.REVISION, model_manifest_sha256=sha(model.MANIFEST),
        input_sha256=sha(INPUTS), protocol_sha256=sha(PROTOCOL), launcher_sha256=sha(__file__),
        cell_sha256=sha(BASE / 'math_restore_order_cell_v1.py'), max_num_seqs=seqs,
        policy=policy, batch_tokens=batch_tokens, fixed_margin_blocks=margin_blocks, requests_planned=1024,
        output_root=str(OUTROOT), unit_budget_s=1200, child_launch_count=0, gpu_child_status='NOT_STARTED',
        preflight_attempts=[], scope='Only selection in the first16 contiguous PREEMPTED waiting candidates changes; never cross fresh requests.',
        resource_profile=dict(raw_outputs='private tmpfs', executable_cache_root=str(CACHE),
            root_reserve_bytes=ROOT_RESERVE, shm_reserve_bytes=SHM_RESERVE,
            ram_headroom_bytes=RAM_RESERVE, environment={k: os.environ[k] for k in
            ('VLLM_CACHE_ROOT', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
             'TORCH_EXTENSIONS_DIR', 'FLASHINFER_WORKSPACE_BASE', 'XDG_CACHE_HOME',
             'TMPDIR', 'CUDA_CACHE_PATH')}))
    save(out / 'launcher-receipt.json', record)
    fd = None
    try:
        fd = os.open(resource.LOCK, os.O_RDWR | os.O_NOFOLLOW)
        model.lock_identity(fd)
        while True:
            require(time_left(deadline) > 0, 'unit resource wait expired before GPU launch')
            attempt = dict(index=len(record['preflight_attempts']), start_unix_s=time.time())
            record['preflight_attempts'].append(attempt)
            locked, ready = False, False
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    model.lock_identity(fd)
                except BlockingIOError:
                    attempt['status'] = 'WAITING_SHARED_LOCK'
                if locked:
                    attempt['before_model_verification'] = resource_snapshot()
                    if not attempt['before_model_verification']['reasons']:
                        manifest = model.frozen_manifest()
                        identity = model.stage_identity(manifest)
                        files = model.inventory(manifest, complete=True)
                        attempt['model_verified_unix_s'] = time.time()
                        save(out / 'model-reverified.json', dict(status='VERIFIED',
                            preflight_attempt=attempt['index'], files=files, **identity))
                        attempt['after_model_verification'] = resource_snapshot()
                        ready = not attempt['after_model_verification']['reasons']
                    attempt['status'] = 'READY' if ready else 'WAITING_RESOURCES'
                save(out / 'launcher-receipt.json', record)
                if ready:
                    remaining = int(time_left(deadline))
                    require(remaining > 0, 'unit deadline reached after model verification')
                    record.update(status='RUNNING', child_launch_count=1, gpu_child_status='RUNNING',
                        gpu_before=attempt['after_model_verification']['gpu'],
                        gpu_child_start_unix_s=time.time(), child_timeout_s=remaining)
                    save(out / 'launcher-receipt.json', record)
                    resource.bounded([sys.executable, str(BASE / 'math_restore_order_cell_v1.py'),
                        '--model-dir', str(model.STAGE), '--inputs', str(INPUTS),
                        '--output', str(out / 'native'), '--expected-gpu-uuid', resource.GPU,
                        '--kv-bytes', '8589934592', '--max-seqs', str(seqs),
                        '--batch-tokens', str(batch_tokens), '--max-model-len', '4096',
                        '--policy', policy, '--fixed-margin-blocks', str(margin_blocks)], 'native.log', remaining, fd)
                    for filename, expected in [('status.json', 'COMPLETE'),
                            ('native-drain.json', 'QUALIFIED'), ('measured-pressure.json', 'COMPLETE')]:
                        require(json.loads((out / 'native' / filename).read_text())['status'] == expected,
                                'incomplete native control: ' + filename)
                    policy_receipt = 'restore-fixed-margin-policy.json' if policy == 'restore-fixed-margin' else 'restore-cost-order-policy.json'
                    require(json.loads((out / 'native' / policy_receipt).read_text())['status'] == 'COMPLETE',
                            'restoration policy receipt incomplete')
                    record.update(status='GPU_COMPLETE_POSTPROCESSING', gpu_child_status='COMPLETE',
                                  gpu_after=resource.gpu_idle(), gpu_child_end_unix_s=time.time())
                    save(out / 'launcher-receipt.json', record)
                    break  # No path returns to preflight after child launch.
            finally:
                if locked:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                attempt['end_unix_s'] = time.time()
                save(out / 'launcher-receipt.json', record)
            print(json.dumps(dict(status=attempt['status'], unit=name,
                attempt=attempt['index'], time=time.time())), flush=True)
            remaining = time_left(deadline)
            require(remaining > 0, 'unit resource wait expired before GPU launch')
            time.sleep(min(15, remaining))  # Shared lock is released before every wait.
        # The child exited and the shared GPU lock was released. Compression is
        # optional housekeeping: it never changes a completed GPU result status.
        try:
            compressed = compress_completed(out / 'native', out / 'compression-receipt.json', deadline)
            record['compression_status'] = compressed['status']
        except (InterruptedError, KeyboardInterrupt):
            raise
        except Exception as exc:
            record.update(compression_status='INCOMPLETE',
                          compression_error=f'{type(exc).__name__}: {exc}')
            print(json.dumps(dict(status='COMPRESSION_INCOMPLETE_RAW_OR_VERIFIED_ARCHIVE_RETAINED',
                                  unit=name, error=record['compression_error'])), flush=True)
        record.update(status='COMPLETE', postprocessing_end_unix_s=time.time())
    except BaseException as exc:
        record.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
        if record['gpu_child_status'] == 'RUNNING':
            record['gpu_child_status'] = 'INCOMPLETE'
        raise
    finally:
        if fd is not None:
            os.close(fd)
        record['end_unix_s'] = time.time(); save(out / 'launcher-receipt.json', record)
    return record


def main():
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    model.install_signals()
    protocol = json.loads(PROTOCOL.read_text())
    for name, digest in protocol['execution_sha256'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and sha(BASE / path) == digest,
                'execution file differs: ' + name)
    arms = protocol['arms']
    require([(a['max_seqs'], a['policy'], a['batch_tokens'], a['fixed_margin_blocks']) for a in arms] ==
            [(1024, p, 4096, 48) for p in ('restore-fixed-margin','restore-first-fit','restore-min-recompute')] and
            len({a['name'] for a in arms}) == 3, 'restoration-order protocol differs')
    OUTROOT.mkdir(mode=0o700, exist_ok=False)
    previous = Path('/dev/shm/c-math-scale-runs-20261002-v2')
    require(CACHE.is_dir() and not CACHE.is_symlink() and CACHE.resolve() == CACHE,
            'owned executable cache missing or symlinked')
    for previous_seqs in (512, 1024):
        receipt = json.loads((previous / ('math1024-seqs' + str(previous_seqs)) / 'launcher-receipt.json').read_text())
        require(receipt['status'] == 'COMPLETE' and
                receipt['protocol_sha256'] == sha(BASE / 'math_scale_protocol_v2.json') and
                receipt['resource_profile']['executable_cache_root'] == str(CACHE),
                'owned previous runtime cache provenance differs')
    env = dict(VLLM_CACHE_ROOT=str(CACHE / 'vllm'),
        TORCHINDUCTOR_CACHE_DIR=str(CACHE / 'inductor'), TRITON_CACHE_DIR=str(CACHE / 'triton'),
        TORCH_EXTENSIONS_DIR=str(CACHE / 'extensions'),
        FLASHINFER_WORKSPACE_BASE=str(CACHE / 'flashinfer'), XDG_CACHE_HOME=str(CACHE / 'xdg'),
        TMPDIR=str(CACHE / 'tmp'), TEMP=str(CACHE / 'tmp'), TMP=str(CACHE / 'tmp'),
        CUDA_CACHE_PATH=str(previous / 'cuda-driver-cache'))
    for value in set(env.values()):
        cache_path = Path(value)
        require(cache_path.is_dir() and not cache_path.is_symlink() and cache_path.resolve() == cache_path,
                'owned existing runtime cache path differs')
    os.environ.update(env, CUDA_CACHE_MAXSIZE='268435456', PYTHONDONTWRITEBYTECODE='1')
    save(OUTROOT / 'owner.json', dict(purpose='C research restoration-object choice action experiment',
        launcher_pid=os.getpid(), protocol_sha256=sha(PROTOCOL), created_unix_s=time.time()))
    for arm in arms:
        unit(arm['name'], arm['max_seqs'], arm['policy'], arm['batch_tokens'], arm['fixed_margin_blocks'])


if __name__ == '__main__':
    main()

"""Normal-capacity B probe; reuse the existing whole-GPU controller boundaries."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('b_original_controller', ROOT.parent/'run_group.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
write = base.write


def acquire_gpu_lock(fd, wait_seconds, receipt, receipt_path):
    """Wait on this one open lock fd; no child or GPU work precedes acquisition."""
    started = time.monotonic()
    deadline = started + wait_seconds
    waiting = False
    while True:
        if waiting and time.monotonic() >= deadline:
            raise TimeoutError(f'GPU lock wait exceeded {wait_seconds:g} seconds')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if wait_seconds == 0:
                raise
            if not waiting:
                waiting = True
                receipt.update(status='WAIT_GPU', pid=os.getpid(), lock_wait=dict(
                    started_unix_s=time.time(), timeout_seconds=wait_seconds))
                write(receipt_path, receipt)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'GPU lock wait exceeded {wait_seconds:g} seconds')
            time.sleep(min(5.0, remaining))
        else:
            receipt.update(status='RUNNING', acquired_unix_s=time.time())
            if waiting:
                receipt['lock_wait'].update(acquired_unix_s=time.time(),
                    elapsed_seconds=time.monotonic()-started)
            write(receipt_path, receipt)
            return


def interrupted(signum, frame):
    raise InterruptedError(f'Controller interrupted by signal {signum}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--wait-lock-seconds', type=float, default=0)
    args = parser.parse_args()
    if not math.isfinite(args.wait_lock_seconds) or args.wait_lock_seconds < 0:
        parser.error('--wait-lock-seconds must be finite and nonnegative')
    p = json.loads(args.plan.read_text())
    session = Path(p['session_dir'])
    session.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='PREFLIGHT', plan=p, cells=[])
    write(session/'receipt.json', receipt)
    fd = os.open(p['lock_path'], os.O_RDWR | os.O_NOFOLLOW)
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in old_handlers:
            signal.signal(sig, interrupted)
        acquire_gpu_lock(fd, args.wait_lock_seconds, receipt, session/'receipt.json')
        os.dup2(fd, 9, inheritable=True)
        receipt['lock'] = dict(device=os.fstat(fd).st_dev, inode=os.fstat(fd).st_ino)
        if int(Path('/sys/fs/cgroup/memory.max').read_text()) > p['host_limit_bytes']:
            raise RuntimeError('Host budget differs')
        receipt['source_sha256'] = {str(f.relative_to(ROOT.parent)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [*sorted(ROOT.glob('*.py')), *sorted((ROOT.parent/'pkg').glob('*.py'))]}
        tasks = [(cap, 'native') for cap in p['baseline_caps']]
        if p['baseline_caps'] != [64, 192, 256] or p['engine_max_num_seqs'] != 256:
            raise ValueError('Only the three predeclared operating points are allowed')
        kv_bytes = 'auto'
        i = 0
        while i < len(tasks):
            state = base.checked_boundary(p['gpu_uuid'])
            receipt.setdefault('cell_boundaries', []).append(state)
            write(session/'receipt.json', receipt)
            if not state['empty']:
                raise RuntimeError('GPU_BUSY_OR_UNKNOWN before cell; no automatic retry')
            cap, mode = tasks[i]
            cell = session/f'cell-{i:02d}-cap{cap}-{mode}'
            cell.mkdir()
            env = {k: v for k, v in os.environ.items() if not k.startswith(('A_', 'B_', 'H1_', 'VLLM_', 'HF_', 'HUGGINGFACE_', 'TRANSFORMERS_'))}
            env.update(HF_HOME=p['hf_cache'], HF_HUB_CACHE=p['hf_cache']+'/hub', HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES=p['gpu_uuid'],
                VLLM_USE_FLASHINFER_SAMPLER='0', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                A_NATIVE_OLDEST_ADMISSION='native', A_NATIVE_OLDEST_REPEAT='1',
                A_RECOVERY_LEASE_MODE='off', B_RECOVERY_ORDER=mode, LD_LIBRARY_PATH=p['library_path'])
            for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONINSPECT', 'PYTHONUSERBASE', 'VIRTUAL_ENV'):
                env.pop(key, None)
            if json.loads((Path(p['runtime_overlay'])/'repair.json').read_text())['status'] != 'CPU_IMPORT_OK':
                raise RuntimeError('Private torch import view not qualified')
            env['PYTHONPATH'] = p['runtime_overlay']
            for key in ('XDG_CACHE_HOME','TMPDIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','VLLM_CACHE_ROOT','TORCH_HOME','CUDA_CACHE_PATH'):
                d = cell/'cache'/key
                d.mkdir(parents=True)
                env[key] = str(d)
            command = [p['python'], str(ROOT/'run_cell.py'), '--inputs', str(ROOT/'inputs'/f'cap{cap}'),
                '--warmup-inputs', str(ROOT.parent/'pkg/warmups'), '--variant', 'native_full_ordinary_only',
                '--ordinary-backfill', '--measurement-mode', 'performance', '--output-dir', str(cell/'output'),
                '--max-num-seqs', str(p['engine_max_num_seqs']), '--gpu-kv-bytes', str(kv_bytes),
                '--max-seconds', str(p['measurement_max_seconds'])]
            write(cell/'command.json', dict(argv=command, environment_overrides={k:v for k,v in env.items() if os.environ.get(k) != v}))
            start = time.monotonic()
            with (cell/'run.log').open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(9,), start_new_session=True)
                try:
                    code = child.wait(timeout=p['per_cell_seconds'])
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise
            receipt['cells'].append(dict(cap=cap, mode=mode, exit_code=code, wall_s=time.monotonic()-start))
            write(session/'receipt.json', receipt)
            if code:
                raise RuntimeError('Cell failed; all raw retained, no automatic retry')
            memory = json.loads((cell/'output/memory-after-init.json').read_text())
            actual = memory['kv_storage_bytes']
            if kv_bytes == 'auto':
                kv_bytes = actual
                receipt['fixed_gpu_kv_bytes'] = actual
            elif actual != kv_bytes:
                raise RuntimeError('Actual GPU KV capacity changed between cells')
            events = json.loads((cell/'output/recovery-order.json').read_text())['events']
            orders = [e for e in events if e['kind'] == 'reorder']
            opportunity = any(e['before'] != e['legal_flush_order'] and e['flush_fallback'] is None for e in orders)
            receipt['cells'][-1]['legal_flush_opportunity'] = opportunity
            if i == 2:
                # Only the predeclared high-pressure point can expand into a policy contrast.
                if not opportunity:
                    receipt['status'] = 'BASELINES_COMPLETE_NO_HIGH_PRESSURE_FLUSH_REORDER'
                    break
                age_equal = all(e['before'] == sorted(e['before']) for e in orders)
                sequence = ['flush_first', 'flush_first', 'native'] if age_equal else ['age', 'flush_first', 'flush_first', 'age', 'native']
                tasks.extend((256, mode) for mode in sequence)
                receipt['simple_age_merged_with_native'] = age_equal
            i += 1
        else:
            receipt['status'] = 'COMPLETE'
        receipt['final_gpu'] = base.checked_boundary(p['gpu_uuid'])
    except Exception as error:
        receipt.update(status='ABORTED', error=f'{type(error).__name__}: {error}')
    finally:
        receipt['end_unix_s'] = time.time()
        write(session/'receipt.json', receipt)
        os.close(9) if fd != 9 and 'lock' in receipt else None
        os.close(fd)
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
    print(json.dumps(receipt))
    return 0 if receipt['status'].startswith(('COMPLETE', 'BASELINES_COMPLETE')) else 1


if __name__ == '__main__':
    sys.exit(main())

"""One whole-group lock, original 128-request workload; refuse busy/unknown GPU."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def boundary(uuid):
    flags = ['--format=csv,noheader,nounits']
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.used,utilization.gpu', *flags], text=True).strip()
    procs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', *flags], text=True).strip()
    row = [x.strip() for x in gpu.split(',')]
    return dict(unix_s=time.time(), gpu=gpu, processes=procs,
                empty=len(gpu.splitlines()) == 1 and row[0] == uuid and int(row[-2]) <= 256 and int(row[-1]) == 0 and not procs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.plan.read_text())
    root = Path(__file__).resolve().parent
    session = Path(p['session_dir'])
    session.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='PREFLIGHT', cells=[], plan=p)
    write(session/'receipt.json', receipt)
    fd = os.open(p['lock_path'], os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.dup2(fd, 9, inheritable=True)
        receipt['lock'] = dict(device=os.fstat(fd).st_dev, inode=os.fstat(fd).st_ino)
        state = boundary(p['gpu_uuid'])
        receipt['initial_gpu'] = state
        if not state['empty']:
            raise RuntimeError('GPU_BUSY_OR_UNKNOWN: no CUDA initialization or experiment permitted')
        if int(Path('/sys/fs/cgroup/memory.max').read_text()) > p['host_limit_bytes']:
            raise RuntimeError('Host budget differs')
        receipt['source_sha256'] = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [root/'run_group.py', *sorted((root/'pkg').glob('*.py'))]}
        modes = ['native', 'age', 'flush_first', 'flush_first', 'age', 'native']
        i = 0
        while i < len(modes):
            if not boundary(p['gpu_uuid'])['empty']:
                raise RuntimeError('GPU_BUSY_OR_UNKNOWN between cells')
            mode = modes[i]
            cell = session / f'cell-{i:02d}-{mode}'
            cell.mkdir()
            env = {k: v for k, v in os.environ.items() if not k.startswith(('A_', 'B_', 'H1_', 'VLLM_', 'HF_', 'HUGGINGFACE_', 'TRANSFORMERS_'))}
            env.update(HF_HOME=p['hf_cache'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                HF_HUB_CACHE=p['hf_cache']+'/hub', CUDA_VISIBLE_DEVICES=p['gpu_uuid'],
                VLLM_USE_FLASHINFER_SAMPLER='0', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                A_NATIVE_OLDEST_ADMISSION='native', A_NATIVE_OLDEST_REPEAT='1',
                A_RECOVERY_LEASE_MODE='off', B_RECOVERY_ORDER=mode,
                LD_LIBRARY_PATH=p['library_path'])
            for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONINSPECT', 'PYTHONUSERBASE', 'VIRTUAL_ENV'):
                env.pop(key, None)
            if p.get('runtime_overlay'):
                repair = json.loads((Path(p['runtime_overlay'])/'repair.json').read_text())
                if repair['status'] != 'CPU_IMPORT_OK':
                    raise RuntimeError('Private torch import view not qualified')
                env['PYTHONPATH'] = p['runtime_overlay']
            for key in ('XDG_CACHE_HOME','TMPDIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','VLLM_CACHE_ROOT','TORCH_HOME','CUDA_CACHE_PATH'):
                d = cell/'cache'/key
                d.mkdir(parents=True)
                env[key] = str(d)
            command = [p['python'], str(root/'pkg/run_recovery_cadence.py'), '--inputs', str(root/'pkg/inputs'),
                '--warmup-inputs', str(root/'pkg/warmups'), '--variant', 'native_full_ordinary_only',
                '--ordinary-backfill', '--measurement-mode', 'performance', '--output-dir', str(cell/'output')]
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
            receipt['cells'].append(dict(mode=mode, exit_code=code, wall_s=time.monotonic()-start))
            write(session/'receipt.json', receipt)
            if code:
                raise RuntimeError('Cell failed; raw retained, no automatic retry')
            if i == 0:
                events = json.loads((cell/'output/recovery-order.json').read_text())['events']
                orders = [e for e in events if e['kind'] == 'reorder']
                opportunity = any(e['before'] != e['legal_flush_order'] and e['flush_fallback'] is None for e in orders)
                receipt['natural_flush_opportunity'] = opportunity
                if not opportunity:
                    receipt['status'] = 'BASELINE_COMPLETE_NO_NATURAL_FLUSH_REORDER'
                    break
                if all(e['before'] == sorted(e['before']) for e in orders):
                    modes = ['native', 'flush_first', 'flush_first', 'native']
                    receipt['simple_age_merged_with_native'] = True
            i += 1
        else:
            receipt['status'] = 'COMPLETE'
        receipt['final_gpu'] = boundary(p['gpu_uuid'])
    except Exception as error:
        receipt.update(status='ABORTED', error=f'{type(error).__name__}: {error}')
    finally:
        receipt['end_unix_s'] = time.time()
        write(session/'receipt.json', receipt)
        os.close(fd)
    print(json.dumps(receipt))
    return 0 if receipt['status'].startswith(('COMPLETE','BASELINE_COMPLETE')) else 1


if __name__ == '__main__':
    sys.exit(main())

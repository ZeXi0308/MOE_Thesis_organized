"""Bounded whole-group common lock; abort on any foreign GPU process."""
import argparse
import fcntl
import json
import os
import signal
import shutil
from pathlib import Path
import subprocess
import sys
import time
from selector import validate_target_spec
from run_cell import telemetry_arm_modes, store_replay_arm_modes, host_load_arm_modes

p=argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--gpu-uuid', required=True, help='Exact GPU verified for this run')
p.add_argument('--arms', default='host,recompute,recompute,host')
p.add_argument('--interval', type=float, default=0)
p.add_argument('--threshold', type=int, default=0)
p.add_argument('--output-tokens', type=int, default=256)
p.add_argument('--requests', type=int, default=64)
p.add_argument('--workload', type=Path, required=True)
p.add_argument('--kv-bytes', type=int, default=None)
p.add_argument('--host-gib', type=int, default=32)
p.add_argument('--max-num-seqs', type=int, default=256)
p.add_argument('--batch-tokens', type=int, default=2048)
p.add_argument('--execution-mode', choices=['eager','default'], default='eager',
               help='One native execution mode shared by warmups and formal arms')
p.add_argument('--group-timeout-seconds', type=float,
               help='Optional stricter whole-child bound, including engine startup and all episodes')
p.add_argument('--max-seconds', type=float, default=600)
p.add_argument('--fixed-output', action='store_true', help='Controlled diagnostic only; normal groups respect EOS')
p.add_argument('--timing-observer', action='store_true', help='Passive timing diagnostic; identical in every arm')
p.add_argument('--native-transfer-probe', action='store_true',
               help='Passive native transfer-start wall boundary; no CUDA synchronization')
p.add_argument('--host-history-observer', action='store_true',
               help='Passive STORE cursor/job fields; same setting for warmup and every arm')
p.add_argument('--store-replay-modes',
               help='Causal one-entry R-only cursor replay: off/on per formal arm; warmup always off')
p.add_argument('--host-load-modes',
               help='Native Host async/inline per formal arm; warmup always async')
p.add_argument('--gpu-telemetry-modes', help='Diagnostic on/off sequence; does not change per-step CPU timing')
p.add_argument('--telemetry-script', type=Path, help='Explicit passive observer source to record and execute')
p.add_argument('--target-spec', type=Path,
               help='Optional single legal recovery intervention; all other recovery decisions use native Host')
p.add_argument('--wait-lock-seconds', type=float, default=0,
               help='Bounded queue wait before loading the model; default is nonblocking')
a=p.parse_args()
assert a.workload.is_file(), f'Workload file does not exist: {a.workload}'
assert a.requests > 0 and a.wait_lock_seconds >= 0
assert a.group_timeout_seconds is None or a.group_timeout_seconds > 0
assert all(arm in ('host','recompute','length','headroom') for arm in a.arms.split(','))
telemetry_arm_modes(a.gpu_telemetry_modes, len(a.arms.split(',')), a.timing_observer)
replay_modes = store_replay_arm_modes(a.store_replay_modes, len(a.arms.split(',')))
host_modes = host_load_arm_modes(a.host_load_modes, len(a.arms.split(',')))
if 'inline' in host_modes:
    assert all(arm == 'host' for arm in a.arms.split(','))
    assert a.target_spec is None and 'on' not in replay_modes
if 'on' in replay_modes:
    assert a.target_spec is not None and a.host_history_observer
    assert all(arm == 'recompute' for arm in a.arms.split(','))
if a.telemetry_script is not None:
    a.telemetry_script = a.telemetry_script.resolve()
    assert a.telemetry_script.is_file()
if a.target_spec is not None:
    assert a.target_spec.is_file()
    validate_target_spec(json.loads(a.target_spec.read_text()))
    assert all(arm in ('host','recompute') for arm in a.arms.split(','))
root=Path(__file__).resolve().parent
out=root/a.name
out.mkdir(exist_ok=False)
def dump(name, value):
    (out/name).write_text(json.dumps(value,indent=2)+'\n')
lock=open('/root/autodl-tmp/moe-research-gpu.lock','a')
def lock_timeout(signum, frame):
    raise TimeoutError('common GPU lock wait expired')
try:
    if a.wait_lock_seconds:
        dump('status.json',dict(status='WAITING_FOR_LOCK',controller_pid=os.getpid(),
                               deadline_unix=time.time()+a.wait_lock_seconds,time=time.time()))
        previous_alarm=signal.signal(signal.SIGALRM,lock_timeout)
        signal.setitimer(signal.ITIMER_REAL,a.wait_lock_seconds)
        try:
            fcntl.flock(lock,fcntl.LOCK_EX)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,previous_alarm)
    else:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except (BlockingIOError,TimeoutError):
    dump('status.json',dict(status='ABORT_LOCK_BUSY',time=time.time()))
    sys.exit(73)
env=os.environ.copy()
env.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_FLASHINFER_SAMPLER='0',
    E_EXPECTED_GPU_UUID=a.gpu_uuid,
    PYTHONPATH=str(root/'package_view'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    WISP_PLUGIN_DISABLE='1',CUDA_VISIBLE_DEVICES='0',
    LD_LIBRARY_PATH='/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib:'+env.get('LD_LIBRARY_PATH',''))
if a.execution_mode == 'default':
    # Triton/Inductor load generated shared libraries. /dev/shm is mounted
    # noexec on this host, so use an executable E-owned cache directory.
    cache = Path('/tmp/moe-e-restore-choice-20261004')/'compile_cache'/out.name
    cache.mkdir(parents=True, exist_ok=False)
    env.update(VLLM_CACHE_ROOT=str(cache/'vllm'),
               TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
               TRITON_CACHE_DIR=str(cache/'triton'),
               CUDA_CACHE_PATH=str(cache/'cuda'),
               XDG_CACHE_HOME=str(cache/'xdg'))
    dump('execution_cache.json', dict(path=str(cache), requires_executable_filesystem=True))
done=[]
try:
    # One bounded child performs the complete same-engine group. It exits after
    # the specified arms; there is no command loop or lock held for analysis.
    for index, arm in enumerate(['same_engine']):
        # Prior complete summary group used 465 MiB of raw output. Leave more
        # than twice that before loading a model on this nearly full volume.
        disk = shutil.disk_usage(out)
        dump('disk_before.json', dict(free_bytes=disk.free, required_free_bytes=1024**3))
        assert disk.free >= 1024**3, 'ABORT_INSUFFICIENT_OUTPUT_DISK_SPACE'
        # The previous owner's CUDA context can outlive its lock release.
        # Hold this same lock only for a bounded drain, before model loading.
        drain_deadline=time.monotonic()+60
        drain_samples=[]
        while True:
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,temperature.gpu,power.draw,clocks.current.sm','--format=csv,noheader'],text=True)
            procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
            snapshot=dict(gpu=gpu,processes=procs,time=time.time())
            drain_samples.append(snapshot)
            dump('gpu_handoff.json',drain_samples)
            assert a.gpu_uuid in gpu, 'ABORT_GPU_CHANGED'
            if not procs.strip():
                break
            assert time.monotonic() < drain_deadline, 'ABORT_GPU_NOT_DRAINED'
            dump('status.json',dict(status='WAITING_GPU_DRAIN',controller_pid=os.getpid(),time=time.time()))
            time.sleep(1)
        dump(f'{index:02d}_gpu_before.json',snapshot)
        if a.execution_mode == 'default':
            memory_max = int(Path('/sys/fs/cgroup/memory.max').read_text())
            memory_current = int(Path('/sys/fs/cgroup/memory.current').read_text())
            available = memory_max-memory_current
            required = (a.host_gib+16)*1024**3
            cache_free = shutil.disk_usage(cache).free
            dump('compilation_memory_before.json', dict(
                memory_max_bytes=memory_max, memory_current_bytes=memory_current,
                headroom_bytes=available, required_headroom_bytes=required,
                cache_volume_free_bytes=cache_free, required_cache_volume_free_bytes=4*1024**3))
            assert available >= required, 'ABORT_INSUFFICIENT_CONTAINER_MEMORY_FOR_HOST_AND_COMPILE'
            assert cache_free >= 4*1024**3, 'ABORT_INSUFFICIENT_PRIVATE_COMPILE_CACHE_SPACE'
        name=f'{index:02d}_{arm}'
        cmd=[sys.executable,str(root/'run_cell.py'),'--out',str(out/name),'--policy',arm,
            '--threshold',str(a.threshold),'--interval',str(a.interval),'--output-tokens',str(a.output_tokens),
            '--model','/root/autodl-tmp/moe-research-20261002/model',
            '--workload',str(a.workload.resolve()),'--requests',str(a.requests),
            '--host-gib',str(a.host_gib),'--max-num-seqs',str(a.max_num_seqs),
            '--batch-tokens',str(a.batch_tokens),'--max-seconds',str(a.max_seconds),
            '--policies',a.arms,'--execution-mode',a.execution_mode]
        cmd[cmd.index('--policy')+1] = a.arms.split(',')[0]
        if a.kv_bytes is not None:
            cmd += ['--kv-bytes',str(a.kv_bytes)]
        if a.timing_observer:
            cmd += ['--timing-observer']
        if a.native_transfer_probe:
            cmd += ['--native-transfer-probe']
        if a.host_history_observer:
            cmd += ['--host-history-observer']
        if a.store_replay_modes is not None:
            cmd += ['--store-replay-modes', a.store_replay_modes]
        if a.host_load_modes is not None:
            cmd += ['--host-load-modes', a.host_load_modes]
        if a.gpu_telemetry_modes is not None:
            cmd += ['--gpu-telemetry-modes', a.gpu_telemetry_modes]
        if a.telemetry_script is not None:
            cmd += ['--telemetry-script', str(a.telemetry_script)]
        if a.target_spec:
            cmd += ['--target-spec',str(a.target_spec.resolve())]
        if not a.fixed_output:
            cmd += ['--natural']
        dump('status.json',dict(status='RUNNING',controller_pid=os.getpid(),cell=name,completed=done,command=cmd))
        with (out/f'{name}.log').open('x') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
                           timeout=a.group_timeout_seconds or 240 + 2*a.max_seconds*len(a.arms.split(',')))
        done.append(name)
    dump('status.json',dict(status='COMPLETE',completed=done,time=time.time()))
except BaseException as exc:
    dump('status.json',dict(status='FAILED',completed=done,error=repr(exc),time=time.time()))
    raise
finally:
    fcntl.flock(lock,fcntl.LOCK_UN)

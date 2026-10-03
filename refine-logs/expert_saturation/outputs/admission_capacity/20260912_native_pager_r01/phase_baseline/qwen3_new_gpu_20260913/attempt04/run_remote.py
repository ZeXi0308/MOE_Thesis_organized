"""One detached, group-locked Qwen load/localization/static session; no retries."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

PACKAGE = Path(__file__).resolve().parent
OUTPUT = Path('/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-r04')
SHARDS = Path('/root/qwen-streaming-shards-20260914-r04')
LOCK = Path('/root/autodl-tmp/moe-research-gpu.lock')
PYTHON = '/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python'
INSTALLED = Path('/root/autodl-tmp/expert-saturation/vllm-0.26/lib/python3.12/site-packages/vllm')


def read_memory():
    c = Path('/sys/fs/cgroup')
    fields = {name: {k: int(v) for k, v in (line.split() for line in (c / ('memory.' + name)).read_text().splitlines())} for name in ('stat', 'events')}
    return dict(current=int((c / 'memory.current').read_text()), limit=int((c / 'memory.max').read_text()), **fields)


def read_host_identity():
    raw = Path('/proc/1/stat').read_text()
    ticks = int(raw.rsplit(')', 1)[1].split()[19])
    hz = os.sysconf('SC_CLK_TCK')
    boot_unix = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
    return dict(unix_s=time.time(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid1_stat=raw.strip(), pid1_start_ticks=ticks, clock_ticks_per_s=hz,
                kernel_boot_unix_s=boot_unix, pid1_start_unix_s=boot_unix + ticks / hz,
                uptime=Path('/proc/uptime').read_text().strip())


def storage_checks(out, shards):
    rows = {}
    for name, path, minimum in [('results', out, 2**30), ('shards', shards, 6 * 2**30)]:
        fs = os.statvfs(path)
        rows[name] = dict(path=str(path.resolve()), device=os.stat(path).st_dev,
                          available_bytes=fs.f_bavail * fs.f_frsize, minimum_bytes=minimum)
    rows['different_devices'] = rows['results']['device'] != rows['shards']['device']
    rows['passed'] = all(rows[k]['available_bytes'] >= rows[k]['minimum_bytes'] for k in ('results', 'shards'))
    return rows


def launchers():
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if len(argv) > 1 and b'python' in Path(argv[0].decode(errors='replace')).name.encode() and any(Path(x.decode(errors='replace')).name.startswith(('run_native_pager', 'run_wisp', 'run_session', 'run_expert_union')) and x.endswith(b'.py') for x in argv[1:]):
            found.append(int(entry.name))
    return found


def descendant(pid, child_pid):
    for _ in range(32):
        if pid == child_pid:
            return True
        if pid <= 1:
            return False
        try:
            pid = int(next(line.split()[1] for line in Path('/proc', str(pid), 'status').read_text().splitlines() if line.startswith('PPid:')))
        except (OSError, StopIteration, ValueError):
            return False
    return False


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Some hosts deny signal-0 even for an already reaped group; query, never assume absence.
        table = subprocess.run(['ps', '-axo', 'pgid='], capture_output=True, text=True, check=True)
        return pgid in {int(x) for x in table.stdout.split()}


def stop_owned(process, receipt, grace=20):
    if process is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        process.poll()
        if not group_exists(process.pid):
            break
        os.killpg(process.pid, sig)
        receipt.setdefault('cleanup_signals', []).append(dict(pgid=process.pid, signal=int(sig), unix_s=time.time()))
        deadline = time.monotonic() + (grace if sig == signal.SIGTERM else 2)
        while group_exists(process.pid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(.05)
    receipt['owned_group_present_after_cleanup'] = group_exists(process.pid)
    receipt['child_exit_code'] = process.wait(timeout=2)


def command_for(p, out, shards):
    command = ['taskset', '-c', '0-7', PYTHON, '-u', str(p / 'run_native_pager.py'), '--output', str(out / 'comparison'), '--prepared', str(p / 'prepared'), '--model', str(p / 'model_metadata'), '--qwen-static-prefill', '--qualification-prepared', str(p / 'qualification_prepared'), '--expert-cap', '48', '--requests', '4', '--token-budget', '64', '--kv-bytes', '536870912', '--prompt-tokens', '64', '--output-tokens', '8', '--arrival-interval', '0.05', '--prefill-limit', '32', '--trace-retention', 'episode', '--execution', 'expert', '--max-seconds', '600']
    for flag, value in [('manifest', p / 'qwen3.manifest.json'), ('index', p / 'qwen3.index.json'), ('workspace', shards), ('receipt', out / 'loader_receipt.jsonl')]:
        command += ['--loader-' + flag, str(value)]
    return command


def preflight(p):
    with (p / 'preflight.jsonl').open('x') as log:
        for tick in range(3):
            query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
            active = launchers()
            row = dict(unix_s=time.time(), returncode=query.returncode, gpu=query.stdout.strip(), stderr=query.stderr, launchers=active)
            log.write(json.dumps(row) + '\n'); log.flush()
            if query.returncode or query.stdout.strip() or active:
                raise RuntimeError('GPU/launcher busy or query failed; no initialization attempted')
            if tick < 2:
                time.sleep(5)


def main(p=PACKAGE, out=OUTPUT, lock_path=LOCK, shards=SHARDS):
    launch = dict(status='PREPARING', start_unix_s=time.time(), results=str(out), qualification_only=False, startup_numerical_localization=True)
    with (p / 'launch.json').open('x') as f:
        json.dump(launch, f)
    def save():
        temp = p / 'launch.json.tmp'
        temp.write_text(json.dumps(launch, indent=2) + '\n'); temp.replace(p / 'launch.json')
    def interrupted(signum, _frame):
        raise RuntimeError('parent signal ' + signal.Signals(signum).name)
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    process = lock = nv = None; initialized = False; exit_code = 1
    try:
        out.mkdir(exist_ok=False); shards.mkdir(exist_ok=False)
        lock = lock_path.open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch.update(lock_path=str(lock_path), lock_acquired_unix_s=time.time(), lock_owner_pid=os.getpid()); save()
        launch['host_identity_before'] = read_host_identity(); save()
        protocol = json.loads((p / 'protocol.json').read_text())
        for name, sha in protocol['package_files'].items():
            assert hashlib.sha256((p / name).read_bytes()).hexdigest() == sha, name
        for name, sha in protocol['installed_source_sha256'].items():
            assert hashlib.sha256((INSTALLED / name).read_bytes()).hexdigest() == sha, name
        import pynvml as nv
        nv.nvmlInit(); initialized = True; gpu = nv.nvmlDeviceGetHandleByIndex(0)
        uuid = nv.nvmlDeviceGetUUID(gpu); uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
        assert uuid == protocol['new_host']['gpu_uuid'] and nv.nvmlDeviceGetMemoryInfo(gpu).total == protocol['memory_budget']['gpu_total_bytes']
        launch['gpu_identity'] = dict(uuid=uuid, total_bytes=nv.nvmlDeviceGetMemoryInfo(gpu).total); save()
        preflight(p)
        memory_start = read_memory(); storage = storage_checks(out, shards)
        launch['storage_before'] = storage; save()
        assert memory_start['limit'] == 90 * 2**30 and memory_start['stat']['anon'] < 8 * 2**30
        assert storage['passed'], 'Require root shard filesystem>=6GiB and data results filesystem>=1GiB'
        env = dict(os.environ, OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', WISP_PLUGIN_DISABLE='1', WISP_PREFETCH='0', WISP_DYNAMIC='0', PYTHONPATH='/root/autodl-tmp/qwen3-research-wisp-86f69720/source/src:' + str(p))
        command = command_for(p, out, shards)
        launch.update(status='RUNNING', memory_start=memory_start, disk_available_before=storage['results']['available_bytes'], gpu_uuid=uuid, command=command, child_start_unix_s=time.time()); save()
        with (p / 'run.log').open('x') as log, (p / 'hardware.jsonl').open('x') as hw:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
            launch.update(child_pid=process.pid, child_pgid=process.pid, child_sid=process.pid, inherited_lock_fd=lock.fileno()); save()
            tick = 0
            while process.poll() is None:
                memory = read_memory(); info = nv.nvmlDeviceGetMemoryInfo(gpu)
                procs = [dict(pid=x.pid, used_bytes=x.usedGpuMemory) for x in nv.nvmlDeviceGetComputeRunningProcesses(gpu)]
                foreign = [x['pid'] for x in procs if not descendant(x['pid'], process.pid)]
                row = dict(unix_s=time.time(), processes=procs, foreign=foreign, used_bytes=info.used, temperature_c=nv.nvmlDeviceGetTemperature(gpu, nv.NVML_TEMPERATURE_GPU), power_mw=nv.nvmlDeviceGetPowerUsage(gpu), memory=memory)
                hw.write(json.dumps(row) + '\n'); hw.flush()
                if foreign: raise RuntimeError('Foreign GPU process; stop only owned worker group')
                if memory['limit'] != 90 * 2**30 or memory['stat']['anon'] > 82 * 2**30: raise RuntimeError('90GiB cgroup/82GiB anonymous boundary failed')
                if any(memory['events'].get(k, 0) > memory_start['events'].get(k, 0) for k in ('oom', 'oom_kill')): raise RuntimeError('Host OOM event increased')
                if time.time() - launch['child_start_unix_s'] > 21600: raise RuntimeError('21600s complete process limit exceeded')
                if tick % 40 == 0:
                    receipt = out / 'loader_receipt.jsonl'
                    print(json.dumps(dict(progress_s=time.time()-launch['child_start_unix_s'], loader_latest=receipt.read_text().splitlines()[-1:] if receipt.exists() else [], temporary_shards=[dict(name=x.name, bytes=x.stat().st_size) for x in shards.rglob('*.safetensors')])), flush=True)
                tick += 1; time.sleep(.5)
        exit_code = process.returncode
        if group_exists(process.pid): raise RuntimeError('Worker leader exited with surviving owned process group')
        final = nv.nvmlDeviceGetComputeRunningProcesses(gpu)
        launch['gpu_final_pids'] = [x.pid for x in final]
        if final: raise RuntimeError('Final GPU process set not empty')
    except BaseException as exc:
        launch['error'] = f'{type(exc).__name__}: {exc}'; exit_code = 1
    finally:
        for s in previous: signal.signal(s, signal.SIG_IGN)
        try:
            stop_owned(process, launch)
        except BaseException as exc:
            launch['cleanup_error'] = repr(exc); exit_code = 1
        if initialized:
            try: nv.nvmlShutdown()
            except BaseException as exc: launch['nvml_shutdown_error'] = repr(exc); exit_code = 1
        if process is not None: launch.update(child_exit_code=process.poll(), child_end_unix_s=time.time())
        try: launch['host_identity_after'] = read_host_identity()
        except BaseException as exc: launch['host_identity_after_error'] = repr(exc)
        launch.update(status='EXITED', exit_code=exit_code, end_unix_s=time.time())
        launch['parent_wall_s'] = launch['end_unix_s'] - launch['start_unix_s']
        save()
        if lock is not None: lock.close()  # No LOCK_UN: inherited worker FD protects abnormal-parent exits.
        for s, handler in previous.items(): signal.signal(s, handler)
        print(json.dumps(launch), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())

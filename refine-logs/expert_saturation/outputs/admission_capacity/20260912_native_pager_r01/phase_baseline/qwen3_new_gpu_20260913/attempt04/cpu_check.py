"""CPU-only checks of unchanged arguments, inherited flock and owned cleanup."""
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('qwen_parent', ROOT / 'run_remote.py')
parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)


def fixture(root, label, code, failure=False):
    p = root / label; p.mkdir(); out = root / (label + '_out')
    protocol = dict(package_files={}, installed_source_sha256={}, new_host=dict(gpu_uuid='CPU_FAKE'), memory_budget=dict(gpu_total_bytes=123))
    (p / 'protocol.json').write_text(json.dumps(protocol))
    parent.command_for = lambda *_: [sys.executable, '-c', code]
    parent.preflight = lambda *_: None
    parent.read_memory = lambda: dict(current=0, limit=90*2**30, stat=dict(anon=0), events={})
    parent.read_host_identity = lambda: dict(boot_id='CPU_FAKE', pid1_start_ticks=123, pid1_start_unix_s=456.)
    calls = [0]
    def processes(_):
        calls[0] += 1
        if failure and calls[0] > 1:
            raise RuntimeError('injected monitor query failure')
        return []
    nv = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda _: 0, nvmlDeviceGetUUID=lambda _: 'CPU_FAKE',
        nvmlDeviceGetMemoryInfo=lambda _: SimpleNamespace(total=123, used=0),
        nvmlDeviceGetComputeRunningProcesses=processes, nvmlDeviceGetTemperature=lambda *_: 0,
        nvmlDeviceGetPowerUsage=lambda _: 0, NVML_TEMPERATURE_GPU=0)
    sys.modules['pynvml'] = nv
    with contextlib.redirect_stdout(io.StringIO()):
        rc = parent.main(p, out, root / 'gpu.lock', root / (label + '_shards'))
    result = json.loads((p / 'launch.json').read_text())
    assert result['exit_code'] == rc and result['status'] == 'EXITED'
    return rc, result


def main():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    old = json.loads((ROOT.parent / 'attempt03/protocol.json').read_text())
    administrative = {'administrative_launch', 'new_host', 'execution_guard', 'current_gpu_identity_evidence'}
    assert all(protocol[k] == v for k, v in old.items() if k not in administrative)
    assert protocol['administrative_launch']['complete_process_limit_s'] == 21600
    assert protocol['administrative_launch']['capture_limit_s'] == 600
    assert protocol['administrative_launch']['shard_workspace'] == str(parent.SHARDS)
    for name, sha in old['package_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    assert (ROOT / 'detached_launch.py').read_bytes() == (ROOT.parent / 'detached_launch.py').read_bytes()
    args = parent.command_for(Path('/stage'), Path('/out'), Path('/shards'))
    # Extract only the original literal command construction, not its GPU parent.
    import ast
    tree = ast.parse((ROOT.parent / 'attempt02/run_remote.py').read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'command' for t in n.targets)]
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple) and [x.id for x in n.target.elts if isinstance(x, ast.Name)] == ['flag', 'value']]
    env = dict(p=Path('/stage'), out=Path('/out'), shards=Path('/shards'), python=parent.PYTHON)
    exec(compile(ast.Module(body=nodes + loops, type_ignores=[]), '<original-command>', 'exec'), env)
    assert args == env['command']
    with tempfile.TemporaryDirectory(prefix='qwen-r04-cpu-') as tmp:
        root = Path(tmp); lock_path = root / 'gpu.lock'
        from unittest.mock import patch
        # Separate mount capacities: sufficient sum cannot hide either deficient filesystem.
        data, shards = root / 'data', root / 'root-shards'; data.mkdir(); shards.mkdir()
        for data_bytes, shard_bytes, passed in [(2**30, 6*2**30, True), (2**30-1, 20*2**30, False), (20*2**30, 6*2**30-1, False)]:
            def statvfs(path):
                return SimpleNamespace(f_bavail=data_bytes if path == data else shard_bytes, f_frsize=1)
            with patch.object(parent.os, 'statvfs', side_effect=statvfs):
                storage = parent.storage_checks(data, shards)
            assert storage['passed'] == passed
            assert storage['results']['available_bytes'] == data_bytes and storage['shards']['available_bytes'] == shard_bytes
        with lock_path.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            rc, result = fixture(root, 'busy', 'raise AssertionError("must not start")')
            assert rc == 1 and 'child_pid' not in result and 'BlockingIOError' in result['error']
        # Confirm the actual worker inherits an open FD for the held lock.
        child = 'import os,time,pathlib; target=os.stat(' + repr(str(lock_path)) + '); found=[]\nfor fd in range(3,256):\n try:\n  s=os.fstat(fd)\n  if (s.st_dev,s.st_ino)==(target.st_dev,target.st_ino): found.append(fd)\n except OSError: pass\nassert found; time.sleep(.1)'
        for code in (0, 7):
            rc, result = fixture(root, 'normal' + str(code), child + '\nraise SystemExit(' + str(code) + ')')
            assert rc == code and result['child_exit_code'] == code and not result['owned_group_present_after_cleanup']
            assert result['host_identity_before'] == result['host_identity_after']
            assert result['storage_before']['passed'] and result['gpu_identity']['uuid'] == 'CPU_FAKE'
        outsider = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            rc, result = fixture(root, 'query_failure', 'import time; time.sleep(30)', failure=True)
            assert rc == 1 and 'injected monitor query failure' in result['error']
            assert result['cleanup_signals'] and all(x['pgid'] == result['child_pid'] for x in result['cleanup_signals'])
            assert result.get('owned_group_present_after_cleanup') is False and outsider.poll() is None, result
            rc, result = fixture(root, 'parent_term', 'import os,signal,time; time.sleep(.1); os.kill(os.getppid(),signal.SIGTERM); time.sleep(30)')
            assert rc == 1 and 'parent signal SIGTERM' in result['error']
            assert not result['owned_group_present_after_cleanup'] and outsider.poll() is None
        finally:
            outsider.terminate(); outsider.wait(timeout=3)
        with lock_path.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return dict(status='PASS_CPU_ONLY', checks=['30 payload / 17 runtime / detached launcher unchanged; command and scientific fields equal; root>=6GiB and data>=1GiB independently enforced', 'busy never launches; inherited worker lock FD, identity fields and real exit0/7 retained; lock released', 'monitor failure and parent SIGTERM reap own group, preserve unrelated group, write terminal'], scope='Fake NVML/memory/host identity/preflight and explicit storage-capacity fixtures; real CPU subprocesses and OS flock. No GPU, network, model or numerical qualification. SIGKILL/host-loss receipts are not guaranteed.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path); args = parser.parse_args()
    result = main()
    if args.out:
        with args.out.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2))

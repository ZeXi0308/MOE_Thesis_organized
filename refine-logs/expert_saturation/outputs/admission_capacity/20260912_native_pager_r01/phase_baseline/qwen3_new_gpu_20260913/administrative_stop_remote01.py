import json, os, signal, time
from pathlib import Path

stage = Path('/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r01')
out = Path('/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-r01')
launch = json.loads((stage / 'launch.json').read_text())
assert launch['status'] == 'RUNNING' and launch['child_pid'] == 3326
assert launch['child_start_unix_s'] == 1789284674.754814 or abs(launch['child_start_unix_s'] - 1789284674.754) < 0.01
worker = 3326
parent = 3314
argv = Path('/proc', str(worker), 'cmdline').read_bytes().split(b'\0')
assert str(stage / 'run_native_pager.py').encode() in argv
assert str(out / 'comparison').encode() in argv
status = Path('/proc', str(worker), 'status').read_text()
assert int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:'))) == parent
assert os.getpgid(worker) == worker and os.getsid(worker) == worker
assert str(stage / 'run_remote.py').encode() in Path('/proc', str(parent), 'cmdline').read_bytes().split(b'\0')
comparison = out / 'comparison'
assert not list(comparison.glob('repeat_*_static*')), 'Performance already started; no administrative interruption permitted by this script'
start_ticks = Path('/proc', str(worker), 'stat').read_text().rsplit(')', 1)[1].split()[19]
record = dict(status='STOP_REQUESTED', unix_s=time.time(), worker=worker, parent=parent, worker_start_ticks=start_ticks, argv=[x.decode() for x in argv if x], reason='Administrative loading-timeout revision. First shard about774s, second408s, third3-4MB/s;7200s whole-process deadline lacks margin. No performance cells executed. New attempt will use21600s with identical runtime/input/scientific configuration.', action='SIGTERM only verified own worker process group; existing parent collects exit and writes original launch receipt', retained_temporary_files=[dict(path=str(p),bytes=p.stat().st_size) for p in (out/'shard_workspace').rglob('*.safetensors')])
with (stage / 'administrative_stop01.json').open('x') as f:
    json.dump(record, f, indent=2)
os.killpg(worker, signal.SIGTERM)
print(json.dumps(record), flush=True)
for _ in range(60):
    current = json.loads((stage / 'launch.json').read_text())
    if current['status'] == 'EXITED':
        print(json.dumps(dict(parent_terminal=current, worker_proc_exists=Path('/proc',str(worker)).exists())), flush=True)
        break
    time.sleep(.5)
else:
    raise RuntimeError('Parent terminal receipt not observed in30s; no additional signal sent')

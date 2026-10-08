"""Execute a prepared native KV campaign on an existing, authenticated GPU host.

No credentials, package installation, model download or automatic retry. Each
cell is archived and verified locally before the next one. SSH failure stops
the driver: inspect the original remote process before any recovery.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time


LABELS = ('repeat0-budget95', 'repeat0-budget90',
          'repeat1-budget90', 'repeat1-budget95')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--port', required=True)
    p.add_argument('--control-path', required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--remote-dir', required=True)
    p.add_argument('--labels', nargs='+', default=LABELS,
                   help='explicit order accepted by the frozen run_one_cell entrypoint')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--stage-only', action='store_true', help='upload/verify only; a busy GPU is allowed')
    modes.add_argument('--resume-staged', action='store_true', help='run an untouched STAGED package without uploading again')
    args = p.parse_args()
    out = args.output.resolve()
    if not args.resume_staged:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir():
        raise FileNotFoundError('staged output directory does not exist')
    ssh = ['ssh', '-p', args.port, '-S', args.control_path,
           '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=15',
           '-o', 'ServerAliveCountMax=4', args.host]
    scp = ['scp', '-P', args.port, '-o', f'ControlPath={args.control_path}',
           '-o', 'BatchMode=yes']
    remote = args.remote_dir
    py = '/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python'
    def run(command, **kwargs):
        return subprocess.run(command, check=True, **kwargs)
    def query(code):
        try:
            return json.loads(subprocess.check_output(
                ssh + [shlex.join([py, '-c', code])], text=True, stderr=subprocess.PIPE))
        except subprocess.CalledProcessError as error:
            (attempt_dir/'query-failure.json').write_text(json.dumps(dict(returncode=error.returncode, stdout=error.output, stderr=error.stderr), indent=2)+'\n')
            raise
    identity = dict(host=args.host, port=args.port, gpu_uuid=args.gpu_uuid,
                    remote_dir=remote, source=str(args.source.resolve()), labels=list(args.labels))
    attempt_dir = out/'attempts'/str(time.time_ns())
    attempt_dir.mkdir(parents=True, exist_ok=False)
    attempt = dict(status='STARTING', driver_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), started_unix_s=time.time(), mode='resume' if args.resume_staged else 'stage' if args.stage_only else 'execute', **identity)
    (attempt_dir/'attempt.json').write_text(json.dumps(attempt, indent=2)+'\n')
    record, locked, started_run = None, False, False
    lock = out/'.driver.lock'
    def save():
        temp = out/'execution.json.tmp'
        temp.write_text(json.dumps(record, indent=2)+'\n')
        temp.replace(out/'execution.json')
    try:
        with lock.open('x') as stream:
            stream.write(str(attempt_dir)+'\n')
        locked = True
        record = json.loads((out/'execution.json').read_text()) if args.resume_staged else dict(status='PREFLIGHT', started_unix_s=time.time(), cells=[], **identity)
        if args.resume_staged:
            assert record['status'] == 'STAGED' and record['cells'] == [], 'resume requires intact STAGED state with no started cell'
            assert all(record.get(k) == v for k, v in identity.items()), 'staged source/host/port/GPU/remote-dir/labels mismatch'
            if any((out/'gpu_results').iterdir()):
                record['status'] = 'STOPPED_LOCAL_ACTIVITY'
                raise RuntimeError('staged local results are not empty; refusing rerun')
        save()
        source_bytes = (args.source/'execution.tar.gz').read_bytes()
        digest = hashlib.sha256(source_bytes).hexdigest()
        archive = out/'execution.tar.gz'
        if args.resume_staged:
            assert digest == record['archive_sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest(), 'staged/local/source archive mismatch'
        else:
            archive.write_bytes(source_bytes)
            record['archive_sha256'] = digest
        if args.stage_only or args.resume_staged:
            with tarfile.open(archive) as bundle:
                manifest = json.load(bundle.extractfile('campaign.json'))
                if args.resume_staged:
                    assert all((out/'frozen'/m.name).read_bytes() == bundle.extractfile(m).read() for m in bundle.getmembers() if m.isfile()), 'local frozen files differ from staged archive'
            assert list(args.labels) == [c['label'] for c in manifest['cells']], 'labels differ from frozen campaign order'
        preflight = query('''import hashlib,json,pathlib,subprocess
import torch,vllm,transformers
r=pathlib.Path('/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5')
index=json.loads((r/'model.safetensors.index.json').read_text())
shards=[]
for name in sorted(set(index['weight_map'].values())):
 p=r/name
 with p.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
 expected=p.resolve().name
 assert actual==expected,(name,'cache content hash mismatch')
 shards.append(dict(name=name,bytes=p.stat().st_size,sha256=actual))
def smi(*args):return subprocess.check_output(['nvidia-smi',*args],text=True).strip()
print(json.dumps(dict(torch=torch.__version__,vllm=vllm.__version__,transformers=transformers.__version__,shards=shards,gpu=smi('--query-gpu=uuid,name,memory.total','--format=csv,noheader'),processes=smi('--query-compute-apps=pid','--format=csv,noheader'))))''')
        (attempt_dir/'preflight.json').write_text(json.dumps(preflight, indent=2)+'\n')
        if not args.resume_staged:
            (out/'preflight.json').write_text(json.dumps(preflight, indent=2)+'\n')
        assert args.gpu_uuid in preflight['gpu'], 'requested GPU UUID is absent'
        assert (preflight['torch'], preflight['vllm'], preflight['transformers']) == (
            '2.11.0+cu130', '0.26.0', '5.15.1')
        if not args.stage_only and not args.resume_staged and preflight['processes']:
            raise RuntimeError('GPU busy; no cell started')
        if args.resume_staged:
            staged = query(f'''import hashlib,json,pathlib,tarfile
r=pathlib.Path({remote!r}); archive=r/'execution.tar.gz'
with tarfile.open(archive) as bundle:
 same=all(hashlib.sha256((r/m.name).read_bytes()).digest()==hashlib.sha256(bundle.extractfile(m).read()).digest() for m in bundle.getmembers() if m.isfile())
print(json.dumps(dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),source_matches=same,results_empty=not (r/'results').exists() or not any((r/'results').iterdir()))))''')
            (attempt_dir/'staged-verification.json').write_text(json.dumps(staged, indent=2)+'\n')
            assert staged['sha256'] == digest and staged['source_matches'], 'remote staged archive/source mismatch'
            if not staged['results_empty']:
                record['status'] = 'STOPPED_REMOTE_ACTIVITY'
                raise RuntimeError('remote cell activity already exists; refusing rerun')
        else:
            run(ssh + [shlex.join(['mkdir', remote])])
            run(scp + [str(archive), f'{args.host}:{remote}/execution.tar.gz'])
            actual = query(f"import hashlib,json,pathlib; print(json.dumps(hashlib.sha256(pathlib.Path({remote!r}+'/execution.tar.gz').read_bytes()).hexdigest()))")
            assert digest == actual, 'uploaded archive mismatch'
            run(ssh + [shlex.join(['tar', '-xzf', remote+'/execution.tar.gz', '-C', remote])])
            with tarfile.open(archive) as bundle:
                bundle.extractall(out/'frozen', filter='data')
            if (args.source/'analyze_kv_budget.py').exists():
                (out/'analyze_kv_budget.py').write_bytes((args.source/'analyze_kv_budget.py').read_bytes())
            (out/'inputs_preparation').symlink_to('frozen/inputs_preparation', target_is_directory=True)
        if args.resume_staged and preflight['processes']:
            raise RuntimeError('GPU busy; staged package retained, no cell started')
        results = out/'gpu_results'
        results.mkdir(exist_ok=args.resume_staged)
        if args.stage_only:
            record['status'] = 'STAGED'
            record['staged_unix_s'] = time.time()
            attempt['status'] = 'STAGED'
            return
        started_run = True
        record['status'] = 'RUNNING'; save()
        for label in args.labels:
            state = dict(label=label, status='RUNNING', started_unix_s=time.time())
            record['cells'].append(state); save()
            command = 'cd '+shlex.quote(remote)+' && '+shlex.join([py, '-u', 'run_one_cell.py', label])
            with (out/f'{label}.ssh.log').open('x') as log:
                process = subprocess.run(ssh+[command], stdout=log, stderr=subprocess.STDOUT)
            state.update(returncode=process.returncode, finished_unix_s=time.time())
            save()
            if process.returncode == 255:
                raise RuntimeError('SSH disconnected; inspect original remote execution; never relaunch automatically')
            filename = label+'.tar.gz'
            run(ssh+[shlex.join(['tar', '-czf', remote+'/'+filename, '-C', remote+'/results',
                label, label+'-execution.json', label+'.stdout.log', label+'.stderr.log'])])
            run(scp+[f'{args.host}:{remote}/{filename}', str(out/filename)])
            expected = query(f"import hashlib,json,pathlib; print(json.dumps(hashlib.sha256(pathlib.Path({remote!r}+'/{filename}').read_bytes()).hexdigest()))")
            assert hashlib.sha256((out/filename).read_bytes()).hexdigest() == expected
            with tarfile.open(out/filename) as bundle:
                bundle.extractall(results, filter='data')
            terminal = json.loads((results/label/'status.json').read_text())
            state.update(status='READ_BACK', archive_sha256=expected, terminal=terminal)
            save()
            print(label, terminal, flush=True)
            if process.returncode or terminal['status'] != 'COMPLETE':
                raise RuntimeError(f'{label} did not complete; all available files retained')
        record['status'] = 'COMPLETE'
        attempt['status'] = 'COMPLETE'
    except BaseException as error:
        attempt.update(status='REJECTED_OR_FAILED', error=f'{type(error).__name__}: {error}')
        if record is not None and (not args.resume_staged or started_run):
            record.update(status='STOPPED', error=attempt['error'])
        raise
    finally:
        attempt['finished_unix_s'] = time.time()
        (attempt_dir/'attempt.json').write_text(json.dumps(attempt, indent=2)+'\n')
        if record is not None and locked:
            record['updated_unix_s'] = time.time(); save()
        if locked:
            lock.unlink()


if __name__ == '__main__':
    main()

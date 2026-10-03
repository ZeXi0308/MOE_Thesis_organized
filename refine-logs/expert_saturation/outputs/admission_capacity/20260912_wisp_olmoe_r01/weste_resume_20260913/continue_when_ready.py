"""Bounded resource wait, then one sealed qualification and performance attempt."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
Q = Path('/root/autodl-tmp/full-stage-qualification-20260913-weste-r02')
P = Path('/root/autodl-tmp/full-stage-performance-20260913-weste-r02')
MODEL = Path('/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5')
PYTHON = '/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python'


def read(path): return json.loads(path.read_text())
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def sealed_inputs():
    manifest = read(ROOT / 'continuation_inputs.json')
    for filename, expected in manifest.items():
        if digest(Path(filename)) != expected: raise RuntimeError('sealed input changed: ' + filename)
    for bundle in (Q, P):
        for name, expected in read(bundle / 'sources.json').items():
            if digest(bundle / 'source' / name) != expected: raise RuntimeError('runtime changed: ' + name)
    for name, expected in read(P / 'analysis_sources.json')['sources'].items():
        if digest(P / 'analysis_source' / name) != expected: raise RuntimeError('analysis changed: ' + name)
    for package, expected in {'vllm':'0.26.0', 'torch':'2.11.0', 'transformers':'5.15.1',
                             'triton':'3.6.0', 'tokenizers':'0.22.2', 'flashinfer-python':'0.6.14'}.items():
        if importlib.metadata.version(package) != expected: raise RuntimeError('environment changed: ' + package)
    return len(manifest)


def preflight():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
        '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
    if result.returncode: raise RuntimeError('GPU query failed: ' + result.stderr)
    gpu = [line for line in result.stdout.splitlines() if line.strip()]
    names = sorted(set(read(MODEL / 'model.safetensors.index.json')['weight_map'].values()))
    missing = [name for name in names if not (MODEL / name).is_file()]
    partial = list((MODEL.parent.parent / 'blobs').glob('*.incomplete'))
    return dict(unix_s=time.time(), gpu=gpu, missing_shards=missing,
        incomplete_bytes=sum(p.stat().st_size for p in partial),
        ready=not gpu and not missing)


def event(status, **values):
    entry = dict(unix_s=time.time(), status=status)
    entry.update(values)
    with (ROOT / 'events.jsonl').open('a') as f: f.write(json.dumps(entry) + '\n')
    temporary = ROOT / 'state.tmp'
    temporary.write_text(json.dumps(entry, indent=2) + '\n')
    temporary.replace(ROOT / 'state.json')
    print(json.dumps(entry), flush=True)


def run(stage, command, **env):
    event(stage, command=command)
    started = time.perf_counter()
    with (ROOT / (stage.lower() + '.log')).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **env))
    event(stage + '_FINISHED', returncode=result.returncode, wall_s=time.perf_counter() - started)
    if result.returncode: raise RuntimeError(stage + ' failed; no automatic retry')


def archive():
    path = ROOT / 'attempt01.tar.gz'
    with tarfile.open(path, 'x:gz') as tar:
        for bundle, label in ((Q, 'qualification'), (P, 'performance')):
            if (bundle / 'results').exists(): tar.add(bundle / 'results', arcname=label + '/results')
        for path_in in sorted(ROOT.iterdir()):
            if path_in.is_file() and path_in != path and path_in.name != 'archive.json':
                tar.add(path_in, arcname='continuation/' + path_in.name)
    (ROOT / 'archive.json').write_text(json.dumps(dict(name=path.name, bytes=path.stat().st_size,
        sha256=digest(path)), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    checked = sealed_inputs()
    if args.check_only:
        print(json.dumps(dict(checked_inputs=checked, preflight=preflight()))); return
    with (ROOT / 'started.json').open('x') as f:
        json.dump(dict(pid=os.getpid(), unix_s=time.time(), wait_limit_s=21600), f)
    deadline = time.monotonic() + 21600
    try:
        for bundle in (Q, P):
            if (bundle / 'results').exists(): raise RuntimeError('existing attempt: ' + str(bundle))
        ready_count = 0
        while time.monotonic() < deadline:
            sample = preflight(); event('WAITING', **sample)
            ready_count = ready_count + 1 if sample['ready'] else 0
            if ready_count == 2: break
            time.sleep(30)
        else: raise TimeoutError('six-hour resource wait expired; GPU execution not started')
        sealed_inputs(); event('VERIFYING_MODEL')
        shards = []
        for name in sorted(set(read(MODEL / 'model.safetensors.index.json')['weight_map'].values())):
            path = MODEL / name; expected = path.resolve().name; actual = digest(path)
            if len(expected) != 64 or actual != expected: raise RuntimeError('model LFS digest mismatch: ' + name)
            shards.append(dict(name=name, bytes=path.stat().st_size, sha256=actual))
        (ROOT / 'bootstrap_model.json').write_text(json.dumps(dict(snapshot=str(MODEL), shards=shards), indent=2) + '\n')
        if not preflight()['ready']: raise RuntimeError('resources changed after model verification; ABORT')
        run('QUALIFYING', [PYTHON, '-u', str(Q / 'run_new_host_qualification.py')])
        run('CHECKING_QUALIFICATION', [PYTHON, str(ROOT / 'check_qualification.py'), '--results', str(Q / 'results'),
            '--source', str(Q / 'source'), '--out', str(ROOT / 'qualification_check.json')],
            MOE_QUALIFICATION_HELPERS=str(P / 'analysis_source'))
        result = read(ROOT / 'qualification_check.json')
        if result['status'] != 'QUALIFIED_NEW_HOST_LIFECYCLE_ONLY' or result['issues']:
            raise RuntimeError('qualification has issues; performance UNRUN')
        sealed_inputs()
        if not preflight()['ready']: raise RuntimeError('resources changed before performance; ABORT')
        run('PERFORMANCE', [PYTHON, '-u', str(P / 'run_attempt.py')])
        run('ANALYZING', [PYTHON, str(P / 'analysis_source/analyze_shared_pool_execution.py'),
            '--input-dir', str(P), '--out', str(ROOT / 'performance_analysis.json')])
        result = read(ROOT / 'performance_analysis.json')
        if result['status'] != 'DESCRIPTIVE_NATIVE_SHARED_POOL_EXECUTION' or result['issues']:
            raise RuntimeError('performance analysis has issues')
        event('COMPLETE', scope='Frozen campaign completed; descriptive evidence only, research goal not adjudicated.')
    except BaseException as exc:
        event('STOPPED', error=repr(exc)); raise
    finally:
        archive()


if __name__ == '__main__': main()

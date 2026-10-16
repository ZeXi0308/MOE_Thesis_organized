#!/usr/bin/env python3
"""Download and verify the queued matrix, then acknowledge each local archive.

Uses an already-authenticated SSH control socket; no credentials are stored.
Only the explicitly listed HORIZON sessions are eligible for cleanup.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/root/autodl-tmp/research_A_horizon'
SESSIONS = ['qwen-calibration-r11', 'olmoe24-w2-corrected-frontier',
            'olmoe48-w2-corrected-frontier', 'olmoe24-w1-corrected-frontier',
            'olmoe48-w1-corrected-frontier', 'olmoe24-w3-corrected-frontier',
            'olmoe48-w3-corrected-frontier']
RAW_NAMES = ['raw.json', 'horizon.json', 'offload.json', 'offload-drain.json',
             'warmup-0.json', 'warmup-1.json', 'warmup-drain.json']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def unpack(archive, name):
    """Reject links/path escapes; never overwrite different local evidence."""
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        for member in members:
            parts = PurePosixPath(member.name).parts
            if (not member.isfile() or not parts or parts[0] != name
                    or any(p in ('..', '') for p in parts)):
                raise ValueError(f'Unexpected archive member: {member.name}')
            target = ROOT / 'runs' / member.name
            if not target.resolve().is_relative_to((ROOT / 'runs' / name).resolve()):
                raise ValueError('Archive path escapes session')
        for member in members:
            target = ROOT / 'runs' / member.name
            data = tar.extractfile(member).read()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f'Local evidence differs: {target}')
            else:
                with target.open('xb') as f:
                    f.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', required=True)
    parser.add_argument('--qwen-session', choices=['qwen-calibration-r11', 'qwen-calibration-r11b'],
                        default='qwen-calibration-r11')
    args = parser.parse_args()
    ssh = ['ssh', '-S', args.socket, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
           '-p', '50328', 'root@connect.westb.seetacloud.com']
    def remote(code):
        result = subprocess.run(ssh + ['/root/miniconda3/bin/python -'], input=code, text=True,
                                capture_output=True, check=True, timeout=180)
        return result.stdout
    for name in [args.qwen_session] + SESSIONS[1:]:
        print('WAITING_ARCHIVE', name, flush=True)
        while True:
            # Archive metadata is published only after tar creation finishes.
            info = json.loads(remote(f'''import json,pathlib
p=pathlib.Path({REMOTE!r})/'runs'/{name!r}
r=json.loads((p/'receipt.json').read_text()) if (p/'receipt.json').exists() else {{}}
a=json.loads((p/'archive.json').read_text()) if (p/'archive.json').exists() else None
print(json.dumps(dict(status=r.get('status'),archive=a)))
'''))
            if info['status'] in ('COMPLETE', 'STOPPED') and info['archive']:
                break
            time.sleep(30)
        metadata = info['archive']
        expected_remote = f'{REMOTE}/runs/{name}.raw.tar.gz'
        if metadata['path'] != expected_remote:
            raise ValueError('Unexpected remote archive path')
        local = ROOT / 'runs' / (name + '.raw.tar.gz')
        if not local.exists():
            part = local.with_suffix(local.suffix + '.part')
            subprocess.run(['scp', '-o', f'ControlPath={args.socket}', '-o', 'BatchMode=yes',
                            '-P', '50328', 'root@connect.westb.seetacloud.com:' + expected_remote,
                            str(part)], check=True, timeout=600)
            if part.stat().st_size != metadata['bytes'] or digest(part) != metadata['sha256']:
                raise ValueError('Downloaded archive mismatch')
            part.rename(local)
        if local.stat().st_size != metadata['bytes'] or digest(local) != metadata['sha256']:
            raise ValueError('Existing local archive mismatch')
        unpack(local, name)
        session = ROOT / 'runs' / name
        (session / 'archive.json').write_text(json.dumps(metadata, indent=2) + '\n')
        subprocess.run([sys.executable, '-B', str(ROOT / 'code/analyze.py'),
                        '--session', str(session)], check=True)
        if info['status'] != 'COMPLETE':
            raise RuntimeError(f'{name} stopped: archived failure; continuation not acknowledged')
        verified = {k: metadata[k] for k in ('sha256', 'bytes')}
        # Recheck remote terminal state and the immutable archive before cleanup.
        # Deletion is restricted to cache and exact raw names present in this tar.
        result = remote(f'''import hashlib,json,pathlib,shutil,tarfile
p=pathlib.Path({REMOTE!r})/'runs'/{name!r}
a=json.loads((p/'archive.json').read_text());v={verified!r}
assert json.loads((p/'receipt.json').read_text())['status']=='COMPLETE'
assert all(a[k]==v[k] for k in v)
t=pathlib.Path(a['path']);assert t.stat().st_size==v['bytes']
assert hashlib.sha256(t.read_bytes()).hexdigest()==v['sha256']
with tarfile.open(t) as archive:
    archived={{m.name for m in archive.getmembers() if m.isfile()}}
for cell in p.glob('cell-*'):
    if not cell.is_dir() or cell.is_symlink():continue
    for name in {RAW_NAMES!r}:
        f=cell/name
        if f.exists():
            assert str(f.relative_to(p.parent)) in archived
            assert not f.is_symlink()
            f.unlink()
cache=p/'cache'
if cache.exists():
    assert not cache.is_symlink()
    shutil.rmtree(cache)
ack=p/'local-archive-verified.json'
if ack.exists():assert json.loads(ack.read_text())==v
else:
    temp=ack.with_suffix('.tmp');temp.write_text(json.dumps(v)+'\\n');temp.replace(ack)
print('ARCHIVED_AND_ACKNOWLEDGED',p.name)
''')
        (session / 'local-archive-verified.json').write_text(json.dumps(verified) + '\n')
        print(result.strip(), flush=True)
    print('COMPLETE archive continuation', flush=True)


if __name__ == '__main__':
    main()

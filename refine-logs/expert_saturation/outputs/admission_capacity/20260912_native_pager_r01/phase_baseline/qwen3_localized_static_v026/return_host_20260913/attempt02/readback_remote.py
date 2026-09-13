"""Read-only terminal r02 archive: tar on stdout, SHA receipt on stderr."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time

stage = Path('/root/autodl-tmp/qwen3-localized-static-v026-launch-r02')
source = Path('/root/autodl-tmp/qwen3-localized-static-v026-r02')


def quiescent():
    launch = json.loads((stage / 'launch.json').read_text())
    assert launch['status'] == 'EXITED', 'parent must have terminal EXITED receipt'
    child = launch.get('child_pid')
    if child is not None:
        assert launch.get('child_exit_code') is not None, 'missing child terminal code'
        assert not Path('/proc', str(child)).exists(), 'recorded worker PID still exists'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        assert not any(arg.startswith(str(stage).encode() + b'/') and arg.endswith(b'.py')
                       for arg in argv), 'stage process still exists'
    return launch


def snapshot():
    files = []
    for root, prefix in ((source, 'results'), (stage, 'launch')):
        assert root.is_dir(), 'required archive root missing: ' + str(root)
        for path in sorted(root.rglob('*')):
            if (not path.is_file() or path.is_symlink() or '__pycache__' in path.parts
                    or 'shard_workspace' in path.parts or path.suffix == '.safetensors'):
                continue
            stat = path.stat()
            files.append((path, prefix + '/' + path.relative_to(root).as_posix(),
                          stat.st_size, stat.st_mtime_ns))
    return files


class HashIO:
    def __init__(self, stream):
        self.stream, self.digest, self.count = stream, hashlib.sha256(), 0

    def write(self, data):
        self.digest.update(data); self.count += len(data)
        return self.stream.write(data)

    def read(self, amount):
        data = self.stream.read(amount)
        self.digest.update(data); self.count += len(data)
        return data


launch = quiescent()
files = snapshot(); members = []; sink = HashIO(sys.stdout.buffer)
with tarfile.open(fileobj=sink, mode='w|gz') as archive:
    for path, name, size, mtime in files:
        with path.open('rb') as stream:
            reader = HashIO(stream)
            archive.addfile(archive.gettarinfo(str(path), arcname=name), reader)
        assert reader.count == size, 'file size changed during archive'
        members.append(dict(name=name, bytes=size, sha256=reader.digest.hexdigest()))
assert snapshot() == files, 'archive member set or metadata changed'
assert quiescent() == launch, 'terminal launch receipt changed'
sys.stdout.buffer.flush()
print(json.dumps(dict(status='QUIESCENT_TERMINAL_READBACK', unix_s=time.time(),
    sha256=sink.digest.hexdigest(), bytes=sink.count, files=members,
    parent_status=launch['status'], child_exit_code=launch.get('child_exit_code'),
    scope='Complete regular launch/results files including layer47.pt; excludes pycache, symlinks and temporary safetensors. No remote files changed.')),
    file=sys.stderr, flush=True)

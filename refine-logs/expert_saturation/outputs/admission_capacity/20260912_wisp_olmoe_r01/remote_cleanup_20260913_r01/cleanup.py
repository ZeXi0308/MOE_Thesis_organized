"""Delete only complete unpacked cells byte-identical to retained archives."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time


def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()


def signature(path):
    s = path.stat(follow_symlinks=False)
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def no_consumers(roots):
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = str((proc / 'cwd').resolve(strict=True))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(root in args or cwd == root or cwd.startswith(root + '/') for root in roots):
            raise RuntimeError('active consumer: ' + proc.name)


root = Path(__file__).resolve().parent
plan = json.loads((root / 'plan.json').read_text())
receipt = dict(status='VERIFYING', started_unix_s=time.time(),
               disk_before=shutil.disk_usage('/root/autodl-tmp')._asdict(), cells=[])


def save():
    temp = root / 'receipt.tmp'
    temp.write_text(json.dumps(receipt, indent=2) + '\n')
    temp.replace(root / 'receipt.json')


if (root / 'receipt.json').exists():
    raise FileExistsError('retain previous cleanup attempt')
try:
    roots = sorted({str(Path(c['remote_target']).parent.parent) for c in plan['cells']})
    no_consumers(roots)
    for c in plan['cells']:
        target, archive = Path(c['remote_target']), Path(c['remote_archive'])
        if target.is_symlink() or archive.is_symlink() or not target.is_dir():
            raise ValueError('unexpected target/archive type: ' + str(target))
        with archive.open('rb') as stream:
            if digest(stream) != c['archive_sha256']:
                raise ValueError('remote archive differs from verified local archive')
        if json.loads((target / 'status.json').read_text())['status'] != 'COMPLETE':
            raise ValueError('cell not complete')
        files, dirs, seen, absent = {}, [], set(), []
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                p = PurePosixPath(member.name)
                if p.is_absolute() or '..' in p.parts or member.name in seen:
                    raise ValueError('unsafe or duplicate archive member')
                seen.add(member.name)
                # Archives also contain sibling driver logs; they are retained.
                if p.parts[0] != c['label']:
                    continue
                path = target.parent / p
                if path.is_symlink():
                    raise ValueError('unexpected symlink')
                if not path.exists():
                    absent.append(str(path))
                    continue
                if member.isdir():
                    if not path.is_dir():
                        raise ValueError('missing archived directory')
                    dirs.append(str(path))
                elif member.isfile():
                    before = signature(path)
                    if before[2] != member.size:
                        raise ValueError('size differs')
                    with path.open('rb') as stream, tar.extractfile(member) as archived:
                        if digest(stream) != digest(archived):
                            raise ValueError('unpacked content differs: ' + str(path))
                    if signature(path) != before:
                        raise ValueError('file changed during verification')
                    files[str(path)] = before
                else:
                    raise ValueError('unsupported archive entry')
        actual = {str(p) for p in target.rglob('*')}
        if any(p.is_symlink() for p in target.rglob('*')) or actual != (set(files) | set(dirs)) - {str(target)}:
            raise ValueError('unarchived entries: ' + str(target))
        receipt['cells'].append(dict(target=str(target), archive=str(archive),
            archive_sha256=c['archive_sha256'], archive_signature=signature(archive),
            files=files, dirs=dirs, already_absent_archived_members=absent,
            verified_bytes=sum(s[2] for s in files.values()), status='VERIFIED'))
        save()
        print('verified', c['label'], flush=True)
    no_consumers(roots)
    receipt['status'] = 'DELETING'
    save()
    for c in receipt['cells']:
        no_consumers([str(Path(c['target']).parent.parent)])
        if signature(Path(c['archive'])) != c['archive_signature']:
            raise ValueError('archive changed before deletion')
        for path, expected in c['files'].items():
            if signature(Path(path)) != expected:
                raise ValueError('file changed before deletion')
        c['deleted_files'] = []
        for path in c['files']:
            Path(path).unlink()
            c['deleted_files'].append(path)
        for path in sorted(c['dirs'], key=lambda x: len(Path(x).parts), reverse=True):
            Path(path).rmdir()
        c['status'] = 'DELETED_DUPLICATE_ARCHIVE_RETAINED'
        save()
    receipt['status'] = 'COMPLETE'
except BaseException as exc:
    receipt.update(status='STOPPED', error=repr(exc))
    raise
finally:
    receipt.update(finished_unix_s=time.time(), disk_after=shutil.disk_usage('/root/autodl-tmp')._asdict())
    save()

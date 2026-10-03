"""Read-only gzip cache collector for the already running Qwen new-host r02 loader."""
import argparse, gzip, hashlib, json, os, re, shutil, stat, time
from pathlib import Path

RESULTS = Path('/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-r02')
MANIFEST_SHA = 'dc3710a91ad37c2f55a73d71543a761b5c9fecdf420e5988a2c6a0634d4df8a8'
BLOCK, GUARD = 256 * 1024, 8 * 2**30


class Stop(RuntimeError):
    pass


def disk_guard(paths, free=lambda p: shutil.disk_usage(p).free):
    if any(free(p) < GUARD + 1024 * 1024 for p in paths):
        raise Stop('disk_guard_8GiB_plus_1MiB_write_margin')


def collect_fd(source, name, spec, destination, *, running, guard, emit):
    partial, final = destination / (name + '.gz.partial'), destination / (name + '.gz')
    digest, count = hashlib.sha256(), 0
    try:
        guard()
        with partial.open('xb', buffering=0) as sink:
            with gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=sink, mtime=0) as packed:
                while count < spec['bytes']:
                    if not running():
                        raise Stop('loading_ended')
                    guard()
                    chunk = source.read(min(BLOCK, spec['bytes'] - count))
                    if not chunk:
                        time.sleep(.1)  # A regular file at temporary EOF can still grow, even after unlink.
                        continue
                    packed.write(chunk); digest.update(chunk); count += len(chunk)
                if source.read(1) or os.fstat(source.fileno()).st_size != count or digest.hexdigest() != spec['sha256']:
                    raise ValueError('size_or_sha256_mismatch')
            sink.flush(); os.fsync(sink.fileno())
        guard()
        os.rename(partial, final)  # Same private directory; no other collector owns these names.
        emit('CACHED', shard=name, path=str(final), bytes=count, sha256=digest.hexdigest(), gzip_bytes=final.stat().st_size)
        return final
    except (OSError, ValueError, Stop) as exc:
        emit('PARTIAL', shard=name, path=str(partial), exists=partial.exists(), bytes=count,
             received_sha256=digest.hexdigest(), reason=f'{type(exc).__name__}: {exc}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, action='append', required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--worker-pid', type=int, default=5127)
    args = parser.parse_args()
    workspace, receipt = RESULTS / 'shard_workspace', RESULTS / 'loader_receipt.jsonl'
    data = args.manifest.read_bytes()
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA or not 1 <= len(args.cache_dir) <= 2:
        raise ValueError('requires frozen r02 manifest and one or two private cache directories')
    specs = json.loads(data)['shards']
    if any(not re.fullmatch(r'model-\d{5}-of-00016\.safetensors', n) or not 0 < s['bytes']
           or not re.fullmatch(r'[0-9a-f]{64}', s['sha256']) for n, s in specs.items()):
        raise ValueError('invalid frozen shard contract')
    roots = [p.resolve() for p in args.cache_dir]
    if len(set(roots)) != len(roots) or any(p == RESULTS or p.is_relative_to(RESULTS) for p in roots):
        raise ValueError('cache must be separate from the source results')
    if not any(args.log.resolve().is_relative_to(p) for p in roots):
        raise ValueError('log must be inside a new private cache directory')
    for p in roots:
        p.mkdir(parents=True, exist_ok=False)
    log = args.log.open('x')
    def emit(event, **fields):
        log.write(json.dumps(dict(event=event, unix_s=time.time(), **fields)) + '\n'); log.flush()
    def worker_identity():
        try:
            fields = Path(f'/proc/{args.worker_pid}/stat').read_text().rsplit(')', 1)[1].split()
            return None if fields[0] == 'Z' else fields[19]
        except (OSError, IndexError):
            return None
    identity = worker_identity()
    pending, seen, cached, owned = {}, set(), [], None
    def guard():
        disk_guard(roots + [workspace])
    def running():
        nonlocal owned
        if identity is None or worker_identity() != identity:
            return False
        try:
            with receipt.open('rb') as stream: contents = stream.read(65537)
        except FileNotFoundError:
            return True
        if len(contents) > 65536: raise ValueError('unexpected oversized loader receipt')
        events = [json.loads(line) for line in contents.splitlines(keepends=True) if line.endswith(b'\n')]
        if events:
            first = events[0]
            directory = Path(first['owned_directory']).resolve()
            if (first['event'] != 'START' or first['manifest_sha256'] != MANIFEST_SHA
                    or directory.parent != workspace.resolve() or not directory.name.startswith('serial-shards-')):
                raise ValueError('receipt is outside the fixed r02 loader scope')
            owned = directory
        if any(e['event'] in ('COMPLETE', 'FAILED') for e in events):
            return False
        if owned is not None:
            for name in specs.keys() - seen:
                try:
                    fd = os.open(owned / name, os.O_RDONLY | os.O_NOFOLLOW)
                except FileNotFoundError:
                    continue
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    os.close(fd); raise ValueError('nonregular source')
                pending[name] = os.fdopen(fd, 'rb', buffering=0); seen.add(name)
                emit('OPENED', shard=name, inode=info.st_ino, device=info.st_dev, observed_size=info.st_size)
        return True
    code, reason = 0, 'loading_ended'
    try:
        emit('START', worker_pid=args.worker_pid, worker_start_ticks=identity, workspace=str(workspace),
             manifest_sha256=MANIFEST_SHA, cache_dirs=list(map(str, roots)), block_bytes=BLOCK, free_guard_bytes=GUARD)
        while running():
            guard()
            if not pending:
                time.sleep(.1); continue
            name = next(iter(pending)); source = pending.pop(name)
            try:
                destination = max(roots, key=lambda p: shutil.disk_usage(p).free)
                collect_fd(source, name, specs[name], destination, running=running, guard=guard, emit=emit)
                cached.append(name)
            finally:
                source.close()
            if len(cached) == len(specs):
                reason = 'all_manifest_shards_cached'; break
    except (OSError, ValueError, Stop) as exc:
        reason = f'{type(exc).__name__}: {exc}'; code = 0 if str(exc) == 'loading_ended' else 2
    finally:
        for source in pending.values(): source.close()
        emit('STOP', reason=reason, cached=cached, uncached=sorted(specs.keys() - set(cached)), exit_code=code)
        log.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())

"""Three small filesystem fixtures; no network, worker interaction, or GPU."""
import gzip, hashlib, json, tempfile, threading, time
from pathlib import Path
from collect_shards import BLOCK, GUARD, Stop, collect_fd, disk_guard


def main():
    root = Path(tempfile.mkdtemp(prefix='moe_shard_collector_cpu_'))
    payload = bytes(range(256)) * 2100
    spec = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    events = []
    def emit(event, **fields): events.append(dict(event=event, **fields))
    destination = root / 'cache'; destination.mkdir()
    source = root / 'slow.safetensors'; source.write_bytes(payload[:19])
    writer_errors = []
    def writer():
        try:
            with source.open('ab', buffering=0) as stream:
                time.sleep(.03); source.unlink()
                for offset in range(19, len(payload), 8192):
                    stream.write(payload[offset:offset+8192]); time.sleep(.001)
        except Exception as exc: writer_errors.append(repr(exc))
    with source.open('rb', buffering=0) as stream:
        thread = threading.Thread(target=writer); thread.start()
        good = collect_fd(stream, 'slow.safetensors', spec, destination, running=lambda: True, guard=lambda: None, emit=emit)
        thread.join()
    assert not writer_errors and not source.exists() and gzip.decompress(good.read_bytes()) == payload
    assert events[-1]['event'] == 'CACHED' and events[-1]['sha256'] == spec['sha256']
    source = root / 'wrong.safetensors'; source.write_bytes(payload)
    before = source.read_bytes(); before_stat = source.stat()
    with source.open('rb', buffering=0) as stream:
        try:
            collect_fd(stream, source.name, dict(spec, sha256='0'*64), destination, running=lambda: True, guard=lambda: None, emit=emit)
        except ValueError: pass
        else: raise AssertionError('bad digest promoted')
    assert not (destination / (source.name+'.gz')).exists()
    assert (destination / (source.name+'.gz.partial')).is_file()
    assert source.read_bytes() == before and source.stat().st_ino == before_stat.st_ino
    assert events[-1]['event'] == 'PARTIAL' and 'sha256_mismatch' in events[-1]['reason']
    source = root / 'disk.safetensors'; source.write_bytes(payload)
    before = source.read_bytes(); before_stat = source.stat(); checks = [0]
    def guard():
        checks[0] += 1
        disk_guard([destination], free=lambda _: GUARD+2**20 if checks[0] <= 2 else GUARD-1)
    with source.open('rb', buffering=0) as stream:
        try:
            collect_fd(stream, source.name, spec, destination, running=lambda: True, guard=guard, emit=emit)
        except Stop: pass
        else: raise AssertionError('disk guard did not stop')
    assert source.read_bytes() == before and source.stat().st_ino == before_stat.st_ino
    assert (destination / (source.name+'.gz.partial')).is_file() and not (destination / (source.name+'.gz')).exists()
    assert events[-1]['bytes'] == BLOCK and 'disk_guard' in events[-1]['reason']
    result = dict(status='PASS', scope='stdlib CPU filesystem fixtures only; collector not deployed',
        slow_append_after_unlink_exact_sha=True, wrong_sha_no_promotion=True, disk_guard_source_unchanged=True,
        read_block_bytes=BLOCK, minimum_free_bytes=GUARD, artifacts=str(root), events=events,
        source_sha256=hashlib.sha256(Path(__file__).with_name('collect_shards.py').read_bytes()).hexdigest())
    with Path(__file__).with_name('cpu_checks.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'events'}, indent=2))


if __name__ == '__main__':
    main()

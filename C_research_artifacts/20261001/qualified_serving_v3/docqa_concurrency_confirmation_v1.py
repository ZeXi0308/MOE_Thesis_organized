#!/usr/bin/env python3
"""One fixed reverse-order timing replication:64 then128; no parameter sweep."""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import docqa_concurrency64_launch_v1 as capped
import native_docqa128_launch_v1 as native

BASE = Path(__file__).resolve().parent
PROTOCOL = BASE / 'docqa_concurrency_confirmation_protocol_v1.json'
RECEIPT = BASE / 'docqa-concurrency-confirmation-v1.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol['driver_sha256'] == digest(__file__)
    for name, expected in protocol['files_sha256'].items():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert digest(BASE / path) == expected, name
    assert protocol['order'] == ['concurrency64', 'native128']
    assert not RECEIPT.exists()
    record = dict(status='RUNNING', start_unix_s=time.time(), driver_sha256=digest(__file__),
                  protocol_sha256=digest(PROTOCOL), order=protocol['order'], units=[])
    def save():
        tmp = RECEIPT.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2) + '\n'); tmp.replace(RECEIPT)
    save()
    try:
        for label, module, name in (
            ('concurrency64', capped, 'qwen7b-docqa-concurrency64-confirm-v1'),
            ('native128', native, 'qwen7b-native-docqa128-confirm-v1')):
            original = module.OUT
            module.OUT = BASE / name
            unit = dict(label=label, original_output_name=original.name, output_name=name,
                        status='RUNNING', start_unix_s=time.time())
            record['units'].append(unit); save()
            try:
                code = module.main()
                unit.update(exit_code=code, status='COMPLETE' if code == 0 else 'STOPPED',
                            end_unix_s=time.time())
                save()
                if code != 0:
                    raise RuntimeError(f'{label} stopped with exit code {code}; no automatic retry')
            finally:
                module.OUT = original
        record['status'] = 'COMPLETE'
    except BaseException as exc:
        record.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['end_unix_s'] = time.time(); save()


if __name__ == '__main__':
    native.qualified.install_signals()
    main()
